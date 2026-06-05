"""V5.5 Macro & Credit Intelligence Engine.

This module turns raw FRED-style macro series into forward-looking risk indicators.
It is API-neutral: it works with whichever macro columns are available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").ffill()
    mu = s.rolling(window, min_periods=max(20, window // 5)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 5)).std()
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    lookup = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
        if n.lower() in lookup:
            return pd.to_numeric(df[lookup[n.lower()]], errors="coerce")
    return None


def build_macro_credit_dashboard(macro: pd.DataFrame) -> pd.DataFrame:
    """Create recession, credit stress, inflation pressure and equity risk scores.

    Expected-but-optional columns include: US10Y, us_10y_yield, us_2y_yield,
    high_yield_spread, HY_SPREAD, IG_SPREAD, CPI_YOY, fed_funds_rate, vix, dxy.
    """
    if macro is None or macro.empty:
        return pd.DataFrame()
    m = macro.copy()
    if "date" in m.columns:
        m["date"] = pd.to_datetime(m["date"])
        m = m.sort_values("date").set_index("date")
    else:
        m.index = pd.to_datetime(m.index)
    out = pd.DataFrame(index=m.index)

    us10y = _col(m, "US10Y", "us_10y_yield", "DGS10")
    us2y = _col(m, "US2Y", "us_2y_yield", "DGS2")
    hy = _col(m, "HY_SPREAD", "high_yield_spread", "BAMLH0A0HYM2")
    ig = _col(m, "IG_SPREAD", "investment_grade_spread", "BAMLC0A0CM")
    cpi = _col(m, "CPI_YOY", "cpi_yoy", "CPIAUCSL")
    fed = _col(m, "fed_funds_rate", "FEDFUNDS", "fed_funds")
    vix = _col(m, "vix", "VIXCLS")
    dxy = _col(m, "dxy", "DTWEXBGS")

    term_spread = (us10y - us2y) if us10y is not None and us2y is not None else pd.Series(0, index=m.index)
    credit_spread = hy if hy is not None else (ig if ig is not None else pd.Series(0, index=m.index))
    inflation = cpi if cpi is not None else pd.Series(0, index=m.index)
    policy = fed if fed is not None else pd.Series(0, index=m.index)
    volatility = vix if vix is not None else pd.Series(0, index=m.index)
    dollar = dxy if dxy is not None else pd.Series(0, index=m.index)

    # Higher = worse risk.
    yield_curve_risk = (-_zscore(term_spread)).clip(-3, 3)
    credit_stress = _zscore(credit_spread).clip(-3, 3)
    inflation_pressure = _zscore(inflation).clip(-3, 3)
    policy_tightness = _zscore(policy).clip(-3, 3)
    market_stress = (_zscore(volatility) * 0.7 + _zscore(dollar) * 0.3).clip(-3, 3)

    composite = (
        0.30 * yield_curve_risk
        + 0.30 * credit_stress
        + 0.15 * inflation_pressure
        + 0.15 * policy_tightness
        + 0.10 * market_stress
    ).clip(-3, 3)

    recession_probability = (1 / (1 + np.exp(-1.35 * composite))).clip(0.01, 0.99)
    equity_risk_score = ((composite + 3) / 6 * 100).clip(0, 100)
    credit_stress_score = ((credit_stress + 3) / 6 * 100).clip(0, 100)

    out["term_spread"] = term_spread.reindex(out.index).ffill().fillna(0)
    out["credit_spread_proxy"] = credit_spread.reindex(out.index).ffill().fillna(0)
    out["yield_curve_risk"] = yield_curve_risk
    out["credit_stress_score"] = credit_stress_score
    out["inflation_pressure"] = inflation_pressure
    out["policy_tightness"] = policy_tightness
    out["market_stress"] = market_stress
    out["macro_credit_composite"] = composite
    out["recession_probability_6m"] = recession_probability
    out["equity_risk_score"] = equity_risk_score
    out["risk_regime"] = pd.cut(
        out["equity_risk_score"],
        bins=[-1, 35, 60, 100],
        labels=["Risk-On", "Neutral", "Risk-Off"],
    ).astype(str)
    out["equity_budget_multiplier"] = out["risk_regime"].map({"Risk-On": 1.0, "Neutral": 0.75, "Risk-Off": 0.40}).astype(float)
    return out.reset_index().rename(columns={"index": "date"})


def latest_macro_summary(dashboard: pd.DataFrame) -> dict:
    if dashboard is None or dashboard.empty:
        return {"risk_regime": "Unknown", "recession_probability_6m": None, "equity_risk_score": None}
    row = dashboard.sort_values("date").iloc[-1].to_dict()
    return {
        "risk_regime": row.get("risk_regime", "Unknown"),
        "recession_probability_6m": float(row.get("recession_probability_6m", 0)),
        "equity_risk_score": float(row.get("equity_risk_score", 0)),
        "credit_stress_score": float(row.get("credit_stress_score", 0)),
        "equity_budget_multiplier": float(row.get("equity_budget_multiplier", 0.75)),
    }
