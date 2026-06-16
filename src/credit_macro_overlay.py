"""Credit-macro overlay for risk-on/risk-off allocation.

This module now supports both legacy uppercase columns and the newer lowercase
columns produced by src.macro_data. It no longer collapses to all-zero scores
when FRED is unavailable; Yahoo-derived VIX/yield proxies can still create a
useful risk-on/risk-off time series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get(m: pd.DataFrame, *names: str) -> pd.Series | None:
    lookup = {str(c).lower(): c for c in m.columns}
    for n in names:
        if n in m.columns:
            return pd.to_numeric(m[n], errors="coerce")
        if n.lower() in lookup:
            return pd.to_numeric(m[lookup[n.lower()]], errors="coerce")
    return None


def _rank_component(s: pd.Series, weight: float = 1.0, inverse: bool = False) -> pd.Series:
    if s is None:
        return None
    x = pd.to_numeric(s, errors="coerce").ffill()
    comp = (x.rank(pct=True) - 0.5) * 2.0
    if inverse:
        comp = -comp
    return comp.fillna(0.0) * weight


def credit_macro_score(macro: pd.DataFrame) -> pd.DataFrame:
    if macro is None or macro.empty:
        return pd.DataFrame()
    m = macro.copy().ffill()
    out = pd.DataFrame(index=m.index)

    hy = _get(m, "HY_SPREAD", "high_yield_spread", "BAMLH0A0HYM2")
    ig = _get(m, "IG_SPREAD", "investment_grade_spread", "BAMLC0A0CM")
    cpi = _get(m, "CPI_YOY", "cpi_yoy")
    us10 = _get(m, "US10Y", "us_10y_yield", "DGS10")
    us2 = _get(m, "US2Y", "us_2y_yield", "DGS2")
    vix = _get(m, "VIXCLS", "vix")
    dxy = _get(m, "DTWEXBGS", "dxy")

    # Higher score = more risk-on. Higher spreads, inflation, yields, volatility, and dollar are risk-off.
    score = pd.Series(0.0, index=m.index)
    total_w = 0.0
    components = []
    for comp, w in [
        (_rank_component(hy, weight=1.2, inverse=True), 1.2),
        (_rank_component(ig, weight=0.8, inverse=True), 0.8),
        (_rank_component(cpi, weight=0.5, inverse=True), 0.5),
        (_rank_component(vix, weight=1.0, inverse=True), 1.0),
        (_rank_component(dxy, weight=0.4, inverse=True), 0.4),
    ]:
        if comp is not None:
            components.append(comp)
            total_w += w
    if us10 is not None:
        components.append(-((us10.diff(20).fillna(0).rank(pct=True) - 0.5) * 2.0) * 0.5)
        total_w += 0.5
    if us10 is not None and us2 is not None:
        term = us10 - us2
        components.append(((term.rank(pct=True) - 0.5) * 2.0) * 0.6)
        total_w += 0.6

    if components:
        score = sum(components) / max(total_w, 1e-9)

    out["credit_macro_score"] = score.clip(-3, 3).fillna(0.0)
    out["overlay_regime"] = pd.cut(
        out["credit_macro_score"],
        bins=[-10, -0.25, 0.35, 10],
        labels=["Risk-Off", "Neutral", "Risk-On"],
    ).astype(str)
    out["equity_risk_budget_multiplier"] = out["overlay_regime"].map({"Risk-Off": 0.45, "Neutral": 0.75, "Risk-On": 1.00}).astype(float)
    return out


def apply_overlay(weights: pd.Series, multiplier: float, cash_symbol: str = "CASH") -> pd.Series:
    w = weights.copy() * float(multiplier)
    w[cash_symbol] = max(0.0, 1.0 - w.sum())
    return w / w.sum()
