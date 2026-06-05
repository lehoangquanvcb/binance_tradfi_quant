"""Credit-macro overlay for risk-on/risk-off allocation."""
from __future__ import annotations

import pandas as pd


def credit_macro_score(macro: pd.DataFrame) -> pd.DataFrame:
    m = macro.copy().ffill()
    out = pd.DataFrame(index=m.index)
    # Expected columns if available: fed_funds, us10y, cpi_yoy, hy_spread, ig_spread, dxy
    score = pd.Series(0.0, index=m.index)
    if "HY_SPREAD" in m:
        score -= (m["HY_SPREAD"].rank(pct=True) - 0.5) * 2
    if "IG_SPREAD" in m:
        score -= (m["IG_SPREAD"].rank(pct=True) - 0.5)
    if "CPI_YOY" in m:
        score -= (m["CPI_YOY"].rank(pct=True) - 0.5) * 0.7
    if "US10Y" in m:
        score -= (m["US10Y"].diff(20).fillna(0).rank(pct=True) - 0.5) * 0.5
    out["credit_macro_score"] = score.clip(-3, 3)
    out["overlay_regime"] = pd.cut(out["credit_macro_score"], bins=[-10,-0.75,0.75,10], labels=["Risk-Off", "Neutral", "Risk-On"])
    out["equity_risk_budget_multiplier"] = out["overlay_regime"].map({"Risk-Off":0.45,"Neutral":0.75,"Risk-On":1.00}).astype(float)
    return out


def apply_overlay(weights: pd.Series, multiplier: float, cash_symbol: str = "CASH") -> pd.Series:
    w = weights.copy() * float(multiplier)
    w[cash_symbol] = max(0.0, 1.0 - w.sum())
    return w / w.sum()
