"""Economic regime classification for V5.5."""
from __future__ import annotations
import pandas as pd


def classify_economic_regime(macro_credit: pd.DataFrame) -> pd.DataFrame:
    if macro_credit is None or macro_credit.empty:
        return pd.DataFrame()
    df = macro_credit.copy()
    rec = df.get("recession_probability_6m", pd.Series(0.5, index=df.index))
    infl = df.get("inflation_pressure", pd.Series(0.0, index=df.index))
    credit = df.get("credit_stress_score", pd.Series(50, index=df.index))
    regimes = []
    for r, i, c in zip(rec, infl, credit):
        if r >= 0.65 and c >= 65:
            regimes.append("Recession / Credit Stress")
        elif i >= 1.0 and r < 0.60:
            regimes.append("Inflation Shock")
        elif r >= 0.55:
            regimes.append("Slowdown")
        elif r <= 0.35 and c <= 45:
            regimes.append("Expansion")
        else:
            regimes.append("Recovery / Neutral")
    out = df[["date"]].copy() if "date" in df.columns else pd.DataFrame({"date": df.index})
    out["economic_regime_v55"] = regimes
    out["recession_probability_6m"] = rec.values
    out["credit_stress_score"] = credit.values
    return out
