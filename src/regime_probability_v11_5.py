"""V11.5 Regime Probability Forecast."""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_regime_probability_forecast(market_regime: pd.DataFrame | None, macro_credit: pd.DataFrame | None = None) -> pd.DataFrame:
    score = 50.0
    regime = "NEUTRAL"
    if market_regime is not None and not market_regime.empty:
        row = market_regime.iloc[-1]
        score = float(row.get("regime_score", row.get("market_score", 50.0)) or 50.0)
        regime = str(row.get("market_regime", row.get("regime", "NEUTRAL"))).upper()
    macro_penalty = 0.0
    if macro_credit is not None and not macro_credit.empty:
        mrow = macro_credit.iloc[-1]
        macro_penalty = float(mrow.get("recession_probability_6m", 0.0) or 0.0) * 20
    adj = max(0, min(100, score - macro_penalty))
    base_on = adj / 100
    base_off = max(0.0, (55 - adj) / 55)
    rows = []
    for h, damp in [("1M", 0.85), ("3M", 0.70), ("6M", 0.55)]:
        risk_on = min(0.90, max(0.05, base_on * damp + 0.10))
        risk_off = min(0.80, max(0.05, base_off * (1.05 - damp) + 0.08))
        neutral = max(0.05, 1.0 - risk_on - risk_off)
        total = risk_on + neutral + risk_off
        rows.append({
            "horizon": h,
            "current_regime": regime,
            "risk_on_prob": round(risk_on / total, 4),
            "neutral_prob": round(neutral / total, 4),
            "risk_off_prob": round(risk_off / total, 4),
            "regime_score": round(adj, 2),
        })
    return pd.DataFrame(rows)
