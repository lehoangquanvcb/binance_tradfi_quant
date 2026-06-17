"""V9.0 regime probability layer."""
from __future__ import annotations

import pandas as pd


def build_regime_probabilities(regime: pd.DataFrame | None) -> pd.DataFrame:
    if regime is None or regime.empty:
        return pd.DataFrame([{"risk_on_prob": 0.33, "neutral_prob": 0.34, "risk_off_prob": 0.33, "dominant_regime": "NEUTRAL"}])
    latest = regime.tail(1).iloc[0]
    score = float(latest.get("regime_score", 50.0)) if "regime_score" in regime.columns else 50.0
    risk_on = max(0.0, min(1.0, (score - 35.0) / 55.0))
    risk_off = max(0.0, min(1.0, (65.0 - score) / 55.0))
    neutral = max(0.0, 1.0 - abs(score - 50.0) / 50.0)
    total = risk_on + risk_off + neutral
    if total <= 0:
        risk_on, neutral, risk_off, total = 0.33, 0.34, 0.33, 1.0
    risk_on, neutral, risk_off = risk_on / total, neutral / total, risk_off / total
    dominant = "RISK_ON" if risk_on >= neutral and risk_on >= risk_off else ("RISK_OFF" if risk_off >= neutral else "NEUTRAL")
    return pd.DataFrame([{
        "date": latest.get("date", None),
        "risk_on_prob": round(risk_on, 4),
        "neutral_prob": round(neutral, 4),
        "risk_off_prob": round(risk_off, 4),
        "dominant_regime": dominant,
        "regime_score": round(score, 2),
    }])
