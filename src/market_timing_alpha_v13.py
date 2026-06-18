"""V13 Institutional Alpha Engine - market timing layer.
Produces a compact risk-on / neutral / risk-off decision from existing regime,
confidence and macro-credit outputs. Defensive by design for Streamlit Cloud.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_v13_market_timing(market_regime: pd.DataFrame | None, confidence: pd.DataFrame | None = None,
                            macro_credit: pd.DataFrame | None = None, calibration_summary: dict | pd.DataFrame | None = None) -> pd.DataFrame:
    today = pd.Timestamp.utcnow().date().isoformat()
    regime = "NEUTRAL"
    regime_score = 50.0
    equity_weight = 0.60
    cash_weight = 0.20
    if market_regime is not None and not market_regime.empty:
        row = market_regime.tail(1).iloc[0]
        regime = str(row.get("market_regime", row.get("regime", "NEUTRAL"))).upper().replace(" ", "_")
        regime_score = _num(row.get("regime_score", row.get("score", 50)), 50)
        equity_weight = _num(row.get("recommended_equity_weight", 0.60), 0.60)
        cash_weight = _num(row.get("recommended_cash_weight", 0.20), 0.20)

    conf_score = 55.0
    if confidence is not None and not confidence.empty:
        conf_score = _num(confidence.tail(1).iloc[0].get("confidence_score", 55), 55)

    macro_penalty = 0.0
    if macro_credit is not None and not macro_credit.empty:
        m = macro_credit.tail(1).iloc[0]
        cm_score = _num(m.get("credit_macro_score", m.get("equity_risk_score", 0)), 0)
        macro_penalty = max(0.0, min(20.0, cm_score / 5.0))

    cal_status = "OK"
    if isinstance(calibration_summary, pd.DataFrame) and not calibration_summary.empty:
        cal_status = str(calibration_summary.tail(1).iloc[0].get("status", "OK")).upper()
    elif isinstance(calibration_summary, dict):
        cal_status = str(calibration_summary.get("status", "OK")).upper()
    cal_penalty = 8.0 if "ALERT" in cal_status else (4.0 if "WATCH" in cal_status else 0.0)

    raw_score = 0.50 * regime_score + 0.35 * conf_score + 15.0 - macro_penalty - cal_penalty
    score = float(np.clip(raw_score, 0, 100))

    if score >= 68:
        signal = "RISK_ON"
        target_equity = min(0.90, max(equity_weight, 0.75))
        target_cash = min(max(cash_weight, 0.08), 0.15)
    elif score <= 42:
        signal = "RISK_OFF"
        target_equity = min(equity_weight, 0.45)
        target_cash = max(cash_weight, 0.35)
    else:
        signal = "NEUTRAL"
        target_equity = min(max(equity_weight, 0.55), 0.70)
        target_cash = min(max(cash_weight, 0.18), 0.30)

    return pd.DataFrame([{
        "date": today,
        "market_timing_score": round(score, 2),
        "market_timing_signal": signal,
        "base_regime": regime,
        "regime_score": round(regime_score, 2),
        "confidence_score": round(conf_score, 2),
        "macro_penalty": round(macro_penalty, 2),
        "calibration_status": cal_status,
        "target_equity_weight": round(target_equity, 4),
        "target_cash_weight": round(target_cash, 4),
        "target_defensive_weight": round(max(0.0, 1.0 - target_equity - target_cash), 4),
        "decision": "DEPLOY_RISK" if signal == "RISK_ON" else ("CUT_RISK" if signal == "RISK_OFF" else "BALANCED_RISK"),
    }])
