"""V11.5 Adaptive Retraining Engine.
Decides whether a model should be retrained based on drift, monitoring status and validation metrics.
"""
from __future__ import annotations
import pandas as pd


def build_adaptive_retraining_plan(monitoring: dict | pd.DataFrame | None, metrics: dict | None = None, readiness: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = metrics or {}
    if isinstance(monitoring, pd.DataFrame) and not monitoring.empty:
        m = monitoring.iloc[-1].to_dict()
    elif isinstance(monitoring, dict):
        m = monitoring
    else:
        m = {}
    drift = float(m.get("drift_score", 0.0) or 0.0)
    auc = float(metrics.get("auc", m.get("auc", 0.0)) or 0.0)
    acc = float(metrics.get("accuracy", m.get("accuracy", 0.0)) or 0.0)
    readiness_score = None
    if readiness is not None and not readiness.empty:
        readiness_score = float(readiness.iloc[-1].get("institutional_readiness_score", 0.0) or 0.0)
    if drift >= 1.5:
        action = "MANDATORY_RETRAIN"
        urgency = "HIGH"
        reason = "Drift score above 1.5; production distribution has materially changed."
    elif drift >= 1.0:
        action = "RETRAIN_RECOMMENDED"
        urgency = "MEDIUM"
        reason = "Drift score above 1.0; refresh model before increasing risk budget."
    elif auc < 0.58 or acc < 0.55:
        action = "RETRAIN_RECOMMENDED"
        urgency = "MEDIUM"
        reason = "Predictive quality below research threshold."
    elif readiness_score is not None and readiness_score < 55:
        action = "MONITOR_AND_RECALIBRATE"
        urgency = "LOW"
        reason = "Institutional readiness is still low; monitor calibration and sizing."
    else:
        action = "NO_RETRAIN_REQUIRED"
        urgency = "LOW"
        reason = "Model metrics and drift are within monitoring range."
    return pd.DataFrame([{
        "drift_score": round(drift, 4),
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "readiness_score": readiness_score,
        "retraining_action": action,
        "urgency": urgency,
        "reason": reason,
    }])
