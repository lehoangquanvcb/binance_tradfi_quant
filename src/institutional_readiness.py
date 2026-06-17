"""V10.5 Institutional Readiness Score."""
from __future__ import annotations

import pandas as pd


def _clip(x, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def build_institutional_readiness(metrics: dict, backtest: dict, confidence: pd.DataFrame | dict | None = None, monitoring: dict | None = None) -> pd.DataFrame:
    auc = float(metrics.get("auc", 0.5) or 0.5)
    acc = float(metrics.get("accuracy", 0.5) or 0.5)
    sharpe = float(backtest.get("sharpe", 0.0) or 0.0)
    mdd = float(backtest.get("max_drawdown", 1.0) or 1.0)
    conf = 50.0
    if isinstance(confidence, pd.DataFrame) and not confidence.empty:
        conf = float(confidence.tail(1).iloc[0].get("confidence_score", 50.0) or 50.0)
    elif isinstance(confidence, dict):
        conf = float(confidence.get("confidence_score", 50.0) or 50.0)
    drift = 0.0
    if isinstance(monitoring, dict):
        drift = float(monitoring.get("drift_score", 0.0) or 0.0)
    auc_score = _clip((auc - 0.50) / 0.15 * 100)
    acc_score = _clip((acc - 0.50) / 0.12 * 100)
    sharpe_score = _clip(sharpe / 1.5 * 100)
    dd_score = _clip((0.35 - mdd) / 0.25 * 100)
    conf_score = _clip(conf)
    drift_score = _clip(100 - drift * 25)
    total = 0.20*auc_score + 0.15*acc_score + 0.25*sharpe_score + 0.20*dd_score + 0.10*conf_score + 0.10*drift_score
    if total >= 85:
        label = "INSTITUTIONAL"
    elif total >= 75:
        label = "CHAMPION_READY"
    elif total >= 60:
        label = "CANDIDATE"
    elif total >= 45:
        label = "WATCH"
    else:
        label = "RESEARCH_ONLY"
    return pd.DataFrame([{
        "institutional_readiness_score": round(total, 2),
        "readiness_label": label,
        "auc_score": round(auc_score, 2),
        "accuracy_score": round(acc_score, 2),
        "sharpe_score": round(sharpe_score, 2),
        "drawdown_score": round(dd_score, 2),
        "confidence_score_component": round(conf_score, 2),
        "drift_score_component": round(drift_score, 2),
        "auc": auc,
        "accuracy": acc,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "confidence": conf,
        "drift_score": drift,
    }])
