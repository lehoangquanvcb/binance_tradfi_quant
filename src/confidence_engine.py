"""V9.0 confidence scoring for CIO dashboard."""
from __future__ import annotations

import pandas as pd


def _f(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def build_confidence_score(metrics: dict, monitoring: dict | pd.DataFrame | None, regime: pd.DataFrame | None, backtest_metrics: dict | None = None) -> pd.DataFrame:
    metrics = metrics or {}
    backtest_metrics = backtest_metrics or {}
    auc = _f(metrics.get("auc"), 0.5)
    acc = _f(metrics.get("accuracy"), 0.5)
    sharpe = _f(backtest_metrics.get("sharpe"), 0.0)
    mdd = _f(backtest_metrics.get("max_drawdown"), 0.50)

    drift = 0.0
    if isinstance(monitoring, pd.DataFrame) and not monitoring.empty:
        drift = _f(monitoring.tail(1).iloc[0].get("drift_score"), 0.0)
    elif isinstance(monitoring, dict):
        drift = _f(monitoring.get("drift_score"), 0.0)

    regime_score = 50.0
    if regime is not None and not regime.empty and "regime_score" in regime.columns:
        regime_score = _f(regime.tail(1).iloc[0].get("regime_score"), 50.0)

    predictive = max(0.0, min(100.0, (auc - 0.50) / 0.20 * 100.0)) * 0.45 + max(0.0, min(100.0, (acc - 0.50) / 0.15 * 100.0)) * 0.15
    backtest = max(0.0, min(100.0, sharpe / 1.5 * 100.0)) * 0.20 + max(0.0, min(100.0, (0.35 - mdd) / 0.35 * 100.0)) * 0.10
    stability = max(0.0, 100.0 - drift * 35.0) * 0.05 + regime_score * 0.05
    score = max(0.0, min(100.0, predictive + backtest + stability))
    label = "HIGH" if score >= 75 else ("MODERATE" if score >= 55 else ("LOW" if score >= 35 else "VERY_LOW"))
    return pd.DataFrame([{
        "confidence_score": round(score, 2),
        "confidence_label": label,
        "auc": auc,
        "accuracy": acc,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "drift_score": drift,
        "regime_score": regime_score,
    }])
