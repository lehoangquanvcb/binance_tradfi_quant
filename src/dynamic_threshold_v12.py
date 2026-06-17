"""V12 Dynamic Threshold Engine.
Converts static buy/sell thresholds into adaptive thresholds based on model quality,
portfolio risk, drift and confidence.
"""
from __future__ import annotations
import pandas as pd


def _float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _last(df, col, default=0.0):
    try:
        if df is None or df.empty or col not in df.columns:
            return default
        return _float(df.tail(1).iloc[0].get(col), default)
    except Exception:
        return default


def build_dynamic_threshold(metrics: dict | None = None, backtest: dict | pd.DataFrame | None = None,
                            confidence: pd.DataFrame | None = None, monitoring: dict | pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = metrics or {}
    auc = _float(metrics.get('auc'), 0.5)
    acc = _float(metrics.get('accuracy'), 0.5)
    if isinstance(backtest, pd.DataFrame):
        sharpe = _last(backtest, 'sharpe', 0.0)
        mdd = _last(backtest, 'max_drawdown', 1.0)
    else:
        backtest = backtest or {}
        sharpe = _float(backtest.get('sharpe'), 0.0)
        mdd = _float(backtest.get('max_drawdown'), 1.0)
    conf = _last(confidence, 'confidence_score', 50.0) if isinstance(confidence, pd.DataFrame) else 50.0
    drift = _last(monitoring, 'drift_score', 0.0) if isinstance(monitoring, pd.DataFrame) else _float((monitoring or {}).get('drift_score'), 0.0)

    quality = max(0.0, min(100.0, (auc - 0.50) / 0.20 * 45 + (acc - 0.50) / 0.15 * 25 + min(sharpe, 2.0) / 2.0 * 30))
    risk_penalty = max(0.0, min(25.0, max(mdd - 0.15, 0) * 100))
    drift_penalty = max(0.0, min(25.0, drift * 8.0))
    final_quality = max(0.0, min(100.0, quality + (conf - 50.0) * 0.25 - risk_penalty - drift_penalty))

    # Higher quality allows lower threshold; high drift/risk raises it.
    buy_threshold = 0.58 - (final_quality - 50.0) / 1000.0 + min(drift, 2.0) * 0.015 + max(mdd - 0.20, 0) * 0.08
    buy_threshold = max(0.52, min(0.68, buy_threshold))
    strong_buy_threshold = min(0.78, buy_threshold + 0.08)
    reduce_threshold = max(0.42, buy_threshold - 0.12)
    exit_threshold = max(0.30, reduce_threshold - 0.10)

    if drift > 1.5:
        mode = 'DRIFT_CONTROL'
    elif mdd > 0.25:
        mode = 'DRAWDOWN_CONTROL'
    elif final_quality >= 70:
        mode = 'OFFENSIVE'
    elif final_quality >= 55:
        mode = 'BALANCED'
    else:
        mode = 'DEFENSIVE'

    return pd.DataFrame([{
        'model_version': 'v12.0',
        'adaptive_mode': mode,
        'quality_score': round(final_quality, 2),
        'buy_threshold': round(buy_threshold, 4),
        'strong_buy_threshold': round(strong_buy_threshold, 4),
        'reduce_threshold': round(reduce_threshold, 4),
        'exit_threshold': round(exit_threshold, 4),
        'auc': round(auc, 4),
        'accuracy': round(acc, 4),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(mdd, 4),
        'confidence_score': round(conf, 2),
        'drift_score': round(drift, 4),
    }])
