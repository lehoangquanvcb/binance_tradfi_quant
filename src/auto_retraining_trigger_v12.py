"""V12 Auto Retraining Trigger.
Produces a clear production action from drift, calibration and readiness diagnostics.
"""
from __future__ import annotations
import pandas as pd


def _last(df, col, default=0.0):
    try:
        if df is None or df.empty or col not in df.columns:
            return default
        v = df.tail(1).iloc[0].get(col)
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def build_auto_retraining_trigger(monitoring: pd.DataFrame | dict | None = None, readiness: pd.DataFrame | None = None,
                                  calibration_summary: dict | pd.DataFrame | None = None,
                                  thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    if isinstance(monitoring, pd.DataFrame):
        drift = _last(monitoring, 'drift_score', 0.0)
        status = str(monitoring.tail(1).iloc[0].get('status', 'OK')) if not monitoring.empty else 'OK'
    else:
        monitoring = monitoring or {}
        drift = float(monitoring.get('drift_score', 0.0) or 0.0)
        status = str(monitoring.get('status', 'OK'))
    ready = _last(readiness, 'institutional_readiness_score', 50.0)
    if isinstance(calibration_summary, pd.DataFrame):
        cal_error = _last(calibration_summary, 'calibration_error', 0.0)
        cal_status = str(calibration_summary.tail(1).iloc[0].get('status', 'OK')) if not calibration_summary.empty else 'OK'
    else:
        calibration_summary = calibration_summary or {}
        cal_error = float(calibration_summary.get('calibration_error', 0.0) or 0.0)
        cal_status = str(calibration_summary.get('status', 'OK'))
    if drift >= 1.5 or status.upper() == 'ALERT':
        action = 'MANDATORY_RETRAIN'
        priority = 'HIGH'
    elif drift >= 0.8 or cal_error >= 0.10:
        action = 'SCHEDULE_RETRAIN'
        priority = 'MEDIUM'
    elif ready < 55:
        action = 'WATCH_AND_COLLECT_DATA'
        priority = 'MEDIUM'
    else:
        action = 'NO_RETRAIN_REQUIRED'
        priority = 'LOW'
    return pd.DataFrame([{
        'model_version': 'v12.0',
        'retraining_action': action,
        'priority': priority,
        'drift_score': round(drift,4),
        'monitoring_status': status,
        'readiness_score': round(ready,2),
        'calibration_error': round(cal_error,4),
        'calibration_status': cal_status,
        'recommended_next_step': 'Retrain before live use' if action == 'MANDATORY_RETRAIN' else ('Review within next cycle' if priority == 'MEDIUM' else 'Continue monitoring')
    }])
