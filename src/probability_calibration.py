"""V10 Probability Calibration Engine."""
from __future__ import annotations
import numpy as np
import pandas as pd


def calibration_report(df: pd.DataFrame, prob_col='prob_up', target_col='target_up_1d', bins=10) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty or prob_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(), {'brier_score': None, 'calibration_error': None, 'status': 'NO_DATA'}
    x = df[[prob_col, target_col]].dropna().copy()
    x[prob_col] = pd.to_numeric(x[prob_col], errors='coerce').clip(0,1)
    x[target_col] = pd.to_numeric(x[target_col], errors='coerce').clip(0,1)
    x = x.dropna()
    if x.empty:
        return pd.DataFrame(), {'brier_score': None, 'calibration_error': None, 'status': 'NO_DATA'}
    x['bucket'] = pd.cut(x[prob_col], bins=np.linspace(0,1,bins+1), include_lowest=True)
    rep = x.groupby('bucket', observed=False).agg(
        avg_pred=(prob_col,'mean'), realized=(target_col,'mean'), count=(target_col,'size')
    ).reset_index()
    rep['abs_error'] = (rep['avg_pred'] - rep['realized']).abs()
    brier = float(((x[prob_col] - x[target_col]) ** 2).mean())
    ece = float((rep['abs_error'] * rep['count'] / rep['count'].sum()).sum()) if rep['count'].sum() else None
    status = 'OK' if ece is not None and ece < 0.05 else ('WATCH' if ece is not None and ece < 0.10 else 'ALERT')
    return rep, {'brier_score': brier, 'calibration_error': ece, 'status': status}
