"""Model drift and performance monitoring."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .database import log_event

def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors='coerce').dropna()
    a = pd.to_numeric(actual, errors='coerce').dropna()
    if len(e) < 30 or len(a) < 30:
        return 0.0
    quantiles = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        return 0.0
    e_counts = np.histogram(e, bins=quantiles)[0] / len(e)
    a_counts = np.histogram(a, bins=quantiles)[0] / len(a)
    e_counts = np.where(e_counts == 0, 1e-6, e_counts)
    a_counts = np.where(a_counts == 0, 1e-6, a_counts)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))

def monitor_model(df: pd.DataFrame, metrics: dict, signals: pd.DataFrame, run_id: str,
                  baseline_path: Path, features: list[str] | None = None) -> dict:
    features = features or metrics.get('features', [])
    status = 'OK'
    drift_scores = []
    current = df.sort_values('date').groupby('symbol').tail(250)

    if baseline_path.exists():
        baseline = pd.read_parquet(baseline_path)
        for f in features:
            if f in baseline.columns and f in current.columns:
                drift_scores.append(population_stability_index(baseline[f], current[f]))
    else:
        df.sort_values('date').groupby('symbol').head(250).to_parquet(baseline_path, index=False)

    drift_score = float(np.nanmean(drift_scores)) if drift_scores else 0.0
    auc = metrics.get('auc')
    acc = metrics.get('accuracy')
    avg_prob = float(signals['prob_up'].mean()) if not signals.empty else None
    hit_rate = None
    if acc is not None:
        hit_rate = float(acc)
    if (auc is not None and auc < 0.52) or (acc is not None and acc < 0.50) or drift_score > 0.25:
        status = 'WATCH'
    if (auc is not None and auc < 0.50) or drift_score > 0.50:
        status = 'ALERT'
    out = {
        'run_id': run_id, 'auc': auc, 'accuracy': acc, 'hit_rate': hit_rate,
        'avg_prob': avg_prob, 'data_rows': int(len(df)), 'drift_score': drift_score, 'status': status
    }
    log_event('MODEL_MONITORING', status, json.dumps(out, default=str))
    return out
