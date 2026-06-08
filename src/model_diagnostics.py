from __future__ import annotations
import pandas as pd


def flatten_model_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for model, vals in metrics.get('model_metrics', {}).items():
        row = {'model': model}
        if isinstance(vals, dict):
            row.update(vals)
        rows.append(row)
    rows.append({
        'model': 'ensemble',
        'auc': metrics.get('auc'),
        'accuracy': metrics.get('accuracy'),
        'precision': metrics.get('precision'),
        'recall': metrics.get('recall'),
        'n_train': metrics.get('n_train'),
        'n_test': metrics.get('n_test'),
    })
    return pd.DataFrame(rows)


def signal_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty or 'signal' not in signals:
        return pd.DataFrame()
    return signals['signal'].value_counts(dropna=False).rename_axis('signal').reset_index(name='count')
