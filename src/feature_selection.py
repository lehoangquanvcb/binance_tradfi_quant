from __future__ import annotations
import pandas as pd


def univariate_feature_power(df: pd.DataFrame, features: list[str], target: str = 'target_up_1d') -> pd.DataFrame:
    rows = []
    if target not in df:
        return pd.DataFrame()
    for f in features:
        if f not in df:
            continue
        tmp = df[[f, target]].dropna()
        if len(tmp) < 50:
            continue
        rows.append({'feature': f, 'spearman_ic': tmp[f].corr(tmp[target], method='spearman'), 'coverage': len(tmp)})
    return pd.DataFrame(rows).sort_values('spearman_ic', key=lambda s: s.abs(), ascending=False)
