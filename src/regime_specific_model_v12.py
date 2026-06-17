"""V12 Regime-Specific Model Diagnostics.
Evaluates model behavior by market regime so PMs can see where signals work best.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_regime_specific_diagnostics(ds: pd.DataFrame | None, signals: pd.DataFrame | None, regimes: pd.DataFrame | None) -> pd.DataFrame:
    if ds is None or ds.empty or signals is None or signals.empty:
        return pd.DataFrame(columns=['market_regime','rows','avg_prob','hit_rate','recommended_weight'])
    d = ds.copy()
    sig = signals.copy()
    if 'symbol' not in d.columns or 'symbol' not in sig.columns:
        return pd.DataFrame()
    latest_sig = sig[['symbol','prob_up']].drop_duplicates('symbol') if 'prob_up' in sig.columns else sig[['symbol']].assign(prob_up=0.5)
    d = d.merge(latest_sig, on='symbol', how='left')
    if 'market_regime' not in d.columns:
        if regimes is not None and not regimes.empty and {'date','market_regime'}.issubset(regimes.columns):
            d = d.merge(regimes[['date','market_regime']].drop_duplicates('date'), on='date', how='left')
        else:
            d['market_regime'] = 'NEUTRAL'
    if 'target_up_1d' not in d.columns:
        d = d.sort_values(['symbol','date'])
        d['target_up_1d'] = (d.groupby('symbol')['close'].shift(-1) > d['close']).astype(float)
    d['pred_up'] = (pd.to_numeric(d['prob_up'], errors='coerce') >= 0.5).astype(float)
    grp = d.dropna(subset=['market_regime']).groupby('market_regime')
    out = grp.apply(lambda x: pd.Series({
        'rows': len(x),
        'avg_prob': pd.to_numeric(x['prob_up'], errors='coerce').mean(),
        'hit_rate': (x['pred_up'] == x['target_up_1d']).mean() if x['target_up_1d'].notna().any() else np.nan,
        'avg_return_20d': pd.to_numeric(x.get('ret_20d', 0), errors='coerce').mean() if 'ret_20d' in x.columns else 0,
    })).reset_index()
    out['recommended_weight'] = (out['hit_rate'].fillna(0.5).clip(0.45,0.65) - 0.45) / 0.20
    return out.sort_values('hit_rate', ascending=False).reset_index(drop=True)
