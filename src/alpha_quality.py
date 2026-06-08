from __future__ import annotations
import numpy as np
import pandas as pd


def compute_alpha_quality(dataset: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or signals.empty:
        return pd.DataFrame()
    nxt = dataset.sort_values(['symbol', 'date']).copy()
    nxt['next_ret'] = nxt.groupby('symbol')['close'].shift(-1) / nxt['close'] - 1
    latest = nxt.sort_values('date').groupby('symbol').tail(1)[['symbol', 'next_ret']]
    s = signals.merge(latest, on='symbol', how='left')
    if 'prob_up' not in s:
        return pd.DataFrame()
    s['rank_prob'] = s['prob_up'].rank(pct=True)
    ic = s[['rank_prob', 'next_ret']].corr(method='spearman').iloc[0, 1] if s['next_ret'].notna().sum() > 2 else np.nan
    buys = s[s.get('ensemble_signal', s.get('signal', 'HOLD')) == 'BUY'] if 'signal' in s else s.iloc[0:0]
    return pd.DataFrame([{
        'rank_ic_proxy': float(ic) if pd.notna(ic) else np.nan,
        'avg_prob_up': float(s['prob_up'].mean()),
        'buy_count': int(len(buys)),
        'avg_net_edge_bps': float(s['net_edge_bps'].mean()) if 'net_edge_bps' in s else np.nan,
        'positive_net_edge_count': int((s['net_edge_bps'] > 0).sum()) if 'net_edge_bps' in s else np.nan,
    }])
