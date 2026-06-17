"""V12 Meta Model Overlay.
Combines legacy stock ranking, alpha attribution and V11.5 ensemble into a single CIO score.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(50)


def _symbol_df(df: pd.DataFrame | None, cols=None) -> pd.DataFrame:
    if df is None or df.empty or 'symbol' not in df.columns:
        return pd.DataFrame(columns=['symbol'])
    keep = ['symbol'] + [c for c in (cols or []) if c in df.columns]
    return df[keep].drop_duplicates('symbol')


def build_meta_model_overlay(v115_ensemble: pd.DataFrame | None, alpha_attr: pd.DataFrame | None,
                             stock_selection: pd.DataFrame | None, thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    base = _symbol_df(stock_selection, ['stock_score','prob_up','decision','sector_bucket','sector_action','market_regime'])
    if base.empty:
        base = _symbol_df(v115_ensemble, ['v115_score','ensemble_score','prob_up','decision','sector_bucket'])
    if base.empty:
        return pd.DataFrame(columns=['symbol','v12_meta_score','v12_signal','confidence_bucket'])

    ens = _symbol_df(v115_ensemble, ['v115_score','ensemble_score','cio_vote_score','v115_signal','decision'])
    alp = _symbol_df(alpha_attr, ['total_score','ml_component','momentum_component','sector_component','risk_component','regime_component'])
    out = base.merge(ens, on='symbol', how='left').merge(alp, on='symbol', how='left')

    ml_proxy = out['prob_up'] * 100 if 'prob_up' in out.columns else out.get('stock_score', 50)
    components = []
    weights = []
    for col, w in [('stock_score',0.25), ('total_score',0.25), ('v115_score',0.20), ('ensemble_score',0.15), ('cio_vote_score',0.15)]:
        if col in out.columns:
            components.append(_norm(out[col]) * w); weights.append(w)
    if not components:
        components = [_norm(pd.to_numeric(ml_proxy, errors='coerce'))]; weights=[1.0]
    score = sum(components) / max(sum(weights), 1e-9)

    buy_th = 0.58
    strong_th = 0.66
    reduce_th = 0.46
    exit_th = 0.36
    if thresholds is not None and not thresholds.empty:
        r = thresholds.tail(1).iloc[0]
        buy_th = float(r.get('buy_threshold', buy_th))
        strong_th = float(r.get('strong_buy_threshold', strong_th))
        reduce_th = float(r.get('reduce_threshold', reduce_th))
        exit_th = float(r.get('exit_threshold', exit_th))
    # Convert thresholds from probability to 0-100 score.
    buy_s, strong_s, reduce_s, exit_s = buy_th*100, strong_th*100, reduce_th*100, exit_th*100
    out['v12_meta_score'] = score.round(2)
    out['v12_signal'] = np.select(
        [score >= strong_s, score >= buy_s, score <= exit_s, score <= reduce_s],
        ['STRONG_BUY', 'BUY', 'EXIT', 'REDUCE'],
        default='HOLD'
    )
    out['confidence_bucket'] = pd.cut(score, bins=[-1,45,55,65,75,101], labels=['LOW','WATCH','MEDIUM','HIGH','VERY_HIGH']).astype(str)
    out['score_explanation'] = out.apply(lambda r: f"Meta {r.v12_meta_score:.1f}; signal={r.v12_signal}; threshold BUY={buy_s:.1f}", axis=1)
    return out.sort_values('v12_meta_score', ascending=False).reset_index(drop=True)
