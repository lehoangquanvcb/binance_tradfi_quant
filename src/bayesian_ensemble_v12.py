"""V12 Bayesian Ensemble Layer.
Applies simple Bayesian shrinkage to meta scores so noisy signals are pulled toward neutral.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_bayesian_ensemble(meta_scores: pd.DataFrame | None, regime_diag: pd.DataFrame | None = None,
                            thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    if meta_scores is None or meta_scores.empty or 'symbol' not in meta_scores.columns:
        return pd.DataFrame(columns=['symbol','bayesian_score','bayesian_signal','posterior_prob'])
    out = meta_scores.copy()
    score = pd.to_numeric(out.get('v12_meta_score', out.get('stock_score', 50)), errors='coerce').fillna(50)
    prior = 50.0
    confidence = pd.to_numeric(out.get('v12_meta_score', 50), errors='coerce').rank(pct=True).fillna(0.5)
    if regime_diag is not None and not regime_diag.empty and 'hit_rate' in regime_diag.columns:
        regime_quality = float(pd.to_numeric(regime_diag['hit_rate'], errors='coerce').mean() or 0.5)
    else:
        regime_quality = 0.5
    k = (0.35 + confidence * 0.45 + max(regime_quality - 0.5, 0) * 1.5).clip(0.25, 0.90)
    bayes = prior * (1 - k) + score * k
    out['bayesian_score'] = bayes.round(2)
    out['posterior_prob'] = (bayes / 100).clip(0.01, 0.99).round(4)
    buy_s, strong_s, reduce_s, exit_s = 58, 66, 46, 36
    if thresholds is not None and not thresholds.empty:
        r = thresholds.tail(1).iloc[0]
        buy_s = float(r.get('buy_threshold', 0.58)) * 100
        strong_s = float(r.get('strong_buy_threshold', 0.66)) * 100
        reduce_s = float(r.get('reduce_threshold', 0.46)) * 100
        exit_s = float(r.get('exit_threshold', 0.36)) * 100
    out['bayesian_signal'] = np.select(
        [bayes >= strong_s, bayes >= buy_s, bayes <= exit_s, bayes <= reduce_s],
        ['STRONG_BUY','BUY','EXIT','REDUCE'],
        default='HOLD'
    )
    return out.sort_values('bayesian_score', ascending=False).reset_index(drop=True)
