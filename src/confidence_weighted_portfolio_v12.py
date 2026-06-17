"""V12 Confidence Weighted Portfolio.
Uses Bayesian ensemble score, confidence and drift-aware controls to size positions.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_confidence_weighted_portfolio(ensemble: pd.DataFrame | None, fallback_portfolio: pd.DataFrame | None = None,
                                        confidence: pd.DataFrame | None = None, nav: float = 100000.0,
                                        max_weight: float = 0.10, cash_buffer: float = 0.15) -> pd.DataFrame:
    if ensemble is None or ensemble.empty or 'symbol' not in ensemble.columns:
        if fallback_portfolio is not None and not fallback_portfolio.empty:
            return fallback_portfolio.copy()
        return pd.DataFrame(columns=['symbol','target_weight','target_usd','decision','rationale'])
    df = ensemble.copy()
    score = pd.to_numeric(df.get('bayesian_score', df.get('v12_meta_score', 50)), errors='coerce').fillna(50)
    signal = df.get('bayesian_signal', df.get('v12_signal', 'HOLD')).astype(str)
    eligible = df[signal.isin(['STRONG_BUY','BUY','HOLD'])].copy()
    if eligible.empty:
        return pd.DataFrame([{'symbol':'CASH','target_weight':1.0,'target_usd':nav,'decision':'DEFENSIVE_CASH','rationale':'No eligible assets'}])
    score_e = pd.to_numeric(eligible.get('bayesian_score', eligible.get('v12_meta_score', 50)), errors='coerce').fillna(50)
    raw = (score_e - 45).clip(lower=0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=eligible.index)
    invested = 1.0 - cash_buffer
    # If confidence is low, raise cash.
    if confidence is not None and not confidence.empty:
        conf = float(confidence.tail(1).iloc[0].get('confidence_score', 50) or 50)
        if conf < 55:
            invested = min(invested, 0.65)
        elif conf < 70:
            invested = min(invested, 0.80)
    weights = raw / raw.sum() * invested
    weights = weights.clip(upper=max_weight)
    if weights.sum() > invested:
        weights = weights / weights.sum() * invested
    out = eligible[['symbol']].copy()
    out['target_weight'] = weights.round(4)
    out['target_usd'] = (out['target_weight'] * nav).round(2)
    out['decision'] = signal.loc[eligible.index].values
    out['bayesian_score'] = score_e.round(2).values
    out['rationale'] = out.apply(lambda r: f"{r.decision}; Bayesian score {r.bayesian_score:.1f}; confidence-weighted sizing", axis=1)
    cash = max(0.0, 1.0 - float(out['target_weight'].sum()))
    if cash > 0.001:
        out = pd.concat([out, pd.DataFrame([{'symbol':'CASH','target_weight':round(cash,4),'target_usd':round(cash*nav,2),'decision':'CASH_BUFFER','bayesian_score':0,'rationale':'Residual cash / risk control'}])], ignore_index=True)
    return out.sort_values('target_weight', ascending=False).reset_index(drop=True)
