"""Multi-strategy ensemble signals."""
from __future__ import annotations
import pandas as pd

def build_ensemble_signals(df: pd.DataFrame, ml_signals: pd.DataFrame) -> pd.DataFrame:
    if ml_signals.empty:
        return ml_signals
    latest = df.sort_values('date').groupby('symbol').tail(1).copy()
    base = ml_signals.merge(latest[['symbol','ema_trend','rsi_14','ret_5d','ret_20d','vol_ratio_20','market_regime']], on='symbol', how='left')
    base['trend_score'] = (base['ema_trend'] > 0).astype(float)
    base['momentum_score'] = ((base['rsi_14'] > 55) & (base['ret_20d'] > 0)).astype(float)
    base['mean_reversion_score'] = ((base['rsi_14'] < 35) & (base['ret_5d'] < 0)).astype(float)
    base['macro_score'] = base['market_regime'].map({'RISK_ON':1.0, 'AI_TECH_MOMENTUM':0.8, 'NEUTRAL':0.5, 'HIGH_INFLATION':0.3, 'RECESSION_RISK':0.0, 'RISK_OFF':0.0}).fillna(0.5)
    base['ml_score'] = base['prob_up'].clip(0,1)
    base['ensemble_score'] = (
        0.40*base['ml_score'] + 0.20*base['trend_score'] +
        0.15*base['momentum_score'] + 0.10*base['mean_reversion_score'] + 0.15*base['macro_score']
    )
    base['ensemble_signal'] = base['ensemble_score'].map(lambda x: 'BUY' if x>=0.62 else ('SELL' if x<=0.38 else 'HOLD'))
    return base
