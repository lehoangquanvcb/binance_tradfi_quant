"""V10 Alpha Attribution Engine.
Breaks a composite stock score into interpretable components for CIO/PM review.
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


def build_alpha_attribution(stock_selection: pd.DataFrame, sector_rotation: pd.DataFrame | None = None,
                            regime: dict | pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty:
        return pd.DataFrame(columns=['symbol','total_score','ml_component','momentum_component','sector_component','risk_component','regime_component','explanation'])
    df = stock_selection.copy()
    if 'symbol' not in df.columns:
        return pd.DataFrame()
    base = pd.to_numeric(df.get('stock_score', df.get('score', df.get('prob_up', 0.5))), errors='coerce').fillna(50)
    if base.max() <= 1.5:
        base = base * 100
    ml = _norm(pd.to_numeric(df.get('prob_up', base/100), errors='coerce')) * 0.25
    mom = _norm(pd.to_numeric(df.get('momentum_3m', df.get('relative_strength_60d', base)), errors='coerce')) * 0.25
    risk = (100 - _norm(pd.to_numeric(df.get('volatility_20d', df.get('drawdown_60d', 50)), errors='coerce'))) * 0.20
    sector = pd.Series(50.0, index=df.index)
    if sector_rotation is not None and not sector_rotation.empty:
        key = 'symbol' if 'symbol' in sector_rotation.columns else None
        if key and 'sector_score' in sector_rotation.columns:
            smap = sector_rotation.set_index(key)['sector_score'].to_dict()
            sector = df['sector_bucket'].map(smap).fillna(50) if 'sector_bucket' in df.columns else sector
    sector = _norm(sector) * 0.20
    regime_bonus = 10.0
    if isinstance(regime, dict):
        r = str(regime.get('regime', regime.get('market_regime','NEUTRAL'))).upper()
        regime_bonus = 12 if 'RISK_ON' in r else (7 if 'NEUTRAL' in r else 3)
    total = (ml + mom + risk + sector + regime_bonus).clip(0, 100)
    out = pd.DataFrame({
        'symbol': df['symbol'],
        'total_score': total.round(2),
        'ml_component': ml.round(2),
        'momentum_component': mom.round(2),
        'sector_component': sector.round(2),
        'risk_component': risk.round(2),
        'regime_component': regime_bonus,
    })
    out['explanation'] = out.apply(lambda r: f"ML {r.ml_component:.1f} + Momentum {r.momentum_component:.1f} + Sector {r.sector_component:.1f} + Risk {r.risk_component:.1f} + Regime {r.regime_component:.1f}", axis=1)
    return out.sort_values('total_score', ascending=False).reset_index(drop=True)
