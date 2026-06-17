"""V10 Sector Allocation Recommendation Engine."""
from __future__ import annotations
import pandas as pd


def build_sector_allocation(sector_rotation: pd.DataFrame, equity_budget: float = 0.9) -> pd.DataFrame:
    if sector_rotation is None or sector_rotation.empty:
        return pd.DataFrame(columns=['sector','sector_score','action','target_weight'])
    df = sector_rotation.copy()
    sector_col = 'sector' if 'sector' in df.columns else ('symbol' if 'symbol' in df.columns else df.columns[0])
    score_col = 'sector_score' if 'sector_score' in df.columns else ('score' if 'score' in df.columns else None)
    if score_col is None:
        df['sector_score'] = 50
        score_col = 'sector_score'
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce').fillna(50)
    df = df.sort_values(score_col, ascending=False).copy()
    def action(x):
        if x >= 70: return 'OVERWEIGHT'
        if x >= 55: return 'MARKET_WEIGHT'
        if x >= 40: return 'UNDERWEIGHT'
        return 'EXIT_AVOID'
    df['action'] = df[score_col].apply(action)
    raw = df[score_col].clip(lower=0)
    raw = raw / raw.sum() * equity_budget if raw.sum() else raw
    raw[df['action'].eq('EXIT_AVOID')] = 0
    if raw.sum() > 0:
        raw = raw / raw.sum() * equity_budget
    return pd.DataFrame({'sector': df[sector_col].values, 'sector_score': df[score_col].round(2).values, 'action': df['action'].values, 'target_weight': raw.round(4).values})
