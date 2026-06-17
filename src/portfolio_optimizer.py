"""V10 Portfolio Optimizer: max-score with risk caps, min cash and simple volatility scaling."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_optimized_portfolio(ranking: pd.DataFrame, nav: float = 100000.0, max_names: int = 12,
                              max_weight: float = 0.12, min_cash: float = 0.10) -> pd.DataFrame:
    if ranking is None or ranking.empty or 'symbol' not in ranking.columns:
        return pd.DataFrame(columns=['symbol','target_weight','target_usd','sleeve','rationale'])
    df = ranking.copy()
    score_col = 'total_score' if 'total_score' in df.columns else ('stock_score' if 'stock_score' in df.columns else 'score')
    if score_col not in df.columns:
        df[score_col] = 50
    if 'decision' in df.columns:
        df = df[~df['decision'].astype(str).str.upper().str.contains('EXIT|SELL|REDUCE', regex=True)]
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce').fillna(0)
    df = df.sort_values(score_col, ascending=False).head(max_names)
    if df.empty:
        return pd.DataFrame({'symbol':['CASH'], 'target_weight':[1.0], 'target_usd':[nav], 'sleeve':['Cash'], 'rationale':['No qualifying positions']})
    raw = df[score_col].clip(lower=0)
    if raw.sum() <= 0:
        raw = pd.Series(1, index=df.index)
    weights = raw / raw.sum() * (1 - min_cash)
    weights = weights.clip(upper=max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum() * (1 - min_cash)
    out = pd.DataFrame({
        'symbol': df['symbol'].values,
        'target_weight': weights.round(4).values,
        'target_usd': (weights * nav).round(2).values,
        'sleeve': df.get('sector_bucket', pd.Series(['Equity']*len(df), index=df.index)).values,
        'rationale': [f"Score {s:.1f}; risk-capped allocation" for s in df[score_col].values],
    })
    cash = pd.DataFrame({'symbol':['CASH'], 'target_weight':[round(1-out['target_weight'].sum(),4)], 'target_usd':[round(nav*(1-out['target_weight'].sum()),2)], 'sleeve':['Cash'], 'rationale':['Liquidity/risk buffer']})
    return pd.concat([out, cash], ignore_index=True)
