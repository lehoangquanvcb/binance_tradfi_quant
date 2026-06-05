import numpy as np
import pandas as pd


def optimize_portfolio(signals: pd.DataFrame, risk: pd.DataFrame, nav: float = 100000.0,
                       max_weight: float = 0.12, min_prob: float = 0.60) -> pd.DataFrame:
    """Simple robust optimizer without scipy dependency.

    Score = probability edge / volatility. Weights are capped and normalized.
    This is safer than unconstrained max-Sharpe for small/noisy samples.
    """
    if signals.empty:
        return pd.DataFrame()
    df = signals.merge(risk[['symbol','vol_annual','var_95_1d','max_drawdown']], on='symbol', how='left')
    df = df[df['prob_up'] >= min_prob].copy()
    if df.empty:
        return pd.DataFrame(columns=['symbol','prob_up','target_weight','target_notional'])
    df['edge'] = (df['prob_up'] - 0.50).clip(lower=0)
    df['risk_denom'] = df['vol_annual'].replace(0, np.nan).fillna(df['vol_annual'].median())
    df['score'] = df['edge'] / df['risk_denom'].clip(lower=0.05)
    df = df.sort_values('score', ascending=False)
    df['raw_weight'] = df['score'] / df['score'].sum()
    df['target_weight'] = df['raw_weight'].clip(upper=max_weight)
    # redistribute after caps, still not exceeding max_weight materially
    if df['target_weight'].sum() > 0:
        df['target_weight'] = df['target_weight'] / df['target_weight'].sum()
        df['target_weight'] = df['target_weight'].clip(upper=max_weight)
        df['target_weight'] = df['target_weight'] / df['target_weight'].sum()
    df['target_notional'] = df['target_weight'] * nav
    return df[['symbol','prob_up','close','vol_annual','var_95_1d','max_drawdown','target_weight','target_notional','score']]
