import numpy as np
import pandas as pd

def risk_metrics(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.sort_values(['symbol','date']).copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    rows = []
    for sym, g in df.groupby('symbol'):
        r = g['ret'].dropna()
        if r.empty: continue
        cum = (1+r).cumprod()
        dd = cum / cum.cummax() - 1
        rows.append({
            'symbol': sym,
            'vol_annual': r.std()*np.sqrt(252),
            'var_95_1d': r.quantile(0.05),
            'cvar_95_1d': r[r <= r.quantile(0.05)].mean(),
            'max_drawdown': dd.min(),
            'sharpe_annual': (r.mean()*252)/(r.std()*np.sqrt(252)) if r.std() else np.nan,
        })
    return pd.DataFrame(rows)

def position_size(nav: float, entry: float, atr: float, risk_pct=0.01, atr_mult=2.0):
    stop = entry - atr_mult * atr
    risk_per_unit = max(entry - stop, 1e-9)
    qty = (nav * risk_pct) / risk_per_unit
    notional = qty * entry
    return {'entry': entry, 'stop': stop, 'qty': qty, 'notional': notional, 'risk_usd': nav*risk_pct}
