from __future__ import annotations
import pandas as pd


def attach_regime_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'vix' in out.columns:
        out['vix_bucket'] = pd.cut(out['vix'], bins=[-999, 18, 25, 999], labels=['LOW_VOL', 'MID_VOL', 'HIGH_VOL'])
    else:
        out['vix_bucket'] = 'UNKNOWN'
    if 'high_yield_spread' in out.columns:
        q = out['high_yield_spread'].rank(pct=True)
        out['credit_bucket'] = pd.cut(q, bins=[-0.01, 0.33, 0.66, 1.01], labels=['EASY_CREDIT', 'NORMAL_CREDIT', 'STRESS_CREDIT'])
    else:
        out['credit_bucket'] = 'UNKNOWN'
    out['v6_regime_bucket'] = out['vix_bucket'].astype(str) + '_' + out['credit_bucket'].astype(str)
    return out
