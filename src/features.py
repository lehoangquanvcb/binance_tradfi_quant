import pandas as pd
import numpy as np

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol','date']).copy()
    g = df.groupby('symbol', group_keys=False)
    df['ret_1d'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_20d'] = g['close'].pct_change(20)
    for w in [10,20,50,200]:
        df[f'ema_{w}'] = g['close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
    df['ema_trend'] = (df['ema_20'] > df['ema_50']).astype(int)
    # RSI 14
    def rsi(x, n=14):
        delta = x.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / down.replace(0, np.nan)
        return 100 - 100/(1+rs)
    df['rsi_14'] = g['close'].transform(rsi)
    # ATR 14
    prev_close = g['close'].shift(1)
    tr = pd.concat([(df['high']-df['low']).abs(), (df['high']-prev_close).abs(), (df['low']-prev_close).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.groupby(df['symbol']).transform(lambda x: x.rolling(14).mean())
    df['vol_ratio_20'] = df['volume'] / g['volume'].transform(lambda x: x.rolling(20).mean())
    df['target_up_1d'] = (g['close'].shift(-1) > df['close']).astype(int)
    return df

def merge_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out['date'] = pd.to_datetime(out['date'])
    if macro_df is None or macro_df.empty:
        return out
    macro = macro_df.copy(); macro['date'] = pd.to_datetime(macro['date'])
    out = pd.merge_asof(out.sort_values('date'), macro.sort_values('date'), on='date', direction='backward')
    return out.sort_values(['symbol','date'])
