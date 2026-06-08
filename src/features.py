import pandas as pd
import numpy as np


def _safe_numeric(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors='coerce')


def _zscore(x, window=60):
    m = x.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = x.rolling(window, min_periods=max(10, window // 3)).std()
    return (x - m) / sd.replace(0, np.nan)


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build V6.1 alpha features with duplicate-column protection."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
        if c in df.columns:
            df[c] = _safe_numeric(df[c])
    df = df.dropna(subset=['date', 'symbol', 'close']).sort_values(['symbol', 'date']).copy()
    g = df.groupby('symbol', group_keys=False)

    for n in [1, 2, 3, 5, 10, 20, 60, 126, 252]:
        df[f'ret_{n}d'] = g['close'].pct_change(n)

    for w in [10, 20, 50, 100, 200]:
        df[f'ema_{w}'] = g['close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
        df[f'sma_{w}'] = g['close'].transform(lambda x: x.rolling(w, min_periods=max(5, w // 5)).mean())
    df['ema_trend'] = (df['ema_20'] > df['ema_50']).astype(int)
    df['trend_stack'] = ((df['ema_20'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])).astype(int)
    df['price_vs_20dma'] = df['close'] / df['sma_20'].replace(0, np.nan) - 1
    df['price_vs_50dma'] = df['close'] / df['sma_50'].replace(0, np.nan) - 1
    df['price_vs_200dma'] = df['close'] / df['sma_200'].replace(0, np.nan) - 1

    ema12 = g['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']

    def rsi(x, n=14):
        delta = x.diff()
        up = delta.clip(lower=0).rolling(n, min_periods=n).mean()
        down = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
        rs = up / down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    df['rsi_14'] = g['close'].transform(rsi)
    df['rsi_14_z'] = g['rsi_14'].transform(lambda x: _zscore(x, 60))

    prev_close = g['close'].shift(1)
    tr = pd.concat([(df['high'] - df['low']).abs(), (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.groupby(df['symbol']).transform(lambda x: x.rolling(14, min_periods=5).mean())
    df['atr_pct'] = df['atr_14'] / df['close'].replace(0, np.nan)

    df['volatility_20d'] = g['ret_1d'].transform(lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
    df['volatility_60d'] = g['ret_1d'].transform(lambda x: x.rolling(60, min_periods=20).std() * np.sqrt(252))
    df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d'].replace(0, np.nan)
    roll_max_60 = g['close'].transform(lambda x: x.rolling(60, min_periods=10).max())
    df['drawdown_60d'] = df['close'] / roll_max_60.replace(0, np.nan) - 1
    vol_ma20 = g['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    vol_std20 = g['volume'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['vol_ratio_20'] = df['volume'] / vol_ma20.replace(0, np.nan)
    df['volume_zscore'] = (df['volume'] - vol_ma20) / vol_std20.replace(0, np.nan)
    df['dollar_volume'] = df['close'] * df['volume'].fillna(0)
    df['liquidity_rank'] = df.groupby('date')['dollar_volume'].rank(pct=True)

    df['risk_adj_momentum_20d'] = df['ret_20d'] / df['volatility_20d'].replace(0, np.nan)
    df['risk_adj_momentum_60d'] = df['ret_60d'] / df['volatility_60d'].replace(0, np.nan)
    for c in ['ret_20d', 'ret_60d', 'ret_126d', 'price_vs_200dma', 'risk_adj_momentum_20d', 'risk_adj_momentum_60d']:
        if c in df.columns:
            df[f'{c}_rank'] = df.groupby('date')[c].rank(pct=True)
    ranks = [c for c in ['ret_20d_rank', 'ret_60d_rank', 'risk_adj_momentum_20d_rank'] if c in df.columns]
    df['cross_sectional_momentum'] = df[ranks].mean(axis=1) if ranks else np.nan

    df['target_up_1d'] = (g['close'].shift(-1) > df['close']).astype(int)
    return df


def merge_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated()].copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.tz_localize(None)
    if macro_df is None or macro_df.empty:
        return out.sort_values(['symbol', 'date'])
    macro = macro_df.copy()
    macro.columns = [str(c).strip() for c in macro.columns]
    macro = macro.loc[:, ~pd.Index(macro.columns).duplicated()].copy()
    macro['date'] = pd.to_datetime(macro['date'], errors='coerce').dt.tz_localize(None)
    out = pd.merge_asof(out.sort_values('date'), macro.sort_values('date'), on='date', direction='backward')
    return out.sort_values(['symbol', 'date'])
