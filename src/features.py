import pandas as pd
import numpy as np

try:
    from .macro_feature_engine import add_macro_cycle_features
    from .credit_feature_engine import add_credit_stress_features
    from .relative_strength import add_relative_strength_features
except Exception:  # Allows standalone testing
    add_macro_cycle_features = None
    add_credit_stress_features = None
    add_relative_strength_features = None


def _as_numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = _as_numeric_series(df, c, 0.0)

    g = df.groupby("symbol", group_keys=False)

    # Returns and momentum horizons
    df["ret_1d"] = g["close"].pct_change()
    df["ret_3d"] = g["close"].pct_change(3)
    df["ret_5d"] = g["close"].pct_change(5)
    df["ret_10d"] = g["close"].pct_change(10)
    df["ret_20d"] = g["close"].pct_change(20)
    df["ret_60d"] = g["close"].pct_change(60)
    df["ret_120d"] = g["close"].pct_change(120)

    # EMAs and trend features
    for w in [10, 20, 50, 100, 200]:
        df[f"ema_{w}"] = g["close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())
    df["ema_trend"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["price_vs_200dma"] = df["close"] / df["ema_200"].replace(0, np.nan) - 1
    df["ema_20_50_spread"] = df["ema_20"] / df["ema_50"].replace(0, np.nan) - 1
    df["ema_50_200_spread"] = df["ema_50"] / df["ema_200"].replace(0, np.nan) - 1

    # RSI 14
    def rsi(x, n=14):
        delta = x.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    df["rsi_14"] = g["close"].transform(rsi)

    # ATR 14
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)

    # Volatility and drawdown
    df["volatility_20d"] = g["ret_1d"].transform(lambda x: x.rolling(20).std())
    df["volatility_60d"] = g["ret_1d"].transform(lambda x: x.rolling(60).std())
    roll_max = g["close"].transform(lambda x: x.rolling(60).max())
    df["drawdown_60d"] = df["close"] / roll_max.replace(0, np.nan) - 1

    # Volume features
    vol_mean_20 = g["volume"].transform(lambda x: x.rolling(20).mean())
    vol_std_20 = g["volume"].transform(lambda x: x.rolling(20).std())
    df["vol_ratio_20"] = df["volume"] / vol_mean_20.replace(0, np.nan)
    df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)

    # Add relative strength before macro merge. It uses all symbols in the panel.
    if add_relative_strength_features is not None:
        try:
            df = add_relative_strength_features(df, benchmark_symbol="SPY")
        except Exception as e:
            print(f"Relative strength features skipped: {e}")

    # Target
    df["target_up_1d"] = (g["close"].shift(-1) > df["close"]).astype(int)

    return df.replace([np.inf, -np.inf], np.nan)


def merge_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if macro_df is None or macro_df.empty:
        # Still create neutral macro/credit features so V6.2 model columns exist.
        if add_macro_cycle_features is not None:
            out = add_macro_cycle_features(out)
        if add_credit_stress_features is not None:
            out = add_credit_stress_features(out)
        return out.sort_values(["symbol", "date"])

    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")
    macro.columns = [str(c).strip() for c in macro.columns]
    out = pd.merge_asof(out.sort_values("date"), macro.sort_values("date"), on="date", direction="backward")
    out = out.sort_values(["symbol", "date"])

    if add_macro_cycle_features is not None:
        out = add_macro_cycle_features(out)
    if add_credit_stress_features is not None:
        out = add_credit_stress_features(out)
    return out.replace([np.inf, -np.inf], np.nan)
