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


def _rolling_zscore(s: pd.Series, window: int = 60) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(window, min_periods=max(10, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 4)).std()
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """V8.8 alpha feature engine.

    Adds stronger cross-sectional, volatility-adjusted and regime-friendly
    features while preserving the older V6/V7 feature names used elsewhere.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "close"]).sort_values(["symbol", "date"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = _as_numeric_series(df, c, 0.0)

    g = df.groupby("symbol", group_keys=False)

    # Multi-horizon returns and momentum
    for h in [1, 2, 3, 5, 10, 20, 40, 60, 120, 252]:
        df[f"ret_{h}d"] = g["close"].pct_change(h)

    # EMAs and trend features
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"ema_{w}"] = g["close"].transform(lambda x, w=w: x.ewm(span=w, adjust=False).mean())
    df["ema_trend"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["trend_stack"] = ((df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])).astype(int)
    df["price_vs_50dma"] = df["close"] / df["ema_50"].replace(0, np.nan) - 1
    df["price_vs_200dma"] = df["close"] / df["ema_200"].replace(0, np.nan) - 1
    df["ema_20_50_spread"] = df["ema_20"] / df["ema_50"].replace(0, np.nan) - 1
    df["ema_50_200_spread"] = df["ema_50"] / df["ema_200"].replace(0, np.nan) - 1
    df["momentum_accel"] = df["ret_20d"] - df["ret_60d"] / 3.0

    # RSI 14
    def rsi(x, n=14):
        delta = x.diff()
        up = delta.clip(lower=0).rolling(n, min_periods=max(5, n // 2)).mean()
        down = (-delta.clip(upper=0)).rolling(n, min_periods=max(5, n // 2)).mean()
        rs = up / down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    df["rsi_14"] = g["close"].transform(rsi)
    df["rsi_z_60"] = g["rsi_14"].transform(lambda x: _rolling_zscore(x, 60))

    # ATR 14
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)

    # Volatility, downside risk and volatility-adjusted momentum
    df["volatility_20d"] = g["ret_1d"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["volatility_60d"] = g["ret_1d"].transform(lambda x: x.rolling(60, min_periods=20).std())
    downside = df["ret_1d"].where(df["ret_1d"] < 0, 0.0)
    df["downside_vol_60d"] = downside.groupby(df["symbol"]).transform(lambda x: x.rolling(60, min_periods=20).std())
    for h in [20, 60, 120]:
        vol = df["volatility_60d"].replace(0, np.nan)
        df[f"risk_adj_mom_{h}d"] = df[f"ret_{h}d"] / vol

    # Drawdown features
    roll_max_60 = g["close"].transform(lambda x: x.rolling(60, min_periods=20).max())
    roll_max_120 = g["close"].transform(lambda x: x.rolling(120, min_periods=30).max())
    df["drawdown_60d"] = df["close"] / roll_max_60.replace(0, np.nan) - 1
    df["drawdown_120d"] = df["close"] / roll_max_120.replace(0, np.nan) - 1
    df["drawdown_recovery"] = df["drawdown_60d"] - g["drawdown_60d"].shift(10)

    # Volume and liquidity features
    vol_mean_20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    vol_std_20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=10).std())
    vol_mean_60 = g["volume"].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df["vol_ratio_20"] = df["volume"] / vol_mean_20.replace(0, np.nan)
    df["volume_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
    df["liquidity_trend_20_60"] = vol_mean_20 / vol_mean_60.replace(0, np.nan) - 1

    # Cross-sectional features by date: these are often more predictive than raw indicators.
    for col in ["ret_20d", "ret_60d", "ret_120d", "risk_adj_mom_60d", "price_vs_200dma", "volume_zscore_20", "drawdown_60d"]:
        if col in df.columns:
            rank = df.groupby("date")[col].rank(pct=True)
            df[f"cs_{col}_rank"] = rank.fillna(0.5)
    df["cs_momentum_composite"] = (
        0.35 * df.get("cs_ret_20d_rank", 0.5)
        + 0.35 * df.get("cs_ret_60d_rank", 0.5)
        + 0.20 * df.get("cs_risk_adj_mom_60d_rank", 0.5)
        + 0.10 * df.get("cs_price_vs_200dma_rank", 0.5)
    )
    df["cs_defensive_score"] = (
        0.50 * (1 - df.get("cs_drawdown_60d_rank", 0.5))
        + 0.50 * (1 - df.groupby("date")["volatility_60d"].rank(pct=True).fillna(0.5))
    )

    # Add relative strength before macro merge. It uses all symbols in the panel.
    if add_relative_strength_features is not None:
        try:
            df = add_relative_strength_features(df, benchmark_symbol="SPY")
        except Exception as e:
            print(f"Relative strength features skipped: {e}")

    # V8.8 target: next-day direction. Keep index alignment robust after merge/sort.
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["target_up_1d"] = (
        next_close.reset_index(drop=True) > df["close"].reset_index(drop=True)
    ).astype(int)

    return df.replace([np.inf, -np.inf], np.nan)


def merge_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if macro_df is None or macro_df.empty:
        if add_macro_cycle_features is not None:
            out = add_macro_cycle_features(out)
        if add_credit_stress_features is not None:
            out = add_credit_stress_features(out)
        return out.sort_values(["symbol", "date"]).replace([np.inf, -np.inf], np.nan)

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
