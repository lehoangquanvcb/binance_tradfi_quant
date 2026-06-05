import pandas as pd
import numpy as np


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten possible MultiIndex columns and remove duplicate column names."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() if c[0] else str(c[-1]).lower() for c in out.columns]
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _as_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a clean numeric Series even if a duplicated column produced a DataFrame."""
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce")


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe_columns(df)
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_ta_features missing columns: {missing}. Current columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = _as_numeric_series(df, c)
    df = df.dropna(subset=["symbol", "date", "close"]).sort_values(["symbol", "date"]).copy()

    g_close = df.groupby("symbol")["close"]
    df["ret_1d"] = g_close.pct_change()
    df["ret_5d"] = g_close.pct_change(5)
    df["ret_20d"] = g_close.pct_change(20)

    for w in [10, 20, 50, 200]:
        df[f"ema_{w}"] = g_close.transform(lambda x: x.ewm(span=w, adjust=False).mean())

    df["ema_trend"] = (df["ema_20"] > df["ema_50"]).astype(int)

    def rsi(x: pd.Series, n: int = 14) -> pd.Series:
        delta = x.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / down.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    df["rsi_14"] = g_close.transform(rsi)

    prev_close = g_close.shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda x: x.rolling(14).mean())

    vol_ma20 = df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20).mean())
    df["vol_ratio_20"] = df["volume"] / vol_ma20.replace(0, np.nan)
    df["target_up_1d"] = (g_close.shift(-1) > df["close"]).astype(int)
    return df


def merge_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    out = _dedupe_columns(price_df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    if macro_df is None or macro_df.empty:
        return out.sort_values(["symbol", "date"])

    macro = macro_df.copy()
    if isinstance(macro.columns, pd.MultiIndex):
        macro.columns = [str(c[0]).lower() if c[0] else str(c[-1]).lower() for c in macro.columns]
    macro.columns = [str(c).strip().lower() for c in macro.columns]
    macro = macro.loc[:, ~macro.columns.duplicated()].copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.tz_localize(None)
    macro = macro.dropna(subset=["date"]).sort_values("date")

    out = out.dropna(subset=["date"]).sort_values("date")
    out = pd.merge_asof(out, macro, on="date", direction="backward")
    return out.sort_values(["symbol", "date"])
