"""V10.5 Stop Loss Optimizer.

Builds volatility-adjusted stop-loss and take-profit levels from latest prices,
ATR/volatility proxies and dynamic position risk budgets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_market_snapshot(ds: pd.DataFrame) -> pd.DataFrame:
    if ds is None or ds.empty or "symbol" not in ds.columns:
        return pd.DataFrame()
    x = ds.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).copy()
    return x


def build_stop_loss_plan(
    ranking: pd.DataFrame,
    ds: pd.DataFrame,
    sizing: pd.DataFrame | None = None,
    default_atr_mult: float = 2.0,
    rr_ratio: float = 2.0,
) -> pd.DataFrame:
    if ranking is None or ranking.empty or "symbol" not in ranking.columns:
        return pd.DataFrame(columns=["symbol", "entry", "stop", "take_profit", "risk_pct", "risk_usd", "rr_ratio", "rationale"])
    latest = _latest_market_snapshot(ds)
    df = ranking.copy()
    if not latest.empty:
        cols = [c for c in ["symbol", "close", "atr_14", "volatility_20d", "drawdown_60d"] if c in latest.columns]
        df = df.merge(latest[cols], on="symbol", how="left", suffixes=("", "_latest"))
    entry = pd.to_numeric(df.get("close", df.get("entry", np.nan)), errors="coerce")
    if entry.isna().all() and "price" in df.columns:
        entry = pd.to_numeric(df["price"], errors="coerce")
    atr = pd.to_numeric(df.get("atr_14", np.nan), errors="coerce")
    vol = pd.to_numeric(df.get("volatility_20d", np.nan), errors="coerce")
    fallback_atr = entry * vol.fillna(0.02).clip(0.005, 0.15)
    atr = atr.fillna(fallback_atr).fillna(entry * 0.03)
    if "stock_score" in df.columns:
        score = pd.to_numeric(df["stock_score"], errors="coerce").fillna(50)
    elif "total_score" in df.columns:
        score = pd.to_numeric(df["total_score"], errors="coerce").fillna(50)
    else:
        score = pd.Series(50, index=df.index)
    atr_mult = np.where(score >= 75, default_atr_mult * 1.25, np.where(score >= 60, default_atr_mult, default_atr_mult * 0.75))
    stop = (entry - atr * atr_mult).clip(lower=0)
    take_profit = entry + (entry - stop) * rr_ratio
    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "entry": entry.round(4).values,
        "stop": stop.round(4).values,
        "take_profit": take_profit.round(4).values,
        "rr_ratio": rr_ratio,
        "atr_multiple": pd.Series(atr_mult, index=df.index).round(2).values,
        "rationale": ["Volatility-adjusted stop using ATR/realized-vol proxy"] * len(df),
    })
    if sizing is not None and not sizing.empty and "symbol" in sizing.columns:
        out = out.merge(sizing[[c for c in ["symbol", "risk_pct", "risk_usd"] if c in sizing.columns]], on="symbol", how="left")
    return out.reset_index(drop=True)
