"""Market Timing Engine V6.2.

Creates an actionable market timing score for allocation decisions:
Risk-On / Neutral / Risk-Off plus suggested equity and cash allocation.
Works with available price and macro-credit data; falls back gracefully.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _norm_0_100(x: pd.Series, center: float = 0.0, scale: float = 1.0) -> pd.Series:
    z = (pd.to_numeric(x, errors="coerce") - center) / max(scale, 1e-9)
    return (50 + 20 * z).clip(0, 100).fillna(50)


def _last_available(panel: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(panel.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def build_market_timing(close_panel: pd.DataFrame, macro_credit: pd.DataFrame | None = None) -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    if px.empty:
        return pd.DataFrame()
    bench = _last_available(px, ["SPY", "QQQ", "DIA"])
    if bench is None:
        bench = px.columns[0]
    b = pd.to_numeric(px[bench], errors="coerce").ffill()
    ret_20 = b.pct_change(20)
    ret_60 = b.pct_change(60)
    ma_50 = b.rolling(50, min_periods=20).mean()
    ma_200 = b.rolling(200, min_periods=60).mean()
    vol_20 = b.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)

    # Breadth = share of symbols above 50DMA and 200DMA.
    ma50_panel = px.rolling(50, min_periods=20).mean()
    ma200_panel = px.rolling(200, min_periods=60).mean()
    breadth_50 = (px > ma50_panel).mean(axis=1) * 100
    breadth_200 = (px > ma200_panel).mean(axis=1) * 100

    trend_score = ((b > ma_50).astype(float) * 35 + (b > ma_200).astype(float) * 45 + (ma_50 > ma_200).astype(float) * 20).fillna(50)
    momentum_score = (_norm_0_100(ret_20, 0, 0.05) * 0.45 + _norm_0_100(ret_60, 0, 0.10) * 0.55).clip(0, 100)
    volatility_score = (100 - _norm_0_100(vol_20, 0.25, 0.15)).clip(0, 100).fillna(50)
    breadth_score = (0.45 * breadth_50 + 0.55 * breadth_200).fillna(50).clip(0, 100)

    out = pd.DataFrame(index=px.index)
    out["benchmark"] = bench
    out["trend_score"] = trend_score
    out["momentum_score"] = momentum_score
    out["volatility_score"] = volatility_score
    out["breadth_score"] = breadth_score

    if macro_credit is not None and not macro_credit.empty and "date" in macro_credit.columns:
        mc = macro_credit.copy()
        mc["date"] = pd.to_datetime(mc["date"])
        mc = mc.sort_values("date").set_index("date")
        if "equity_risk_score" in mc.columns:
            credit_score = 100 - pd.to_numeric(mc["equity_risk_score"], errors="coerce").reindex(out.index, method="ffill").fillna(50)
        else:
            credit_score = pd.Series(50, index=out.index)
    else:
        credit_score = pd.Series(50, index=out.index)
    out["credit_macro_score"] = credit_score.clip(0, 100)

    out["market_timing_score"] = (
        0.25 * out["trend_score"]
        + 0.25 * out["momentum_score"]
        + 0.20 * out["breadth_score"]
        + 0.15 * out["volatility_score"]
        + 0.15 * out["credit_macro_score"]
    ).clip(0, 100)
    out["timing_regime"] = pd.cut(
        out["market_timing_score"],
        bins=[-1, 40, 70, 101],
        labels=["Risk-Off", "Neutral", "Risk-On"],
    ).astype(str)
    out["suggested_equity_allocation"] = out["market_timing_score"].map(lambda x: 0.25 if x < 40 else (0.60 if x < 70 else 0.85))
    out["suggested_cash_allocation"] = 1.0 - out["suggested_equity_allocation"]
    return out.reset_index().rename(columns={"index": "date"})
