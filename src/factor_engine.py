"""Factor exposure engine for US equity/ETF symbols."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factor_scores(price_panel: pd.DataFrame, benchmark: str | None = None) -> pd.DataFrame:
    """Return per-symbol factor scores: beta, momentum, low_vol, quality_proxy, trend."""
    px = price_panel.copy().ffill().dropna(how="all")
    rets = px.pct_change().dropna(how="all").fillna(0)
    if benchmark and benchmark in rets.columns:
        mkt = rets[benchmark]
    else:
        mkt = rets.mean(axis=1)
    rows = []
    for sym in rets.columns:
        r = rets[sym]
        beta = np.cov(r, mkt)[0, 1] / max(np.var(mkt), 1e-12)
        mom_63 = px[sym].pct_change(63).iloc[-1] if len(px) > 63 else px[sym].pct_change().sum()
        mom_252 = px[sym].pct_change(252).iloc[-1] if len(px) > 252 else mom_63
        vol = r.tail(63).std() * np.sqrt(252)
        low_vol = 1 / max(vol, 1e-6)
        trend = float(px[sym].iloc[-1] / max(px[sym].rolling(200).mean().iloc[-1], 1e-9) - 1) if len(px) >= 200 else 0.0
        quality_proxy = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(252)
        rows.append({
            "symbol": sym,
            "beta": beta,
            "momentum_3m": mom_63,
            "momentum_12m": mom_252,
            "low_vol_score": low_vol,
            "quality_proxy": quality_proxy,
            "trend_vs_200dma": trend,
        })
    return pd.DataFrame(rows).set_index("symbol")


def portfolio_factor_exposure(weights: pd.Series, factor_scores: pd.DataFrame) -> pd.Series:
    aligned = factor_scores.reindex(weights.index).fillna(0)
    w = weights.reindex(aligned.index).fillna(0)
    return aligned.mul(w, axis=0).sum()
