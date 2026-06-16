"""Sector Rotation Engine V6.2."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services",
    "SMH": "Semiconductors", "IWM": "Small Caps", "DIA": "Dow Industrials", "QQQ": "Nasdaq 100", "SPY": "S&P 500",
}


def _score_series(s: pd.Series, scale: float) -> pd.Series:
    return (50 + 20 * (pd.to_numeric(s, errors="coerce") / max(scale, 1e-9))).clip(0, 100).fillna(50)


def build_sector_rotation(close_panel: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    sectors = [s for s in SECTOR_MAP if s in px.columns]
    if not sectors:
        return pd.DataFrame()
    if benchmark not in px.columns:
        benchmark = "SPY" if "SPY" in px.columns else sectors[0]
    bench_ret_60 = px[benchmark].pct_change(60)
    rows = []
    for sym in sectors:
        s = pd.to_numeric(px[sym], errors="coerce").ffill()
        ret_20 = s.pct_change(20).iloc[-1] if len(s) > 20 else np.nan
        ret_60 = s.pct_change(60).iloc[-1] if len(s) > 60 else np.nan
        ret_120 = s.pct_change(120).iloc[-1] if len(s) > 120 else np.nan
        rs_vs_bench = (s.pct_change(60) - bench_ret_60).iloc[-1] if len(s) > 60 else np.nan
        vol_60 = s.pct_change().rolling(60, min_periods=20).std().iloc[-1] * np.sqrt(252) if len(s) > 20 else np.nan
        ma200 = s.rolling(200, min_periods=60).mean().iloc[-1] if len(s) >= 60 else np.nan
        trend = float(s.iloc[-1] > ma200) if pd.notna(ma200) else 0.5
        score = (
            0.30 * float(_score_series(pd.Series([ret_20]), 0.06).iloc[0])
            + 0.30 * float(_score_series(pd.Series([ret_60]), 0.12).iloc[0])
            + 0.20 * float(_score_series(pd.Series([rs_vs_bench]), 0.08).iloc[0])
            + 0.10 * (100 - float(_score_series(pd.Series([vol_60]), 0.30).iloc[0]))
            + 0.10 * (100 if trend >= 1 else 45)
        )
        recommendation = "Overweight" if score >= 70 else ("Underweight" if score < 40 else "Neutral")
        if score < 30:
            recommendation = "Exit"
        rows.append({
            "symbol": sym,
            "sector": SECTOR_MAP.get(sym, sym),
            "ret_20d": ret_20,
            "ret_60d": ret_60,
            "ret_120d": ret_120,
            "relative_strength_vs_benchmark": rs_vs_bench,
            "vol_60d": vol_60,
            "above_200dma": bool(trend >= 1),
            "sector_score": round(float(score), 2),
            "recommendation": recommendation,
        })
    return pd.DataFrame(rows).sort_values("sector_score", ascending=False)
