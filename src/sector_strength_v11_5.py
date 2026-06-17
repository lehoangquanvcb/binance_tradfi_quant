"""V11.5 Sector Relative Strength Engine."""
from __future__ import annotations
import numpy as np
import pandas as pd

SECTOR_MAP = {
    "XLK": "Technology", "QQQ": "Nasdaq 100", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "Real Estate", "SPY": "S&P 500", "DIA": "Dow Industrials", "IWM": "Small Caps"
}


def _score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(50)


def build_sector_strength(close_panel: pd.DataFrame) -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame(columns=["symbol", "sector", "rs_3m", "rs_6m", "rs_12m", "sector_strength_score", "action"])
    cp = close_panel.copy().sort_index().ffill()
    benchmark = cp["SPY"] if "SPY" in cp.columns else cp.mean(axis=1)
    rows = []
    for sym in [c for c in cp.columns if c in SECTOR_MAP]:
        px = cp[sym].dropna()
        if len(px) < 80:
            continue
        def rel(days):
            if len(px) <= days or benchmark.dropna().shape[0] <= days:
                return 0.0
            return (px.iloc[-1] / px.iloc[-days] - 1) - (benchmark.iloc[-1] / benchmark.iloc[-days] - 1)
        rs3, rs6, rs12 = rel(63), rel(126), rel(252)
        raw = 0.45 * rs3 + 0.35 * rs6 + 0.20 * rs12
        rows.append({"symbol": sym, "sector": SECTOR_MAP.get(sym, sym), "rs_3m": rs3, "rs_6m": rs6, "rs_12m": rs12, "raw_strength": raw})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["sector_strength_score"] = _score(out["raw_strength"]).round(2)
    out["action"] = pd.cut(out["sector_strength_score"], bins=[-1, 35, 55, 70, 100], labels=["UNDERWEIGHT", "NEUTRAL", "OVERWEIGHT", "STRONG_OVERWEIGHT"]).astype(str)
    return out.sort_values("sector_strength_score", ascending=False).reset_index(drop=True)
