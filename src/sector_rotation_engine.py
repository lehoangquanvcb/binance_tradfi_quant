"""V8 CIO Sector Rotation Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services",
    "SMH": "Semiconductors", "QQQ": "Nasdaq 100", "SPY": "S&P 500", "DIA": "Dow Industrials", "IWM": "Small Caps",
}


def _score(x: float, scale: float) -> float:
    if pd.isna(x):
        return 50.0
    return float(np.clip(50 + 25 * (x / max(scale, 1e-9)), 0, 100))


def build_sector_rotation_v8(close_panel: pd.DataFrame, market_regime: pd.DataFrame | None = None, benchmark: str = "SPY") -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    sectors = [s for s in SECTOR_MAP if s in px.columns]
    if not sectors:
        return pd.DataFrame()
    if benchmark not in px.columns:
        benchmark = "SPY" if "SPY" in px.columns else sectors[0]

    bench_ret_60 = px[benchmark].pct_change(60)
    latest_regime = "NEUTRAL"
    if market_regime is not None and not market_regime.empty and "market_regime" in market_regime.columns:
        latest_regime = str(market_regime.sort_values("date").iloc[-1]["market_regime"])

    rows = []
    for sym in sectors:
        s = pd.to_numeric(px[sym], errors="coerce").ffill()
        ret20 = s.pct_change(20).iloc[-1] if len(s) > 20 else np.nan
        ret60 = s.pct_change(60).iloc[-1] if len(s) > 60 else np.nan
        ret120 = s.pct_change(120).iloc[-1] if len(s) > 120 else np.nan
        rs60 = (s.pct_change(60) - bench_ret_60).iloc[-1] if len(s) > 60 else np.nan
        vol60 = s.pct_change().rolling(60, min_periods=20).std().iloc[-1] * np.sqrt(252) if len(s) > 20 else np.nan
        ma50 = s.rolling(50, min_periods=20).mean().iloc[-1] if len(s) >= 20 else np.nan
        ma200 = s.rolling(200, min_periods=60).mean().iloc[-1] if len(s) >= 60 else np.nan
        trend = 100.0 if pd.notna(ma200) and s.iloc[-1] > ma200 else 35.0
        if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
            trend = min(100.0, trend + 10)
        risk_penalty = 0.0
        if latest_regime in {"RISK_OFF", "CRISIS"} and sym in {"XLK", "XLY", "SMH", "IWM", "QQQ"}:
            risk_penalty = 8.0
        if latest_regime in {"RISK_ON", "RECOVERY"} and sym in {"XLU", "XLP"}:
            risk_penalty = 6.0
        sector_score = (
            0.22 * _score(ret20, 0.06)
            + 0.28 * _score(ret60, 0.12)
            + 0.18 * _score(ret120, 0.20)
            + 0.20 * _score(rs60, 0.08)
            + 0.07 * (100 - _score(vol60, 0.35))
            + 0.05 * trend
            - risk_penalty
        )
        sector_score = float(np.clip(sector_score, 0, 100))
        if sector_score >= 72:
            action = "OVERWEIGHT"
        elif sector_score >= 55:
            action = "NEUTRAL_PLUS"
        elif sector_score >= 40:
            action = "NEUTRAL"
        elif sector_score >= 28:
            action = "UNDERWEIGHT"
        else:
            action = "EXIT"
        rows.append({
            "symbol": sym,
            "sector": SECTOR_MAP.get(sym, sym),
            "ret_20d": ret20,
            "ret_60d": ret60,
            "ret_120d": ret120,
            "relative_strength_60d": rs60,
            "vol_60d": vol60,
            "trend_score": trend,
            "sector_score": round(sector_score, 2),
            "sector_action": action,
            "market_regime": latest_regime,
        })
    out = pd.DataFrame(rows).sort_values("sector_score", ascending=False)
    out["sector_rank"] = range(1, len(out) + 1)
    return out
