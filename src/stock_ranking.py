"""Stock Ranking Engine V6.2."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _scale_0_100(x: pd.Series, center: float = 0.0, scale: float = 1.0) -> pd.Series:
    return (50 + 20 * ((pd.to_numeric(x, errors="coerce") - center) / max(scale, 1e-9))).clip(0, 100).fillna(50)


def build_stock_ranking(signals: pd.DataFrame, close_panel: pd.DataFrame, sector_rotation: pd.DataFrame | None = None, market_timing: pd.DataFrame | None = None) -> pd.DataFrame:
    if signals is None or signals.empty or close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    latest = signals.copy()
    if "symbol" not in latest.columns:
        return pd.DataFrame()
    rows = []
    bench = "SPY" if "SPY" in px.columns else (px.columns[0] if len(px.columns) else None)
    bench_ret60 = px[bench].pct_change(60).iloc[-1] if bench and len(px) > 60 else 0.0
    market_score = 50.0
    market_regime = "Neutral"
    if market_timing is not None and not market_timing.empty:
        mt = market_timing.sort_values("date").iloc[-1]
        market_score = float(mt.get("market_timing_score", 50) or 50)
        market_regime = str(mt.get("timing_regime", "Neutral"))
    sector_scores = {}
    if sector_rotation is not None and not sector_rotation.empty and {"symbol", "sector_score"}.issubset(sector_rotation.columns):
        sector_scores = dict(zip(sector_rotation["symbol"], sector_rotation["sector_score"]))
    for _, r in latest.iterrows():
        sym = str(r["symbol"])
        if sym not in px.columns:
            continue
        s = pd.to_numeric(px[sym], errors="coerce").ffill()
        if s.dropna().empty:
            continue
        ret20 = s.pct_change(20).iloc[-1] if len(s) > 20 else np.nan
        ret60 = s.pct_change(60).iloc[-1] if len(s) > 60 else np.nan
        ret120 = s.pct_change(120).iloc[-1] if len(s) > 120 else np.nan
        rs60 = (ret60 - bench_ret60) if pd.notna(ret60) else np.nan
        vol60 = s.pct_change().rolling(60, min_periods=20).std().iloc[-1] * np.sqrt(252) if len(s) > 20 else np.nan
        ma200 = s.rolling(200, min_periods=60).mean().iloc[-1] if len(s) >= 60 else np.nan
        dd_120 = (s.iloc[-1] / s.rolling(120, min_periods=30).max().iloc[-1] - 1) if len(s) >= 30 else 0.0
        prob = float(r.get("prob_up", r.get("prob_ensemble", 0.5)) or 0.5)
        ensemble_score = float(r.get("ensemble_score", prob) or prob)
        sector_score = float(sector_scores.get(sym, 50.0))
        score = (
            0.25 * (prob * 100)
            + 0.20 * (ensemble_score * 100 if ensemble_score <= 1 else ensemble_score)
            + 0.15 * float(_scale_0_100(pd.Series([rs60]), 0, 0.08).iloc[0])
            + 0.15 * float(_scale_0_100(pd.Series([ret60]), 0, 0.12).iloc[0])
            + 0.10 * sector_score
            + 0.10 * market_score
            + 0.05 * (100 - float(_scale_0_100(pd.Series([vol60]), 0.35, 0.20).iloc[0]))
        )
        if pd.notna(ma200) and s.iloc[-1] < ma200:
            score -= 10
        if pd.notna(dd_120) and dd_120 < -0.20:
            score -= 10
        score = float(np.clip(score, 0, 100))
        action = "BUY" if score >= 70 else ("EXIT" if score < 40 else "HOLD")
        if market_regime == "Risk-Off" and score < 55:
            action = "EXIT"
        rows.append({
            "symbol": sym,
            "date": r.get("date"),
            "close": r.get("close", s.iloc[-1]),
            "prob_up": prob,
            "ensemble_score": ensemble_score,
            "ret_20d": ret20,
            "ret_60d": ret60,
            "ret_120d": ret120,
            "relative_strength_60d": rs60,
            "drawdown_120d": dd_120,
            "vol_60d": vol60,
            "sector_score": sector_score,
            "market_timing_score": market_score,
            "stock_score": round(score, 2),
            "action": action,
        })
    out = pd.DataFrame(rows).sort_values("stock_score", ascending=False)
    if not out.empty:
        out["rank"] = range(1, len(out) + 1)
    return out
