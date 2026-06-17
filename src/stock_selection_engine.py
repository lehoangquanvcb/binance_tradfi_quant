"""V8 CIO Stock Selection Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_BUCKET = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "SMH", "AMD": "SMH", "AVGO": "SMH", "INTC": "SMH", "TSM": "SMH",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "COST": "XLP", "WMT": "XLP",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF",
    "XOM": "XLE", "CVX": "XLE", "SLV": "GLD", "GLD": "GLD", "TLT": "TLT", "BTC": "BTC", "ETH": "ETH",
}


def _score(x: float, scale: float) -> float:
    if pd.isna(x):
        return 50.0
    return float(np.clip(50 + 25 * (x / max(scale, 1e-9)), 0, 100))


def build_stock_selection_v8(signals: pd.DataFrame, close_panel: pd.DataFrame, sector_rotation: pd.DataFrame | None, market_regime: pd.DataFrame | None) -> pd.DataFrame:
    if signals is None or signals.empty or close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    latest = signals.copy()
    if "symbol" not in latest.columns:
        return pd.DataFrame()

    benchmark = "SPY" if "SPY" in px.columns else (px.columns[0] if len(px.columns) else None)
    bench_ret60 = px[benchmark].pct_change(60).iloc[-1] if benchmark and len(px) > 60 else 0.0

    sector_scores = {}
    sector_actions = {}
    if sector_rotation is not None and not sector_rotation.empty:
        if {"symbol", "sector_score"}.issubset(sector_rotation.columns):
            sector_scores = dict(zip(sector_rotation["symbol"].astype(str), sector_rotation["sector_score"]))
        if {"symbol", "sector_action"}.issubset(sector_rotation.columns):
            sector_actions = dict(zip(sector_rotation["symbol"].astype(str), sector_rotation["sector_action"]))

    regime_score = 50.0
    regime = "NEUTRAL"
    if market_regime is not None and not market_regime.empty:
        row = market_regime.sort_values("date").iloc[-1]
        regime_score = float(row.get("regime_score", 50) or 50)
        regime = str(row.get("market_regime", "NEUTRAL"))

    rows = []
    for _, r in latest.iterrows():
        sym = str(r.get("symbol"))
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
        max120 = s.rolling(120, min_periods=30).max().iloc[-1] if len(s) >= 30 else s.max()
        dd120 = (s.iloc[-1] / max(max120, 1e-9) - 1) if pd.notna(max120) else 0.0
        ma200 = s.rolling(200, min_periods=60).mean().iloc[-1] if len(s) >= 60 else np.nan
        trend_ok = bool(pd.notna(ma200) and s.iloc[-1] > ma200)

        prob = float(r.get("prob_up", r.get("prob_ensemble", 0.5)) or 0.5)
        ens = float(r.get("ensemble_score", prob) or prob)
        if ens <= 1:
            ens *= 100
        else:
            ens = min(ens, 100)
        sector_bucket = SECTOR_BUCKET.get(sym, sym if sym in sector_scores else "")
        sector_score = float(sector_scores.get(sector_bucket, sector_scores.get(sym, 50.0)))
        sector_action = str(sector_actions.get(sector_bucket, sector_actions.get(sym, "NEUTRAL")))

        # V8.8 alpha overlay: reward cross-sectional momentum/defensive quality
        # features created in features.py when they are available in the latest signal row.
        cs_mom = float(r.get('cs_momentum_composite', 0.5) or 0.5)
        cs_def = float(r.get('cs_defensive_score', 0.5) or 0.5)
        rs_rank = float(r.get('rs_rank_60d', r.get('relative_strength_rank_60d', 0.5)) or 0.5)
        alpha_overlay = 10.0 * (cs_mom - 0.5) + 5.0 * (rs_rank - 0.5) + 3.0 * (cs_def - 0.5)

        macro_adj = 0.0
        if regime in {"RISK_ON", "RECOVERY"}:
            macro_adj += 5.0
        if regime in {"RISK_OFF", "CRISIS"}:
            macro_adj -= 10.0
        if sector_action in {"UNDERWEIGHT", "EXIT"}:
            macro_adj -= 8.0
        if sector_action in {"OVERWEIGHT", "NEUTRAL_PLUS"}:
            macro_adj += 5.0

        score = (
            0.20 * prob * 100
            + 0.18 * ens
            + 0.17 * _score(rs60, 0.08)
            + 0.15 * _score(ret60, 0.12)
            + 0.10 * _score(ret120, 0.20)
            + 0.10 * sector_score
            + 0.05 * regime_score
            + 0.05 * (100 - _score(vol60, 0.35))
            + macro_adj
            + alpha_overlay
        )
        if not trend_ok:
            score -= 8
        if pd.notna(dd120) and dd120 < -0.20:
            score -= 8
        score = float(np.clip(score, 0, 100))
        if score >= 75:
            decision = "TOP_BUY"
        elif score >= 65:
            decision = "BUY"
        elif score >= 45:
            decision = "HOLD"
        elif score >= 35:
            decision = "REDUCE"
        else:
            decision = "SELL"
        rows.append({
            "symbol": sym,
            "date": r.get("date"),
            "close": r.get("close", s.iloc[-1]),
            "prob_up": prob,
            "ensemble_score": ens,
            "ret_20d": ret20,
            "ret_60d": ret60,
            "ret_120d": ret120,
            "relative_strength_60d": rs60,
            "vol_60d": vol60,
            "drawdown_120d": dd120,
            "above_200dma": trend_ok,
            "sector_bucket": sector_bucket,
            "sector_score": sector_score,
            "sector_action": sector_action,
            "market_regime": regime,
            "stock_score": round(score, 2),
            "alpha_overlay": round(alpha_overlay, 2),
            "cs_momentum_composite": round(cs_mom, 4),
            "cs_defensive_score": round(cs_def, 4),
            "decision": decision,
        })
    out = pd.DataFrame(rows).sort_values("stock_score", ascending=False)
    if not out.empty:
        out["rank"] = range(1, len(out) + 1)
    return out
