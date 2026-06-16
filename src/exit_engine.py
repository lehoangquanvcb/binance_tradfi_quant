"""Exit Watchlist Engine V6.2."""
from __future__ import annotations

import pandas as pd


def build_exit_watchlist(stock_ranking: pd.DataFrame, close_panel: pd.DataFrame, market_timing: pd.DataFrame | None = None, sector_rotation: pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_ranking is None or stock_ranking.empty or close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    latest_market_regime = "Neutral"
    if market_timing is not None and not market_timing.empty:
        latest_market_regime = str(market_timing.sort_values("date").iloc[-1].get("timing_regime", "Neutral"))
    underweight = set()
    if sector_rotation is not None and not sector_rotation.empty and {"symbol", "recommendation"}.issubset(sector_rotation.columns):
        underweight = set(sector_rotation.loc[sector_rotation["recommendation"].isin(["Underweight", "Exit"]), "symbol"].astype(str))
    rows = []
    for _, r in stock_ranking.iterrows():
        sym = str(r.get("symbol"))
        if sym not in px.columns:
            continue
        s = pd.to_numeric(px[sym], errors="coerce").ffill()
        if s.dropna().empty:
            continue
        reasons = []
        ma200 = s.rolling(200, min_periods=60).mean().iloc[-1] if len(s) >= 60 else None
        if ma200 is not None and pd.notna(ma200) and s.iloc[-1] < ma200:
            reasons.append("Price below 200DMA")
        dd_120 = r.get("drawdown_120d", 0)
        try:
            if float(dd_120) < -0.20:
                reasons.append("Drawdown worse than -20%")
        except Exception:
            pass
        try:
            if float(r.get("relative_strength_60d", 0)) < -0.05:
                reasons.append("Underperforming benchmark")
        except Exception:
            pass
        if sym in underweight:
            reasons.append("Sector/asset bucket underweight")
        if latest_market_regime == "Risk-Off" and float(r.get("stock_score", 50) or 50) < 60:
            reasons.append("Market timing Risk-Off")
        if str(r.get("action", "")).upper() == "EXIT":
            reasons.append("Stock ranking exit signal")
        if reasons:
            severity = "High" if len(reasons) >= 3 or "Market timing Risk-Off" in reasons else ("Medium" if len(reasons) == 2 else "Low")
            rows.append({
                "symbol": sym,
                "close": r.get("close", s.iloc[-1]),
                "stock_score": r.get("stock_score"),
                "action": "EXIT" if severity in ["High", "Medium"] else "WATCH",
                "severity": severity,
                "reasons": "; ".join(dict.fromkeys(reasons)),
            })
    return pd.DataFrame(rows).sort_values(["severity", "stock_score"], ascending=[True, True]) if rows else pd.DataFrame()
