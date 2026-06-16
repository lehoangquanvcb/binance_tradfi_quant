"""V8 CIO Exit Watchlist Engine."""
from __future__ import annotations

import pandas as pd


def build_exit_watchlist_v8(stock_selection: pd.DataFrame, close_panel: pd.DataFrame, market_regime: pd.DataFrame | None = None, sector_rotation: pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty or close_panel is None or close_panel.empty:
        return pd.DataFrame()
    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    regime = "NEUTRAL"
    if market_regime is not None and not market_regime.empty and "market_regime" in market_regime.columns:
        regime = str(market_regime.sort_values("date").iloc[-1]["market_regime"])
    weak_sectors = set()
    if sector_rotation is not None and not sector_rotation.empty and {"symbol", "sector_action"}.issubset(sector_rotation.columns):
        weak_sectors = set(sector_rotation.loc[sector_rotation["sector_action"].isin(["UNDERWEIGHT", "EXIT"]), "symbol"].astype(str))

    rows = []
    for _, r in stock_selection.iterrows():
        sym = str(r.get("symbol"))
        if sym not in px.columns:
            continue
        reasons = []
        score = float(r.get("stock_score", 50) or 50)
        decision = str(r.get("decision", "HOLD"))
        if decision in {"SELL", "REDUCE"}:
            reasons.append(f"Model decision: {decision}")
        try:
            if float(r.get("relative_strength_60d", 0) or 0) < -0.05:
                reasons.append("Weak relative strength vs benchmark")
        except Exception:
            pass
        try:
            if float(r.get("drawdown_120d", 0) or 0) < -0.20:
                reasons.append("Drawdown worse than -20%")
        except Exception:
            pass
        if not bool(r.get("above_200dma", True)):
            reasons.append("Price below 200DMA")
        bucket = str(r.get("sector_bucket", ""))
        if bucket in weak_sectors:
            reasons.append(f"Weak sector bucket: {bucket}")
        if regime in {"RISK_OFF", "CRISIS"} and score < 60:
            reasons.append(f"Market regime is {regime}")
        if reasons:
            if len(reasons) >= 3 or "CRISIS" in "; ".join(reasons):
                exit_action = "IMMEDIATE_EXIT"
                severity = "High"
            elif len(reasons) == 2:
                exit_action = "REDUCE"
                severity = "Medium"
            else:
                exit_action = "MONITOR"
                severity = "Low"
            rows.append({
                "symbol": sym,
                "close": r.get("close"),
                "stock_score": score,
                "exit_action": exit_action,
                "severity": severity,
                "reasons": "; ".join(dict.fromkeys(reasons)),
            })
    order = {"High": 0, "Medium": 1, "Low": 2}
    out = pd.DataFrame(rows)
    if not out.empty:
        out["severity_order"] = out["severity"].map(order)
        out = out.sort_values(["severity_order", "stock_score"], ascending=[True, True]).drop(columns=["severity_order"])
    return out
