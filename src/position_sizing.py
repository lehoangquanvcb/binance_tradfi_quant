"""V9.0 position sizing engine for CIO portfolio recommendations."""
from __future__ import annotations

import pandas as pd


def build_position_sizing(stock_selection: pd.DataFrame, nav: float = 100000.0, risk_pct: float = 0.01) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty:
        return pd.DataFrame(columns=["symbol", "decision", "suggested_notional", "risk_usd", "position_tier", "reason"])
    df = stock_selection.copy()
    score_col = "stock_score" if "stock_score" in df.columns else ("ensemble_score" if "ensemble_score" in df.columns else None)
    if score_col is None:
        df["stock_score"] = 50.0
        score_col = "stock_score"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(50.0)
    if "decision" not in df.columns:
        df["decision"] = df[score_col].apply(lambda x: "TOP_BUY" if x >= 75 else ("BUY" if x >= 60 else ("HOLD" if x >= 45 else "REDUCE")))

    def tier(row):
        s = float(row[score_col])
        d = str(row.get("decision", "HOLD"))
        if "EXIT" in d or "SELL" in d:
            return 0.0, "No new capital", "Exit or sell signal"
        if s >= 75 or d == "TOP_BUY":
            return 0.08, "Core", "High alpha score"
        if s >= 65 or d == "BUY":
            return 0.05, "Satellite", "Positive alpha score"
        if s >= 55:
            return 0.025, "Monitor", "Moderate alpha score"
        return 0.0, "No new capital", "Insufficient alpha score"

    rows = []
    risk_usd = float(nav) * float(risk_pct)
    for _, r in df.sort_values(score_col, ascending=False).head(30).iterrows():
        weight, tier_name, reason = tier(r)
        rows.append({
            "symbol": r.get("symbol"),
            "decision": r.get("decision", "HOLD"),
            "stock_score": round(float(r[score_col]), 2),
            "suggested_weight": weight,
            "suggested_notional": round(float(nav) * weight, 2),
            "risk_usd": round(risk_usd, 2),
            "position_tier": tier_name,
            "reason": reason,
        })
    return pd.DataFrame(rows)
