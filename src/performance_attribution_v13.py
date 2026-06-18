"""V13 Performance Attribution."""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_v13_performance_attribution(portfolio: pd.DataFrame | None, returns_panel: pd.DataFrame | None) -> pd.DataFrame:
    if portfolio is None or portfolio.empty or returns_panel is None or returns_panel.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "recent_return_20d", "contribution_20d", "attribution_bucket"])
    if "symbol" not in portfolio.columns or "target_weight" not in portfolio.columns:
        return pd.DataFrame()
    rets = returns_panel.copy()
    if isinstance(rets, pd.Series):
        rets = rets.to_frame()
    recent = rets.tail(20).add(1).prod() - 1
    rows = []
    for _, r in portfolio.iterrows():
        sym = str(r.get("symbol"))
        if sym == "CASH":
            rr = 0.0
        else:
            rr = float(recent.get(sym, 0.0)) if sym in recent.index else 0.0
        w = float(r.get("target_weight", 0.0))
        contrib = w * rr
        rows.append({
            "symbol": sym,
            "target_weight": round(w, 4),
            "recent_return_20d": round(rr, 4),
            "contribution_20d": round(contrib, 4),
            "attribution_bucket": "Positive" if contrib > 0 else ("Negative" if contrib < 0 else "Neutral"),
        })
    return pd.DataFrame(rows).sort_values("contribution_20d", ascending=False).reset_index(drop=True)
