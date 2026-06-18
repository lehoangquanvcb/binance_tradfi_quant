"""V13 Portfolio Construction with risk-budgeted cash allocation."""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_v13_portfolio(stock_alpha: pd.DataFrame | None, market_timing: pd.DataFrame | None = None,
                        sector_rotation: pd.DataFrame | None = None, nav: float = 100000.0,
                        max_weight: float = 0.10) -> pd.DataFrame:
    if stock_alpha is None or stock_alpha.empty or "symbol" not in stock_alpha.columns:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_usd", "decision", "rationale"])
    df = stock_alpha.copy()
    df = df[df.get("v13_decision", "HOLD").isin(["STRONG_BUY", "BUY", "HOLD"])].copy()
    if df.empty:
        return pd.DataFrame([{"symbol": "CASH", "target_weight": 1.0, "target_usd": nav, "decision": "CASH_BUFFER", "rationale": "No investable V13 ideas"}])

    signal = "NEUTRAL"
    cash = 0.20
    if market_timing is not None and not market_timing.empty:
        row = market_timing.tail(1).iloc[0]
        signal = str(row.get("market_timing_signal", "NEUTRAL")).upper()
        cash = float(row.get("target_cash_weight", 0.20))
    # Avoid excessive cash when regime is constructive.
    if signal == "RISK_ON":
        cash = min(max(cash, 0.08), 0.15)
    elif signal == "RISK_OFF":
        cash = max(cash, 0.35)
    else:
        cash = min(max(cash, 0.15), 0.25)

    investable = 1.0 - cash
    score = pd.to_numeric(df.get("v13_alpha_score", 50), errors="coerce").fillna(50).clip(lower=0)
    raw = score / score.sum() if score.sum() else pd.Series(1/len(df), index=df.index)
    weights = raw * investable
    weights = weights.clip(upper=max_weight)
    if weights.sum() > 0:
        weights = weights / weights.sum() * investable
    out = df.copy()
    out["target_weight"] = weights.round(4)
    out["target_usd"] = (out["target_weight"] * nav).round(2)
    out["decision"] = out.get("v13_decision", "HOLD")
    out["rationale"] = out.apply(lambda r: f"{r.decision}; V13 alpha={float(r.v13_alpha_score):.1f}; risk={signal}", axis=1)
    keep = ["symbol", "target_weight", "target_usd", "decision", "v13_alpha_score", "rationale"]
    out = out[[c for c in keep if c in out.columns]].sort_values("target_weight", ascending=False).head(20)
    cash_row = pd.DataFrame([{"symbol": "CASH", "target_weight": round(cash, 4), "target_usd": round(cash * nav, 2), "decision": "CASH_BUFFER", "v13_alpha_score": 0.0, "rationale": f"Cash/risk control under {signal}"}])
    return pd.concat([out, cash_row], ignore_index=True)
