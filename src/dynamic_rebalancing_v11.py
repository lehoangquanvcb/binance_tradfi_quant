"""V11 Dynamic Rebalancing Engine."""
from __future__ import annotations
import pandas as pd
import numpy as np

def _readiness_score(readiness) -> float:
    try:
        if isinstance(readiness, pd.DataFrame) and not readiness.empty:
            return float(readiness.tail(1).iloc[0].get("institutional_readiness_score", 50))
        if isinstance(readiness, dict):
            return float(readiness.get("institutional_readiness_score", 50))
    except Exception:
        pass
    return 50.0

def build_dynamic_rebalance_plan(portfolio: pd.DataFrame, readiness=None, confidence=None, regime: dict | None=None, min_trade_weight: float=0.01) -> pd.DataFrame:
    if portfolio is None or portfolio.empty:
        return pd.DataFrame(columns=["symbol","current_weight","target_weight","trade_weight","rebalance_action","priority","reason"])
    df=portfolio.copy()
    if "symbol" not in df.columns:
        return pd.DataFrame()
    target_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
    if target_col is None:
        df["target_weight"] = 1/len(df)
    else:
        df["target_weight"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    # No broker positions are available; use equal/current proxy and make it explicit.
    df["current_weight"] = 1.0/len(df)
    readiness_score=_readiness_score(readiness)
    regime_name=str((regime or {}).get("market_regime", (regime or {}).get("regime", "NEUTRAL"))).upper()
    risk_scale = 0.70 if readiness_score < 60 else (0.85 if readiness_score < 75 else 1.0)
    if "RISK_OFF" in regime_name or "CRISIS" in regime_name:
        risk_scale *= 0.70
    df["target_weight"] = (df["target_weight"] * risk_scale).clip(lower=0)
    if df["target_weight"].sum() > 0:
        df["target_weight"] = df["target_weight"] / df["target_weight"].sum() * min(risk_scale, 1.0)
    df["trade_weight"] = df["target_weight"] - df["current_weight"]
    def act(x):
        if x > min_trade_weight: return "BUY/ADD"
        if x < -min_trade_weight: return "SELL/REDUCE"
        return "HOLD"
    df["rebalance_action"] = df["trade_weight"].apply(act)
    df["priority"] = df["trade_weight"].abs().rank(ascending=False, method="first").astype(int)
    df["reason"] = "V11 rebalance: target minus current, scaled by readiness/regime risk budget"
    return df[["symbol","current_weight","target_weight","trade_weight","rebalance_action","priority","reason"]].sort_values("priority").reset_index(drop=True)
