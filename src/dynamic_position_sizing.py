"""V10.5 Dynamic Position Sizing Engine.

Converts model confidence, regime, volatility and drawdown risk into per-symbol
risk budgets. This is a decision-support module; it does not send orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_confidence(confidence: pd.DataFrame | dict | None) -> float:
    if confidence is None:
        return 50.0
    if isinstance(confidence, dict):
        return float(confidence.get("confidence_score", 50.0) or 50.0)
    if isinstance(confidence, pd.DataFrame) and not confidence.empty:
        row = confidence.tail(1).iloc[0]
        return float(row.get("confidence_score", 50.0) or 50.0)
    return 50.0


def _regime_multiplier(regime: pd.DataFrame | dict | None) -> float:
    r = "NEUTRAL"
    if isinstance(regime, dict):
        r = str(regime.get("market_regime", regime.get("regime", "NEUTRAL"))).upper()
    elif isinstance(regime, pd.DataFrame) and not regime.empty:
        row = regime.tail(1).iloc[0]
        r = str(row.get("market_regime", row.get("regime", "NEUTRAL"))).upper()
    if "RISK_ON" in r:
        return 1.20
    if "RECOVERY" in r:
        return 1.10
    if "RISK_OFF" in r or "CRISIS" in r:
        return 0.50
    return 0.80


def build_dynamic_position_sizing(
    ranking: pd.DataFrame,
    nav: float = 100000.0,
    confidence: pd.DataFrame | dict | None = None,
    regime: pd.DataFrame | dict | None = None,
    base_risk_pct: float = 0.01,
    max_risk_pct: float = 0.025,
) -> pd.DataFrame:
    if ranking is None or ranking.empty or "symbol" not in ranking.columns:
        return pd.DataFrame(columns=["symbol", "decision", "risk_pct", "risk_usd", "confidence_score", "rationale"])
    df = ranking.copy()
    conf = _latest_confidence(confidence)
    conf_mult = np.interp(conf, [40, 50, 60, 70, 80, 90], [0.35, 0.50, 0.75, 1.00, 1.50, 2.00])
    reg_mult = _regime_multiplier(regime)
    score_col = "total_score" if "total_score" in df.columns else ("stock_score" if "stock_score" in df.columns else "prob_up")
    df[score_col] = pd.to_numeric(df.get(score_col, 50), errors="coerce").fillna(50)
    if df[score_col].max() <= 1.5:
        df[score_col] = df[score_col] * 100
    score_mult = (df[score_col] / 70.0).clip(0.25, 1.50)
    if "volatility_20d" in df.columns:
        vol = pd.to_numeric(df["volatility_20d"], errors="coerce").fillna(df["volatility_20d"].median())
        vol_mult = (0.25 / vol.replace(0, np.nan)).clip(0.35, 1.25).fillna(0.75)
    else:
        vol_mult = 1.0
    if "decision" in df.columns:
        d = df["decision"].astype(str).str.upper()
        decision_mult = np.where(d.str.contains("TOP|STRONG|BUY"), 1.0, np.where(d.str.contains("HOLD"), 0.55, 0.0))
    else:
        decision_mult = 1.0
    risk_pct = (base_risk_pct * conf_mult * reg_mult * score_mult * vol_mult * decision_mult).clip(0, max_risk_pct)
    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "decision": df.get("decision", pd.Series(["HOLD"] * len(df), index=df.index)).values,
        "risk_pct": pd.Series(risk_pct, index=df.index).round(4).values,
        "risk_usd": (pd.Series(risk_pct, index=df.index) * float(nav)).round(2).values,
        "confidence_score": round(conf, 2),
        "rationale": [f"confidence={conf:.1f}, regime_mult={reg_mult:.2f}, score={s:.1f}" for s in df[score_col].values],
    })
    return out.sort_values("risk_pct", ascending=False).reset_index(drop=True)
