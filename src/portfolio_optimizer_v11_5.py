"""V11.5 Robust Portfolio Optimizer.
Provides practical max-score / risk-capped portfolio weights for Streamlit Cloud.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_v115_portfolio_optimizer(scores: pd.DataFrame, returns_panel: pd.DataFrame | None = None, nav: float = 100000.0, max_weight: float = 0.12, cash_buffer: float = 0.10) -> pd.DataFrame:
    if scores is None or scores.empty or "symbol" not in scores.columns:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_usd", "score", "risk_cap", "rationale"])
    df = scores.copy()
    score_col = "v115_score" if "v115_score" in df.columns else ("total_score" if "total_score" in df.columns else "stock_score")
    df[score_col] = pd.to_numeric(df.get(score_col, 50), errors="coerce").fillna(50)
    df = df.sort_values(score_col, ascending=False).head(15).copy()
    raw = df[score_col].clip(lower=0)
    if raw.sum() == 0:
        raw = pd.Series(1.0, index=df.index)
    weights = raw / raw.sum() * (1 - cash_buffer)
    # Volatility cap if returns available.
    if returns_panel is not None and not returns_panel.empty:
        vol = returns_panel.std() * np.sqrt(252)
        caps = df["symbol"].map(lambda s: max_weight * 0.75 if float(vol.get(s, 0.0) or 0.0) > 0.55 else max_weight)
    else:
        caps = pd.Series(max_weight, index=df.index)
    weights = np.minimum(weights, caps.astype(float))
    if weights.sum() > 0:
        weights = weights / weights.sum() * (1 - cash_buffer)
    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "target_weight": pd.Series(weights, index=df.index).round(4).values,
        "target_usd": (pd.Series(weights, index=df.index) * float(nav)).round(2).values,
        "score": df[score_col].round(2).values,
        "risk_cap": pd.Series(caps, index=df.index).round(4).values,
        "rationale": [f"V11.5 score {x:.1f}; risk-capped allocation" for x in df[score_col].values],
    })
    cash = pd.DataFrame([{"symbol": "CASH", "target_weight": round(cash_buffer, 4), "target_usd": round(float(nav) * cash_buffer, 2), "score": None, "risk_cap": None, "rationale": "Strategic liquidity buffer"}])
    return pd.concat([out, cash], ignore_index=True)
