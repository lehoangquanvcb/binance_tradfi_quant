"""V10.5 lightweight ensemble voting overlay.

This module does not retrain heavy models on Streamlit Cloud. It blends existing
ML probability with ranking, sector and regime evidence to create a stable CIO
vote for sizing and ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_cio_ensemble_vote(stock_selection: pd.DataFrame, alpha_attribution: pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty or "symbol" not in stock_selection.columns:
        return pd.DataFrame(columns=["symbol", "cio_vote_score", "cio_vote", "vote_rationale"])
    df = stock_selection.copy()
    base = pd.to_numeric(df.get("prob_up", df.get("stock_score", 50)), errors="coerce").fillna(0.5)
    if base.max() <= 1.5:
        ml = base * 100
    else:
        ml = base
    score = pd.to_numeric(df.get("stock_score", ml), errors="coerce").fillna(50)
    sector = pd.Series(50.0, index=df.index)
    if "sector_action" in df.columns:
        sector = df["sector_action"].astype(str).str.upper().map({"OVERWEIGHT": 85, "NEUTRAL_PLUS": 65, "MARKET_WEIGHT": 55, "NEUTRAL": 50, "UNDERWEIGHT": 35, "EXIT": 15}).fillna(50)
    attr = pd.Series(50.0, index=df.index)
    if alpha_attribution is not None and not alpha_attribution.empty and "symbol" in alpha_attribution.columns:
        amap = alpha_attribution.set_index("symbol").get("total_score")
        if amap is not None:
            attr = df["symbol"].map(amap).fillna(50)
    vote_score = (0.35*ml + 0.30*score + 0.20*sector + 0.15*attr).clip(0, 100)
    def label(x):
        if x >= 75: return "STRONG_BUY"
        if x >= 62: return "BUY"
        if x >= 50: return "HOLD"
        if x >= 40: return "REDUCE"
        return "EXIT"
    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "cio_vote_score": vote_score.round(2).values,
        "cio_vote": [label(x) for x in vote_score],
        "vote_rationale": [f"ML={m:.1f}; score={s:.1f}; sector={sec:.1f}; attribution={a:.1f}" for m, s, sec, a in zip(ml, score, sector, attr)],
    })
    return out.sort_values("cio_vote_score", ascending=False).reset_index(drop=True)
