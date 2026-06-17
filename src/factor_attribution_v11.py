"""V11 Portfolio Factor Attribution."""
from __future__ import annotations
import pandas as pd
import numpy as np

def build_factor_attribution(portfolio: pd.DataFrame, dataset: pd.DataFrame, factor_scores: pd.DataFrame | None=None) -> pd.DataFrame:
    if portfolio is None or portfolio.empty or dataset is None or dataset.empty:
        return pd.DataFrame(columns=["factor","portfolio_exposure","contribution","interpretation"])
    port=portfolio.copy()
    if "symbol" not in port.columns:
        return pd.DataFrame()
    if "target_weight" not in port.columns:
        port["target_weight"] = 1/len(port)
    latest=dataset.sort_values("date").groupby("symbol").tail(1).copy() if "date" in dataset.columns else dataset.groupby("symbol").tail(1).copy()
    merged=port[["symbol","target_weight"]].merge(latest, on="symbol", how="left")
    factors=["ret_20d","ret_60d","rsi_14","volatility_20d","drawdown_60d","relative_strength_60d","price_vs_200dma"]
    rows=[]
    for f in factors:
        if f in merged.columns:
            vals=pd.to_numeric(merged[f], errors="coerce").fillna(0)
            w=pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0)
            exposure=float((vals*w).sum()) if w.sum()!=0 else 0.0
            interp="Positive exposure" if exposure>0 else ("Negative exposure" if exposure<0 else "Neutral exposure")
            rows.append({"factor":f,"portfolio_exposure":exposure,"contribution":abs(exposure),"interpretation":interp})
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)
