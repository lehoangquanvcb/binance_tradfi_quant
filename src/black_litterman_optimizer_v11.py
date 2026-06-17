"""V11 Black-Litterman-lite Portfolio Optimizer.
A robust, dependency-light approximation using alpha views and covariance shrinkage.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def build_black_litterman_portfolio(alpha: pd.DataFrame, returns_panel: pd.DataFrame | None=None, nav: float=100000.0, max_weight: float=0.12, cash_buffer: float=0.10) -> pd.DataFrame:
    if alpha is None or alpha.empty or "symbol" not in alpha.columns:
        return pd.DataFrame(columns=["symbol","target_weight","target_usd","view_score","risk_penalty","method"])
    df=alpha.copy().head(30)
    score_col="total_score" if "total_score" in df.columns else ("stock_score" if "stock_score" in df.columns else None)
    if score_col is None:
        df["view_score"]=50.0
    else:
        df["view_score"]=pd.to_numeric(df[score_col], errors="coerce").fillna(50)
    risk_penalty=pd.Series(1.0, index=df.index)
    if returns_panel is not None and not returns_panel.empty:
        vols=returns_panel.std()*np.sqrt(252)
        risk_penalty=df["symbol"].map(vols.to_dict()).fillna(vols.median() if len(vols)>0 else 0.25)
        risk_penalty=(risk_penalty/risk_penalty.median()).clip(0.5,2.0)
    raw=(df["view_score"].clip(40,100)-40)/risk_penalty
    if raw.sum()<=0:
        weights=pd.Series((1-cash_buffer)/len(df), index=df.index)
    else:
        weights=raw/raw.sum()*(1-cash_buffer)
    weights=weights.clip(0,max_weight)
    if weights.sum()>0:
        weights=weights/weights.sum()*(1-cash_buffer)
    out=pd.DataFrame({
        "symbol":df["symbol"].values,
        "target_weight":weights.round(4).values,
        "target_usd":(weights*float(nav)).round(2).values,
        "view_score":df["view_score"].round(2).values,
        "risk_penalty":risk_penalty.round(3).values,
        "method":"Black-Litterman-lite"
    })
    cash=pd.DataFrame([{"symbol":"CASH","target_weight":cash_buffer,"target_usd":float(nav)*cash_buffer,"view_score":0,"risk_penalty":0,"method":"Cash buffer"}])
    return pd.concat([out, cash], ignore_index=True)
