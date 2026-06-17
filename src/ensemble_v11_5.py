"""V11.5 Ensemble Voting Engine.
Creates a CIO-friendly ensemble score from ML probability, stock score, alpha attribution and regime/sector context.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(50.0)


def build_v115_ensemble(stock_selection: pd.DataFrame, alpha_attribution: pd.DataFrame | None = None, sector_strength: pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty or "symbol" not in stock_selection.columns:
        return pd.DataFrame(columns=["symbol", "v115_score", "v115_signal", "ml_score", "alpha_score", "sector_score", "risk_adjustment"])
    df = stock_selection.copy()
    base = pd.to_numeric(df.get("stock_score", df.get("score", df.get("prob_up", 0.5))), errors="coerce").fillna(0.5)
    if base.max() <= 1.5:
        base = base * 100
    ml_score = _norm(base)
    alpha_score = pd.Series(50.0, index=df.index)
    if alpha_attribution is not None and not alpha_attribution.empty and {"symbol", "total_score"}.issubset(alpha_attribution.columns):
        amap = alpha_attribution.set_index("symbol")["total_score"].to_dict()
        alpha_score = df["symbol"].map(amap).fillna(50.0)
    sector_score = pd.Series(50.0, index=df.index)
    if sector_strength is not None and not sector_strength.empty:
        key = "symbol" if "symbol" in sector_strength.columns else ("sector" if "sector" in sector_strength.columns else None)
        if key and "sector_strength_score" in sector_strength.columns:
            smap = sector_strength.set_index(key)["sector_strength_score"].to_dict()
            sector_score = df["symbol"].map(smap).fillna(sector_score)
        elif "sector_bucket" in df.columns and "sector" in sector_strength.columns and "sector_strength_score" in sector_strength.columns:
            smap = sector_strength.set_index("sector")["sector_strength_score"].to_dict()
            sector_score = df["sector_bucket"].map(smap).fillna(sector_score)
    risk_adj = pd.Series(0.0, index=df.index)
    if "drawdown_60d" in df.columns:
        risk_adj -= _norm(pd.to_numeric(df["drawdown_60d"], errors="coerce").abs()).fillna(50) * 0.08
    if "volatility_20d" in df.columns:
        risk_adj -= _norm(df["volatility_20d"]).fillna(50) * 0.05
    score = (0.45 * ml_score + 0.30 * alpha_score + 0.20 * sector_score + risk_adj).clip(0, 100)
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "v115_score": score.round(2),
        "ml_score": ml_score.round(2),
        "alpha_score": pd.to_numeric(alpha_score, errors="coerce").round(2),
        "sector_score": pd.to_numeric(sector_score, errors="coerce").round(2),
        "risk_adjustment": risk_adj.round(2),
    })
    out["v115_signal"] = pd.cut(out["v115_score"], bins=[-1, 40, 55, 70, 100], labels=["EXIT", "HOLD", "BUY", "STRONG_BUY"]).astype(str)
    return out.sort_values("v115_score", ascending=False).reset_index(drop=True)
