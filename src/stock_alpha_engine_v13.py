"""V13 Stock Alpha Engine: clean alpha score focused on investable decisions."""
from __future__ import annotations
import pandas as pd
import numpy as np


def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(50)


def build_v13_stock_alpha(stock_selection: pd.DataFrame | None, bayesian: pd.DataFrame | None = None,
                          sector_rotation: pd.DataFrame | None = None, market_timing: pd.DataFrame | None = None) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty:
        return pd.DataFrame(columns=["symbol", "v13_alpha_score", "v13_decision"])
    df = stock_selection.copy()
    if "symbol" not in df.columns:
        return pd.DataFrame()

    score = 0.35 * _norm(df.get("stock_score", df.get("score", df.get("prob_up", 0.5))))
    score += 0.25 * _norm(df.get("prob_up", df.get("ensemble_score", 0.5)))
    score += 0.20 * _norm(df.get("relative_strength_60d", df.get("momentum_3m", 50)))

    if bayesian is not None and not bayesian.empty and "symbol" in bayesian.columns:
        bcol = "v12_meta_score" if "v12_meta_score" in bayesian.columns else ("bayesian_score" if "bayesian_score" in bayesian.columns else None)
        if bcol:
            bmap = bayesian.set_index("symbol")[bcol].to_dict()
            score += 0.20 * _norm(df["symbol"].map(bmap).fillna(50))
        else:
            score += 10
    else:
        score += 10

    # Sector boost/penalty.
    sector_bonus = pd.Series(0.0, index=df.index)
    if sector_rotation is not None and not sector_rotation.empty:
        key = "symbol" if "symbol" in sector_rotation.columns else "sector"
        if key in sector_rotation.columns and "v13_sector_score" in sector_rotation.columns:
            smap = sector_rotation.set_index(key)["v13_sector_score"].to_dict()
            if "sector_bucket" in df.columns:
                sector_bonus = (_norm(df["sector_bucket"].map(smap).fillna(50)) - 50) * 0.10
            else:
                sector_bonus = (_norm(df["symbol"].map(smap).fillna(50)) - 50) * 0.10
    score = (score + sector_bonus).clip(0, 100)

    risk_signal = "NEUTRAL"
    if market_timing is not None and not market_timing.empty:
        risk_signal = str(market_timing.tail(1).iloc[0].get("market_timing_signal", "NEUTRAL")).upper()
    buy_thr = 68 if risk_signal == "RISK_ON" else 72
    reduce_thr = 48
    exit_thr = 38

    out = df.copy()
    out["v13_alpha_score"] = score.round(2)
    out["v13_decision"] = "HOLD"
    out.loc[out["v13_alpha_score"] >= buy_thr, "v13_decision"] = "BUY"
    out.loc[out["v13_alpha_score"] >= buy_thr + 8, "v13_decision"] = "STRONG_BUY"
    out.loc[out["v13_alpha_score"] < reduce_thr, "v13_decision"] = "REDUCE"
    out.loc[out["v13_alpha_score"] < exit_thr, "v13_decision"] = "EXIT"
    out["rank"] = out["v13_alpha_score"].rank(ascending=False, method="first").astype(int)
    out["score_explanation"] = out.apply(lambda r: f"Alpha {r.v13_alpha_score:.1f}; decision={r.v13_decision}; risk={risk_signal}", axis=1)
    return out.sort_values("v13_alpha_score", ascending=False).reset_index(drop=True)
