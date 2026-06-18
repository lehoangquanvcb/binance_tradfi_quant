"""V13 Sector Rotation Alpha Engine."""
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


def build_v13_sector_rotation(sector_v8: pd.DataFrame | None = None, sector_strength: pd.DataFrame | None = None,
                              sector_alloc: pd.DataFrame | None = None, market_timing: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = []
    if sector_v8 is not None and not sector_v8.empty:
        df = sector_v8.copy()
        if "sector" not in df.columns and "symbol" in df.columns:
            df["sector"] = df["symbol"]
        df["v8_score"] = pd.to_numeric(df.get("sector_score", df.get("score", 50)), errors="coerce")
        frames.append(df[[c for c in ["symbol", "sector", "v8_score", "sector_action"] if c in df.columns]])
    if sector_strength is not None and not sector_strength.empty:
        ss = sector_strength.copy()
        if "sector" not in ss.columns and "symbol" in ss.columns:
            ss["sector"] = ss["symbol"]
        ss["strength_score"] = pd.to_numeric(ss.get("sector_strength_score", ss.get("score", 50)), errors="coerce")
        frames.append(ss[[c for c in ["symbol", "sector", "strength_score", "action"] if c in ss.columns]])
    if not frames:
        return pd.DataFrame(columns=["sector", "v13_sector_score", "v13_action", "target_sector_weight"])

    base = frames[0]
    for f in frames[1:]:
        key = "sector" if "sector" in base.columns and "sector" in f.columns else "symbol"
        base = base.merge(f, on=key, how="outer", suffixes=("", "_y"))
        if "sector_y" in base.columns and "sector" not in base.columns:
            base["sector"] = base["sector_y"]
    if "sector" not in base.columns:
        base["sector"] = base.get("symbol", pd.Series(range(len(base))))

    score_cols = [c for c in ["v8_score", "strength_score"] if c in base.columns]
    if score_cols:
        score = sum(_norm(base[c]) for c in score_cols) / len(score_cols)
    else:
        score = pd.Series(50.0, index=base.index)

    risk_on = True
    if market_timing is not None and not market_timing.empty:
        sig = str(market_timing.tail(1).iloc[0].get("market_timing_signal", "NEUTRAL"))
        risk_on = sig.upper() == "RISK_ON"
    if risk_on:
        score = (score * 1.05).clip(0, 100)

    out = base.copy()
    out["v13_sector_score"] = score.round(2)
    out["v13_action"] = pd.cut(out["v13_sector_score"], bins=[-1, 35, 55, 70, 101], labels=["UNDERWEIGHT", "NEUTRAL", "OVERWEIGHT", "STRONG_OVERWEIGHT"]).astype(str)
    pos = out["v13_sector_score"].clip(lower=0)
    if pos.sum() == 0:
        out["target_sector_weight"] = 1 / len(out)
    else:
        out["target_sector_weight"] = (pos / pos.sum()).round(4)
    cols = ["symbol", "sector", "v13_sector_score", "v13_action", "target_sector_weight"]
    return out[[c for c in cols if c in out.columns]].sort_values("v13_sector_score", ascending=False).reset_index(drop=True)
