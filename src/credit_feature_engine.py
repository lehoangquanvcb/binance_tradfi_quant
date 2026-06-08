"""V6.2 Credit Feature Engine.

Builds market-stress features from VIX, HY spread and related risk proxies. It
works even when FRED is unavailable by filling missing fields with neutral 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _safe_series(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    col = _first_existing(df, candidates)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").ffill().fillna(default)


def add_credit_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vix = _safe_series(out, ["vix", "VIX"])
    hy = _safe_series(out, ["high_yield_spread", "HY_SPREAD", "BAMLH0A0HYM2"])
    ig = _safe_series(out, ["ig_spread", "IG_SPREAD"])
    dxy = _safe_series(out, ["dxy", "DXY"])

    out["vix_change_5d"] = vix.diff(5).fillna(0)
    out["vix_change_20d"] = vix.diff(20).fillna(0)
    out["hy_spread_change_20d"] = hy.diff(20).fillna(0)
    out["hy_spread_change_60d"] = hy.diff(60).fillna(0)
    out["ig_spread_change_20d"] = ig.diff(20).fillna(0)
    out["dxy_change_20d"] = dxy.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0)

    out["credit_stress_score_v62"] = (
        vix.rank(pct=True).fillna(0.5)
        + hy.rank(pct=True).fillna(0.5)
        + out["hy_spread_change_20d"].rank(pct=True).fillna(0.5)
        + out["vix_change_20d"].rank(pct=True).fillna(0.5)
        + out["dxy_change_20d"].rank(pct=True).fillna(0.5)
    ) / 5.0

    out["risk_off_flag_v62"] = (out["credit_stress_score_v62"] >= 0.70).astype(int)
    return out
