"""V6.2 Macro Feature Engine.

Adds macro-cycle features from the merged macro columns already present in the
model dataset. The functions are defensive: if a column is missing, the feature
is created with 0 so Streamlit Cloud will not crash when FRED data/API keys are
unavailable.
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


def add_macro_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    us10y = _safe_series(out, ["us_10y_yield", "US10Y", "DGS10", "us10y"])
    us2y = _safe_series(out, ["us_2y_yield", "US2Y", "DGS2", "us2y"])
    fed = _safe_series(out, ["fed_funds_rate", "FEDFUNDS", "fed_funds"])
    cpi = _safe_series(out, ["cpi_index", "CPIAUCSL", "cpi"])
    unemp = _safe_series(out, ["unemployment_rate", "UNRATE", "unemployment"])
    indpro = _safe_series(out, ["industrial_production", "INDPRO"])
    m2 = _safe_series(out, ["m2_money_supply", "M2SL", "m2"])

    out["yield_curve_10y2y"] = us10y - us2y
    out["us10y_change_20d"] = us10y.diff(20).fillna(0)
    out["us2y_change_20d"] = us2y.diff(20).fillna(0)
    out["fed_change_60d"] = fed.diff(60).fillna(0)

    out["cpi_mom"] = cpi.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["cpi_trend_6m"] = cpi.pct_change(6).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["unemployment_change_3m"] = unemp.diff(3).fillna(0)
    out["indpro_growth_6m"] = indpro.pct_change(6).replace([np.inf, -np.inf], np.nan).fillna(0)
    out["m2_growth_6m"] = m2.pct_change(6).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Simple macro risk score: higher means more hostile for equities.
    out["macro_risk_score_v62"] = (
        (-out["yield_curve_10y2y"]).rank(pct=True).fillna(0.5)
        + out["us10y_change_20d"].rank(pct=True).fillna(0.5)
        + out["fed_change_60d"].rank(pct=True).fillna(0.5)
        + out["unemployment_change_3m"].rank(pct=True).fillna(0.5)
        - out["indpro_growth_6m"].rank(pct=True).fillna(0.5)
    ) / 5.0

    return out
