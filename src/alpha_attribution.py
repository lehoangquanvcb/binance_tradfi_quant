"""V10.5 Alpha Attribution Engine.

Breaks a composite stock score into interpretable components for CIO/PM review.
This version is defensive against missing columns, scalar fallbacks, and index
alignment issues that can occur after Streamlit/Yahoo data refreshes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(value, index: pd.Index, default: float = 0.0) -> pd.Series:
    """Return a numeric Series aligned to index.

    pandas DataFrame.get can return a scalar fallback when a column is missing.
    This helper normalizes Series/scalars/arrays into a safe numeric Series.
    """
    if isinstance(value, pd.Series):
        s = value.reindex(index)
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == len(index):
            s = pd.Series(value, index=index)
        else:
            s = pd.Series(default, index=index)
    else:
        s = pd.Series(default if value is None else value, index=index)

    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _norm(value, index: pd.Index | None = None, default: float = 50.0) -> pd.Series:
    """Robust 0-100 percentile-style normalization."""
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    else:
        if index is None:
            index = pd.RangeIndex(1)
        s = _as_series(value, index, default=default)

    if s.notna().sum() < 2:
        return pd.Series(default, index=s.index)

    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(default, index=s.index)

    return ((s.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(default)


def _sector_component(df: pd.DataFrame, sector_rotation: pd.DataFrame | None) -> pd.Series:
    sector = pd.Series(50.0, index=df.index)

    if sector_rotation is None or sector_rotation.empty:
        return sector

    sr = sector_rotation.copy()

    # Case 1: stock table has sector_bucket and sector table has sector/sector_name.
    if "sector_bucket" in df.columns and "sector_score" in sr.columns:
        sector_key = None
        for candidate in ["sector", "sector_name", "sector_bucket", "symbol"]:
            if candidate in sr.columns:
                sector_key = candidate
                break
        if sector_key:
            smap = sr.set_index(sector_key)["sector_score"].to_dict()
            mapped = df["sector_bucket"].map(smap)
            sector = pd.to_numeric(mapped, errors="coerce").fillna(50.0)
            sector.index = df.index
            return sector

    # Case 2: sector_rotation has symbol-level ETF scores and df has symbol.
    if "symbol" in df.columns and "symbol" in sr.columns and "sector_score" in sr.columns:
        smap = sr.set_index("symbol")["sector_score"].to_dict()
        mapped = df["symbol"].map(smap)
        sector = pd.to_numeric(mapped, errors="coerce").fillna(50.0)
        sector.index = df.index

    return sector


def build_alpha_attribution(
    stock_selection: pd.DataFrame,
    sector_rotation: pd.DataFrame | None = None,
    regime: dict | pd.DataFrame | None = None,
) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "total_score",
                "ml_component",
                "momentum_component",
                "sector_component",
                "risk_component",
                "regime_component",
                "explanation",
            ]
        )

    df = stock_selection.copy().reset_index(drop=True)
    if "symbol" not in df.columns:
        return pd.DataFrame()

    idx = df.index

    base = _as_series(
        df.get("stock_score", df.get("score", df.get("prob_up", 0.5))),
        idx,
        default=0.5,
    ).fillna(0.5)
    if base.max(skipna=True) <= 1.5:
        base = base * 100.0

    prob = _as_series(df.get("prob_up", base / 100.0), idx, default=0.5)
    if prob.max(skipna=True) <= 1.5:
        prob_score = prob * 100.0
    else:
        prob_score = prob

    momentum_raw = _as_series(
        df.get("momentum_3m", df.get("relative_strength_60d", base)),
        idx,
        default=50.0,
    )

    vol = _as_series(df.get("volatility_20d", 0.0), idx, default=0.0).abs()
    dd = _as_series(df.get("drawdown_60d", 0.0), idx, default=0.0).abs()

    # Lower volatility/drawdown should contribute positively.
    vol_quality = 100.0 - _norm(vol, idx)
    dd_quality = 100.0 - _norm(dd, idx)
    risk_quality = (vol_quality * 0.50 + dd_quality * 0.50).clip(0, 100)

    sector_raw = _sector_component(df, sector_rotation)

    ml = _norm(prob_score, idx) * 0.25
    mom = _norm(momentum_raw, idx) * 0.25
    sector = _norm(sector_raw, idx) * 0.20
    risk = _norm(risk_quality, idx) * 0.20

    regime_bonus = 10.0
    if isinstance(regime, dict):
        r = str(regime.get("regime", regime.get("market_regime", "NEUTRAL"))).upper()
        if "RISK_ON" in r or "RECOVERY" in r:
            regime_bonus = 12.0
        elif "NEUTRAL" in r:
            regime_bonus = 7.0
        elif "RISK_OFF" in r or "CRISIS" in r:
            regime_bonus = 3.0

    total = (ml + mom + risk + sector + regime_bonus).clip(0, 100)

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str),
            "total_score": total.round(2),
            "ml_component": ml.round(2),
            "momentum_component": mom.round(2),
            "sector_component": sector.round(2),
            "risk_component": risk.round(2),
            "regime_component": float(regime_bonus),
        }
    )

    out["explanation"] = out.apply(
        lambda r: (
            f"ML {r.ml_component:.1f} + "
            f"Momentum {r.momentum_component:.1f} + "
            f"Sector {r.sector_component:.1f} + "
            f"Risk {r.risk_component:.1f} + "
            f"Regime {r.regime_component:.1f}"
        ),
        axis=1,
    )

    return out.sort_values("total_score", ascending=False).reset_index(drop=True)
