"""Multi-strategy ensemble signals.

Robust version for Streamlit Cloud / pandas 2.x / yfinance data.
It avoids KeyError when some technical columns are missing or duplicated after
feature generation and merge operations.
"""
from __future__ import annotations

import pandas as pd


def _first_existing(df: pd.DataFrame, names: list[str], default=None):
    """Return the first existing column as a Series; otherwise return default Series."""
    for name in names:
        if name in df.columns:
            x = df[name]
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            return x
    return pd.Series(default, index=df.index)


def _ensure_numeric(series: pd.Series, default=0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def build_ensemble_signals(df: pd.DataFrame, ml_signals: pd.DataFrame) -> pd.DataFrame:
    """Build ensemble signals from ML signal + trend/momentum/regime features.

    The previous implementation assumed columns like rsi_14 and ret_20d were
    always present after merge. On Streamlit Cloud, yfinance/pandas column
    handling can create missing or suffixed columns, causing KeyError. This
    version is defensive and fills missing inputs with neutral defaults.
    """
    if ml_signals is None or ml_signals.empty:
        return ml_signals

    base = ml_signals.copy()

    if df is not None and not df.empty and "symbol" in df.columns:
        latest = df.sort_values("date").groupby("symbol", as_index=False).tail(1).copy()
        wanted = [
            "symbol", "ema_trend", "rsi_14", "rsi", "ret_5d", "ret_20d",
            "vol_ratio_20", "market_regime", "regime"
        ]
        keep = [c for c in wanted if c in latest.columns]
        if "symbol" in keep:
            latest = latest[keep].loc[:, ~latest[keep].columns.duplicated()].copy()
            base = base.merge(latest, on="symbol", how="left", suffixes=("", "_ds"))

    # Robust column resolution after merge. Prefer original signal columns,
    # then dataset-suffixed alternatives, then neutral defaults.
    prob_up = _ensure_numeric(_first_existing(base, ["prob_up", "prob_up_ds", "probability"], 0.5), 0.5).clip(0, 1)
    ema_trend = _ensure_numeric(_first_existing(base, ["ema_trend", "ema_trend_ds"], 0), 0)
    rsi_14 = _ensure_numeric(_first_existing(base, ["rsi_14", "rsi_14_ds", "rsi"], 50), 50)
    ret_5d = _ensure_numeric(_first_existing(base, ["ret_5d", "ret_5d_ds"], 0), 0)
    ret_20d = _ensure_numeric(_first_existing(base, ["ret_20d", "ret_20d_ds"], 0), 0)
    regime = _first_existing(base, ["market_regime", "market_regime_ds", "regime"], "NEUTRAL").fillna("NEUTRAL")

    base["prob_up"] = prob_up
    base["rsi_14"] = rsi_14
    base["ret_5d"] = ret_5d
    base["ret_20d"] = ret_20d
    base["market_regime"] = regime.astype(str).str.upper()

    base["trend_score"] = (ema_trend > 0).astype(float)
    base["momentum_score"] = ((rsi_14 > 55) & (ret_20d > 0)).astype(float)
    base["mean_reversion_score"] = ((rsi_14 < 35) & (ret_5d < 0)).astype(float)
    base["macro_score"] = base["market_regime"].map({
        "RISK_ON": 1.0,
        "AI_TECH_MOMENTUM": 0.8,
        "NEUTRAL": 0.5,
        "HIGH_INFLATION": 0.3,
        "RECESSION_RISK": 0.0,
        "RISK_OFF": 0.0,
        "EXPANSION": 1.0,
        "RECOVERY": 0.8,
        "SLOWDOWN": 0.4,
        "RECESSION": 0.0,
        "INFLATION_SHOCK": 0.2,
    }).fillna(0.5)
    base["ml_score"] = base["prob_up"]
    base["ensemble_score"] = (
        0.40 * base["ml_score"]
        + 0.20 * base["trend_score"]
        + 0.15 * base["momentum_score"]
        + 0.10 * base["mean_reversion_score"]
        + 0.15 * base["macro_score"]
    )
    base["ensemble_signal"] = base["ensemble_score"].map(
        lambda x: "BUY" if x >= 0.62 else ("SELL" if x <= 0.38 else "HOLD")
    )
    return base
