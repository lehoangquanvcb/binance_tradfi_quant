"""V8 CIO Market Regime Engine.

Transforms multi-asset prices and macro-credit indicators into a CIO-friendly
regime classification: CRISIS, RISK_OFF, NEUTRAL, RECOVERY, RISK_ON.
The module is deliberately API-neutral and works with the price/macro data that
already exists in the package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(x, index):
    if x is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(x, errors="coerce").reindex(index).ffill()


def _score_from_return(r: pd.Series, scale: float) -> pd.Series:
    return (50 + 25 * (pd.to_numeric(r, errors="coerce") / max(scale, 1e-9))).clip(0, 100).fillna(50)


def _score_inverse_vol(vol: pd.Series, center: float = 0.25, scale: float = 0.20) -> pd.Series:
    z = (pd.to_numeric(vol, errors="coerce") - center) / max(scale, 1e-9)
    return (55 - 25 * z).clip(0, 100).fillna(50)


def build_market_regime(close_panel: pd.DataFrame, macro_credit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a time series of market regime scores and recommended risk budget."""
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()

    px = close_panel.copy().sort_index().ffill().dropna(how="all")
    if px.empty:
        return pd.DataFrame()
    idx = px.index

    equity = "SPY" if "SPY" in px.columns else ("QQQ" if "QQQ" in px.columns else px.columns[0])
    growth = "QQQ" if "QQQ" in px.columns else equity
    bonds = "TLT" if "TLT" in px.columns else None
    gold = "GLD" if "GLD" in px.columns else None
    btc = "BTC" if "BTC" in px.columns else ("BTC-USD" if "BTC-USD" in px.columns else None)

    eq = _as_series(px[equity], idx)
    gr = _as_series(px[growth], idx)
    bd = _as_series(px[bonds], idx) if bonds else pd.Series(np.nan, index=idx)
    gd = _as_series(px[gold], idx) if gold else pd.Series(np.nan, index=idx)
    bt = _as_series(px[btc], idx) if btc else pd.Series(np.nan, index=idx)

    eq_ret_20 = eq.pct_change(20)
    eq_ret_60 = eq.pct_change(60)
    gr_ret_60 = gr.pct_change(60)
    bd_ret_60 = bd.pct_change(60)
    gd_ret_60 = gd.pct_change(60)
    bt_ret_60 = bt.pct_change(60)

    ma50 = eq.rolling(50, min_periods=20).mean()
    ma200 = eq.rolling(200, min_periods=60).mean()
    trend_score = ((eq > ma50).astype(float) * 35 + (eq > ma200).astype(float) * 45 + (ma50 > ma200).astype(float) * 20).fillna(50)
    momentum_score = (0.45 * _score_from_return(eq_ret_20, 0.05) + 0.55 * _score_from_return(eq_ret_60, 0.12)).clip(0, 100)

    vol_20 = eq.pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
    volatility_score = _score_inverse_vol(vol_20)

    breadth = (px > px.rolling(200, min_periods=60).mean()).mean(axis=1) * 100
    breadth_score = breadth.fillna(50).clip(0, 100)

    # Defensive assets rising while equities fall is a risk-off signal.
    defensive_score = pd.Series(50.0, index=idx)
    if bonds:
        defensive_score -= _score_from_return(bd_ret_60.fillna(0) - eq_ret_60.fillna(0), 0.08) * 0.20
    if gold:
        defensive_score -= _score_from_return(gd_ret_60.fillna(0) - eq_ret_60.fillna(0), 0.08) * 0.10
    if btc:
        defensive_score += _score_from_return(bt_ret_60.fillna(0), 0.25) * 0.10
    defensive_score = defensive_score.clip(0, 100).fillna(50)

    macro_risk = pd.Series(50.0, index=idx)
    if macro_credit is not None and not macro_credit.empty and "date" in macro_credit.columns:
        mc = macro_credit.copy()
        mc["date"] = pd.to_datetime(mc["date"])
        mc = mc.sort_values("date").set_index("date")
        if "equity_risk_score" in mc.columns:
            macro_risk = pd.to_numeric(mc["equity_risk_score"], errors="coerce").reindex(idx, method="ffill").fillna(50)
    macro_score = (100 - macro_risk).clip(0, 100)

    regime_score = (
        0.25 * trend_score
        + 0.20 * momentum_score
        + 0.20 * breadth_score
        + 0.15 * volatility_score
        + 0.10 * defensive_score
        + 0.10 * macro_score
    ).clip(0, 100)

    def label(score: float) -> str:
        if score < 25:
            return "CRISIS"
        if score < 40:
            return "RISK_OFF"
        if score < 55:
            return "NEUTRAL"
        if score < 70:
            return "RECOVERY"
        return "RISK_ON"

    out = pd.DataFrame(index=idx)
    out["benchmark"] = equity
    out["trend_score"] = trend_score
    out["momentum_score"] = momentum_score
    out["breadth_score"] = breadth_score
    out["volatility_score"] = volatility_score
    out["defensive_asset_score"] = defensive_score
    out["macro_score"] = macro_score
    out["regime_score"] = regime_score
    out["market_regime"] = regime_score.map(label)
    out["confidence"] = ((regime_score - 50).abs() / 50).clip(0, 1)
    out["recommended_equity_weight"] = regime_score.map(lambda x: 0.20 if x < 25 else (0.35 if x < 40 else (0.55 if x < 55 else (0.75 if x < 70 else 0.90))))
    out["recommended_bond_weight"] = regime_score.map(lambda x: 0.40 if x < 40 else (0.25 if x < 55 else 0.10))
    out["recommended_gold_weight"] = regime_score.map(lambda x: 0.20 if x < 40 else (0.10 if x < 70 else 0.05))
    out["recommended_cash_weight"] = (1.0 - out["recommended_equity_weight"] - out["recommended_bond_weight"] - out["recommended_gold_weight"]).clip(0, 1)
    return out.reset_index().rename(columns={"index": "date"})


def latest_regime_row(regime: pd.DataFrame) -> dict:
    if regime is None or regime.empty:
        return {"market_regime": "UNKNOWN", "regime_score": 50.0, "confidence": 0.0}
    return regime.sort_values("date").iloc[-1].to_dict()
