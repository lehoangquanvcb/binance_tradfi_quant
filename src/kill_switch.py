"""V8.7 institutional kill-switch.

This version avoids blocking the entire strategy because one high-volatility
asset (for example BTC/ETH/VFS) has a large standalone drawdown. The hard gate
is portfolio-level; asset-level breaches are warnings used by CIO/risk review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _portfolio_stats(portfolio_returns: pd.Series | None) -> dict:
    if portfolio_returns is None:
        return {}
    r = pd.to_numeric(portfolio_returns, errors="coerce").dropna()
    if r.empty:
        return {}
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    var_95 = float(r.quantile(0.05))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95
    vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = float((r.mean() * 252) / (r.std() * np.sqrt(252))) if r.std() not in (0, np.nan) else 0.0
    return {
        "portfolio_var_95_1d": var_95,
        "portfolio_cvar_95_1d": cvar_95,
        "portfolio_max_drawdown": float(dd.min()),
        "portfolio_vol_annual": vol,
        "portfolio_sharpe_annual": sharpe,
    }


def _risk_score(max_dd: float, var_95: float, vol: float = 0.0) -> float:
    """100 = low risk; 0 = severe risk."""
    dd_penalty = min(abs(max_dd) / 0.35, 1.0) * 45
    var_penalty = min(abs(var_95) / 0.10, 1.0) * 35
    vol_penalty = min(abs(vol) / 0.60, 1.0) * 20
    return round(max(0.0, 100.0 - dd_penalty - var_penalty - vol_penalty), 1)


def evaluate_kill_switch(
    risk_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    portfolio_returns: pd.Series | None = None,
    daily_loss_pct: float = 0.03,
    max_drawdown_pct: float = 0.20,
    var_limit_pct: float = 0.08,
    extreme_asset_drawdown_pct: float = 0.85,
    api_error_count: int = 0,
    api_error_limit: int = 3,
    slippage_pct: float = 0.01,
) -> dict:
    """Central safety gate.

    Hard blocks:
    - Portfolio drawdown or portfolio VaR breaches limits.
    - Operational failures such as API errors or extreme slippage.

    Warnings:
    - Individual assets with large drawdown or high VaR. These should trigger
      review, not automatically block the whole CIO dashboard.
    """
    breaches: list[str] = []
    warnings: list[str] = []
    stats = _portfolio_stats(portfolio_returns)

    if stats:
        p_dd = abs(_safe_float(stats.get("portfolio_max_drawdown")))
        p_var = abs(_safe_float(stats.get("portfolio_var_95_1d")))
        p_vol = abs(_safe_float(stats.get("portfolio_vol_annual")))
        if p_dd >= max_drawdown_pct:
            breaches.append(f"Portfolio max drawdown breach: {p_dd:.2%} >= {max_drawdown_pct:.2%}")
        if p_var >= var_limit_pct:
            breaches.append(f"Portfolio VaR breach: {p_var:.2%} >= {var_limit_pct:.2%}")
        score = _risk_score(p_dd, p_var, p_vol)
    else:
        # Fallback when no portfolio return series is supplied: do not block on
        # standalone asset drawdowns; use median asset risk as an approximate score.
        p_dd = p_var = p_vol = 0.0
        score = 70.0

    if risk_df is not None and not risk_df.empty:
        if "max_drawdown" in risk_df.columns:
            worst_dd = abs(_safe_float(risk_df["max_drawdown"].min()))
            if worst_dd >= extreme_asset_drawdown_pct:
                warnings.append(f"Extreme single-asset drawdown observed: {worst_dd:.2%}; review universe/position limits.")
        if "var_95_1d" in risk_df.columns:
            worst_var = abs(_safe_float(risk_df["var_95_1d"].min()))
            if worst_var >= var_limit_pct:
                warnings.append(f"Single-asset VaR warning: {worst_var:.2%} >= {var_limit_pct:.2%}; do not block portfolio automatically.")

    if trades_df is not None and not trades_df.empty and "realized_pnl_pct" in trades_df.columns:
        today_loss = _safe_float(pd.to_numeric(trades_df["realized_pnl_pct"], errors="coerce").dropna().tail(20).sum())
        if today_loss <= -abs(daily_loss_pct):
            breaches.append(f"Daily/trailing realized loss breach: {today_loss:.2%} <= -{daily_loss_pct:.2%}")

    if api_error_count >= api_error_limit:
        breaches.append(f"API error breach: {api_error_count} >= {api_error_limit}")
    if slippage_pct >= 0.02:
        breaches.append(f"Slippage breach: {slippage_pct:.2%}")

    if score >= 80:
        risk_level = "LOW"
    elif score >= 60:
        risk_level = "MODERATE"
    elif score >= 40:
        risk_level = "ELEVATED"
    else:
        risk_level = "HIGH"

    return {
        "allow_trading": len(breaches) == 0,
        "status": "OK" if len(breaches) == 0 else "BLOCKED",
        "institutional_risk_score": score,
        "risk_level": risk_level,
        "portfolio_max_drawdown": stats.get("portfolio_max_drawdown"),
        "portfolio_var_95_1d": stats.get("portfolio_var_95_1d"),
        "portfolio_vol_annual": stats.get("portfolio_vol_annual"),
        "breaches": breaches,
        "warnings": warnings,
    }
