"""V9.0 institutional portfolio backtest utilities.

Designed for Streamlit Cloud: deterministic, light-weight, and dependency-free.
It builds a simple long-only portfolio return series from CIO recommended weights
and close prices, then reports institutional metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    peak = equity.cummax().replace(0, np.nan)
    dd = equity / peak - 1.0
    return abs(float(dd.min())) if dd.notna().any() else 0.0


def _annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 30 or r.std() == 0 or pd.isna(r.std()):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / r.std())


def _sortino(returns: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    downside = r[r < 0]
    if len(r) < 30 or downside.std() == 0 or pd.isna(downside.std()):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / downside.std())


def build_v9_backtest(close_panel: pd.DataFrame, recommended_portfolio: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Build an institutional backtest from close prices and latest weights.

    Parameters
    ----------
    close_panel: DataFrame with date index and one column per symbol.
    recommended_portfolio: optional DataFrame with symbol and target_weight.

    Returns
    -------
    equity_curve, metrics_dict
    """
    if close_panel is None or close_panel.empty:
        metrics = {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0, "volatility": 0.0, "hit_rate": 0.0}
        return pd.DataFrame(columns=["date", "portfolio_return", "equity_curve", "drawdown"]), metrics

    prices = close_panel.copy().sort_index().ffill().dropna(how="all")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Use CIO target weights when available. Otherwise equal-weight the universe.
    weights = None
    if recommended_portfolio is not None and not recommended_portfolio.empty and {"symbol", "target_weight"}.issubset(recommended_portfolio.columns):
        w = recommended_portfolio[["symbol", "target_weight"]].copy()
        w["target_weight"] = pd.to_numeric(w["target_weight"], errors="coerce").fillna(0.0)
        w = w[w["symbol"].isin(returns.columns)]
        if not w.empty and w["target_weight"].sum() > 0:
            weights = w.set_index("symbol")["target_weight"]
            weights = weights / weights.sum()
    if weights is None:
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

    weights = weights.reindex(returns.columns).fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    weights = weights / weights.sum()

    port_ret = (returns * weights).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()
    dd = equity / equity.cummax().replace(0, np.nan) - 1.0

    n_years = max(len(port_ret) / 252.0, 1e-9)
    ending = _safe_float(equity.iloc[-1], 1.0) if len(equity) else 1.0
    cagr = ending ** (1.0 / n_years) - 1.0 if ending > 0 else -1.0
    mdd = _max_drawdown(equity)
    sharpe = _annualized_sharpe(port_ret)
    sortino = _sortino(port_ret)
    vol = float(port_ret.std() * np.sqrt(252)) if len(port_ret) > 1 else 0.0
    calmar = float(cagr / mdd) if mdd > 0 else 0.0
    hit_rate = float((port_ret > 0).mean()) if len(port_ret) else 0.0

    curve = pd.DataFrame({
        "date": prices.index,
        "portfolio_return": port_ret.values,
        "equity_curve": equity.values,
        "drawdown": dd.fillna(0.0).values,
    })
    metrics = {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(mdd),
        "volatility": float(vol),
        "hit_rate": float(hit_rate),
        "equity_final": float(ending),
    }
    return curve, metrics
