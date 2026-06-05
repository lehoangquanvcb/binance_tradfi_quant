"""Institutional risk metrics: VaR, CVaR/ES, stress, concentration, liquidity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(returns.columns).fillna(0)
    return returns.fillna(0).dot(w)


def var_cvar(ret: pd.Series, alpha: float = 0.95) -> dict:
    r = ret.dropna()
    if r.empty:
        return {"VaR": 0.0, "CVaR": 0.0, "ExpectedShortfall": 0.0}
    q = r.quantile(1-alpha)
    tail = r[r <= q]
    cvar = tail.mean() if not tail.empty else q
    return {"VaR": -q, "CVaR": -cvar, "ExpectedShortfall": -cvar}


def concentration_metrics(weights: pd.Series) -> dict:
    w = weights[weights > 0]
    hhi = float((w ** 2).sum())
    top1 = float(w.max()) if len(w) else 0
    top5 = float(w.sort_values(ascending=False).head(5).sum())
    return {"HHI": hhi, "Top1Weight": top1, "Top5Weight": top5}


def stress_test(weights: pd.Series, scenarios: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows=[]
    for name, shocks in scenarios.items():
        loss = 0.0
        for sym, w in weights.items():
            loss += w * shocks.get(sym, shocks.get("DEFAULT", 0.0))
        rows.append({"scenario": name, "portfolio_return": loss, "portfolio_loss": -loss})
    return pd.DataFrame(rows)


def standard_scenarios() -> dict[str, dict[str, float]]:
    return {
        "Fed shock / duration selloff": {"DEFAULT": -0.08, "QQQ": -0.14, "SPY": -0.10, "TLT": -0.12, "GLD": -0.03},
        "US recession": {"DEFAULT": -0.18, "QQQ": -0.22, "SPY": -0.20, "XLE": -0.16, "GLD": 0.05},
        "AI bubble burst": {"DEFAULT": -0.12, "NVDA": -0.40, "AMD": -0.32, "QQQ": -0.25, "SPY": -0.15},
        "Liquidity squeeze": {"DEFAULT": -0.15, "QQQ": -0.20, "SPY": -0.16, "BTC": -0.28, "ETH": -0.32},
    }


def risk_dashboard_table(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    pr = portfolio_returns(returns, weights)
    metrics = var_cvar(pr)
    metrics.update(concentration_metrics(weights))
    metrics["AnnVol"] = float(pr.std() * np.sqrt(252))
    metrics["AnnReturn"] = float(pr.mean() * 252)
    metrics["Sharpe"] = metrics["AnnReturn"] / max(metrics["AnnVol"], 1e-9)
    return pd.DataFrame([metrics])
