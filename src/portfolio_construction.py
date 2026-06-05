"""Institutional portfolio construction engines.

Includes mean-variance, risk parity, Black-Litterman lite, and HRP-lite.
Designed to run without heavy optional packages. If scipy is available it uses it;
otherwise it falls back to robust inverse-volatility allocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return r.fillna(0.0)


def annualized_expected_returns(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    r = _clean_returns(returns)
    return r.mean() * periods


def covariance_matrix(returns: pd.DataFrame, periods: int = 252, shrinkage: float = 0.10) -> pd.DataFrame:
    r = _clean_returns(returns)
    cov = r.cov() * periods
    diag = np.diag(np.diag(cov.values))
    shrunk = (1 - shrinkage) * cov.values + shrinkage * diag
    return pd.DataFrame(shrunk, index=cov.index, columns=cov.columns)


def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    vol = _clean_returns(returns).std().replace(0, np.nan)
    inv = 1 / vol
    w = inv / inv.sum()
    return w.fillna(0)


def mean_variance_weights(
    returns: pd.DataFrame,
    risk_aversion: float = 5.0,
    long_only: bool = True,
    max_weight: float = 0.35,
) -> pd.Series:
    """Approximate max utility weights: argmax mu'w - lambda/2 w'Sw.

    Uses closed-form then clips/renormalizes for robustness. This is suitable as
    a transparent baseline; replace with cvxpy for production constrained MVO.
    """
    r = _clean_returns(returns)
    mu = annualized_expected_returns(r)
    cov = covariance_matrix(r)
    try:
        raw = np.linalg.pinv(cov.values).dot(mu.values) / max(risk_aversion, 1e-6)
        w = pd.Series(raw, index=mu.index)
    except Exception:
        w = inverse_vol_weights(r)
    if long_only:
        w = w.clip(lower=0)
    if w.abs().sum() == 0:
        w = inverse_vol_weights(r)
    w = w / w.sum()
    w = w.clip(upper=max_weight)
    return w / w.sum()


def risk_parity_weights(returns: pd.DataFrame, max_iter: int = 2000, tol: float = 1e-8) -> pd.Series:
    """Simple equal-risk-contribution solver using multiplicative updates."""
    r = _clean_returns(returns)
    cov = covariance_matrix(r).values
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = float(w.T @ cov @ w)
        mrc = cov @ w
        rc = w * mrc / max(port_var, 1e-12)
        target = np.ones(n) / n
        err = rc - target
        if np.max(np.abs(err)) < tol:
            break
        w = w * (target / np.maximum(rc, 1e-8)) ** 0.05
        w = np.maximum(w, 0)
        w = w / w.sum()
    return pd.Series(w, index=r.columns)


def black_litterman_lite(
    returns: pd.DataFrame,
    views: dict[str, float] | None = None,
    confidence: float = 0.35,
    max_weight: float = 0.35,
) -> pd.Series:
    """Black-Litterman-inspired blend of market prior and user/model views.

    views maps symbol -> annual expected return view. confidence controls how
    strongly views replace historical expected returns.
    """
    r = _clean_returns(returns)
    prior = annualized_expected_returns(r)
    posterior = prior.copy()
    if views:
        for sym, view in views.items():
            if sym in posterior.index:
                posterior.loc[sym] = (1 - confidence) * prior.loc[sym] + confidence * float(view)
    adjusted = r.copy()
    # shift daily returns so sample annual mean approximates posterior
    delta_daily = (posterior - prior) / 252
    for c in adjusted.columns:
        adjusted[c] = adjusted[c] + delta_daily.get(c, 0.0)
    return mean_variance_weights(adjusted, max_weight=max_weight)


def hrp_lite_weights(returns: pd.DataFrame) -> pd.Series:
    """HRP-lite: cluster by correlation sign/strength using greedy grouping.

    Full HRP requires scipy clustering; this fallback groups high-correlation
    assets and allocates inverse-vol inside and across groups.
    """
    r = _clean_returns(returns)
    corr = r.corr().fillna(0)
    unused = set(r.columns)
    groups: list[list[str]] = []
    while unused:
        seed = sorted(unused)[0]
        group = [seed]
        unused.remove(seed)
        peers = [c for c in list(unused) if corr.loc[seed, c] > 0.65]
        for p in peers:
            group.append(p); unused.remove(p)
        groups.append(group)
    group_returns = pd.DataFrame({f"G{i}": r[g].mean(axis=1) for i, g in enumerate(groups)})
    group_w = inverse_vol_weights(group_returns)
    final = pd.Series(0.0, index=r.columns)
    for i, g in enumerate(groups):
        inner = inverse_vol_weights(r[g])
        final.loc[g] = inner * group_w.iloc[i]
    return final / final.sum()


def build_target_weights(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    views: dict[str, float] | None = None,
) -> pd.Series:
    method = method.lower()
    if method in {"mvo", "mean_variance"}:
        return mean_variance_weights(returns)
    if method in {"black_litterman", "bl"}:
        return black_litterman_lite(returns, views=views)
    if method in {"hrp", "hrp_lite"}:
        return hrp_lite_weights(returns)
    return risk_parity_weights(returns)
