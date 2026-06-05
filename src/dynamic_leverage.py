"""Dynamic leverage controls based on volatility, VaR, drawdown and regime."""
from __future__ import annotations


def leverage_multiplier(vol_annual: float, var_95: float, drawdown: float, regime: str) -> float:
    mult = 1.0
    if vol_annual > 0.35: mult *= 0.65
    if vol_annual > 0.55: mult *= 0.45
    if var_95 > 0.035: mult *= 0.70
    if abs(drawdown) > 0.12: mult *= 0.60
    if str(regime).lower() in {"risk-off", "bear", "high volatility bear"}: mult *= 0.50
    return max(0.0, min(1.0, mult))
