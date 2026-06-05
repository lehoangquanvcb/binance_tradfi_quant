"""AI-style research narrative generator without external LLM dependency."""
from __future__ import annotations

import pandas as pd


def explain_signal(symbol: str, row: dict) -> str:
    prob = row.get("prob_up", row.get("probability", None))
    signal = row.get("signal", "HOLD")
    regime = row.get("regime", "Unknown")
    rsi = row.get("rsi", None)
    trend = row.get("trend_score", None)
    var = row.get("var_95", None)
    parts = [f"{symbol}: recommendation = {signal}."]
    if prob is not None:
        parts.append(f"Model probability-up is {prob:.1%}.")
    if rsi is not None:
        parts.append(f"RSI is {rsi:.1f}, indicating {'positive' if rsi >= 55 else 'weak/neutral'} momentum.")
    if trend is not None:
        parts.append(f"Trend score is {trend:.2f}.")
    parts.append(f"Current macro regime: {regime}.")
    if var is not None:
        parts.append(f"Estimated 95% daily VaR is {var:.2%}.")
    parts.append("Trade should pass liquidity, concentration, kill-switch and human-approval gates before live execution.")
    return " ".join(parts)


def portfolio_brief(weights: pd.Series, factor_exposure: pd.Series | None = None, regime: str = "Unknown") -> str:
    top = weights.sort_values(ascending=False).head(5)
    text = f"Portfolio is currently in {regime} regime. Top target weights: " + ", ".join([f"{k} {v:.1%}" for k,v in top.items()]) + "."
    if factor_exposure is not None and len(factor_exposure):
        beta = factor_exposure.get("beta", None)
        mom = factor_exposure.get("momentum_3m", None)
        if beta is not None:
            text += f" Portfolio beta is approximately {beta:.2f}."
        if mom is not None:
            text += f" 3-month momentum exposure is {mom:.2%}."
    return text
