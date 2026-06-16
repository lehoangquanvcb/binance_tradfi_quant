"""V8 CIO Dashboard Summary Builder."""
from __future__ import annotations

import pandas as pd


def build_cio_summary(market_regime: pd.DataFrame, sector_rotation: pd.DataFrame, stock_selection: pd.DataFrame, exit_watchlist: pd.DataFrame, portfolio: pd.DataFrame) -> str:
    if market_regime is not None and not market_regime.empty:
        mr = market_regime.sort_values("date").iloc[-1]
        regime = mr.get("market_regime", "UNKNOWN")
        score = float(mr.get("regime_score", 50) or 50)
        eq_weight = float(mr.get("recommended_equity_weight", 0.60) or 0.60)
    else:
        regime, score, eq_weight = "UNKNOWN", 50.0, 0.60
    top_sectors = []
    weak_sectors = []
    if sector_rotation is not None and not sector_rotation.empty:
        top_sectors = sector_rotation.head(3)["sector"].astype(str).tolist() if "sector" in sector_rotation.columns else []
        weak_sectors = sector_rotation.tail(3)["sector"].astype(str).tolist() if "sector" in sector_rotation.columns else []
    top_names = []
    exits = []
    if stock_selection is not None and not stock_selection.empty:
        top_names = stock_selection.head(5)["symbol"].astype(str).tolist()
    if exit_watchlist is not None and not exit_watchlist.empty:
        exits = exit_watchlist.head(5)["symbol"].astype(str).tolist()
    return (
        f"Current market regime: {regime} (score {score:.1f}/100).\n"
        f"Suggested equity allocation: {eq_weight:.0%}.\n"
        f"Overweight sectors: {', '.join(top_sectors) if top_sectors else 'N/A'}.\n"
        f"Underweight / avoid sectors: {', '.join(weak_sectors) if weak_sectors else 'N/A'}.\n"
        f"Top buy/hold candidates: {', '.join(top_names) if top_names else 'N/A'}.\n"
        f"Exit / reduce watchlist: {', '.join(exits) if exits else 'None'}.\n"
        "Use this as a decision-support layer, not as an instruction to trade without human review."
    )
