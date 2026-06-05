"""Dynamic asset allocation driven by macro-credit regime."""
from __future__ import annotations
import pandas as pd

POLICY = {
    "Risk-On": {"Equity": 0.70, "Bond": 0.15, "Gold": 0.10, "Cash": 0.05},
    "Neutral": {"Equity": 0.50, "Bond": 0.25, "Gold": 0.15, "Cash": 0.10},
    "Risk-Off": {"Equity": 0.25, "Bond": 0.40, "Gold": 0.20, "Cash": 0.15},
}

def strategic_allocation(regime: str) -> pd.DataFrame:
    p = POLICY.get(regime, POLICY["Neutral"])
    return pd.DataFrame([{"asset_class": k, "target_weight": v} for k, v in p.items()])


def map_symbol_to_asset_class(symbol: str) -> str:
    s = symbol.upper()
    if any(x in s for x in ["TLT", "IEF", "BOND", "TREASURY"]):
        return "Bond"
    if any(x in s for x in ["GLD", "GOLD"]):
        return "Gold"
    if any(x in s for x in ["USDT", "CASH"]):
        return "Cash"
    return "Equity"
