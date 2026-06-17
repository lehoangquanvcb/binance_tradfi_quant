"""V11 Cross-Asset Allocation Engine."""
from __future__ import annotations
import pandas as pd
import numpy as np

def build_cross_asset_allocation(close_panel: pd.DataFrame, macro_credit: pd.DataFrame | None=None, regime: pd.DataFrame | dict | None=None, nav: float=100000.0) -> pd.DataFrame:
    regime_name="NEUTRAL"; regime_score=50.0
    try:
        if isinstance(regime, pd.DataFrame) and not regime.empty:
            r=regime.tail(1).iloc[0]
            regime_name=str(r.get("market_regime", r.get("regime","NEUTRAL"))).upper()
            regime_score=float(r.get("regime_score",50))
        elif isinstance(regime, dict):
            regime_name=str(regime.get("market_regime", regime.get("regime","NEUTRAL"))).upper()
            regime_score=float(regime.get("regime_score",50))
    except Exception:
        pass
    sleeves=[
        ("Equity Growth", "QQQ", 0.30), ("Equity Core", "SPY", 0.25), ("Defensive Equity", "XLV", 0.10),
        ("Bonds", "TLT", 0.15), ("Gold", "GLD", 0.10), ("Cash", "CASH", 0.10)
    ]
    if "RISK_ON" in regime_name or regime_score>=65:
        sleeves=[("Equity Growth","QQQ",0.35),("Equity Core","SPY",0.30),("Financials","XLF",0.10),("Bonds","TLT",0.08),("Gold","GLD",0.07),("Cash","CASH",0.10)]
    elif "RISK_OFF" in regime_name or "CRISIS" in regime_name or regime_score<40:
        sleeves=[("Equity Core","SPY",0.20),("Defensive Equity","XLV",0.10),("Bonds","TLT",0.30),("Gold","GLD",0.20),("Cash","CASH",0.20)]
    rows=[]
    for sleeve, symbol, weight in sleeves:
        available = (symbol == "CASH") or (close_panel is not None and symbol in close_panel.columns)
        rows.append({"sleeve":sleeve,"symbol":symbol,"target_weight":weight,"target_usd":weight*float(nav),"available_in_universe":available,"regime":regime_name})
    return pd.DataFrame(rows)
