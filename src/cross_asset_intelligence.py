"""Cross-asset intelligence: equity, bonds, gold, USD and crypto risk signals."""
from __future__ import annotations
import numpy as np
import pandas as pd


def cross_asset_signals(price_panel: pd.DataFrame) -> pd.DataFrame:
    if price_panel is None or price_panel.empty:
        return pd.DataFrame()
    px = price_panel.copy().ffill()
    ret_20 = px.pct_change(20).iloc[-1].fillna(0)
    vol_20 = px.pct_change().tail(20).std().fillna(0) * np.sqrt(252)
    rows = []
    for sym in px.columns:
        mom = float(ret_20.get(sym, 0))
        vol = float(vol_20.get(sym, 0))
        score = mom / max(vol, 1e-6)
        rows.append({"symbol": sym, "return_20d": mom, "ann_vol_20d": vol, "cross_asset_score": score})
    out = pd.DataFrame(rows)
    risk_on_assets = [s for s in out.symbol if any(x in s.upper() for x in ["SPY", "QQQ", "NVDA", "AAPL", "MSFT", "BTC", "ETH"])]
    safe_assets = [s for s in out.symbol if any(x in s.upper() for x in ["GLD", "GOLD", "TLT", "BOND", "USDT"])]
    risk_on = out[out.symbol.isin(risk_on_assets)]["cross_asset_score"].mean() if risk_on_assets else out["cross_asset_score"].mean()
    safe = out[out.symbol.isin(safe_assets)]["cross_asset_score"].mean() if safe_assets else 0
    out["global_risk_signal"] = "Risk-On" if risk_on > safe else "Risk-Off"
    return out
