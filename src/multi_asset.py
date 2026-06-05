"""Multi-asset universe definitions."""
from __future__ import annotations

MULTI_ASSET_UNIVERSE = {
    "us_equity_single_names": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"],
    "equity_etfs": ["SPY", "QQQ", "DIA", "IWM"],
    "rates_etfs": ["TLT", "IEF", "SHY"],
    "commodities": ["GLD", "USO"],
    "crypto": ["BTC", "ETH"],
    "cash": ["CASH"],
}


def flatten_universe() -> list[str]:
    out=[]
    for vals in MULTI_ASSET_UNIVERSE.values():
        out.extend(vals)
    return list(dict.fromkeys(out))
