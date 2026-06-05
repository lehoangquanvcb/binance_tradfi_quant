"""Market microstructure and liquidity scoring."""
from __future__ import annotations

import pandas as pd


def orderbook_metrics(orderbook: dict) -> dict:
    bids = orderbook.get("bids", [])[:10]
    asks = orderbook.get("asks", [])[:10]
    if not bids or not asks:
        return {"spread_bps": None, "imbalance": None, "liquidity_score": 0.0}
    best_bid = float(bids[0][0]); best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2
    spread_bps = (best_ask - best_bid) / max(mid, 1e-9) * 10000
    bid_qty = sum(float(x[1]) for x in bids)
    ask_qty = sum(float(x[1]) for x in asks)
    imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1e-9)
    liquidity_score = max(0, 100 - spread_bps) * (1 - min(abs(imbalance), 1) * 0.5)
    return {"spread_bps": spread_bps, "imbalance": imbalance, "liquidity_score": liquidity_score}


def liquidity_from_ohlcv(df: pd.DataFrame) -> dict:
    d = df.copy()
    dollar_vol = (d["close"] * d.get("volume", 0)).tail(20).mean()
    atr_proxy = ((d["high"] - d["low"]) / d["close"]).tail(20).mean() * 10000
    score = min(100, max(0, dollar_vol / 1_000_000)) * max(0.25, 1 - atr_proxy / 1000)
    return {"avg_dollar_volume_20d": dollar_vol, "range_bps_20d": atr_proxy, "liquidity_score": score}
