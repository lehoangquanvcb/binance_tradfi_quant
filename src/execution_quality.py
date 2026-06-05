"""Execution quality analytics."""
from __future__ import annotations

import pandas as pd


def execution_quality_report(orders: pd.DataFrame) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame(columns=["symbol","fill_ratio","avg_slippage_bps","orders"])
    df = orders.copy()
    if "filled_qty" not in df: df["filled_qty"] = df.get("quantity", 0)
    if "quantity" not in df: df["quantity"] = 0
    df["fill_ratio"] = df["filled_qty"] / df["quantity"].replace(0, pd.NA)
    if "slippage_bps" not in df: df["slippage_bps"] = 0.0
    return df.groupby("symbol").agg(fill_ratio=("fill_ratio","mean"), avg_slippage_bps=("slippage_bps","mean"), orders=("symbol","count")).reset_index()
