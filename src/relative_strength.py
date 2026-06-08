"""V6.2 Relative Strength Engine."""
from __future__ import annotations

import pandas as pd


def add_relative_strength_features(df: pd.DataFrame, benchmark_symbol: str = "SPY") -> pd.DataFrame:
    out = df.copy().sort_values(["symbol", "date"])
    if not {"date", "symbol", "close"}.issubset(out.columns):
        return out

    panel = out.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    if benchmark_symbol not in panel.columns:
        # Fallback: use equal-weight market proxy.
        bench = panel.mean(axis=1)
    else:
        bench = panel[benchmark_symbol]

    for w in [5, 20, 60, 120]:
        sym_ret = panel.pct_change(w)
        bench_ret = bench.pct_change(w)
        rel = sym_ret.sub(bench_ret, axis=0)
        rel_long = rel.stack().rename(f"rel_strength_{w}d").reset_index()
        rel_long.columns = ["date", "symbol", f"rel_strength_{w}d"]
        out = out.merge(rel_long, on=["date", "symbol"], how="left")

    return out
