"""V8 CIO Portfolio Constructor."""
from __future__ import annotations

import numpy as np
import pandas as pd


def construct_portfolio_v8(stock_selection: pd.DataFrame, sector_rotation: pd.DataFrame | None, market_regime: pd.DataFrame | None, nav: float = 100000.0, max_positions: int = 12) -> pd.DataFrame:
    if stock_selection is None or stock_selection.empty:
        return pd.DataFrame()
    regime_row = {}
    if market_regime is not None and not market_regime.empty:
        regime_row = market_regime.sort_values("date").iloc[-1].to_dict()
    equity_budget = float(regime_row.get("recommended_equity_weight", 0.60) or 0.60)
    cash_weight = float(regime_row.get("recommended_cash_weight", 0.20) or 0.20)
    bond_weight = float(regime_row.get("recommended_bond_weight", 0.10) or 0.10)
    gold_weight = float(regime_row.get("recommended_gold_weight", 0.10) or 0.10)

    candidates = stock_selection[stock_selection["decision"].isin(["TOP_BUY", "BUY", "HOLD"])].copy()
    candidates = candidates.sort_values("stock_score", ascending=False).head(max_positions)
    if candidates.empty:
        return pd.DataFrame([{"symbol": "CASH", "target_weight": 1.0, "target_usd": nav, "sleeve": "Cash", "rationale": "No eligible risk assets"}])

    scores = pd.to_numeric(candidates["stock_score"], errors="coerce").fillna(50).clip(lower=1)
    raw = scores / scores.sum()
    # Cap single-name concentration.
    weights = raw.clip(upper=0.18)
    weights = weights / weights.sum() * equity_budget

    rows = []
    for (_, r), w in zip(candidates.iterrows(), weights):
        rows.append({
            "symbol": r["symbol"],
            "target_weight": round(float(w), 4),
            "target_usd": round(float(w) * nav, 2),
            "sleeve": "Equity/ETF",
            "decision": r.get("decision"),
            "stock_score": r.get("stock_score"),
            "rationale": f"Score {r.get('stock_score')}; regime {regime_row.get('market_regime','NEUTRAL')}; sector {r.get('sector_bucket','')}",
        })
    if bond_weight > 0:
        rows.append({"symbol": "TLT", "target_weight": round(bond_weight, 4), "target_usd": round(bond_weight * nav, 2), "sleeve": "Bond hedge", "decision": "HEDGE", "stock_score": None, "rationale": "Regime-based duration hedge"})
    if gold_weight > 0:
        rows.append({"symbol": "GLD", "target_weight": round(gold_weight, 4), "target_usd": round(gold_weight * nav, 2), "sleeve": "Real asset hedge", "decision": "HEDGE", "stock_score": None, "rationale": "Regime-based inflation/stress hedge"})
    remaining = 1.0 - sum(r["target_weight"] for r in rows)
    if remaining > 0.005:
        rows.append({"symbol": "CASH", "target_weight": round(float(remaining), 4), "target_usd": round(float(remaining) * nav, 2), "sleeve": "Cash", "decision": "RESERVE", "stock_score": None, "rationale": "Residual risk budget / liquidity reserve"})
    out = pd.DataFrame(rows)
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce").fillna(0)
    return out.sort_values("target_weight", ascending=False)
