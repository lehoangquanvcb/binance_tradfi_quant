"""Transaction cost and liquidity guardrails."""
from __future__ import annotations
import pandas as pd

def estimate_transaction_cost(price: float, qty: float, side: str = 'BUY', fee_bps: float = 4.0,
                              spread_bps: float = 8.0, slippage_bps: float = 10.0,
                              funding_bps_daily: float = 0.0, holding_days: float = 1.0) -> dict:
    notional = abs(float(price) * float(qty))
    fee = notional * fee_bps / 10000
    half_spread = notional * spread_bps / 10000 / 2
    slippage = notional * slippage_bps / 10000
    funding = notional * funding_bps_daily / 10000 * holding_days
    total = fee + half_spread + slippage + funding
    effective_price = price + (total / max(abs(qty), 1e-12)) * (1 if side.upper() == 'BUY' else -1)
    return {
        'notional': notional, 'fee': fee, 'half_spread': half_spread,
        'slippage': slippage, 'funding': funding, 'total_cost': total,
        'cost_bps': (total / notional * 10000) if notional else 0.0,
        'effective_price': effective_price
    }

def apply_cost_filter(signals: pd.DataFrame, min_edge_bps: float = 20.0,
                      fee_bps: float = 4.0, spread_bps: float = 8.0, slippage_bps: float = 10.0) -> pd.DataFrame:
    if signals.empty:
        return signals
    df = signals.copy()
    # Convert probability edge into a simple bps proxy. This is intentionally conservative.
    df['expected_edge_bps'] = (df['prob_up'] - 0.50) * 200.0
    df['estimated_cost_bps'] = fee_bps + spread_bps / 2 + slippage_bps
    df['net_edge_bps'] = df['expected_edge_bps'] - df['estimated_cost_bps']
    df['cost_adjusted_signal'] = df.apply(
        lambda r: r['signal'] if r['net_edge_bps'] >= min_edge_bps else 'HOLD_COST_FILTER', axis=1
    )
    return df
