"""Capital allocation across strategy sleeves."""
from __future__ import annotations
import pandas as pd

def allocate_capital(nav: float, regime: str = 'NEUTRAL', max_strategy_weight: float = 0.45) -> pd.DataFrame:
    base = {
        'trend_following': 0.30,
        'momentum': 0.25,
        'mean_reversion': 0.15,
        'macro_regime': 0.20,
        'ml_signal': 0.10,
    }
    if regime in ('RISK_ON','AI_TECH_MOMENTUM'):
        base.update({'trend_following':0.35, 'momentum':0.30, 'mean_reversion':0.10, 'macro_regime':0.15, 'ml_signal':0.10})
    elif regime in ('RISK_OFF','RECESSION_RISK'):
        base.update({'trend_following':0.15, 'momentum':0.10, 'mean_reversion':0.20, 'macro_regime':0.40, 'ml_signal':0.15})
    elif regime == 'HIGH_INFLATION':
        base.update({'trend_following':0.20, 'momentum':0.15, 'mean_reversion':0.15, 'macro_regime':0.35, 'ml_signal':0.15})
    # Cap and renormalize.
    capped = {k:min(v, max_strategy_weight) for k,v in base.items()}
    total = sum(capped.values()) or 1
    rows = [{'strategy':k, 'weight':v/total, 'capital_usd':nav*v/total} for k,v in capped.items()]
    return pd.DataFrame(rows)
