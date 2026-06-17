"""V10 Regime Transition Matrix."""
from __future__ import annotations
import pandas as pd


def build_regime_transition_matrix(regimes: pd.DataFrame, regime_col='market_regime') -> pd.DataFrame:
    if regimes is None or regimes.empty or regime_col not in regimes.columns:
        return pd.DataFrame()
    r = regimes.sort_values('date') if 'date' in regimes.columns else regimes.copy()
    s = r[regime_col].astype(str).replace('', 'NEUTRAL')
    trans = pd.DataFrame({'from_regime': s.shift(1), 'to_regime': s}).dropna()
    if trans.empty:
        return pd.DataFrame()
    mat = pd.crosstab(trans['from_regime'], trans['to_regime'], normalize='index').round(4)
    return mat.reset_index()


def latest_transition_prob(matrix: pd.DataFrame, current_regime: str) -> dict:
    if matrix is None or matrix.empty or 'from_regime' not in matrix.columns:
        return {}
    row = matrix[matrix['from_regime'].astype(str) == str(current_regime)]
    if row.empty:
        return {}
    return row.drop(columns=['from_regime']).iloc[0].to_dict()
