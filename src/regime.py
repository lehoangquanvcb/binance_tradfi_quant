import numpy as np
import pandas as pd


def classify_market_regime(dataset: pd.DataFrame) -> pd.DataFrame:
    """Classify broad US market regime using macro + price trend features.

    Output is intentionally rules-based so it is explainable and robust when
    macro series are missing. Required columns are optional; missing values are
    handled gracefully.
    """
    df = dataset.sort_values(['symbol', 'date']).copy()
    # Use SPY-equivalent if available, otherwise cross-sectional median trend.
    spy = df[df['symbol'].astype(str).str.contains('SPY', case=False, na=False)].copy()
    base = spy if not spy.empty else df.groupby('date', as_index=False).median(numeric_only=True)
    base = base.sort_values('date').copy()

    if 'close' in base:
        base['market_ret_63d'] = base['close'].pct_change(63)
        base['market_ma_200'] = base['close'].rolling(200).mean()
        base['market_above_ma200'] = base['close'] > base['market_ma_200']
    else:
        base['market_ret_63d'] = np.nan
        base['market_above_ma200'] = False

    # Common FRED/yfinance macro names from macro_data.py; tolerate alternatives.
    def col(*names):
        for n in names:
            if n in base.columns:
                return n
        return None

    vix_col = col('VIXCLS', 'vix', '^VIX')
    fed_col = col('FEDFUNDS', 'fed_funds', 'policy_rate')
    cpi_col = col('CPIAUCSL_yoy', 'cpi_yoy', 'inflation_yoy')
    term_col = col('term_spread', 'T10Y2Y', 'us10y_minus_2y')

    regimes = []
    for _, r in base.iterrows():
        vix = r.get(vix_col, np.nan) if vix_col else np.nan
        fed = r.get(fed_col, np.nan) if fed_col else np.nan
        cpi = r.get(cpi_col, np.nan) if cpi_col else np.nan
        term = r.get(term_col, np.nan) if term_col else np.nan
        mom = r.get('market_ret_63d', np.nan)
        above = bool(r.get('market_above_ma200', False))

        if (pd.notna(vix) and vix >= 30) or (pd.notna(mom) and mom <= -0.12):
            regime = 'Risk-off / Stress'
        elif pd.notna(cpi) and cpi >= 4.0 and pd.notna(fed) and fed >= 4.0:
            regime = 'High inflation / Tight policy'
        elif pd.notna(term) and term < -0.25 and (pd.isna(mom) or mom < 0.04):
            regime = 'Recession risk'
        elif above and pd.notna(mom) and mom > 0.06:
            regime = 'Risk-on / Momentum'
        else:
            regime = 'Neutral'
        regimes.append({'date': r['date'], 'market_regime': regime, 'market_ret_63d': mom, 'vix': vix, 'cpi_yoy': cpi, 'fed_funds': fed})

    out = pd.DataFrame(regimes).drop_duplicates('date')
    return out


def attach_regime(dataset: pd.DataFrame) -> pd.DataFrame:
    regimes = classify_market_regime(dataset)
    return dataset.merge(regimes[['date', 'market_regime']], on='date', how='left')
