import pandas as pd
import yfinance as yf
from .config import FRED_API_KEY

FRED_SERIES = {
    'fed_funds_rate': 'FEDFUNDS',
    'us_10y_yield': 'DGS10',
    'us_2y_yield': 'DGS2',
    'cpi_index': 'CPIAUCSL',
    'unemployment_rate': 'UNRATE',
    'industrial_production': 'INDPRO',
    'm2_money_supply': 'M2SL',
    'high_yield_spread': 'BAMLH0A0HYM2',
}
YF_MACRO = {'vix': '^VIX', 'dxy': 'DX-Y.NYB', 'sp500': '^GSPC', 'nasdaq': '^IXIC'}

def fetch_fred(start='2015-01-01') -> pd.DataFrame:
    if not FRED_API_KEY:
        return pd.DataFrame()
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    frames = []
    for name, sid in FRED_SERIES.items():
        s = fred.get_series(sid, observation_start=start).rename(name)
        frames.append(s)
    return pd.concat(frames, axis=1).sort_index().rename_axis('date').reset_index()

def fetch_yfinance_macro(start='2015-01-01') -> pd.DataFrame:
    frames = []
    for name, ticker in YF_MACRO.items():
        try:
            d = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if not d.empty:
                s = d['Close'].rename(name)
                frames.append(s)
        except Exception:
            pass
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index().rename_axis('date').reset_index()

def load_macro(start='2015-01-01') -> pd.DataFrame:
    fred = fetch_fred(start)
    yfmac = fetch_yfinance_macro(start)
    if fred.empty and yfmac.empty: return pd.DataFrame()
    if fred.empty: df = yfmac
    elif yfmac.empty: df = fred
    else: df = pd.merge(fred, yfmac, on='date', how='outer')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').ffill()
    return df
