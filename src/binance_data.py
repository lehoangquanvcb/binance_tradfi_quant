import time
from typing import List, Optional
import requests
import pandas as pd

BASE = 'https://fapi.binance.com'
TRADFI_HINTS = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','SPY','QQQ','GLD','SLV','XAU','USOIL','SAMSUNG','HYUNDAI','SKHYNIX']

def get_exchange_info() -> dict:
    r = requests.get(f'{BASE}/fapi/v1/exchangeInfo', timeout=20)
    r.raise_for_status()
    return r.json()

def discover_tradfi_symbols(fallback: Optional[List[str]] = None) -> List[str]:
    """Discover Binance USDⓈ-M symbols likely representing TradFi perpetuals.
    Binance's public exchangeInfo does not always expose a clean 'TradFi' tag, so we filter
    by known stock/index/commodity symbol hints and keep perpetual USDT contracts.
    """
    try:
        info = get_exchange_info()
        out = []
        for s in info.get('symbols', []):
            symbol = s.get('symbol','')
            contract = s.get('contractType','')
            status = s.get('status','')
            if status == 'TRADING' and contract == 'PERPETUAL' and symbol.endswith('USDT'):
                if any(h in symbol.upper() for h in TRADFI_HINTS):
                    out.append(symbol)
        return sorted(set(out)) or (fallback or [])
    except Exception:
        return fallback or []

def fetch_klines(symbol: str, interval='1d', start_ms: Optional[int]=None, end_ms: Optional[int]=None, limit=1500) -> pd.DataFrame:
    rows = []
    current = start_ms
    while True:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        if current: params['startTime'] = current
        if end_ms: params['endTime'] = end_ms
        r = requests.get(f'{BASE}/fapi/v1/klines', params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data: break
        rows.extend(data)
        last_close = data[-1][6]
        if len(data) < limit or (end_ms and last_close >= end_ms): break
        current = last_close + 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame()
    cols = ['open_time','open','high','low','close','volume','close_time','quote_volume','n_trades','taker_buy_base','taker_buy_quote','ignore']
    df = pd.DataFrame(rows, columns=cols)
    num_cols = ['open','high','low','close','volume','quote_volume','taker_buy_base','taker_buy_quote']
    df[num_cols] = df[num_cols].astype(float)
    df['date'] = pd.to_datetime(df['open_time'], unit='ms').dt.date
    df['symbol'] = symbol
    return df[['date','symbol','open','high','low','close','volume','quote_volume','n_trades']]
