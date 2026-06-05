import ccxt
from .config import BINANCE_API_KEY, BINANCE_API_SECRET, TRADING_MODE

def get_exchange():
    return ccxt.binanceusdm({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

def place_order(symbol: str, side: str, amount: float, order_type='market'):
    if TRADING_MODE != 'live':
        return {'mode': 'paper', 'symbol': symbol, 'side': side, 'amount': amount, 'order_type': order_type, 'status': 'simulated'}
    ex = get_exchange()
    ccxt_symbol = symbol.replace('USDT','/USDT:USDT')
    return ex.create_order(ccxt_symbol, order_type, side.lower(), amount)
