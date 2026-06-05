from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

TRADE_COLUMNS = ['timestamp','mode','symbol','side','qty','price','notional','prob_up','stop_loss','take_profit','status','reason']


def append_paper_trade(symbol: str, side: str, qty: float, price: float, prob_up: float,
                       stop_loss: float | None = None, take_profit: float | None = None,
                       reason: str = '', ledger_path: Path | str = 'outputs/paper_trades.csv') -> pd.DataFrame:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'mode': 'paper', 'symbol': symbol, 'side': side, 'qty': qty,
        'price': price, 'notional': qty * price, 'prob_up': prob_up,
        'stop_loss': stop_loss, 'take_profit': take_profit,
        'status': 'FILLED_PAPER', 'reason': reason,
    }
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_paper_trades(ledger_path: Path | str = 'outputs/paper_trades.csv') -> pd.DataFrame:
    path = Path(ledger_path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=TRADE_COLUMNS)
