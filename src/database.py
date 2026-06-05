"""SQLite data layer for the Binance TradFi Quant Platform V3."""
from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from .config import ROOT

DB_PATH = ROOT / 'data' / 'quant_platform.sqlite'

SCHEMA = {
    'prices': '''CREATE TABLE IF NOT EXISTS prices (
        date TEXT, symbol TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        quote_volume REAL, n_trades REAL, PRIMARY KEY(date, symbol)
    )''',
    'macro': '''CREATE TABLE IF NOT EXISTS macro (
        date TEXT PRIMARY KEY, vix REAL, dxy REAL, sp500 REAL, nasdaq REAL,
        fed_funds_rate REAL, us_10y_yield REAL, us_2y_yield REAL, high_yield_spread REAL
    )''',
    'signals': '''CREATE TABLE IF NOT EXISTS signals (
        run_id TEXT, date TEXT, symbol TEXT, close REAL, prob_up REAL, signal TEXT,
        rsi_14 REAL, atr_14 REAL, ret_20d REAL, strategy TEXT, PRIMARY KEY(run_id, symbol, strategy)
    )''',
    'orders': '''CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, symbol TEXT,
        side TEXT, qty REAL, price REAL, notional REAL, status TEXT, reason TEXT
    )''',
    'pnl': '''CREATE TABLE IF NOT EXISTS pnl (
        ts TEXT, nav REAL, daily_pnl REAL, cumulative_pnl REAL, drawdown REAL
    )''',
    'compliance_log': '''CREATE TABLE IF NOT EXISTS compliance_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event_type TEXT, user_id TEXT,
        symbol TEXT, action TEXT, details TEXT, live_mode_enabled INTEGER DEFAULT 0
    )''',
    'model_monitoring': '''CREATE TABLE IF NOT EXISTS model_monitoring (
        run_id TEXT PRIMARY KEY, ts TEXT, auc REAL, accuracy REAL, hit_rate REAL,
        avg_prob REAL, data_rows INTEGER, drift_score REAL, status TEXT
    )'''
}

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    for sql in SCHEMA.values():
        con.execute(sql)
    con.commit()
    return con

def upsert_dataframe(df: pd.DataFrame, table: str, db_path: Path = DB_PATH) -> None:
    if df is None or df.empty:
        return
    data = df.copy()
    for c in data.columns:
        if 'date' in c or c == 'ts':
            data[c] = pd.to_datetime(data[c], errors='coerce').astype(str)
    with connect(db_path) as con:
        data.to_sql(table, con, if_exists='append', index=False)

def replace_dataframe(df: pd.DataFrame, table: str, db_path: Path = DB_PATH) -> None:
    with connect(db_path) as con:
        if df is not None and not df.empty:
            x = df.copy()
            for c in x.columns:
                if 'date' in c or c == 'ts':
                    x[c] = pd.to_datetime(x[c], errors='coerce').astype(str)
            x.to_sql(table, con, if_exists='replace', index=False)

def read_table(table: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as con:
        try:
            return pd.read_sql_query(f'SELECT * FROM {table}', con)
        except Exception:
            return pd.DataFrame()

def log_event(event_type: str, action: str, details: str, symbol: str | None = None,
              user_id: str = 'system', live_mode_enabled: bool = False) -> None:
    row = pd.DataFrame([{
        'ts': now_utc(), 'event_type': event_type, 'user_id': user_id,
        'symbol': symbol, 'action': action, 'details': details,
        'live_mode_enabled': int(live_mode_enabled)
    }])
    with connect(DB_PATH) as con:
        row.to_sql('compliance_log', con, if_exists='append', index=False)
