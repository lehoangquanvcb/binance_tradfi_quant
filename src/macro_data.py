"""US macro and market proxy data loader.

Design goals for Streamlit Cloud:
- Works without a FRED API key.
- Uses FRED when available.
- Falls back to Yahoo Finance market proxies so Credit-Macro and Market Regime
  dashboards still have a real time series instead of one neutral placeholder row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .config import FRED_API_KEY


FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "us_10y_yield": "DGS10",
    "us_2y_yield": "DGS2",
    "cpi_index": "CPIAUCSL",
    "unemployment_rate": "UNRATE",
    "industrial_production": "INDPRO",
    "m2_money_supply": "M2SL",
    "high_yield_spread": "BAMLH0A0HYM2",
    "investment_grade_spread": "BAMLC0A0CM",
}

# Yahoo tickers that are usually available without keys.
YF_MACRO = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "us_10y_yield_yahoo": "^TNX",   # index points: 10y yield * 10
    "us_13w_yield_yahoo": "^IRX",   # proxy when 2Y is unavailable without FRED
}


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        level0 = [str(x).lower() for x in out.columns.get_level_values(0)]
        price_names = {"open", "high", "low", "close", "adj close", "volume"}
        if any(x in price_names for x in level0):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    out = out.loc[:, ~pd.Index(out.columns).duplicated()].copy()
    return out


def fetch_fred(start: str = "2015-01-01") -> pd.DataFrame:
    """Fetch FRED data if a key is configured; otherwise return empty."""
    if not FRED_API_KEY:
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        frames = []
        for name, sid in FRED_SERIES.items():
            try:
                s = fred.get_series(sid, observation_start=start).rename(name)
                frames.append(s)
            except Exception as e:
                print(f"FRED skipped {sid}: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index().rename_axis("date").reset_index()
    except Exception as e:
        print(f"FRED unavailable: {e}")
        return pd.DataFrame()


def fetch_yfinance_macro(start: str = "2015-01-01") -> pd.DataFrame:
    """Fetch market proxies from Yahoo Finance.

    This gives the Credit-Macro dashboard useful history even without FRED.
    """
    frames = []
    for name, ticker in YF_MACRO.items():
        try:
            d = yf.download(ticker, start=start, progress=False, auto_adjust=True, group_by="column", threads=False)
            if d is None or d.empty:
                continue
            d = _flatten_yfinance_columns(d)
            col = "Close" if "Close" in d.columns else ("Adj Close" if "Adj Close" in d.columns else None)
            if col is None:
                continue
            s = pd.to_numeric(d[col], errors="coerce").rename(name)
            frames.append(s)
        except Exception as e:
            print(f"Yahoo macro skipped {ticker}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index().rename_axis("date").reset_index()
    return out


def _enrich_macro_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Create canonical aliases used by older and newer engines."""
    if df is None or df.empty:
        return pd.DataFrame()
    m = df.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date"]).sort_values("date")

    # Yahoo yield indices are quoted as yield * 10.
    if "us_10y_yield" not in m.columns and "us_10y_yield_yahoo" in m.columns:
        m["us_10y_yield"] = pd.to_numeric(m["us_10y_yield_yahoo"], errors="coerce") / 10.0
    if "us_2y_yield" not in m.columns and "us_13w_yield_yahoo" in m.columns:
        # Short-rate proxy when FRED DGS2 is unavailable.
        m["us_2y_yield"] = pd.to_numeric(m["us_13w_yield_yahoo"], errors="coerce") / 10.0

    # If CPI is an index, create YoY inflation.
    if "cpi_yoy" not in m.columns and "cpi_index" in m.columns:
        m["cpi_yoy"] = pd.to_numeric(m["cpi_index"], errors="coerce").pct_change(12) * 100

    # Credit spread proxy if true FRED credit spreads are unavailable.
    # It is not a true spread, but it gives a market-stress time series instead of all zeros.
    if "high_yield_spread" not in m.columns and "vix" in m.columns:
        v = pd.to_numeric(m["vix"], errors="coerce")
        m["high_yield_spread"] = (2.5 + (v - 15.0).clip(lower=0) * 0.12).clip(lower=2.0, upper=12.0)
    if "investment_grade_spread" not in m.columns and "high_yield_spread" in m.columns:
        m["investment_grade_spread"] = pd.to_numeric(m["high_yield_spread"], errors="coerce") * 0.35

    # Fallback policy rate when FRED is absent.
    if "fed_funds_rate" not in m.columns:
        if "us_13w_yield_yahoo" in m.columns:
            m["fed_funds_rate"] = pd.to_numeric(m["us_13w_yield_yahoo"], errors="coerce") / 10.0
        else:
            m["fed_funds_rate"] = np.nan

    # Uppercase aliases for legacy modules.
    alias_map = {
        "us_10y_yield": "US10Y",
        "us_2y_yield": "US2Y",
        "fed_funds_rate": "FEDFUNDS",
        "cpi_yoy": "CPI_YOY",
        "high_yield_spread": "HY_SPREAD",
        "investment_grade_spread": "IG_SPREAD",
        "vix": "VIXCLS",
        "dxy": "DTWEXBGS",
    }
    for src, dst in alias_map.items():
        if src in m.columns and dst not in m.columns:
            m[dst] = pd.to_numeric(m[src], errors="coerce")

    m = m.sort_values("date").ffill()
    return m


def load_macro(start: str = "2015-01-01") -> pd.DataFrame:
    fred = fetch_fred(start)
    yfmac = fetch_yfinance_macro(start)
    if fred.empty and yfmac.empty:
        return pd.DataFrame()
    if fred.empty:
        df = yfmac
    elif yfmac.empty:
        df = fred
    else:
        df = pd.merge(fred, yfmac, on="date", how="outer")
    return _enrich_macro_aliases(df)
