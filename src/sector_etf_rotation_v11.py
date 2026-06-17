"""V11 Sector ETF Rotation Engine.
Ranks sector ETFs and recommends overweight/underweight sleeves for CIO allocation.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

SECTOR_MAP = {
    "XLK":"Technology", "XLF":"Financials", "XLE":"Energy", "XLV":"Healthcare",
    "XLI":"Industrials", "XLP":"Consumer Staples", "XLY":"Consumer Discretionary",
    "XLU":"Utilities", "XLB":"Materials", "XLRE":"Real Estate", "XLC":"Communication",
    "QQQ":"Nasdaq 100", "SPY":"S&P 500", "DIA":"Dow Industrials", "IWM":"Small Caps",
}

def _safe_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(50.0, index=s.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s.clip(lo, hi)-lo)/(hi-lo)*100).fillna(50)

def build_sector_etf_rotation(close_panel: pd.DataFrame, legacy_sector: pd.DataFrame | None = None) -> pd.DataFrame:
    if close_panel is None or close_panel.empty:
        return pd.DataFrame(columns=["symbol","sector","score","action","target_weight","momentum_1m","momentum_3m","momentum_6m","volatility_60d","relative_strength_60d"])
    px = close_panel.copy().ffill()
    sector_symbols = [s for s in SECTOR_MAP if s in px.columns]
    if not sector_symbols:
        return pd.DataFrame()
    ret = px[sector_symbols].pct_change()
    benchmark = px["SPY"].pct_change(60) if "SPY" in px.columns else px[sector_symbols].mean(axis=1).pct_change(60)
    rows=[]
    for s in sector_symbols:
        series=px[s].dropna()
        if len(series)<80:
            continue
        r=series.pct_change()
        rows.append({
            "symbol":s,
            "sector":SECTOR_MAP.get(s,s),
            "momentum_1m": float(series.pct_change(21).iloc[-1]) if len(series)>22 else 0.0,
            "momentum_3m": float(series.pct_change(63).iloc[-1]) if len(series)>64 else 0.0,
            "momentum_6m": float(series.pct_change(126).iloc[-1]) if len(series)>127 else 0.0,
            "volatility_60d": float(r.tail(60).std()*np.sqrt(252)) if r.tail(60).notna().sum()>20 else 0.0,
            "relative_strength_60d": float(series.pct_change(60).iloc[-1] - benchmark.iloc[-1]) if len(series)>61 and pd.notna(benchmark.iloc[-1]) else 0.0,
        })
    df=pd.DataFrame(rows)
    if df.empty:
        return df
    score=(0.25*_safe_norm(df["momentum_1m"])+0.25*_safe_norm(df["momentum_3m"])+0.20*_safe_norm(df["momentum_6m"])+0.20*_safe_norm(df["relative_strength_60d"])+0.10*(100-_safe_norm(df["volatility_60d"])))
    df["score"]=score.round(2)
    df["rank"]=df["score"].rank(ascending=False, method="first").astype(int)
    df["action"]=pd.cut(df["score"], bins=[-1,35,50,65,100], labels=["UNDERWEIGHT","NEUTRAL_MINUS","NEUTRAL_PLUS","OVERWEIGHT"]).astype(str)
    # CIO sleeve weights from score, capped and normalized to 100% among sectors.
    raw=(df["score"].clip(20,90)-20)
    if raw.sum()<=0:
        df["target_weight"]=1/len(df)
    else:
        df["target_weight"]=(raw/raw.sum()).clip(0.03,0.25)
        df["target_weight"]=df["target_weight"]/df["target_weight"].sum()
    return df.sort_values(["rank","symbol"]).reset_index(drop=True)
