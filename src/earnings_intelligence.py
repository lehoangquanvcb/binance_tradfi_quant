"""Earnings intelligence placeholders with real schema.

In production, populate this from Finnhub, Polygon, Nasdaq Data Link, or broker API.
"""
from __future__ import annotations
import hashlib
import pandas as pd


def build_earnings_intelligence(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        eps_surprise = ((h % 41) - 20) / 100
        revenue_surprise = (((h // 41) % 31) - 15) / 100
        guidance_score = (((h // 1271) % 201) - 100) / 100
        risk = max(0, abs(eps_surprise) + abs(revenue_surprise) - 0.10)
        thesis_score = 0.45 * eps_surprise + 0.35 * revenue_surprise + 0.20 * guidance_score
        rows.append({
            "symbol": s,
            "eps_surprise_proxy": eps_surprise,
            "revenue_surprise_proxy": revenue_surprise,
            "guidance_score_proxy": guidance_score,
            "earnings_event_risk": risk,
            "earnings_thesis_score": thesis_score,
        })
    return pd.DataFrame(rows)
