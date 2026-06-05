"""Alternative data layer: event calendar and sentiment placeholders.

The module is intentionally API-key neutral. Add NewsAPI, AlphaVantage, Finnhub,
Reddit, or X connectors in production; demo mode uses deterministic placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import pandas as pd


@dataclass
class AltDataConfig:
    enable_news: bool = False
    enable_social: bool = False
    enable_earnings: bool = True


def sentiment_score(text: str) -> float:
    pos = ["beat", "growth", "upgrade", "strong", "bullish", "record", "ai"]
    neg = ["miss", "downgrade", "weak", "bearish", "lawsuit", "recession", "cut"]
    t = text.lower()
    return (sum(w in t for w in pos) - sum(w in t for w in neg)) / max(len(pos), 1)


def demo_symbol_sentiment(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        h = int(hashlib.sha256(s.encode()).hexdigest(), 16)
        score = ((h % 200) - 100) / 100
        rows.append({"symbol": s, "news_sentiment": score * 0.4, "social_sentiment": score * 0.3, "event_risk": abs(score) * 0.5})
    return pd.DataFrame(rows).set_index("symbol")


def macro_event_calendar() -> pd.DataFrame:
    return pd.DataFrame([
        {"event": "FOMC decision", "importance": "High", "typical_risk": "rates/tech duration"},
        {"event": "CPI release", "importance": "High", "typical_risk": "inflation surprise"},
        {"event": "PCE inflation", "importance": "High", "typical_risk": "Fed reaction function"},
        {"event": "Nonfarm payrolls", "importance": "Medium", "typical_risk": "growth/wage pressure"},
        {"event": "Big Tech earnings", "importance": "High", "typical_risk": "AI/mega-cap concentration"},
    ])
