"""V8.6 CIO Dashboard Summary Builder."""
from __future__ import annotations

import pandas as pd


def _fmt_pct(x, default: float = 0.0) -> str:
    try:
        return f"{float(x):.0%}"
    except Exception:
        return f"{default:.0%}"


def _top_list(df: pd.DataFrame, col: str, n: int = 5) -> str:
    if df is None or df.empty or col not in df.columns:
        return "N/A"
    vals = df[col].astype(str).head(n).tolist()
    return ", ".join(vals) if vals else "N/A"


def build_cio_summary(
    market_regime: pd.DataFrame,
    sector_rotation: pd.DataFrame,
    stock_selection: pd.DataFrame,
    exit_watchlist: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> str:
    if market_regime is not None and not market_regime.empty:
        mr = market_regime.sort_values("date").iloc[-1]
        regime = str(mr.get("market_regime", "UNKNOWN"))
        score = float(mr.get("regime_score", 50) or 50)
        confidence = float(mr.get("confidence", 0) or 0)
        eq_weight = float(mr.get("recommended_equity_weight", 0.60) or 0.60)
        cash_weight = float(mr.get("recommended_cash_weight", 0.20) or 0.20)
    else:
        regime, score, confidence, eq_weight, cash_weight = "UNKNOWN", 50.0, 0.0, 0.60, 0.20

    top_sectors = "N/A"
    weak_sectors = "N/A"
    if sector_rotation is not None and not sector_rotation.empty:
        sr = sector_rotation.copy()
        top = sr[sr.get("sector_action", "").isin(["OVERWEIGHT", "NEUTRAL_PLUS"])] if "sector_action" in sr.columns else sr.head(3)
        weak = sr[sr.get("sector_action", "").isin(["UNDERWEIGHT", "EXIT"])] if "sector_action" in sr.columns else sr.tail(3)
        top_sectors = _top_list(top if not top.empty else sr.head(3), "sector", 3)
        weak_sectors = _top_list(weak if not weak.empty else sr.tail(3), "sector", 3)

    buy_df = pd.DataFrame()
    if stock_selection is not None and not stock_selection.empty:
        buy_df = stock_selection[stock_selection.get("decision", "").isin(["TOP_BUY", "BUY"])] if "decision" in stock_selection.columns else stock_selection.head(8)
        if buy_df.empty:
            buy_df = stock_selection.head(8)
    top_names = _top_list(buy_df, "symbol", 8)

    exits = _top_list(exit_watchlist, "symbol", 8)

    port_text = "N/A"
    if portfolio is not None and not portfolio.empty and {"symbol", "target_weight"}.issubset(portfolio.columns):
        pairs = []
        for _, r in portfolio.head(8).iterrows():
            try:
                pairs.append(f"{r['symbol']} {_fmt_pct(r['target_weight'])}")
            except Exception:
                pass
        port_text = ", ".join(pairs) if pairs else "N/A"

    return (
        "### CIO Morning Brief\n"
        f"**Market regime:** {regime} | **Score:** {score:.1f}/100 | **Confidence:** {confidence:.0%}.\n\n"
        f"**Suggested risk budget:** Equity {_fmt_pct(eq_weight)}, Cash {_fmt_pct(cash_weight)}.\n\n"
        f"**Overweight sectors:** {top_sectors}.\n\n"
        f"**Underweight / avoid sectors:** {weak_sectors}.\n\n"
        f"**Top buy ideas:** {top_names}.\n\n"
        f"**Exit / reduce watchlist:** {exits if exits != 'N/A' else 'None'}.\n\n"
        f"**Recommended allocation:** {port_text}.\n\n"
        "Decision rule: use this as a CIO decision-support layer; final trading requires human review, risk limits and liquidity checks."
    )
