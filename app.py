import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DATA_PROCESSED, OUTPUTS, ROOT
from src.pipeline import run_all
from src.risk import position_size
from src.paper_trading import append_paper_trade, load_paper_trades
from src.kill_switch import evaluate_kill_switch
from src.transaction_cost import estimate_transaction_cost
from src.database import read_table, log_event
from src.oms import OrderTicket, create_order, load_orders, approval_required
from src.dynamic_leverage import leverage_multiplier
from src.execution_quality import execution_quality_report


def safe_read_csv(path):
    try:
        if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def show_empty_run_message(label="Run / Refresh model first."):
    st.info(label)


st.set_page_config(page_title="V7 Market Intelligence Platform", layout="wide")
st.title("V7 Market Intelligence Platform")
st.caption(
    "Market Timing → Sector Rotation → Stock Ranking → Exit Watchlist, with portfolio risk, OMS, paper trading and model governance."
)

with st.sidebar:
    st.header("Controls")
    start = st.text_input("Start date", "2018-01-01")
    prefer = st.selectbox("Data source priority", ["yahoo", "binance"])
    nav = st.number_input("Portfolio NAV (USD)", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("Risk per trade", 0.001, 0.03, 0.01, 0.001)
    run_wf = st.checkbox("Run walk-forward backtest", value=False)
    backtest_mode = st.selectbox(
        "Backtest mode", ["fast", "standard", "full"], index=0,
        help="Fast is recommended for Streamlit Cloud; Full can take many minutes."
    )
    testnet_mode = st.checkbox("Binance testnet / sandbox mode", value=True)
    live_mode = st.checkbox(
        "Live mode enabled", value=False,
        help="Real orders remain gated by execution.py, OMS, approval and kill-switch controls."
    )
    if st.button("Run / Refresh V7 model"):
        with st.spinner(f"Running V7 Market Intelligence pipeline... Backtest={run_wf}, mode={backtest_mode}"):
            metrics, signals, risks, regimes, portfolio, kill, bt_summary = run_all(
                start=start, prefer=prefer, nav=nav,
                run_walk_forward=run_wf, backtest_mode=backtest_mode
            )
            st.success(f"Done. AUC={metrics.get('auc')}, Accuracy={safe_float(metrics.get('accuracy')):.2%}")

paths = {
    "signals": DATA_PROCESSED / "latest_signals.csv",
    "ensemble": DATA_PROCESSED / "ensemble_signals.csv",
    "risk": DATA_PROCESSED / "risk_metrics.csv",
    "regimes": DATA_PROCESSED / "market_regimes.csv",
    "portfolio": DATA_PROCESSED / "target_portfolio.csv",
    "monitoring": DATA_PROCESSED / "model_monitoring.csv",
    "xai": DATA_PROCESSED / "feature_explainability.csv",
    "weights": DATA_PROCESSED / "v5_portfolio_construction.csv",
    "factors": DATA_PROCESSED / "v5_factor_scores.csv",
    "exposure": DATA_PROCESSED / "v5_factor_exposure.csv",
    "macro_credit": DATA_PROCESSED / "v55_macro_credit_dashboard.csv",
    "overlay": DATA_PROCESSED / "v5_credit_macro_overlay.csv",
    "inst_risk": DATA_PROCESSED / "v5_institutional_risk.csv",
    "stress": DATA_PROCESSED / "v5_stress_scenarios.csv",
    "note": DATA_PROCESSED / "v5_ai_research_note.txt",
    "governance": DATA_PROCESSED / "v5_model_governance_latest.csv",
    "validation": DATA_PROCESSED / "v5_model_validation.csv",
    "v6_metrics": DATA_PROCESSED / "v6_model_metrics.csv",
    "v6_feature_power": DATA_PROCESSED / "v6_feature_power.csv",
    "v6_alpha_quality": DATA_PROCESSED / "v6_alpha_quality.csv",
    "market_timing": DATA_PROCESSED / "v62_market_timing.csv",
    "sector_rotation": DATA_PROCESSED / "v62_sector_rotation.csv",
    "stock_ranking": DATA_PROCESSED / "v62_stock_ranking.csv",
    "exit_watchlist": DATA_PROCESSED / "v62_exit_watchlist.csv",
}

# Lean tab set: removed duplicated/low-value V5.5/V6 diagnostic tabs and folded useful diagnostics into Governance.
tabs = st.tabs([
    "1. Market Timing",
    "2. Sector Rotation",
    "3. Stock Ranking",
    "4. Exit Watchlist",
    "5. Signals",
    "6. Portfolio",
    "7. Factor Exposure",
    "8. Credit-Macro",
    "9. Institutional Risk",
    "10. OMS & Approval",
    "11. Paper Trading",
    "12. Execution Quality",
    "13. AI Research",
    "14. Model Governance",
    "15. Database",
])

with tabs[0]:
    st.subheader("Market Timing")
    mt = safe_read_csv(paths["market_timing"])
    if not mt.empty:
        latest = mt.sort_values("date").tail(1)
        row = latest.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market Timing Score", f"{safe_float(row.get('market_timing_score')):.1f}/100")
        c2.metric("Regime", str(row.get("timing_regime", "N/A")))
        c3.metric("Suggested Equity", f"{safe_float(row.get('suggested_equity_allocation')):.0%}")
        c4.metric("Suggested Cash", f"{safe_float(row.get('suggested_cash_allocation')):.0%}")
        ycols = [c for c in ["market_timing_score", "trend_score", "momentum_score", "breadth_score", "credit_macro_score"] if c in mt.columns]
        if ycols:
            st.plotly_chart(px.line(mt.tail(750), x="date", y=ycols, title="Market Timing Components"), use_container_width=True)
        st.dataframe(mt.tail(500), use_container_width=True)
    else:
        show_empty_run_message("Run / Refresh V7 model first to generate market timing.")

with tabs[1]:
    st.subheader("Sector Rotation")
    sr = safe_read_csv(paths["sector_rotation"])
    if not sr.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Overweight / leadership sectors**")
            st.dataframe(sr[sr.get("recommendation", "").isin(["Overweight"])].head(10), use_container_width=True)
        with c2:
            st.markdown("**Underweight / exit sectors**")
            st.dataframe(sr[sr.get("recommendation", "").isin(["Underweight", "Exit"])].head(10), use_container_width=True)
        st.plotly_chart(px.bar(sr.sort_values("sector_score"), x="sector_score", y="sector", color="recommendation", orientation="h", title="Sector Rotation Score"), use_container_width=True)
        st.dataframe(sr, use_container_width=True)
    else:
        show_empty_run_message("Run / Refresh V7 model first to generate sector rotation.")

with tabs[2]:
    st.subheader("Stock Ranking")
    rk = safe_read_csv(paths["stock_ranking"])
    if not rk.empty:
        actions = sorted(rk["action"].dropna().unique().tolist()) if "action" in rk.columns else []
        selected = st.multiselect("Action filter", actions, default=actions)
        show = rk[rk["action"].isin(selected)] if selected and "action" in rk.columns else rk
        c1, c2, c3 = st.columns(3)
        c1.metric("Buy candidates", int((rk.get("action") == "BUY").sum()) if "action" in rk else 0)
        c2.metric("Hold candidates", int((rk.get("action") == "HOLD").sum()) if "action" in rk else 0)
        c3.metric("Exit candidates", int((rk.get("action") == "EXIT").sum()) if "action" in rk else 0)
        st.plotly_chart(px.bar(show.head(30).sort_values("stock_score"), x="stock_score", y="symbol", color="action", orientation="h", title="Top Stock Candidates"), use_container_width=True)
        st.dataframe(show.sort_values("stock_score", ascending=False), use_container_width=True)
    else:
        show_empty_run_message("Run / Refresh V7 model first to generate stock ranking.")

with tabs[3]:
    st.subheader("Exit Watchlist")
    ex = safe_read_csv(paths["exit_watchlist"])
    if not ex.empty:
        st.warning("Review these names for de-risking, stop tightening or exit planning.")
        st.dataframe(ex, use_container_width=True)
        if "severity" in ex.columns:
            st.plotly_chart(px.histogram(ex, x="severity", color="action", title="Exit Watchlist by Severity"), use_container_width=True)
    else:
        st.success("No exit candidates under current rules.")

with tabs[4]:
    st.subheader("Signals")
    sig = safe_read_csv(paths["ensemble"])
    if sig.empty:
        sig = safe_read_csv(paths["signals"])
    if not sig.empty:
        show_cols = [c for c in ["date", "symbol", "close", "prob_up", "signal", "cost_adjusted_signal", "ensemble_score", "ensemble_signal", "market_regime", "net_edge_bps"] if c in sig.columns]
        sort_col = "ensemble_score" if "ensemble_score" in sig.columns else ("prob_up" if "prob_up" in sig.columns else show_cols[0])
        st.dataframe(sig[show_cols].sort_values(sort_col, ascending=False), use_container_width=True)
        if sort_col in sig.columns and "symbol" in sig.columns:
            st.plotly_chart(px.bar(sig.sort_values(sort_col), x=sort_col, y="symbol", orientation="h", title="Signal Score"), use_container_width=True)
    else:
        show_empty_run_message()

with tabs[5]:
    st.subheader("Institutional Portfolio")
    w = safe_read_csv(paths["weights"])
    if not w.empty:
        st.dataframe(w, use_container_width=True)
        method = st.selectbox("Portfolio method", [c for c in w.columns if c != "symbol"])
        st.plotly_chart(px.pie(w, names="symbol", values=method, title=f"Target Weights - {method}"), use_container_width=True)
    else:
        show_empty_run_message()

with tabs[6]:
    st.subheader("Factor Exposure")
    factors = safe_read_csv(paths["factors"])
    expo = safe_read_csv(paths["exposure"])
    c1, c2 = st.columns(2)
    with c1:
        if not factors.empty:
            st.dataframe(factors, use_container_width=True)
    with c2:
        if not expo.empty:
            st.dataframe(expo, use_container_width=True)
            if "exposure" in expo.columns:
                st.plotly_chart(px.bar(expo, x=expo.columns[0], y="exposure", title="Portfolio Factor Exposure"), use_container_width=True)
    if factors.empty and expo.empty:
        show_empty_run_message()

with tabs[7]:
    st.subheader("Credit-Macro")
    mc = safe_read_csv(paths["macro_credit"])
    overlay = safe_read_csv(paths["overlay"])
    if not mc.empty:
        latest = mc.sort_values("date").tail(1)
        row = latest.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Regime", str(row.get("risk_regime", "N/A")))
        c2.metric("6M Recession", f"{safe_float(row.get('recession_probability_6m')):.1%}")
        c3.metric("Equity Risk", f"{safe_float(row.get('equity_risk_score')):.1f}/100")
        c4.metric("Credit Stress", f"{safe_float(row.get('credit_stress_score')):.1f}/100")
        ycols = [c for c in ["equity_risk_score", "credit_stress_score", "recession_probability_6m", "macro_credit_composite"] if c in mc.columns]
        if ycols:
            st.plotly_chart(px.line(mc.tail(750), x="date", y=ycols, title="Macro-Credit Risk Dashboard"), use_container_width=True)
        st.dataframe(mc.tail(500), use_container_width=True)
    elif not overlay.empty:
        st.dataframe(overlay.tail(500), use_container_width=True)
        if "credit_macro_score" in overlay.columns:
            st.plotly_chart(px.line(overlay.tail(500), x="date", y="credit_macro_score", title="Credit-Macro Score"), use_container_width=True)
    else:
        show_empty_run_message()

with tabs[8]:
    st.subheader("Institutional Risk")
    risk = safe_read_csv(paths["risk"])
    if not risk.empty:
        kill = evaluate_kill_switch(risk)
        if kill.get("allow_trading", True):
            st.success("Trading allowed: no hard breach detected.")
        else:
            st.error("Trading blocked by kill switch.")
            st.write(kill.get("breaches", []))
        st.dataframe(risk, use_container_width=True)
    inst = safe_read_csv(paths["inst_risk"])
    stress = safe_read_csv(paths["stress"])
    if not inst.empty:
        st.subheader("Portfolio VaR / CVaR / Concentration")
        st.dataframe(inst, use_container_width=True)
    if not stress.empty:
        st.subheader("Stress Scenarios")
        st.dataframe(stress, use_container_width=True)
        if {"scenario", "portfolio_loss"}.issubset(stress.columns):
            st.plotly_chart(px.bar(stress, x="scenario", y="portfolio_loss", title="Scenario Portfolio Loss"), use_container_width=True)
    if risk.empty and inst.empty and stress.empty:
        show_empty_run_message()

with tabs[9]:
    st.subheader("Order Management System & Human Approval Gate")
    sig = safe_read_csv(paths["signals"])
    if not sig.empty:
        symbol = st.selectbox("Symbol", sig["symbol"].astype(str).tolist(), key="oms_symbol")
        row = sig[sig.symbol.astype(str) == symbol].iloc[0]
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        side = st.selectbox("Side", ["BUY", "SELL"], key="oms_side")
        notional = safe_float(row.get("close")) * qty
        need_approval = approval_required(nav, notional)
        lev_mult = leverage_multiplier(0.25, 0.02, 0.05, str(row.get("market_regime", "Neutral")))
        c1, c2 = st.columns(2)
        c1.metric("Order notional", f"${notional:,.0f}")
        c2.metric("Dynamic leverage multiplier", f"{lev_mult:.0%}")
        if need_approval:
            st.warning("Human approval required for this order.")
        else:
            st.info("Small order: approval can be automatic in paper mode.")
        if st.button("Create OMS ticket"):
            ticket = OrderTicket(symbol=symbol, side=side, quantity=qty, reason=f"V7 signal; testnet={testnet_mode}", live_mode=live_mode)
            rec = create_order(ticket)
            log_event("OMS", "CREATE_TICKET", json.dumps(rec), symbol=symbol, live_mode_enabled=live_mode)
            st.success("OMS ticket created. Status = PENDING_APPROVAL.")
    st.dataframe(load_orders(), use_container_width=True)

with tabs[10]:
    st.subheader("Paper Trading")
    sig = safe_read_csv(paths["signals"])
    if not sig.empty:
        symbol = st.selectbox("Symbol for paper order", sig["symbol"].astype(str).tolist(), key="paper_symbol")
        row = sig[sig.symbol.astype(str) == symbol].iloc[0]
        atr = safe_float(row.get("atr_14"), 0.0)
        if atr <= 0:
            atr = max(safe_float(row.get("close"), 1.0) * 0.03, 0.01)
        ps = position_size(nav, safe_float(row.get("close")), atr, risk_pct=risk_pct)
        st.json(ps)
        qty = st.number_input("Paper quantity", min_value=0.0, value=float(max(ps.get("qty", 0), 0)), step=1.0)
        side = st.selectbox("Side for paper trade", ["BUY", "SELL"], key="paper_side")
        if st.button("Record paper trade"):
            cost = estimate_transaction_cost(safe_float(row.get("close")), qty, side)
            append_paper_trade(symbol, side, qty, safe_float(row.get("close")), safe_float(row.get("prob_up")), stop_loss=ps.get("stop"), reason=f"paper_trade_cost={cost}", ledger_path=OUTPUTS / "paper_trades.csv")
            log_event("ORDER", "PAPER_TRADE", json.dumps(cost), symbol=symbol, live_mode_enabled=live_mode)
            st.success("Paper trade recorded.")
        st.dataframe(load_paper_trades(OUTPUTS / "paper_trades.csv"), use_container_width=True)
    else:
        show_empty_run_message()

with tabs[11]:
    st.subheader("Execution Quality Analytics")
    orders = load_orders()
    st.dataframe(execution_quality_report(orders), use_container_width=True)
    st.caption("Execution analytics are populated after OMS/paper/live order activity is recorded.")

with tabs[12]:
    st.subheader("AI Research Assistant")
    if paths["note"].exists():
        st.write(paths["note"].read_text(encoding="utf-8"))
    else:
        show_empty_run_message()
    xai = safe_read_csv(paths["xai"])
    if not xai.empty and {"feature", "importance"}.issubset(xai.columns):
        st.plotly_chart(px.bar(xai.head(25).sort_values("importance"), x="importance", y="feature", orientation="h", title="Feature Importance / SHAP"), use_container_width=True)

with tabs[13]:
    st.subheader("Model Governance")
    gov = safe_read_csv(paths["governance"])
    val = safe_read_csv(paths["validation"])
    mon = safe_read_csv(paths["monitoring"])
    v6m = safe_read_csv(paths["v6_metrics"])
    fpower = safe_read_csv(paths["v6_feature_power"])
    alpha = safe_read_csv(paths["v6_alpha_quality"])
    for title, df in [
        ("Model Inventory", gov), ("Validation", val), ("Monitoring", mon),
        ("V6 Model Metrics", v6m), ("Feature Power", fpower), ("Alpha Quality", alpha),
    ]:
        if not df.empty:
            st.markdown(f"**{title}**")
            st.dataframe(df, use_container_width=True)

with tabs[14]:
    st.subheader("Database")
    st.write(str(ROOT / "data" / "quant_platform.sqlite"))
    table = st.selectbox("Table", ["prices", "macro", "signals", "orders", "pnl", "compliance_log", "model_monitoring"])
    st.dataframe(read_table(table).tail(500), use_container_width=True)
