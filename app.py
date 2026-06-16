import json
import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import DATA_PROCESSED
from src.pipeline import run_all
from src.kill_switch import evaluate_kill_switch
from src.database import log_event
from src.oms import OrderTicket, create_order, load_orders, approval_required
from src.dynamic_leverage import leverage_multiplier


def safe_read_csv(path):
    try:
        if path is None or not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()


def safe_read_text(path):
    try:
        if path.exists() and path.stat().st_size > 0:
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


st.set_page_config(page_title="V8 CIO Market Intelligence Platform", layout="wide")
st.title("V8 CIO Market Intelligence Platform")
st.caption("Market Regime → Sector Rotation → Stock Selection → Exit Watchlist → Portfolio Recommendation, with risk governance and model monitoring.")

with st.sidebar:
    st.header("Controls")
    start = st.text_input("Start date", "2018-01-01")
    prefer = st.selectbox("Data source priority", ["yahoo", "binance"], index=0)
    nav = st.number_input("Portfolio NAV (USD)", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("Risk per trade", 0.001, 0.03, 0.01, 0.001)
    run_wf = st.checkbox("Run walk-forward backtest", value=False)
    backtest_mode = st.selectbox("Backtest mode", ["fast", "standard", "full"], index=0, help="Fast is recommended for Streamlit Cloud.")
    testnet_mode = st.checkbox("Binance testnet / sandbox mode", value=True)
    live_mode = st.checkbox("Live mode enabled", value=False, help="Live orders remain gated by OMS, approval and kill-switch controls.")
    if st.button("Run / Refresh V8 model"):
        with st.spinner(f"Running V8 CIO pipeline... Backtest={run_wf}, mode={backtest_mode}"):
            metrics, signals, risks, regimes, portfolio, kill, bt_summary = run_all(
                start=start,
                prefer=prefer,
                nav=nav,
                run_walk_forward=run_wf,
                backtest_mode=backtest_mode,
            )
            auc = metrics.get("auc")
            acc = metrics.get("accuracy")
            st.success(f"Done. AUC={auc}, Accuracy={acc:.2%}" if acc is not None else "Done.")

paths = {
    "signals": DATA_PROCESSED / "latest_signals.csv",
    "ensemble": DATA_PROCESSED / "ensemble_signals.csv",
    "risk": DATA_PROCESSED / "risk_metrics.csv",
    "portfolio": DATA_PROCESSED / "target_portfolio.csv",
    "v5_weights": DATA_PROCESSED / "v5_portfolio_construction.csv",
    "v5_factors": DATA_PROCESSED / "v5_factor_scores.csv",
    "v5_exposure": DATA_PROCESSED / "v5_factor_exposure.csv",
    "v5_overlay": DATA_PROCESSED / "v5_credit_macro_overlay.csv",
    "v55_macro_credit": DATA_PROCESSED / "v55_macro_credit_dashboard.csv",
    "v5_inst_risk": DATA_PROCESSED / "v5_institutional_risk.csv",
    "v5_stress": DATA_PROCESSED / "v5_stress_scenarios.csv",
    "v5_note": DATA_PROCESSED / "v5_ai_research_note.txt",
    "v5_gov": DATA_PROCESSED / "v5_model_governance_latest.csv",
    "v5_validation": DATA_PROCESSED / "v5_model_validation.csv",
    "monitoring": DATA_PROCESSED / "model_monitoring.csv",
    "xai": DATA_PROCESSED / "feature_explainability.csv",
    "v8_regime": DATA_PROCESSED / "v8_market_regime.csv",
    "v8_sector": DATA_PROCESSED / "v8_sector_rotation.csv",
    "v8_stock": DATA_PROCESSED / "v8_stock_selection.csv",
    "v8_exit": DATA_PROCESSED / "v8_exit_watchlist.csv",
    "v8_portfolio": DATA_PROCESSED / "v8_portfolio_recommendation.csv",
    "v8_summary": DATA_PROCESSED / "v8_cio_summary.txt",
}

tabs = st.tabs([
    "1. CIO Dashboard",
    "2. Market Regime",
    "3. Sector Rotation",
    "4. Stock Selection",
    "5. Exit Watchlist",
    "6. Signals",
    "7. Portfolio Construction",
    "8. Factor Exposure",
    "9. Credit-Macro",
    "10. Institutional Risk",
    "11. OMS Approval",
    "12. AI Research",
    "13. Model Governance",
])

with tabs[0]:
    st.header("CIO Dashboard")
    summary = safe_read_text(paths["v8_summary"])
    regime = safe_read_csv(paths["v8_regime"])
    sector = safe_read_csv(paths["v8_sector"])
    stock = safe_read_csv(paths["v8_stock"])
    exits = safe_read_csv(paths["v8_exit"])
    port = safe_read_csv(paths["v8_portfolio"])
    if summary:
        st.info(summary)
    if not regime.empty:
        latest = regime.sort_values("date").tail(1).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market Regime", str(latest.get("market_regime", "N/A")))
        c2.metric("Regime Score", f"{float(latest.get('regime_score', 0)):.1f}")
        c3.metric("Equity Weight", f"{float(latest.get('recommended_equity_weight', 0)):.0%}")
        c4.metric("Cash Weight", f"{float(latest.get('recommended_cash_weight', 0)):.0%}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Sectors")
        if not sector.empty:
            st.dataframe(sector.head(5), use_container_width=True)
        st.subheader("Top Stock Ideas")
        if not stock.empty:
            cols = [c for c in ["rank", "symbol", "stock_score", "decision", "sector_bucket", "market_regime"] if c in stock.columns]
            st.dataframe(stock[cols].head(10), use_container_width=True)
    with c2:
        st.subheader("Exit / Reduce Watchlist")
        if not exits.empty:
            st.dataframe(exits.head(10), use_container_width=True)
        else:
            st.success("No major exit candidates generated.")
        st.subheader("Recommended Portfolio")
        if not port.empty:
            st.dataframe(port, use_container_width=True)

with tabs[1]:
    st.header("Market Regime Engine")
    regime = safe_read_csv(paths["v8_regime"])
    if regime.empty:
        st.info("Run V8 model first.")
    else:
        st.dataframe(regime.tail(20), use_container_width=True)
        if "regime_score" in regime.columns:
            st.plotly_chart(px.line(regime.tail(750), x="date", y="regime_score", color="market_regime", title="Market Regime Score"), use_container_width=True)
        alloc_cols = [c for c in ["recommended_equity_weight", "recommended_bond_weight", "recommended_gold_weight", "recommended_cash_weight"] if c in regime.columns]
        if alloc_cols:
            st.plotly_chart(px.area(regime.tail(750), x="date", y=alloc_cols, title="Recommended Risk Budget"), use_container_width=True)

with tabs[2]:
    st.header("Sector Rotation")
    sector = safe_read_csv(paths["v8_sector"])
    if sector.empty:
        st.info("Run V8 model first.")
    else:
        st.dataframe(sector, use_container_width=True)
        st.plotly_chart(px.bar(sector.sort_values("sector_score"), x="sector_score", y="sector", color="sector_action", orientation="h", title="Sector Rotation Score"), use_container_width=True)

with tabs[3]:
    st.header("Stock Selection")
    stock = safe_read_csv(paths["v8_stock"])
    if stock.empty:
        st.info("Run V8 model first.")
    else:
        cols = [c for c in ["rank", "symbol", "close", "stock_score", "decision", "prob_up", "ensemble_score", "relative_strength_60d", "sector_bucket", "sector_action", "market_regime"] if c in stock.columns]
        st.dataframe(stock[cols], use_container_width=True)
        st.plotly_chart(px.bar(stock.head(25).sort_values("stock_score"), x="stock_score", y="symbol", color="decision", orientation="h", title="Top Stock Selection Scores"), use_container_width=True)

with tabs[4]:
    st.header("Exit Watchlist")
    exits = safe_read_csv(paths["v8_exit"])
    if exits.empty:
        st.success("No major exit/reduce candidates generated.")
    else:
        st.dataframe(exits, use_container_width=True)
        st.plotly_chart(px.bar(exits.sort_values("stock_score"), x="stock_score", y="symbol", color="severity", orientation="h", title="Exit Watchlist"), use_container_width=True)

with tabs[5]:
    st.header("Signals")
    sig = safe_read_csv(paths["ensemble"])
    if sig.empty:
        sig = safe_read_csv(paths["signals"])
    if sig.empty:
        st.info("Run V8 model first.")
    else:
        score_col = "ensemble_score" if "ensemble_score" in sig.columns else "prob_up"
        cols = [c for c in ["date", "symbol", "close", "prob_up", "signal", "cost_adjusted_signal", "ensemble_score", "ensemble_signal", "market_regime", "net_edge_bps"] if c in sig.columns]
        st.dataframe(sig[cols].sort_values(score_col, ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(sig.sort_values(score_col), x=score_col, y="symbol", orientation="h", title="Signal Score"), use_container_width=True)

with tabs[6]:
    st.header("Portfolio Construction")
    port = safe_read_csv(paths["v8_portfolio"])
    old = safe_read_csv(paths["v5_weights"])
    if not port.empty:
        st.subheader("V8 CIO Recommended Portfolio")
        st.dataframe(port, use_container_width=True)
        st.plotly_chart(px.pie(port, names="symbol", values="target_weight", title="V8 Target Weights"), use_container_width=True)
    elif not old.empty:
        st.subheader("Legacy Portfolio Construction")
        st.dataframe(old, use_container_width=True)
    else:
        st.info("Run V8 model first.")

with tabs[7]:
    st.header("Factor Exposure")
    factors = safe_read_csv(paths["v5_factors"])
    expo = safe_read_csv(paths["v5_exposure"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Symbol Factor Scores")
        if not factors.empty:
            st.dataframe(factors, use_container_width=True)
    with c2:
        st.subheader("Portfolio Factor Exposure")
        if not expo.empty:
            st.dataframe(expo, use_container_width=True)
            if "exposure" in expo.columns:
                st.plotly_chart(px.bar(expo, x=expo.columns[0], y="exposure", title="Factor Exposure"), use_container_width=True)

with tabs[8]:
    st.header("Credit-Macro")
    overlay = safe_read_csv(paths["v5_overlay"])
    macro_credit = safe_read_csv(paths["v55_macro_credit"])
    if not macro_credit.empty:
        st.subheader("Macro-Credit Intelligence")
        st.dataframe(macro_credit.tail(250), use_container_width=True)
        y = [c for c in ["recession_probability_6m", "equity_risk_score", "credit_stress_score", "macro_credit_composite"] if c in macro_credit.columns]
        if y:
            st.plotly_chart(px.line(macro_credit.tail(750), x="date", y=y, title="Macro-Credit Risk Indicators"), use_container_width=True)
    elif not overlay.empty:
        st.subheader("Legacy Credit-Macro Overlay")
        st.dataframe(overlay.tail(250), use_container_width=True)
        if "credit_macro_score" in overlay.columns:
            st.plotly_chart(px.line(overlay.tail(750), x="date", y="credit_macro_score", title="Credit-Macro Score"), use_container_width=True)
    else:
        st.info("No credit-macro data yet.")

with tabs[9]:
    st.header("Institutional Risk")
    risk = safe_read_csv(paths["risk"])
    if not risk.empty:
        kill = evaluate_kill_switch(risk)
        if kill.get("allow_trading", True):
            st.success("Trading allowed: no hard breach detected.")
        else:
            st.error("Trading blocked by kill switch.")
            st.write(kill.get("breaches", []))
        st.dataframe(risk, use_container_width=True)
    inst = safe_read_csv(paths["v5_inst_risk"])
    stress = safe_read_csv(paths["v5_stress"])
    if not inst.empty:
        st.subheader("VaR / CVaR / Concentration")
        st.dataframe(inst, use_container_width=True)
    if not stress.empty:
        st.subheader("Stress Scenarios")
        st.dataframe(stress, use_container_width=True)
        if {"scenario", "portfolio_loss"}.issubset(stress.columns):
            st.plotly_chart(px.bar(stress, x="scenario", y="portfolio_loss", title="Scenario Portfolio Loss"), use_container_width=True)

with tabs[10]:
    st.header("OMS & Human Approval")
    sig = safe_read_csv(paths["signals"])
    if sig.empty:
        st.info("Run V8 model first.")
    else:
        symbol = st.selectbox("Symbol", sig["symbol"].astype(str).tolist(), key="oms_symbol")
        row = sig[sig.symbol.astype(str) == symbol].iloc[0]
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        side = st.selectbox("Side", ["BUY", "SELL"], key="oms_side")
        notional = float(row.close) * qty
        need_approval = approval_required(nav, notional)
        lev_mult = leverage_multiplier(0.25, 0.02, 0.05, str(row.get("market_regime", "Neutral")))
        st.metric("Order notional", f"${notional:,.0f}")
        st.metric("Dynamic leverage multiplier", f"{lev_mult:.0%}")
        if need_approval:
            st.warning("Human approval required for this order.")
        else:
            st.info("Small order: approval can be automatic in paper/testnet mode.")
        if st.button("Create OMS ticket"):
            ticket = OrderTicket(symbol=symbol, side=side, quantity=qty, reason=f"V8 CIO signal; testnet={testnet_mode}", live_mode=live_mode)
            rec = create_order(ticket)
            log_event("OMS", "CREATE_TICKET", json.dumps(rec), symbol=symbol, live_mode_enabled=live_mode)
            st.success(f"Created ticket {rec['order_id']}")
        orders = load_orders()
        if not orders.empty:
            st.subheader("Order Book")
            st.dataframe(orders, use_container_width=True)

with tabs[11]:
    st.header("AI Research")
    note = safe_read_text(paths["v8_summary"])
    legacy = safe_read_text(paths["v5_note"])
    if note:
        st.subheader("CIO Brief")
        st.text(note)
    if legacy:
        st.subheader("Research Note")
        st.text(legacy)
    xai = safe_read_csv(paths["xai"])
    if not xai.empty:
        st.subheader("Feature Explainability")
        st.dataframe(xai, use_container_width=True)

with tabs[12]:
    st.header("Model Governance")
    gov = safe_read_csv(paths["v5_gov"])
    validation = safe_read_csv(paths["v5_validation"])
    monitoring = safe_read_csv(paths["monitoring"])
    if not gov.empty:
        st.subheader("Model Inventory")
        st.dataframe(gov, use_container_width=True)
    if not validation.empty:
        st.subheader("Validation")
        st.dataframe(validation, use_container_width=True)
    if not monitoring.empty:
        st.subheader("Monitoring")
        st.dataframe(monitoring, use_container_width=True)
    if gov.empty and validation.empty and monitoring.empty:
        st.info("Run V8 model first.")
