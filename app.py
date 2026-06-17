import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import DATA_PROCESSED
from src.pipeline import run_all
from src.kill_switch import evaluate_kill_switch


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


def show_df(df: pd.DataFrame, empty_msg: str = "Run model first."):
    if df is None or df.empty:
        st.info(empty_msg)
    else:
        st.dataframe(df, use_container_width=True)


st.set_page_config(page_title="V12 Production CIO Workstation", layout="wide")
st.title("V12 Production CIO Workstation")
st.caption(
    "Market Regime → Sector Rotation → Stock Selection → Exit Watchlist → Portfolio Recommendation. "
    "Designed as a decision-support layer for CIO/PM/Head of Research workflows."
)

with st.sidebar:
    st.header("Controls")
    start = st.text_input("Start date", "2018-01-01")
    prefer = st.selectbox("Data source priority", ["yahoo", "binance"], index=0)
    nav = st.number_input("Portfolio NAV (USD)", min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider("Risk per trade", 0.001, 0.03, 0.01, 0.001)
    run_wf = st.checkbox("Run walk-forward backtest", value=False)
    backtest_mode = st.selectbox("Backtest mode", ["fast", "standard", "full"], index=0, help="Fast is recommended for Streamlit Cloud.")
    testnet_mode = st.checkbox("Binance testnet / sandbox mode", value=True)
    live_mode = st.checkbox("Live mode enabled", value=False, help="Live trading should remain disabled unless the OMS/risk stack is fully tested.")
    if st.button("Run / Refresh V12 model"):
        with st.spinner(f"Running V12 Production CIO Workstation pipeline... Backtest={run_wf}, mode={backtest_mode}"):
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
    "v90_equity": DATA_PROCESSED / "v90_institutional_backtest.csv",
    "v90_backtest": DATA_PROCESSED / "v90_backtest_summary.csv",
    "v90_sizing": DATA_PROCESSED / "v90_position_sizing.csv",
    "v90_regime_prob": DATA_PROCESSED / "v90_regime_probability.csv",
    "v90_confidence": DATA_PROCESSED / "v90_confidence_score.csv",
    "v10_alpha": DATA_PROCESSED / "v10_alpha_attribution.csv",
    "v10_optimizer": DATA_PROCESSED / "v10_optimized_portfolio.csv",
    "v10_transition": DATA_PROCESSED / "v10_regime_transition_matrix.csv",
    "v10_sector_alloc": DATA_PROCESSED / "v10_sector_allocation.csv",
    "v10_calibration": DATA_PROCESSED / "v10_probability_calibration.csv",
    "v10_cal_summary": DATA_PROCESSED / "v10_calibration_summary.csv",
    "v105_vote": DATA_PROCESSED / "v105_cio_vote.csv",
    "v105_sizing": DATA_PROCESSED / "v105_dynamic_position_sizing.csv",
    "v105_stop": DATA_PROCESSED / "v105_stop_loss_plan.csv",
    "v105_readiness": DATA_PROCESSED / "v105_institutional_readiness.csv",
    "v11_sector_etf": DATA_PROCESSED / "v11_sector_etf_rotation.csv",
    "v11_cross_asset": DATA_PROCESSED / "v11_cross_asset_allocation.csv",
    "v11_bl_portfolio": DATA_PROCESSED / "v11_black_litterman_portfolio.csv",
    "v11_rebalance": DATA_PROCESSED / "v11_dynamic_rebalance.csv",
    "v11_factor_attr": DATA_PROCESSED / "v11_factor_attribution.csv",
    "v11_regime_forecast": DATA_PROCESSED / "v11_regime_forecast.csv",
    "v115_retraining": DATA_PROCESSED / "v115_adaptive_retraining.csv",
    "v115_sector_strength": DATA_PROCESSED / "v115_sector_strength.csv",
    "v115_regime_probability": DATA_PROCESSED / "v115_regime_probability.csv",
    "v115_ensemble": DATA_PROCESSED / "v115_ensemble.csv",
    "v115_portfolio": DATA_PROCESSED / "v115_optimized_portfolio.csv",
    "v12_thresholds": DATA_PROCESSED / "v12_dynamic_thresholds.csv",
    "v12_meta": DATA_PROCESSED / "v12_meta_model_overlay.csv",
    "v12_regime_diag": DATA_PROCESSED / "v12_regime_specific_diagnostics.csv",
    "v12_bayesian": DATA_PROCESSED / "v12_bayesian_ensemble.csv",
    "v12_portfolio": DATA_PROCESSED / "v12_confidence_weighted_portfolio.csv",
    "v12_retraining": DATA_PROCESSED / "v12_retraining_trigger.csv",
}

tabs = st.tabs([
    "1. CIO Dashboard",
    "2. Market Regime",
    "3. Sector Rotation",
    "4. Stock Selection",
    "5. Exit Watchlist",
    "6. Signals",
    "7. Portfolio",
    "8. Credit-Macro",
    "9. Institutional Risk",
    "10. AI Research",
    "11. Model Governance",
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
        st.markdown(summary)

    if not regime.empty:
        latest = regime.sort_values("date").tail(1).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market Regime", str(latest.get("market_regime", "N/A")))
        c2.metric("Regime Score", f"{float(latest.get('regime_score', 0)):.1f}/100")
        c3.metric("Equity Weight", f"{float(latest.get('recommended_equity_weight', 0)):.0%}")
        c4.metric("Cash Weight", f"{float(latest.get('recommended_cash_weight', 0)):.0%}")
    else:
        st.info("Run V12 model to populate CIO dashboard.")

    v90_conf = safe_read_csv(paths["v90_confidence"])
    v90_bt = safe_read_csv(paths["v90_backtest"])
    v90_prob = safe_read_csv(paths["v90_regime_prob"])
    if not v90_conf.empty or not v90_bt.empty or not v90_prob.empty:
        st.subheader("V12 Production CIO Workstation Metrics")
        c1, c2, c3, c4 = st.columns(4)
        if not v90_conf.empty:
            row = v90_conf.tail(1).iloc[0]
            c1.metric("CIO Confidence", f"{float(row.get('confidence_score', 0)):.1f}/100", str(row.get('confidence_label', 'N/A')))
        if not v90_prob.empty:
            row = v90_prob.tail(1).iloc[0]
            c2.metric("Risk-On Probability", f"{float(row.get('risk_on_prob', 0)):.0%}")
            c3.metric("Risk-Off Probability", f"{float(row.get('risk_off_prob', 0)):.0%}")
        if not v90_bt.empty:
            row = v90_bt.tail(1).iloc[0]
            c4.metric("Portfolio Sharpe", f"{float(row.get('sharpe', 0)):.2f}")


    readiness = safe_read_csv(paths["v105_readiness"])
    cal_sum = safe_read_csv(paths["v10_cal_summary"])
    if not readiness.empty or not cal_sum.empty:
        st.subheader("V10.5/V11 Readiness & Calibration")
        c1, c2, c3 = st.columns(3)
        if not readiness.empty:
            row = readiness.tail(1).iloc[0]
            c1.metric("Institutional Readiness", f"{float(row.get('institutional_readiness_score', 0)):.1f}/100", str(row.get('readiness_label', 'N/A')))
        if not cal_sum.empty:
            row = cal_sum.tail(1).iloc[0]
            c2.metric("Calibration", str(row.get('status', 'N/A')))
            if pd.notna(row.get('calibration_error', None)):
                c3.metric("Calibration Error", f"{float(row.get('calibration_error', 0)):.3f}")

    v11_cross = safe_read_csv(paths["v11_cross_asset"])
    v11_bl = safe_read_csv(paths["v11_bl_portfolio"])
    v11_reb = safe_read_csv(paths["v11_rebalance"])
    v11_fc = safe_read_csv(paths["v11_regime_forecast"])
    if not v11_cross.empty or not v11_bl.empty or not v11_fc.empty:
        st.subheader("V11 Portfolio Manager Overlay")
        c1, c2, c3 = st.columns(3)
        if not v11_fc.empty:
            r = v11_fc.iloc[0]
            c1.metric("1M Risk-On Probability", f"{float(r.get('risk_on_prob', 0)):.0%}", str(r.get('expected_regime', 'N/A')))
        if not v11_bl.empty and 'target_weight' in v11_bl.columns:
            c2.metric("BL Portfolio Names", int(v11_bl['symbol'].nunique()))
        if not v11_reb.empty:
            c3.metric("Rebalance Trades", int((v11_reb.get('rebalance_action', pd.Series(dtype=str)) != 'HOLD').sum()))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Sectors")
        if not sector.empty:
            cols = [c for c in ["rank", "symbol", "sector", "sector_score", "sector_action", "relative_strength_60d", "trend_strength"] if c in sector.columns]
            st.dataframe(sector[cols].head(8), use_container_width=True)
        st.subheader("Top Stock Ideas")
        if not stock.empty:
            cols = [c for c in ["rank", "symbol", "stock_score", "decision", "sector_bucket", "sector_action", "market_regime"] if c in stock.columns]
            st.dataframe(stock[cols].head(12), use_container_width=True)
    with c2:
        st.subheader("Exit / Reduce Watchlist")
        if not exits.empty:
            st.dataframe(exits.head(12), use_container_width=True)
        else:
            st.success("No major exit candidates generated.")
        st.subheader("Recommended Portfolio")
        v115_portfolio = safe_read_csv(paths.get("v115_portfolio"))
        v115_ensemble = safe_read_csv(paths.get("v115_ensemble"))
        v115_retraining = safe_read_csv(paths.get("v115_retraining"))
        v12_portfolio = safe_read_csv(paths.get("v12_portfolio"))
        v12_bayesian = safe_read_csv(paths.get("v12_bayesian"))
        v12_retraining = safe_read_csv(paths.get("v12_retraining"))
        if not v115_retraining.empty:
            st.caption(f"V11.5 retraining action: {v115_retraining.iloc[0].get('retraining_action', 'N/A')}")
        if not v115_ensemble.empty:
            st.subheader("V11.5 Top Ensemble Ideas")
            st.dataframe(v115_ensemble.head(10), use_container_width=True)
        if not v115_portfolio.empty:
            st.caption("V11.5 robust optimized portfolio with drift-aware risk caps")
            st.dataframe(v115_portfolio, use_container_width=True)
            if {"symbol", "target_weight"}.issubset(v115_portfolio.columns):
                st.plotly_chart(px.pie(v115_portfolio, names="symbol", values="target_weight", title="V11.5 Robust Optimized Allocation"), use_container_width=True)
        else:
            opt_port = safe_read_csv(paths["v10_optimizer"])
            if not opt_port.empty:
                st.caption("V10.5 optimized portfolio with risk caps and cash buffer")
                st.dataframe(opt_port, use_container_width=True)
                if {"symbol", "target_weight"}.issubset(opt_port.columns):
                    st.plotly_chart(px.pie(opt_port, names="symbol", values="target_weight", title="V10.5 Optimized Allocation"), use_container_width=True)
        if v115_portfolio.empty and opt_port.empty and not port.empty:
            st.dataframe(port, use_container_width=True)
            if {"symbol", "target_weight"}.issubset(port.columns):
                st.plotly_chart(px.pie(port, names="symbol", values="target_weight", title="CIO Recommended Allocation"), use_container_width=True)

with tabs[1]:
    st.header("Market Regime Engine")
    regime = safe_read_csv(paths["v8_regime"])
    show_df(regime.tail(20) if not regime.empty else regime)
    if not regime.empty and "regime_score" in regime.columns:
        st.plotly_chart(px.line(regime.tail(750), x="date", y="regime_score", color="market_regime", title="Market Regime Score"), use_container_width=True)
        alloc_cols = [c for c in ["recommended_equity_weight", "recommended_bond_weight", "recommended_gold_weight", "recommended_cash_weight"] if c in regime.columns]
        if alloc_cols:
            st.plotly_chart(px.area(regime.tail(750), x="date", y=alloc_cols, title="Recommended Risk Budget"), use_container_width=True)


    v11_fc = safe_read_csv(paths["v11_regime_forecast"])
    if not v11_fc.empty:
        st.subheader("V11 Regime Forecast 1M / 3M / 6M")
        st.dataframe(v11_fc, use_container_width=True)
        prob_cols = [c for c in ["risk_on_prob", "neutral_prob", "risk_off_prob"] if c in v11_fc.columns]
        if prob_cols:
            st.plotly_chart(px.bar(v11_fc, x="horizon", y=prob_cols, barmode="group", title="Forward Regime Probabilities"), use_container_width=True)

    trans = safe_read_csv(paths["v10_transition"])
    if not trans.empty:
        st.subheader("V10 Regime Transition Matrix")
        st.dataframe(trans, use_container_width=True)

with tabs[2]:
    st.header("Sector Rotation")
    sector = safe_read_csv(paths["v8_sector"])
    show_df(sector)
    if not sector.empty and {"sector_score", "sector"}.issubset(sector.columns):
        st.plotly_chart(px.bar(sector.sort_values("sector_score"), x="sector_score", y="sector", color="sector_action" if "sector_action" in sector.columns else None, orientation="h", title="Sector Rotation Score"), use_container_width=True)
    v11_sector = safe_read_csv(paths["v11_sector_etf"])
    if not v11_sector.empty:
        st.subheader("V11 Sector ETF Rotation")
        st.dataframe(v11_sector, use_container_width=True)
        if {"sector", "score"}.issubset(v11_sector.columns):
            st.plotly_chart(px.bar(v11_sector.sort_values("score"), x="score", y="sector", color="action" if "action" in v11_sector.columns else None, orientation="h", title="V11 Sector ETF Rotation Score"), use_container_width=True)

    sector_alloc = safe_read_csv(paths["v10_sector_alloc"])
    if not sector_alloc.empty:
        st.subheader("V10 Sector Allocation Recommendation")
        st.dataframe(sector_alloc, use_container_width=True)
        if {"sector", "target_weight"}.issubset(sector_alloc.columns):
            st.plotly_chart(px.bar(sector_alloc.sort_values("target_weight"), x="target_weight", y="sector", color="action" if "action" in sector_alloc.columns else None, orientation="h", title="Sector Target Weights"), use_container_width=True)

with tabs[3]:
    st.header("Stock Selection")
    stock = safe_read_csv(paths["v8_stock"])
    if stock.empty:
        st.info("Run V11 model first.")
    else:
        cols = [c for c in ["rank", "symbol", "close", "stock_score", "decision", "prob_up", "ensemble_score", "relative_strength_60d", "sector_bucket", "sector_action", "market_regime"] if c in stock.columns]
        st.dataframe(stock[cols], use_container_width=True)
        st.plotly_chart(px.bar(stock.head(25).sort_values("stock_score"), x="stock_score", y="symbol", color="decision", orientation="h", title="Top Stock Selection Scores"), use_container_width=True)
        alpha = safe_read_csv(paths["v10_alpha"])
        vote = safe_read_csv(paths["v105_vote"])
        if not alpha.empty:
            st.subheader("V10 Alpha Attribution")
            st.dataframe(alpha.head(30), use_container_width=True)
        if not vote.empty:
            st.subheader("V10.5 CIO Ensemble Vote")
            st.dataframe(vote.head(30), use_container_width=True)

with tabs[4]:
    st.header("Exit Watchlist")
    exits = safe_read_csv(paths["v8_exit"])
    if exits.empty:
        st.success("No major exit/reduce candidates generated.")
    else:
        st.dataframe(exits, use_container_width=True)
        if "stock_score" in exits.columns:
            st.plotly_chart(px.bar(exits.sort_values("stock_score"), x="stock_score", y="symbol", color="severity" if "severity" in exits.columns else None, orientation="h", title="Exit Watchlist"), use_container_width=True)

with tabs[5]:
    st.header("Signals")
    sig = safe_read_csv(paths["ensemble"])
    if sig.empty:
        sig = safe_read_csv(paths["signals"])
    if sig.empty:
        st.info("Run V11 model first.")
    else:
        score_col = "ensemble_score" if "ensemble_score" in sig.columns else "prob_up"
        cols = [c for c in ["date", "symbol", "close", "prob_up", "signal", "cost_adjusted_signal", "ensemble_score", "ensemble_signal", "market_regime", "net_edge_bps"] if c in sig.columns]
        st.dataframe(sig[cols].sort_values(score_col, ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(sig.sort_values(score_col), x=score_col, y="symbol", orientation="h", title="Signal Score"), use_container_width=True)

with tabs[6]:
    st.header("Portfolio Recommendation")
    port = safe_read_csv(paths["v8_portfolio"])
    legacy = safe_read_csv(paths["v5_weights"])
    if not port.empty:
        st.subheader("V8 CIO Recommended Portfolio")
        st.dataframe(port, use_container_width=True)
        if {"symbol", "target_weight"}.issubset(port.columns):
            st.plotly_chart(px.pie(port, names="symbol", values="target_weight", title="V11 Target Weights"), use_container_width=True)
        opt_port = safe_read_csv(paths["v10_optimizer"])
        dyn_sizing = safe_read_csv(paths["v105_sizing"])
        stop_plan = safe_read_csv(paths["v105_stop"])
        if not opt_port.empty:
            st.subheader("V10.5 Optimized Portfolio")
            st.dataframe(opt_port, use_container_width=True)
        v11_bl = safe_read_csv(paths["v11_bl_portfolio"])
        v11_cross = safe_read_csv(paths["v11_cross_asset"])
        v11_reb = safe_read_csv(paths["v11_rebalance"])
        v11_attr = safe_read_csv(paths["v11_factor_attr"])
        if not v11_bl.empty:
            st.subheader("V11 Black-Litterman-lite Portfolio")
            st.dataframe(v11_bl, use_container_width=True)
            if {"symbol", "target_weight"}.issubset(v11_bl.columns):
                st.plotly_chart(px.pie(v11_bl, names="symbol", values="target_weight", title="V11 Black-Litterman-lite Allocation"), use_container_width=True)
        if not v11_cross.empty:
            st.subheader("V11 Cross-Asset Allocation")
            st.dataframe(v11_cross, use_container_width=True)
        if not v11_reb.empty:
            st.subheader("V11 Dynamic Rebalancing Plan")
            st.dataframe(v11_reb, use_container_width=True)
        if not v11_attr.empty:
            st.subheader("V11 Portfolio Factor Attribution")
            st.dataframe(v11_attr, use_container_width=True)
        if not dyn_sizing.empty:
            st.subheader("V10.5 Dynamic Position Sizing")
            st.dataframe(dyn_sizing, use_container_width=True)
        if not stop_plan.empty:
            st.subheader("V10.5 Stop-Loss / Take-Profit Plan")
            st.dataframe(stop_plan, use_container_width=True)
        sizing = safe_read_csv(paths["v90_sizing"])
        if not sizing.empty:
            st.subheader("V9 Baseline Position Sizing")
            st.dataframe(sizing, use_container_width=True)
        equity = safe_read_csv(paths["v90_equity"])
        bt = safe_read_csv(paths["v90_backtest"])
        if not bt.empty:
            st.subheader("V11 Institutional Backtest Summary")
            st.dataframe(bt, use_container_width=True)
        if not equity.empty and {"date", "equity_curve"}.issubset(equity.columns):
            st.plotly_chart(px.line(equity, x="date", y="equity_curve", title="V11 Portfolio Equity Curve"), use_container_width=True)
            if "drawdown" in equity.columns:
                st.plotly_chart(px.area(equity, x="date", y="drawdown", title="V11 Portfolio Drawdown"), use_container_width=True)
    elif not legacy.empty:
        st.subheader("Legacy Portfolio Construction")
        st.dataframe(legacy, use_container_width=True)
    else:
        st.info("Run V11 model first.")

with tabs[7]:
    st.header("Credit-Macro")
    macro_credit = safe_read_csv(paths["v55_macro_credit"])
    overlay = safe_read_csv(paths["v5_overlay"])
    if not macro_credit.empty:
        st.subheader("Macro-Credit Intelligence")
        st.dataframe(macro_credit.tail(250), use_container_width=True)
        y = [c for c in ["recession_probability_6m", "equity_risk_score", "credit_stress_score", "macro_credit_composite"] if c in macro_credit.columns]
        if y:
            st.plotly_chart(px.line(macro_credit.tail(750), x="date", y=y, title="Macro-Credit Risk Indicators"), use_container_width=True)
    if not overlay.empty:
        st.subheader("Credit-Macro Overlay")
        st.dataframe(overlay.tail(250), use_container_width=True)
        if "credit_macro_score" in overlay.columns:
            st.plotly_chart(px.line(overlay.tail(750), x="date", y="credit_macro_score", title="Risk-On / Risk-Off Overlay Score"), use_container_width=True)
    if macro_credit.empty and overlay.empty:
        st.info("No credit-macro data yet.")

with tabs[8]:
    st.header("Institutional Risk")
    risk = safe_read_csv(paths["risk"])
    if not risk.empty:
        # V8.7: displayed kill-switch is portfolio-aware in pipeline. In the UI fallback,
        # standalone asset risks create warnings instead of blocking the whole platform.
        kill = evaluate_kill_switch(risk)
        c1, c2, c3 = st.columns(3)
        c1.metric("Institutional Risk Score", f"{float(kill.get('institutional_risk_score', 0)):.1f}/100")
        c2.metric("Risk Level", str(kill.get("risk_level", "N/A")))
        c3.metric("Trading Gate", "Allowed" if kill.get("allow_trading", True) else "Blocked")
        if kill.get("allow_trading", True):
            st.success("Trading allowed: no portfolio-level hard breach detected.")
        else:
            st.error("Trading blocked by kill switch.")
            st.write(kill.get("breaches", []))
        if kill.get("warnings"):
            st.warning("Risk warnings: " + " | ".join(kill.get("warnings", [])))
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

with tabs[9]:
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

with tabs[10]:
    st.header("Model Governance")
    gov = safe_read_csv(paths["v5_gov"])
    validation = safe_read_csv(paths["v5_validation"])
    monitoring = safe_read_csv(paths["monitoring"])
    if not gov.empty:
        st.subheader("Model Inventory")
        latest_gov = gov.tail(1).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Governance Status", str(latest_gov.get("status", "N/A")))
        c2.metric("Model Version", str(latest_gov.get("version", "N/A")))
        c3.metric("Owner", str(latest_gov.get("owner", "N/A")))
        st.dataframe(gov, use_container_width=True)
        status = str(latest_gov.get("status", ""))
        if status == "Champion":
            st.success("Champion: approved for live use under current governance rules.")
        elif status == "Candidate":
            st.info("Candidate: acceptable for research / paper trading, but not approved for live trading.")
        elif status == "Watch":
            st.warning("Watch: useful signal, but not enough governance evidence for portfolio deployment.")
        else:
            st.warning("Research: exploratory model only.")
    if not validation.empty:
        st.subheader("Validation")
        st.dataframe(validation, use_container_width=True)
    if not monitoring.empty:
        st.subheader("Monitoring")
        st.dataframe(monitoring, use_container_width=True)
    bt = safe_read_csv(paths["v90_backtest"])
    conf = safe_read_csv(paths["v90_confidence"])
    if not bt.empty:
        st.subheader("V12 Portfolio Backtest Governance")
        st.dataframe(bt, use_container_width=True)
    if not conf.empty:
        st.subheader("V12 Confidence Governance")
        st.dataframe(conf, use_container_width=True)
    readiness = safe_read_csv(paths["v105_readiness"])
    cal_sum = safe_read_csv(paths["v10_cal_summary"])
    cal_curve = safe_read_csv(paths["v10_calibration"])
    if not readiness.empty:
        st.subheader("V12 Institutional Readiness")
        st.dataframe(readiness, use_container_width=True)
    if not cal_sum.empty:
        st.subheader("V12 Probability Calibration Summary")
        st.dataframe(cal_sum, use_container_width=True)

        v115_retraining = safe_read_csv(paths.get("v115_retraining"))
        v115_regprob = safe_read_csv(paths.get("v115_regime_probability"))
        v115_sector = safe_read_csv(paths.get("v115_sector_strength"))
        if not v115_retraining.empty:
            st.subheader("V11.5 Adaptive Retraining")
            st.dataframe(v115_retraining, use_container_width=True)
        if not v115_regprob.empty:
            st.subheader("V11.5 Regime Probability Forecast")
            st.dataframe(v115_regprob, use_container_width=True)
        if not v115_sector.empty:
            st.subheader("V11.5 Sector Relative Strength")
            st.dataframe(v115_sector, use_container_width=True)
    v12_thresholds = safe_read_csv(paths.get("v12_thresholds"))
    v12_meta = safe_read_csv(paths.get("v12_meta"))
    v12_regime_diag = safe_read_csv(paths.get("v12_regime_diag"))
    v12_bayesian = safe_read_csv(paths.get("v12_bayesian"))
    v12_portfolio = safe_read_csv(paths.get("v12_portfolio"))
    v12_retraining = safe_read_csv(paths.get("v12_retraining"))
    if not v12_thresholds.empty:
        st.subheader("V12 Dynamic Threshold Governance")
        st.dataframe(v12_thresholds, use_container_width=True)
    if not v12_retraining.empty:
        st.subheader("V12 Auto Retraining Trigger")
        st.dataframe(v12_retraining, use_container_width=True)
    if not v12_regime_diag.empty:
        st.subheader("V12 Regime-Specific Diagnostics")
        st.dataframe(v12_regime_diag, use_container_width=True)
    if not v12_bayesian.empty:
        st.subheader("V12 Bayesian Ensemble")
        st.dataframe(v12_bayesian.head(20), use_container_width=True)
    if not v12_portfolio.empty:
        st.subheader("V12 Confidence-Weighted Portfolio")
        st.dataframe(v12_portfolio, use_container_width=True)
    if not cal_curve.empty and {"avg_pred", "realized"}.issubset(cal_curve.columns):
        st.plotly_chart(px.line(cal_curve, x="avg_pred", y="realized", markers=True, title="Reliability Curve"), use_container_width=True)
