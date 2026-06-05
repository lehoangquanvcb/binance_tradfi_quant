import json
import streamlit as st
import pandas as pd
import plotly.express as px
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

st.set_page_config(page_title='V5 Quant Risk & Investment Intelligence Platform', layout='wide')
st.title('V5 Quant Risk & Investment Intelligence Platform')
st.caption('Institutional-grade TradFi quant platform: forecasting, portfolio construction, factor exposure, credit-macro overlay, execution governance, OMS, risk and AI research assistant.')

with st.sidebar:
    st.header('Controls')
    start = st.text_input('Start date', '2018-01-01')
    prefer = st.selectbox('Data source priority', ['yahoo','binance'])
    nav = st.number_input('Portfolio NAV (USD)', min_value=1000.0, value=100000.0, step=1000.0)
    risk_pct = st.slider('Risk per trade', 0.001, 0.03, 0.01, 0.001)
    run_wf = st.checkbox('Run walk-forward backtest', value=False)
    testnet_mode = st.checkbox('Binance testnet / sandbox mode', value=True)
    live_mode = st.checkbox('Live mode enabled', value=False, help='Real orders remain gated by execution.py, OMS, approval and kill-switch controls.')
    if st.button('Run / Refresh V5 model'):
        with st.spinner('Running V5 institutional pipeline...'):
            metrics, signals, risks, regimes, portfolio, kill, bt_summary = run_all(start=start, prefer=prefer, nav=nav, run_walk_forward=run_wf)
            st.success(f'Done. AUC={metrics.get("auc")}, Accuracy={metrics.get("accuracy"):.2%}')

paths = {
    'signals': DATA_PROCESSED/'latest_signals.csv',
    'ensemble': DATA_PROCESSED/'ensemble_signals.csv',
    'risk': DATA_PROCESSED/'risk_metrics.csv',
    'regimes': DATA_PROCESSED/'market_regimes.csv',
    'portfolio': DATA_PROCESSED/'target_portfolio.csv',
    'allocation': DATA_PROCESSED/'strategy_allocation.csv',
    'backtest': DATA_PROCESSED/'walk_forward_backtest.csv',
    'bt_summary': DATA_PROCESSED/'walk_forward_summary.csv',
    'monitoring': DATA_PROCESSED/'model_monitoring.csv',
    'xai': DATA_PROCESSED/'feature_explainability.csv',
    'v5_weights': DATA_PROCESSED/'v5_portfolio_construction.csv',
    'v5_factors': DATA_PROCESSED/'v5_factor_scores.csv',
    'v5_exposure': DATA_PROCESSED/'v5_factor_exposure.csv',
    'v5_overlay': DATA_PROCESSED/'v5_credit_macro_overlay.csv',
    'v5_inst_risk': DATA_PROCESSED/'v5_institutional_risk.csv',
    'v5_stress': DATA_PROCESSED/'v5_stress_scenarios.csv',
    'v5_alt': DATA_PROCESSED/'v5_alternative_sentiment.csv',
    'v5_events': DATA_PROCESSED/'v5_macro_event_calendar.csv',
    'v5_note': DATA_PROCESSED/'v5_ai_research_note.txt',
    'v5_gov': DATA_PROCESSED/'v5_model_governance_latest.csv',
    'v5_validation': DATA_PROCESSED/'v5_model_validation.csv',
}

tabs = st.tabs([
    '1. Signals','2. Institutional Portfolio','3. Factor Exposure','4. Credit-Macro Overlay',
    '5. Institutional Risk','6. OMS & Approval','7. Execution Quality','8. Paper Trading',
    '9. Alternative Data','10. AI Research Assistant','11. Model Governance','12. Database & Compliance'
])

with tabs[0]:
    if paths['ensemble'].exists():
        sig = pd.read_csv(paths['ensemble'])
        show_cols = [c for c in ['date','symbol','close','prob_up','signal','cost_adjusted_signal','ensemble_score','ensemble_signal','market_regime','net_edge_bps'] if c in sig.columns]
        st.dataframe(sig[show_cols].sort_values('ensemble_score', ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(sig.sort_values('ensemble_score'), x='ensemble_score', y='symbol', orientation='h', title='Ensemble Score'), use_container_width=True)
    elif paths['signals'].exists():
        sig = pd.read_csv(paths['signals'])
        st.dataframe(sig.sort_values('prob_up', ascending=False), use_container_width=True)
    else:
        st.info('Click Run / Refresh V5 model first.')

with tabs[1]:
    if paths['v5_weights'].exists():
        w = pd.read_csv(paths['v5_weights'])
        st.dataframe(w, use_container_width=True)
        method = st.selectbox('Portfolio method', [c for c in w.columns if c != 'symbol'])
        st.plotly_chart(px.pie(w, names='symbol', values=method, title=f'Target Weights - {method}'), use_container_width=True)
    else:
        st.info('Run V5 model first.')

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        if paths['v5_factors'].exists():
            factors = pd.read_csv(paths['v5_factors'])
            st.subheader('Symbol factor scores')
            st.dataframe(factors, use_container_width=True)
    with c2:
        if paths['v5_exposure'].exists():
            expo = pd.read_csv(paths['v5_exposure'])
            st.subheader('Portfolio factor exposure')
            st.dataframe(expo, use_container_width=True)
            st.plotly_chart(px.bar(expo, x=expo.columns[0], y='exposure', title='Factor Exposure'), use_container_width=True)
    if not paths['v5_factors'].exists(): st.info('Run V5 model first.')

with tabs[3]:
    if paths['v5_overlay'].exists():
        overlay = pd.read_csv(paths['v5_overlay'])
        st.dataframe(overlay.tail(250), use_container_width=True)
        if 'credit_macro_score' in overlay.columns:
            st.plotly_chart(px.line(overlay.tail(500), x='date', y='credit_macro_score', title='Credit-Macro Risk-On/Risk-Off Score'), use_container_width=True)
    else:
        st.info('No credit-macro overlay yet.')

with tabs[4]:
    if paths['risk'].exists():
        risk = pd.read_csv(paths['risk'])
        kill = evaluate_kill_switch(risk)
        st.subheader('Kill Switch')
        if kill['allow_trading']: st.success('Trading allowed: no hard breach detected.')
        else: st.error('Trading blocked by kill switch.'); st.write(kill['breaches'])
        st.dataframe(risk, use_container_width=True)
    if paths['v5_inst_risk'].exists():
        st.subheader('Portfolio VaR / CVaR / Concentration')
        st.dataframe(pd.read_csv(paths['v5_inst_risk']), use_container_width=True)
    if paths['v5_stress'].exists():
        st.subheader('Stress Scenarios')
        stress = pd.read_csv(paths['v5_stress'])
        st.dataframe(stress, use_container_width=True)
        st.plotly_chart(px.bar(stress, x='scenario', y='portfolio_loss', title='Scenario Portfolio Loss'), use_container_width=True)

with tabs[5]:
    st.subheader('Order Management System & Human Approval Gate')
    sig = pd.read_csv(paths['signals']) if paths['signals'].exists() else pd.DataFrame()
    if not sig.empty:
        symbol = st.selectbox('Symbol', sig['symbol'].tolist(), key='oms_symbol')
        row = sig[sig.symbol == symbol].iloc[0]
        qty = st.number_input('Quantity', min_value=0.0, value=1.0, step=1.0)
        side = st.selectbox('Side', ['BUY','SELL'], key='oms_side')
        notional = float(row.close) * qty
        need_approval = approval_required(nav, notional)
        lev_mult = leverage_multiplier(0.25, 0.02, 0.05, str(row.get('market_regime','Neutral')))
        st.metric('Order notional', f'${notional:,.0f}')
        st.metric('Dynamic leverage multiplier', f'{lev_mult:.0%}')
        st.warning('Human approval required for this order.') if need_approval else st.info('Small order: approval can be automatic in paper mode.')
        if st.button('Create OMS ticket'):
            ticket = OrderTicket(symbol=symbol, side=side, quantity=qty, reason=f'V5 signal; testnet={testnet_mode}', live_mode=live_mode)
            rec = create_order(ticket)
            log_event('OMS', 'CREATE_TICKET', json.dumps(rec), symbol=symbol, live_mode_enabled=live_mode)
            st.success('OMS ticket created. Status = PENDING_APPROVAL.')
    st.dataframe(load_orders(), use_container_width=True)

with tabs[6]:
    st.subheader('Execution Quality Analytics')
    orders = load_orders()
    st.dataframe(execution_quality_report(orders), use_container_width=True)
    st.caption('Production version should enrich this table with exchange timestamps, average fill price, benchmark price, latency, partial fills and rejections.')

with tabs[7]:
    sig_path = paths['signals']
    if sig_path.exists():
        sig = pd.read_csv(sig_path)
        symbol = st.selectbox('Symbol for paper order', sig['symbol'].tolist())
        row = sig[sig.symbol == symbol].iloc[0]
        ps = position_size(nav, float(row.close), float(row.atr_14), risk_pct=risk_pct)
        st.json(ps)
        qty = st.number_input('Paper quantity', min_value=0.0, value=float(max(ps['qty'], 0)), step=1.0)
        side = st.selectbox('Side for paper trade', ['BUY','SELL'])
        if st.button('Record paper trade'):
            cost = estimate_transaction_cost(float(row.close), qty, side)
            append_paper_trade(symbol, side, qty, float(row.close), float(row.prob_up), stop_loss=ps['stop'], reason=f'paper_trade_cost={cost}', ledger_path=OUTPUTS/'paper_trades.csv')
            log_event('ORDER', 'PAPER_TRADE', json.dumps(cost), symbol=symbol, live_mode_enabled=live_mode)
            st.success('Paper trade recorded.')
        st.dataframe(load_paper_trades(OUTPUTS/'paper_trades.csv'), use_container_width=True)
    else:
        st.info('Run model first.')

with tabs[8]:
    c1, c2 = st.columns(2)
    with c1:
        if paths['v5_alt'].exists(): st.dataframe(pd.read_csv(paths['v5_alt']), use_container_width=True)
    with c2:
        if paths['v5_events'].exists(): st.dataframe(pd.read_csv(paths['v5_events']), use_container_width=True)
    st.caption('Demo sentiment is deterministic. Connect NewsAPI/Finnhub/Reddit/X in production through src/alternative_data.py.')

with tabs[9]:
    if paths['v5_note'].exists():
        st.write(paths['v5_note'].read_text(encoding='utf-8'))
    else:
        st.info('Run V5 model first.')
    if paths['xai'].exists():
        xai = pd.read_csv(paths['xai'])
        st.plotly_chart(px.bar(xai.head(20).sort_values('importance'), x='importance', y='feature', orientation='h', title='Feature Importance / SHAP'), use_container_width=True)

with tabs[10]:
    if paths['v5_gov'].exists(): st.dataframe(pd.read_csv(paths['v5_gov']), use_container_width=True)
    if paths['v5_validation'].exists(): st.dataframe(pd.read_csv(paths['v5_validation']), use_container_width=True)
    if paths['monitoring'].exists(): st.dataframe(pd.read_csv(paths['monitoring']), use_container_width=True)

with tabs[11]:
    st.write(str(ROOT/'data'/'quant_platform.sqlite'))
    table = st.selectbox('Table', ['prices','macro','signals','orders','pnl','compliance_log','model_monitoring'])
    st.dataframe(read_table(table).tail(500), use_container_width=True)
