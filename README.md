# Binance TradFi Quant Platform V5

**V5 Institutional Quant Platform** for Binance-listed TradFi-style instruments and mapped US equities/ETFs. The system is designed for research, dashboarding, paper trading and controlled execution governance.

> Safety note: live trading is intentionally gated. Use paper trading/testnet first. This package is not investment advice.

## V5 modules

1. **Data layer**: Binance Futures discovery + fallback Yahoo mapping + US macro/FRED-style data.
2. **Forecasting**: technical features, ML direction model and ensemble signals.
3. **Risk control**: VaR, CVaR, drawdown, ATR stop, kill switch.
4. **Paper trading**: ledger and P&L trail.
5. **Walk-forward backtest**: rolling train/test evaluation.
6. **Database layer**: SQLite storage for prices, macro, signals, logs.
7. **Transaction cost model**: fee, spread, slippage and net edge.
8. **Model monitoring**: drift/status alerting.
9. **Explainable AI**: feature importance/SHAP-compatible output.
10. **OMS & approval**: order tickets, status control, human approval gate.
11. **Dynamic leverage**: leverage multiplier by volatility, VaR, drawdown and regime.
12. **Portfolio construction**: mean-variance, risk parity, Black-Litterman-lite and HRP-lite.
13. **Factor engine**: beta, momentum, low volatility, trend and quality proxy.
14. **Credit-macro overlay**: risk-on/risk-off budget multiplier using US macro/credit signals.
15. **Institutional risk dashboard**: portfolio VaR/CVaR, concentration and scenario stress loss.
16. **Alternative data layer**: news/social/event hooks with demo sentiment placeholders.
17. **AI research assistant**: plain-English explanation of portfolio and trade signals.
18. **Model governance**: inventory, validation checks and champion/challenger status.
19. **Cloud deployment**: Dockerfile, Docker Compose and GitHub Actions sample.
20. **Multi-asset expansion**: US equities, ETFs, rates ETFs, commodities, crypto and cash universe definitions.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env
```

## Run dashboard

```bash
streamlit run app.py
```

Click **Run / Refresh V5 model** in the sidebar.

## Run pipeline only

```bash
python run_daily.py
```

## Key files

```text
app.py                          Streamlit dashboard
src/pipeline.py                 Main data/model/risk pipeline
src/portfolio_construction.py   MVO, risk parity, BL-lite, HRP-lite
src/factor_engine.py            Factor exposure engine
src/credit_macro_overlay.py     Risk-on/risk-off overlay
src/institutional_risk.py       VaR/CVaR/concentration/stress
src/oms.py                      Order management system
src/dynamic_leverage.py         Leverage control
src/alternative_data.py         News/social/event hooks
src/ai_research_assistant.py    Narrative explanation
src/model_governance.py         Model inventory and validation
```

## Recommended operating mode

1. Run in demo/paper mode.
2. Review walk-forward backtest and model monitoring.
3. Review kill switch, VaR/CVaR, concentration and stress losses.
4. Use OMS tickets for trade review.
5. Only after manual approval, enable testnet/sandbox execution.
6. Keep live trading disabled until real API, legal, tax and risk controls are reviewed.

## Docker

```bash
cd deployment
docker compose up --build
```
