# Binance TradFi Quant Platform V5.5

**Macro & Credit Intelligence Edition**

This package upgrades the V5 institutional framework with a practical V5.5 layer focused on macro, credit cycle, regime detection, cross-asset signals, earnings intelligence, and dynamic asset allocation.

## New in V5.5

- `src/macro_credit_intelligence.py`
  - 6-month recession probability proxy
  - equity risk score
  - credit stress score
  - term spread / credit spread / inflation / Fed tightness composite
- `src/economic_regime_v55.py`
  - Expansion / Slowdown / Inflation Shock / Recession-Credit Stress regimes
- `src/cross_asset_intelligence.py`
  - cross-asset momentum and volatility-adjusted risk-on/risk-off score
- `src/earnings_intelligence.py`
  - earnings intelligence schema for EPS surprise, revenue surprise, guidance score and event risk
- `src/dynamic_asset_allocation_v55.py`
  - macro-credit-driven Equity / Bond / Gold / Cash target allocation
- `app.py`
  - expanded from 12 tabs to 17 tabs
  - new V5.5 dashboard tabs
- `src/pipeline.py`
  - writes V5.5 outputs to `data/processed/`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

Click **Run / Refresh V5 model** in the sidebar.

## V5.5 outputs

Generated files:

```text
data/processed/v55_macro_credit_dashboard.csv
data/processed/v55_economic_regime.csv
data/processed/v55_cross_asset_intelligence.csv
data/processed/v55_earnings_intelligence.csv
data/processed/v55_dynamic_asset_allocation.csv
```

## Production data connectors to add next

The V5.5 architecture is ready for real data, but some feeds still need API keys:

- FRED: macro and credit spreads
- Finnhub / Polygon: earnings calendar, EPS and revenue surprise
- SEC EDGAR: 10-K, 10-Q, 8-K filings
- NewsAPI / GDELT: market news sentiment
- Binance Futures API: TradFi symbol discovery and klines where available

## Safety note

Default use should be **research / paper trading**. Live trading should remain gated by OMS, kill switch, approval workflow and `.env` configuration.
