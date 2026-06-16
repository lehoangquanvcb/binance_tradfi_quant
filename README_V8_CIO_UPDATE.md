# V8 CIO / Portfolio Manager Edition

This upgrade reorganizes the platform around the investment decision chain:

1. CIO Dashboard
2. Market Regime
3. Sector Rotation
4. Stock Selection
5. Exit Watchlist
6. Signals
7. Portfolio Construction
8. Factor Exposure
9. Credit-Macro
10. Institutional Risk
11. OMS Approval
12. AI Research
13. Model Governance

New modules added:

- `src/market_regime_engine.py`
- `src/sector_rotation_engine.py`
- `src/stock_selection_engine.py`
- `src/exit_watchlist_engine.py`
- `src/portfolio_constructor.py`
- `src/cio_dashboard.py`

Updated files:

- `app.py`
- `src/pipeline.py`

Removed from the main navigation: low-information duplicate V5.5/V6 diagnostic tabs, paper trading ledger view, execution quality placeholder, and database raw view.
