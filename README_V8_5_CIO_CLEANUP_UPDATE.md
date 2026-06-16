# V8.5 CIO Cleanup Update

This update focuses on the current V8 package and fixes the practical issues observed in Streamlit:

## Updated files

- `app.py`
- `src/pipeline.py`
- `src/macro_data.py`
- `src/credit_macro_overlay.py`

## What changed

### 1. Cleaner CIO dashboard
The dashboard has been reduced to 11 decision-useful tabs:

1. CIO Dashboard
2. Market Regime
3. Sector Rotation
4. Stock Selection
5. Exit Watchlist
6. Signals
7. Portfolio Recommendation
8. Credit-Macro
9. Institutional Risk
10. AI Research
11. Model Governance

Low-value or operational placeholders such as Paper Trading, Execution Quality, OMS Approval, and Database have been removed from the main dashboard.

### 2. Credit-Macro no longer collapses to a neutral placeholder
`src/macro_data.py` now:

- Uses FRED if a FRED API key is available.
- Falls back to Yahoo Finance proxies if FRED is unavailable.
- Adds aliases used by both legacy and V8 engines.
- Builds proxy series for high-yield spread and investment-grade spread from VIX when FRED credit spreads are unavailable.

### 3. Legacy Credit-Macro Overlay supports lowercase and uppercase columns
`src/credit_macro_overlay.py` now supports both:

- `HY_SPREAD`, `US10Y`, `VIXCLS`, etc.
- `high_yield_spread`, `us_10y_yield`, `vix`, etc.

### 4. Yahoo mode keeps the configured universe
`src/pipeline.py` now uses the configured `symbols.yaml` universe directly when `prefer='yahoo'`, instead of allowing Binance discovery to shrink the universe unexpectedly.

## Deployment

Copy these files into the existing Git repository, then run:

```bash
git add .
git commit -m "Upgrade to V8.5 CIO cleanup and macro overlay"
git pull origin main --rebase
git push origin main
```

Then reboot the Streamlit app.
