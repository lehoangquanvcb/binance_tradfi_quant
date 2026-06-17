# V9.0 Institutional CIO Workstation Update

## Files to copy

- `app.py` -> repo root `app.py`
- `src/pipeline.py` -> `src/pipeline.py`
- `src/features.py` -> `src/features.py`
- `src/model.py` -> `src/model.py`
- `src/stock_selection_engine.py` -> `src/stock_selection_engine.py`
- `src/model_governance.py` -> `src/model_governance.py`
- `src/institutional_backtest.py` -> new file
- `src/position_sizing.py` -> new file
- `src/confidence_engine.py` -> new file
- `src/regime_probability.py` -> new file

## What changed

V9.0 adds an institutional CIO layer on top of V8.8:

1. Portfolio equity-curve backtest
2. CAGR, Sharpe, Sortino, Calmar, max drawdown, volatility and hit-rate metrics
3. Position sizing recommendations
4. CIO confidence score
5. Regime probability layer
6. Model Governance now uses portfolio Sharpe and drawdown rather than placeholder zero values
7. Dashboard title and controls updated to V9.0

## Git commands

```bash
git add app.py src/pipeline.py src/features.py src/model.py src/stock_selection_engine.py src/model_governance.py src/institutional_backtest.py src/position_sizing.py src/confidence_engine.py src/regime_probability.py
git commit -m "Upgrade to V9.0 Institutional CIO Workstation"
git pull origin main --rebase
git push origin main
```

Then reboot Streamlit.
