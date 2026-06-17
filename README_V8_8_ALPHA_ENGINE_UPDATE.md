# V8.8 Alpha Engine Update

Files included:

```text
app.py
src/pipeline.py
src/features.py
src/model.py
src/stock_selection_engine.py
README_V8_8_ALPHA_ENGINE_UPDATE.md
```

Main changes:

- Adds stronger alpha features: cross-sectional momentum ranks, volatility-adjusted momentum, downside risk, trend stack, drawdown recovery and liquidity trend.
- Expands `model.py` feature set and uses a deterministic XGBoost configuration with threshold optimization.
- Enhances `stock_selection_engine.py` with alpha overlay inputs from the new features.
- Updates pipeline metadata to `v8.8`.

Copy into the existing V8.7/V8.6 repo, then run:

```bash
git add app.py src/pipeline.py src/features.py src/model.py src/stock_selection_engine.py
git commit -m "Upgrade to V8.8 Alpha Engine"
git pull origin main --rebase
git push origin main
```

Then reboot Streamlit.
