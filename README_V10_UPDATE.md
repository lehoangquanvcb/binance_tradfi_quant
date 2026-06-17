# V10 Institutional CIO Workstation Update

Copy these new modules into `src/` and wire them in `pipeline.py` / `app.py`:

- `src/alpha_attribution.py`
- `src/probability_calibration.py`
- `src/portfolio_optimizer.py`
- `src/regime_transition_matrix.py`
- `src/sector_allocation_engine.py`

Recommended pipeline outputs:

- `v10_alpha_attribution.csv`
- `v10_probability_calibration.csv`
- `v10_probability_calibration_summary.csv`
- `v10_optimized_portfolio.csv`
- `v10_regime_transition_matrix.csv`
- `v10_sector_allocation.csv`

Suggested commit:

```bash
git add src/alpha_attribution.py src/probability_calibration.py src/portfolio_optimizer.py src/regime_transition_matrix.py src/sector_allocation_engine.py README_V10_UPDATE.md
git commit -m "Add V10 Institutional CIO engines"
git pull origin main --rebase
git push origin main
```
