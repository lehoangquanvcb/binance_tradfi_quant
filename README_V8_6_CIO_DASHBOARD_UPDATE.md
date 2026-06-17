# V8.6 CIO Dashboard & Governance Update

Copy these files into the existing repo:

- `app.py` -> repo root `app.py`
- `src/pipeline.py` -> `src/pipeline.py`
- `src/model_governance.py` -> `src/model_governance.py`
- `src/cio_dashboard.py` -> `src/cio_dashboard.py`

## Main changes

- CIO Dashboard now presents a clearer decision brief: market regime, equity/cash risk budget, overweight sectors, top stocks, exit watchlist and recommended allocation.
- Model Governance no longer labels a model as `Champion` when paper/live approval checks fail.
- Governance status now becomes `Champion`, `Candidate`, `Watch`, or `Research` based on validation logic.
- Model version is updated to `v8.6` in the governance inventory.
- AI Research tab no longer duplicates the CIO brief text.

## Git commands

```bash
git add app.py src/pipeline.py src/model_governance.py src/cio_dashboard.py
git commit -m "Upgrade V8.6 CIO dashboard and governance"
git pull origin main --rebase
git push origin main
```
