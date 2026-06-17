# V8.7 CIO Risk Gate Update

Copy these files into the existing V8.6 repository:

- `app.py` -> `app.py`
- `src/pipeline.py` -> `src/pipeline.py`
- `src/kill_switch.py` -> `src/kill_switch.py`
- `src/model_governance.py` -> `src/model_governance.py`
- `src/cio_dashboard.py` -> `src/cio_dashboard.py`

Main changes:

- Version label updated to V8.7.
- Kill switch no longer blocks the whole platform because a single asset has a large historical drawdown.
- Hard risk gate now focuses on portfolio-level proxy returns.
- Institutional Risk tab now shows risk score, risk level, trading gate, warnings and detailed risk table.
- Model governance version updated to V8.7.

Deploy:

```bash
git add app.py src/pipeline.py src/kill_switch.py src/model_governance.py src/cio_dashboard.py
git commit -m "Upgrade to V8.7 CIO risk gate"
git pull origin main --rebase
git push origin main
```
