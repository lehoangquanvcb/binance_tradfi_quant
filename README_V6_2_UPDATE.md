# V6.2 Macro + Credit Feature Engine Update

Copy files into the existing repo:

- `src/features.py` replace old file
- `src/model_engine_v6.py` replace old file
- `src/macro_feature_engine.py` new file
- `src/credit_feature_engine.py` new file
- `src/relative_strength.py` new file

Then commit:

```bash
git add src/features.py src/model_engine_v6.py src/macro_feature_engine.py src/credit_feature_engine.py src/relative_strength.py
git commit -m "Upgrade to V6.2 macro credit feature engine"
git push origin main
```

Notes:
- This update focuses on feature quality, not UI.
- It adds macro cycle, credit stress and relative strength features.
- It is defensive if FRED data is missing.
