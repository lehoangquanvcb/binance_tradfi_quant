"""Explainable AI helpers. Uses SHAP when installed; falls back to feature importance."""
from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

def explain_model(df: pd.DataFrame, model_path: Path, output_path: Path, n_rows: int = 1000) -> pd.DataFrame:
    pack = joblib.load(model_path)
    model, feats = pack['model'], pack['features']
    data = df.dropna(subset=feats).tail(n_rows)
    rows = []
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(data[feats])
        vals = abs(pd.DataFrame(values, columns=feats)).mean().sort_values(ascending=False)
        rows = [{'feature':k, 'importance':float(v), 'method':'mean_abs_shap'} for k,v in vals.items()]
    except Exception:
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None:
            rows = [{'feature':f, 'importance':float(v), 'method':'xgb_feature_importance'} for f,v in zip(feats, imp)]
            rows = sorted(rows, key=lambda x: x['importance'], reverse=True)
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out
