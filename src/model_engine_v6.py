"""Model Engine V6: multi-model ensemble for TradFi direction forecasting.

Design goals:
- Keep V5 model.py untouched as fallback.
- Add XGBoost + LightGBM + RandomForest + Logistic Regression.
- Save one model pack and produce compatible `prob_up` / `signal` outputs.
- Avoid overloading Streamlit Cloud by capping training rows.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

V6_FEATURES = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'ema_trend', 'rsi_14', 'atr_14', 'atr_pct', 'vol_ratio_20',
    'volatility_20d', 'volatility_60d', 'drawdown_60d',
    'price_vs_50dma', 'price_vs_200dma', 'volume_zscore',
    'cross_sectional_momentum', 'ret_20d_rank', 'ret_60d_rank',
    'vix', 'dxy', 'sp500', 'nasdaq', 'fed_funds_rate', 'us_10y_yield',
    'us_2y_yield', 'high_yield_spread', 'unemployment_rate', 'industrial_production',
]


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in V6_FEATURES if c in df.columns]


def _make_models(random_state: int = 42) -> dict:
    models = {}
    try:
        from xgboost import XGBClassifier
        models['xgb'] = XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=0.035,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=random_state, n_jobs=1,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models['lgbm'] = LGBMClassifier(
            n_estimators=180, max_depth=-1, learning_rate=0.035,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            n_jobs=1, verbose=-1,
        )
    except Exception:
        pass
    models['rf'] = RandomForestClassifier(
        n_estimators=180, max_depth=6, min_samples_leaf=25,
        random_state=random_state, n_jobs=1, class_weight='balanced_subsample'
    )
    models['logit'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    return models


def _clean_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = df.dropna(subset=['target_up_1d']).copy()
    for f in features:
        data[f] = pd.to_numeric(data[f], errors='coerce')
    data = data.dropna(subset=features, how='all')
    X = data[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = data['target_up_1d'].astype(int)
    return X, y, data


def train_v6_ensemble(df: pd.DataFrame, model_path: Path, max_train_rows: int = 30000) -> dict:
    features = _available_features(df)
    if not features:
        raise RuntimeError('No V6 features available. Check features.py and macro data.')
    X, y, data = _clean_xy(df, features)
    if len(X) > max_train_rows:
        X = X.tail(max_train_rows)
        y = y.tail(max_train_rows)
    split = int(len(X) * 0.8)
    if split < 200 or len(X) - split < 50:
        split = max(1, int(len(X) * 0.7))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = _make_models()
    fitted, model_metrics, probs = {}, {}, []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
            probs.append(pd.Series(p, index=X_test.index, name=name))
            pred = (p >= 0.5).astype(int)
            model_metrics[name] = {
                'auc': float(roc_auc_score(y_test, p)) if len(set(y_test)) > 1 else None,
                'accuracy': float(accuracy_score(y_test, pred)),
                'precision': float(precision_score(y_test, pred, zero_division=0)),
                'recall': float(recall_score(y_test, pred, zero_division=0)),
            }
            fitted[name] = model
        except Exception as e:
            model_metrics[name] = {'error': str(e)}

    if not fitted:
        raise RuntimeError('All V6 models failed to train.')
    prob_df = pd.concat(probs, axis=1) if probs else pd.DataFrame(index=X_test.index)
    ens_prob = prob_df.mean(axis=1)
    ens_pred = (ens_prob >= 0.5).astype(int)
    metrics = {
        'version': 'v6',
        'auc': float(roc_auc_score(y_test, ens_prob)) if len(set(y_test)) > 1 else None,
        'accuracy': float(accuracy_score(y_test, ens_pred)),
        'precision': float(precision_score(y_test, ens_pred, zero_division=0)),
        'recall': float(recall_score(y_test, ens_pred, zero_division=0)),
        'features': features,
        'model_metrics': model_metrics,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
    }
    pack = {'version': 'v6', 'models': fitted, 'features': features, 'metrics': metrics}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pack, model_path)
    return metrics


def predict_latest_v6(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    pack = joblib.load(model_path)
    models, features = pack['models'], pack['features']
    latest = df.sort_values('date').groupby('symbol').tail(1).copy()
    for f in features:
        latest[f] = pd.to_numeric(latest[f], errors='coerce') if f in latest.columns else np.nan
    X = latest[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    for name, model in models.items():
        try:
            latest[f'prob_{name}'] = model.predict_proba(X)[:, 1]
        except Exception:
            latest[f'prob_{name}'] = np.nan
    prob_cols = [c for c in latest.columns if c.startswith('prob_')]
    latest['prob_ensemble'] = latest[prob_cols].mean(axis=1)
    latest['prob_up'] = latest['prob_ensemble']
    latest['signal'] = latest['prob_up'].map(lambda p: 'BUY' if p >= 0.58 else ('SELL' if p <= 0.42 else 'HOLD'))
    keep = ['date', 'symbol', 'close', 'prob_up', 'prob_ensemble', 'signal', 'rsi_14', 'atr_14', 'ret_20d'] + prob_cols
    keep = [c for c in keep if c in latest.columns]
    return latest[keep]


def v6_feature_importance(model_path: Path) -> pd.DataFrame:
    pack = joblib.load(model_path)
    rows = []
    feats = pack['features']
    for name, model in pack['models'].items():
        imp = getattr(model, 'feature_importances_', None)
        if imp is None and hasattr(model, 'named_steps'):
            coef = getattr(model.named_steps.get('model'), 'coef_', None)
            if coef is not None:
                imp = np.abs(coef).ravel()
        if imp is not None:
            for f, v in zip(feats, imp):
                rows.append({'model': name, 'feature': f, 'importance': float(v)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['importance_norm'] = out.groupby('model')['importance'].transform(lambda x: x / max(float(x.sum()), 1e-12))
    return out
