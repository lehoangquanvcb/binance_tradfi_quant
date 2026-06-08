"""Model Engine V6.2: multi-model ensemble with macro/credit/relative-strength features."""
from __future__ import annotations

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

BASE_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "ema_trend", "rsi_14", "atr_14", "atr_pct", "vol_ratio_20", "volume_zscore_20",
    "volatility_20d", "volatility_60d", "drawdown_60d", "price_vs_200dma",
    "ema_20_50_spread", "ema_50_200_spread",
    "rel_strength_5d", "rel_strength_20d", "rel_strength_60d", "rel_strength_120d",
    "vix", "dxy", "sp500", "nasdaq", "fed_funds_rate", "us_10y_yield", "us_2y_yield",
    "high_yield_spread", "yield_curve_10y2y", "us10y_change_20d", "us2y_change_20d",
    "fed_change_60d", "cpi_mom", "cpi_trend_6m", "unemployment_change_3m",
    "indpro_growth_6m", "m2_growth_6m", "macro_risk_score_v62",
    "vix_change_5d", "vix_change_20d", "hy_spread_change_20d", "hy_spread_change_60d",
    "ig_spread_change_20d", "dxy_change_20d", "credit_stress_score_v62", "risk_off_flag_v62",
]


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in BASE_FEATURES if c in df.columns]


def _prep_data(df: pd.DataFrame):
    feats = _available_features(df)
    data = df.dropna(subset=["target_up_1d"]).copy()
    # Avoid last observation if target is artificial due to shift at symbol tail.
    if "date" in data.columns:
        data = data.sort_values("date")
    X = data[feats].replace([np.inf, -np.inf], np.nan)
    y = data["target_up_1d"].astype(int)
    return data, X, y, feats


def _score(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def train_v6_ensemble(df: pd.DataFrame, model_path: Path):
    data, X, y, feats = _prep_data(df)
    if len(data) < 300 or not feats:
        raise RuntimeError(f"Not enough data/features for V6.2 training. rows={len(data)}, features={len(feats)}")

    split = int(len(data) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {}
    models["xgb"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=220, max_depth=3, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            tree_method="hist", random_state=42,
        )),
    ])
    if HAS_LGBM:
        models["lgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=220, max_depth=4, learning_rate=0.035,
                subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1,
            )),
        ])
    models["rf"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=180, max_depth=5, min_samples_leaf=25,
            n_jobs=-1, random_state=42,
        )),
    ])
    models["logit"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")),
    ])

    metrics = {"features": feats, "n_features": len(feats), "n_rows": int(len(data))}
    test_probs = []
    fitted = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            metrics[name] = _score(y_test, prob)
            test_probs.append(prob)
            fitted[name] = model
        except Exception as e:
            metrics[name] = {"error": str(e)}

    if not test_probs:
        raise RuntimeError("All V6.2 models failed to train")
    ens_prob = np.mean(test_probs, axis=0)
    ens = _score(y_test, ens_prob)
    metrics["auc"] = ens["auc"]
    metrics["accuracy"] = ens["accuracy"]
    metrics["ensemble"] = ens

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "features": feats, "metrics": metrics, "version": "V6.2"}, model_path)
    return metrics


def predict_latest_v6(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    pack = joblib.load(model_path)
    models = pack["models"]
    feats = pack["features"]
    latest = df.sort_values("date").groupby("symbol").tail(1).copy()
    X = latest[feats].replace([np.inf, -np.inf], np.nan)

    probs = []
    for name, model in models.items():
        p = model.predict_proba(X)[:, 1]
        latest[f"prob_{name}"] = p
        probs.append(p)
    latest["prob_up"] = np.mean(probs, axis=0)
    latest["prob_ensemble"] = latest["prob_up"]
    latest["signal"] = latest["prob_up"].map(lambda p: "BUY" if p >= 0.60 else ("SELL" if p <= 0.40 else "HOLD"))
    keep = ["date", "symbol", "close", "prob_up", "prob_ensemble", "signal", "rsi_14", "atr_14", "ret_20d"]
    keep += [c for c in latest.columns if c.startswith("prob_") and c not in keep]
    return latest[[c for c in keep if c in latest.columns]]


def feature_importance_v6(model_path: Path, out_path: Path | None = None) -> pd.DataFrame:
    pack = joblib.load(model_path)
    feats = pack.get("features", [])
    rows = []
    for name, pipe in pack.get("models", {}).items():
        model = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            continue
        for f, v in zip(feats, imp):
            rows.append({"model": name, "feature": f, "importance": float(v)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
