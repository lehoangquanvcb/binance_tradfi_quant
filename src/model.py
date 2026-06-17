from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

# V8.8 Alpha Engine feature set. The list is intentionally broad; train_model
# automatically keeps only columns present in the current dataset.
FEATURES = [
    # legacy / core technical
    'ret_1d','ret_2d','ret_3d','ret_5d','ret_10d','ret_20d','ret_40d','ret_60d','ret_120d','ret_252d',
    'ema_trend','trend_stack','rsi_14','rsi_z_60','atr_14','atr_pct','vol_ratio_20','volume_zscore_20',
    'price_vs_50dma','price_vs_200dma','ema_20_50_spread','ema_50_200_spread','momentum_accel',
    'volatility_20d','volatility_60d','downside_vol_60d','drawdown_60d','drawdown_120d','drawdown_recovery',
    'risk_adj_mom_20d','risk_adj_mom_60d','risk_adj_mom_120d','liquidity_trend_20_60',
    # cross-sectional alpha features
    'cs_ret_20d_rank','cs_ret_60d_rank','cs_ret_120d_rank','cs_risk_adj_mom_60d_rank',
    'cs_price_vs_200dma_rank','cs_volume_zscore_20_rank','cs_drawdown_60d_rank',
    'cs_momentum_composite','cs_defensive_score',
    # relative strength / macro / credit features where available
    'relative_strength_20d','relative_strength_60d','rs_20d','rs_60d','rs_rank_60d',
    'vix','dxy','sp500','nasdaq','fed_funds_rate','us_10y_yield','us_2y_yield','term_spread',
    'high_yield_spread','investment_grade_spread','macro_cycle_score','credit_stress_score','recession_probability_6m',
]


def _clean_training_frame(df: pd.DataFrame, use: list[str]) -> pd.DataFrame:
    data = df.copy()
    for c in use:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data['target_up_1d'] = pd.to_numeric(data['target_up_1d'], errors='coerce')
    # Fill within each symbol first, then global median. This keeps Streamlit runs
    # from dropping too much data when newly added features are sparse.
    data = data.sort_values(['symbol', 'date']) if {'symbol','date'}.issubset(data.columns) else data
    data[use] = data.groupby('symbol')[use].transform(lambda x: x.ffill()) if 'symbol' in data.columns else data[use]
    med = data[use].median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data[use] = data[use].replace([np.inf, -np.inf], np.nan).fillna(med)
    data = data.dropna(subset=['target_up_1d']).copy()
    data['target_up_1d'] = data['target_up_1d'].astype(int)
    return data


def _best_threshold(y_true: pd.Series, prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.50
    grid = np.arange(0.45, 0.56, 0.01)
    scores = []
    for t in grid:
        scores.append((accuracy_score(y_true, (prob >= t).astype(int)), float(t)))
    return max(scores, key=lambda x: x[0])[1]


def train_model(df: pd.DataFrame, model_path: Path):
    use = [c for c in FEATURES if c in df.columns]
    # Fallback to original features if a repo has not yet received V8.8 features.
    if len(use) < 6:
        use = [c for c in ['ret_1d','ret_5d','ret_20d','ema_trend','rsi_14','atr_14','vol_ratio_20'] if c in df.columns]
    data = _clean_training_frame(df, use)
    X, y = data[use], data['target_up_1d']
    split = int(len(data) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Stable, slightly stronger XGBoost setup for noisy daily equity direction.
    model = XGBClassifier(
        n_estimators=550,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=2.0,
        reg_alpha=0.10,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    threshold = _best_threshold(y_test, prob)
    pred = (prob >= threshold).astype(int)
    metrics = {
        'auc': float(roc_auc_score(y_test, prob)) if len(set(y_test)) > 1 else None,
        'accuracy': float(accuracy_score(y_test, pred)),
        'threshold': float(threshold),
        'features': use,
        'feature_count': len(use),
        'model_version': 'v8.8_alpha_engine',
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': use, 'metrics': metrics, 'threshold': threshold}, model_path)
    return metrics


def predict_latest(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    pack = joblib.load(model_path)
    model, feats = pack['model'], pack['features']
    threshold = float(pack.get('threshold', 0.50))
    latest = df.sort_values('date').groupby('symbol').tail(1).copy()
    for c in feats:
        latest[c] = pd.to_numeric(latest[c], errors='coerce') if c in latest.columns else np.nan
    med = df[feats].apply(pd.to_numeric, errors='coerce').median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest[feats] = latest[feats].replace([np.inf, -np.inf], np.nan).fillna(med)
    latest = latest.dropna(subset=feats).copy()
    latest['prob_up'] = model.predict_proba(latest[feats])[:, 1]
    latest['signal'] = latest['prob_up'].map(lambda p: 'BUY' if p >= max(0.58, threshold + 0.05) else ('SELL' if p <= min(0.42, threshold - 0.05) else 'HOLD'))
    cols = ['date','symbol','close','prob_up','signal','rsi_14','atr_14','ret_20d']
    for c in cols:
        if c not in latest.columns:
            latest[c] = np.nan
    return latest[cols]
