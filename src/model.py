from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

FEATURES = ['ret_1d','ret_5d','ret_20d','ema_trend','rsi_14','atr_14','vol_ratio_20','vix','dxy','sp500','nasdaq','fed_funds_rate','us_10y_yield','us_2y_yield','high_yield_spread']

def train_model(df: pd.DataFrame, model_path: Path):
    use = [c for c in FEATURES if c in df.columns]
    data = df.dropna(subset=use + ['target_up_1d']).copy()
    X, y = data[use], data['target_up_1d']
    split = int(len(data)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1]
    pred = (prob >= 0.5).astype(int)
    metrics = {'auc': float(roc_auc_score(y_test, prob)) if len(set(y_test))>1 else None, 'accuracy': float(accuracy_score(y_test, pred)), 'features': use}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model':model, 'features':use, 'metrics':metrics}, model_path)
    return metrics

def predict_latest(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    pack = joblib.load(model_path)
    model, feats = pack['model'], pack['features']
    latest = df.sort_values('date').groupby('symbol').tail(1).dropna(subset=feats).copy()
    latest['prob_up'] = model.predict_proba(latest[feats])[:,1]
    latest['signal'] = latest['prob_up'].map(lambda p: 'BUY' if p>=0.60 else ('SELL' if p<=0.40 else 'HOLD'))
    return latest[['date','symbol','close','prob_up','signal','rsi_14','atr_14','ret_20d']]
