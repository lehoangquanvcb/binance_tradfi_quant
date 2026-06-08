"""V6.1 walk-forward backtest with fast / standard / full modes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from .model_engine_v6 import V6_FEATURES
except Exception:
    V6_FEATURES = ['ret_1d','ret_5d','ret_20d','rsi_14','atr_pct','volatility_20d','price_vs_200dma']

MODE_CONFIG = {
    'fast': {'train_days': 504, 'test_days': 252, 'max_windows': 2, 'max_symbols': 20, 'max_rows': 12000},
    'standard': {'train_days': 756, 'test_days': 126, 'max_windows': 4, 'max_symbols': 35, 'max_rows': 22000},
    'full': {'train_days': 756, 'test_days': 63, 'max_windows': 12, 'max_symbols': None, 'max_rows': 50000},
}

def _select_symbols(df: pd.DataFrame, max_symbols: int | None) -> list[str]:
    if not max_symbols or df['symbol'].nunique() <= max_symbols:
        return sorted(df['symbol'].dropna().unique())
    latest = df.sort_values('date').groupby('symbol').tail(60).copy()
    if 'dollar_volume' in latest.columns:
        score = latest.groupby('symbol')['dollar_volume'].mean().sort_values(ascending=False)
    elif 'quote_volume' in latest.columns:
        score = latest.groupby('symbol')['quote_volume'].mean().sort_values(ascending=False)
    else:
        score = latest.groupby('symbol')['close'].count().sort_values(ascending=False)
    return list(score.head(max_symbols).index)

def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], max_rows: int):
    train = train.dropna(subset=['target_up_1d']).copy()
    test = test.copy()
    for f in features:
        train[f] = pd.to_numeric(train[f], errors='coerce')
        test[f] = pd.to_numeric(test[f], errors='coerce')
    train = train.dropna(subset=features, how='all').tail(max_rows)
    valid = test.dropna(subset=['target_up_1d']).copy()
    if train.empty or valid.empty or train['target_up_1d'].nunique() < 2:
        return pd.DataFrame(), {'auc_train': np.nan, 'accuracy_train': np.nan}
    X_train = train[features].replace([np.inf,-np.inf], np.nan)
    y_train = train['target_up_1d'].astype(int)
    X_test = valid[features].replace([np.inf,-np.inf], np.nan)
    y_test = valid['target_up_1d'].astype(int)
    model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=500, class_weight='balanced', C=0.75))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    valid['prob_up'] = prob
    metrics = {'auc_train': float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else np.nan, 'accuracy_train': float(accuracy_score(y_test, pred))}
    return valid, metrics

def walk_forward_backtest(dataset: pd.DataFrame, model_dir: Path, train_days: int = 756, test_days: int = 63, threshold: float = 0.58, max_windows: int | None = None, mode: str = 'fast') -> pd.DataFrame:
    cfg = MODE_CONFIG.get(str(mode).lower(), MODE_CONFIG['fast']).copy()
    train_days, test_days = cfg['train_days'], cfg['test_days']
    max_windows = cfg['max_windows'] if max_windows is None else max_windows
    df = dataset.sort_values(['date', 'symbol']).copy()
    symbols = _select_symbols(df, cfg['max_symbols'])
    df = df[df['symbol'].isin(symbols)].copy()
    features = [c for c in V6_FEATURES if c in df.columns]
    if not features: return pd.DataFrame()
    dates = pd.Series(sorted(df['date'].dropna().unique()))
    rows, window = [], 0
    start_idx = min(train_days, max(1, len(dates)//2))
    while start_idx + test_days < len(dates):
        train_start = dates.iloc[max(0, start_idx - train_days)]
        train_end = dates.iloc[start_idx - 1]
        test_start = dates.iloc[start_idx]
        test_end = dates.iloc[min(start_idx + test_days - 1, len(dates)-1)]
        train = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        valid, metrics = _fit_predict(train, test, features, max_rows=cfg['max_rows'])
        if not valid.empty:
            valid = valid.sort_values(['symbol','date']).copy()
            valid['next_ret'] = valid.groupby('symbol')['close'].shift(-1) / valid['close'] - 1
            valid = valid.dropna(subset=['next_ret']).copy()
            valid['position'] = (valid['prob_up'] >= threshold).astype(float)
            valid['strategy_ret'] = valid['position'] * valid['next_ret']
            valid['window'] = window; valid['backtest_mode'] = mode
            valid['train_start'] = train_start; valid['train_end'] = train_end
            valid['test_start'] = test_start; valid['test_end'] = test_end
            valid['auc_train'] = metrics.get('auc_train'); valid['accuracy_train'] = metrics.get('accuracy_train')
            rows.append(valid[['date','symbol','prob_up','position','next_ret','strategy_ret','window','backtest_mode','train_start','train_end','test_start','test_end','auc_train','accuracy_train']])
        window += 1
        if max_windows and window >= max_windows: break
        start_idx += test_days
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    daily = out.groupby('date')['strategy_ret'].mean().fillna(0)
    eq = (1 + daily).cumprod()
    return out.merge(eq.rename('equity_curve'), left_on='date', right_index=True, how='left')

def summarize_backtest(bt: pd.DataFrame) -> dict:
    if bt.empty: return {}
    daily = bt.groupby('date')['strategy_ret'].mean().fillna(0)
    equity = (1 + daily).cumprod(); dd = equity / equity.cummax() - 1
    return {'backtest_mode': str(bt.get('backtest_mode', pd.Series(['unknown'])).iloc[0]), 'total_return': float(equity.iloc[-1] - 1), 'annual_return': float((equity.iloc[-1] ** (252 / max(len(daily), 1))) - 1), 'annual_vol': float(daily.std() * np.sqrt(252)), 'sharpe': float((daily.mean()*252)/(daily.std()*np.sqrt(252))) if daily.std() else np.nan, 'max_drawdown': float(dd.min()), 'hit_rate': float((bt.loc[bt['position'] > 0, 'next_ret'] > 0).mean()) if (bt['position'] > 0).any() else np.nan, 'trades': int((bt['position'] > 0).sum()), 'windows': int(bt['window'].nunique()) if 'window' in bt.columns else 0, 'symbols': int(bt['symbol'].nunique()) if 'symbol' in bt.columns else 0}
