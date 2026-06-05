import numpy as np
import pandas as pd
from pathlib import Path
from .model import train_model, predict_latest


def walk_forward_backtest(dataset: pd.DataFrame, model_dir: Path, train_days: int = 756, test_days: int = 63,
                          threshold: float = 0.60, max_windows: int | None = None) -> pd.DataFrame:
    """Rolling train/test backtest: train ~3 years, test ~3 months, roll forward.

    It stores only forecast-period results, avoiding look-ahead bias in a simple,
    transparent way. Strategy return = next-day return if prob_up >= threshold,
    otherwise 0. Shorting is disabled by default for risk control.
    """
    df = dataset.sort_values(['date', 'symbol']).copy()
    dates = pd.Series(sorted(df['date'].dropna().unique()))
    rows = []
    start_idx = train_days
    window = 0
    while start_idx + test_days < len(dates):
        train_start = dates.iloc[start_idx - train_days]
        train_end = dates.iloc[start_idx - 1]
        test_start = dates.iloc[start_idx]
        test_end = dates.iloc[min(start_idx + test_days - 1, len(dates)-1)]

        train = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
        model_path = model_dir / f'walk_forward_model_{window:03d}.joblib'
        metrics = train_model(train, model_path)

        # Predict on train+test so feature columns and latest helper remain compatible.
        # We create per-row probabilities manually using the saved pipeline when possible.
        import joblib
        model = joblib.load(model_path)
        feature_cols = model.get('features', [])
        estimator = model.get('model')
        test = test.sort_values(['symbol','date']).copy()
        test['next_ret'] = test.groupby('symbol')['close'].shift(-1) / test['close'] - 1
        valid = test.dropna(subset=feature_cols + ['target_up_1d', 'next_ret']).copy()
        if valid.empty:
            start_idx += test_days
            window += 1
            continue
        valid['prob_up'] = estimator.predict_proba(valid[feature_cols])[:, 1]
        valid['position'] = (valid['prob_up'] >= threshold).astype(float)
        valid['strategy_ret'] = valid['position'] * valid['next_ret']
        valid['window'] = window
        valid['train_start'] = train_start
        valid['train_end'] = train_end
        valid['test_start'] = test_start
        valid['test_end'] = test_end
        valid['auc_train'] = metrics.get('auc')
        valid['accuracy_train'] = metrics.get('accuracy')
        rows.append(valid[['date','symbol','prob_up','position','next_ret','strategy_ret','window','train_start','train_end','test_start','test_end','auc_train','accuracy_train']])
        window += 1
        if max_windows and window >= max_windows:
            break
        start_idx += test_days

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out['equity_curve'] = (1 + out.groupby('date')['strategy_ret'].mean().fillna(0)).cumprod()
    return out


def summarize_backtest(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    daily = bt.groupby('date')['strategy_ret'].mean().fillna(0)
    equity = (1 + daily).cumprod()
    dd = equity / equity.cummax() - 1
    return {
        'total_return': float(equity.iloc[-1] - 1),
        'annual_return': float((equity.iloc[-1] ** (252 / max(len(daily), 1))) - 1),
        'annual_vol': float(daily.std() * np.sqrt(252)),
        'sharpe': float((daily.mean()*252)/(daily.std()*np.sqrt(252))) if daily.std() else np.nan,
        'max_drawdown': float(dd.min()),
        'hit_rate': float((bt.loc[bt['position'] > 0, 'next_ret'] > 0).mean()) if (bt['position'] > 0).any() else np.nan,
        'trades': int((bt['position'] > 0).sum()),
    }
