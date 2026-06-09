from pathlib import Path
import yaml
import pandas as pd
import yfinance as yf
from .config import ROOT, DATA_RAW, DATA_PROCESSED, MODELS
from .binance_data import discover_tradfi_symbols, fetch_klines
from .macro_data import load_macro
from .features import add_ta_features, merge_macro
from .model import train_model, predict_latest
from .risk import risk_metrics
from .regime import attach_regime, classify_market_regime
from .backtest import walk_forward_backtest, summarize_backtest
from .portfolio import optimize_portfolio
from .kill_switch import evaluate_kill_switch
from .database import replace_dataframe, upsert_dataframe, log_event, now_utc
from .transaction_cost import apply_cost_filter
from .model_monitoring import monitor_model
from .ensemble import build_ensemble_signals
from .capital_allocation import allocate_capital
from .explainability import explain_model
from .alerts import send_risk_alert
from .portfolio_construction import build_target_weights
from .factor_engine import compute_factor_scores, portfolio_factor_exposure
from .alternative_data import demo_symbol_sentiment, macro_event_calendar
from .credit_macro_overlay import credit_macro_score, apply_overlay
from .institutional_risk import risk_dashboard_table, stress_test, standard_scenarios
from .ai_research_assistant import portfolio_brief
from .model_governance import register_model, validation_check
from .dynamic_leverage import leverage_multiplier
from .macro_credit_intelligence import build_macro_credit_dashboard, latest_macro_summary
from .economic_regime_v55 import classify_economic_regime
from .cross_asset_intelligence import cross_asset_signals
from .earnings_intelligence import build_earnings_intelligence
from .dynamic_asset_allocation_v55 import strategic_allocation


def load_cfg():
    return yaml.safe_load(open(ROOT/'config'/'symbols.yaml', encoding='utf-8'))


def _flatten_yfinance_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output across versions/Streamlit Cloud.

    yfinance may return single-level columns or MultiIndex columns such as
    ('Close', 'AAPL') / ('AAPL', 'Close'). This helper converts them back to
    Open/High/Low/Close/Volume so downstream feature engineering receives one
    clean OHLCV table per symbol.
    """
    out = d.copy()
    if isinstance(out.columns, pd.MultiIndex):
        level0 = [str(x).lower() for x in out.columns.get_level_values(0)]
        level1 = [str(x).lower() for x in out.columns.get_level_values(1)]
        price_names = {"open", "high", "low", "close", "adj close", "volume"}
        if any(x in price_names for x in level0):
            out.columns = out.columns.get_level_values(0)
        elif any(x in price_names for x in level1):
            out.columns = out.columns.get_level_values(1)
        else:
            out.columns = ["_".join([str(i) for i in c if str(i) != ""]) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated()].copy()
    return out


def _download_yahoo_ohlcv(ticker: str, bsym: str, start: str) -> pd.DataFrame:
    d = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=False,
    )
    if d.empty:
        return pd.DataFrame()
    d = _flatten_yfinance_columns(d)
    d = d.reset_index()
    rename_map = {
        "Date": "date", "Datetime": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Adj Close": "close", "Volume": "volume",
    }
    d = d.rename(columns=rename_map)
    d.columns = [str(c).strip().lower() for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()].copy()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Yahoo {ticker} missing {missing}. Columns={list(d.columns)}")

    d["symbol"] = bsym
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date", "close"])
    d["quote_volume"] = d["close"] * d["volume"].fillna(0)
    d["n_trades"] = None
    return d[["date", "symbol", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]]


def fetch_prices(start='2018-01-01', interval='1d', prefer='yahoo') -> pd.DataFrame:
    cfg = load_cfg()
    fallback = cfg['tradfi_symbols']
    symbols = discover_tradfi_symbols(fallback)
    frames = []
    if prefer == 'binance':
        for s in symbols:
            try:
                d = fetch_klines(s, interval=interval)
                if not d.empty:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)
                    d.columns = [str(c).strip().lower() for c in d.columns]
                    d = d.loc[:, ~d.columns.duplicated()].copy()
                    frames.append(d)
            except Exception as e:
                print(f'Binance failed {s}: {e}')
    if not frames:
        mapping = cfg.get('yahoo_mapping', {})
        for bsym in symbols:
            ticker = mapping.get(bsym)
            if not ticker:
                continue
            try:
                d = _download_yahoo_ohlcv(ticker, bsym, start)
                if not d.empty:
                    frames.append(d)
            except Exception as e:
                print(f'Yahoo failed {bsym}/{ticker}: {e}')
    if not frames:
        raise RuntimeError('No price data downloaded. Check internet/API availability.')
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    for c in ['open','high','low','close','volume','quote_volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['date','symbol','close']).sort_values(['symbol','date'])
    df.to_parquet(DATA_RAW/'prices.parquet', index=False)
    try:
        replace_dataframe(df, 'prices')
    except Exception as e:
        print(f'Database price write skipped: {e}')
    return df


def build_dataset(start='2018-01-01', prefer='yahoo'):
    prices = fetch_prices(start=start, prefer=prefer)
    macro = load_macro(start=start)
    if not macro.empty:
        macro.to_parquet(DATA_RAW/'macro_us.parquet', index=False)
        try:
            replace_dataframe(macro, 'macro')
        except Exception as e:
            print(f'Database macro write skipped: {e}')
    ds = merge_macro(add_ta_features(prices), macro)
    ds = attach_regime(ds)
    ds.to_parquet(DATA_PROCESSED/'model_dataset.parquet', index=False)
    return ds


def run_all(
    start='2018-01-01',
    prefer='yahoo',
    nav=100000.0,
    run_walk_forward=False,
    backtest_mode='fast',
    **kwargs
):
    run_id = now_utc().replace(':','').replace('+','_')
    log_event('RUN', 'START', f'start={start}, prefer={prefer}, nav={nav}, walk_forward={run_walk_forward}')
    ds = build_dataset(start=start, prefer=prefer)
    metrics = train_model(ds, MODELS/'xgb_direction_model.joblib')
    signals = predict_latest(ds, MODELS/'xgb_direction_model.joblib')
    signals = apply_cost_filter(signals)
    ensemble = build_ensemble_signals(ds, signals)
    risks = risk_metrics(ds)
    regimes = classify_market_regime(ds)
    latest_regime = regimes.sort_values('date').tail(1)['market_regime'].iloc[0] if not regimes.empty else 'NEUTRAL'
    strategy_alloc = allocate_capital(nav=nav, regime=latest_regime)
    # Optimizer uses ensemble score where available, without losing original ML probability.
    if not ensemble.empty:
        opt_input = ensemble.copy()
        opt_input['ml_prob_up'] = opt_input['prob_up']
        opt_input['prob_up'] = opt_input['ensemble_score']
        opt_input['signal'] = opt_input['ensemble_signal']
    else:
        opt_input = signals
    portfolio = optimize_portfolio(opt_input, risks, nav=nav)
    kill = evaluate_kill_switch(risks)
    monitoring = monitor_model(ds, metrics, signals, run_id, MODELS/'baseline_features.parquet')
    xai = explain_model(ds, MODELS/'xgb_direction_model.joblib', DATA_PROCESSED/'feature_explainability.csv')

    # V5 institutional modules
    close_panel = ds.pivot_table(index='date', columns='symbol', values='close').ffill().dropna(how='all')
    returns_panel = close_panel.pct_change().dropna(how='all').fillna(0)
    if len(returns_panel.columns) >= 2:
        weights_rp = build_target_weights(returns_panel, method='risk_parity')
        weights_mvo = build_target_weights(returns_panel, method='mean_variance')
        weights_hrp = build_target_weights(returns_panel, method='hrp_lite')
    else:
        weights_rp = pd.Series(1.0, index=returns_panel.columns)
        weights_mvo = weights_rp.copy(); weights_hrp = weights_rp.copy()

    macro_cols = [c for c in ds.columns if c.isupper() or c in ['fed_funds','us10y']]
    macro_daily = ds[['date'] + macro_cols].drop_duplicates('date').set_index('date') if macro_cols else pd.DataFrame(index=close_panel.index)
    overlay = credit_macro_score(macro_daily) if not macro_daily.empty else pd.DataFrame({'credit_macro_score':[0.0], 'overlay_regime':['Neutral'], 'equity_risk_budget_multiplier':[0.75]}, index=[close_panel.index[-1]])

    # V5.5 Macro & Credit Intelligence
    macro_credit = build_macro_credit_dashboard(macro_daily.reset_index()) if not macro_daily.empty else pd.DataFrame()
    macro_summary = latest_macro_summary(macro_credit)
    econ_regime = classify_economic_regime(macro_credit)
    v55_multiplier = macro_summary.get('equity_budget_multiplier') or float(overlay['equity_risk_budget_multiplier'].iloc[-1])
    mult = float(v55_multiplier)
    weights_overlay = apply_overlay(weights_rp, mult)
    strategic_alloc = strategic_allocation(str(macro_summary.get('risk_regime', 'Neutral')))

    factors = compute_factor_scores(close_panel)
    exposures = portfolio_factor_exposure(weights_overlay.drop(labels=['CASH'], errors='ignore'), factors)
    inst_risk = risk_dashboard_table(returns_panel, weights_overlay.drop(labels=['CASH'], errors='ignore')) if not returns_panel.empty else pd.DataFrame()
    stress = stress_test(weights_overlay, standard_scenarios())
    alt = demo_symbol_sentiment(list(close_panel.columns))
    events = macro_event_calendar()
    cross_asset = cross_asset_signals(close_panel)
    earnings = build_earnings_intelligence(list(close_panel.columns))
    latest_regime_v5 = str(macro_summary.get('risk_regime', overlay['overlay_regime'].iloc[-1]))
    research_note = portfolio_brief(weights_overlay, exposures, regime=latest_regime_v5)
    recession_prob = float(macro_summary.get("recession_probability_6m") or 0.0)
    equity_score = float(macro_summary.get("equity_risk_score") or 0.0)
    credit_score = float(macro_summary.get("credit_stress_score") or 0.0)
    risk_regime = str(macro_summary.get("risk_regime") or "Neutral")
    research_note += (
        f"\n\nV5.5 Macro-Credit View: "
        f"regime={risk_regime}, "
        f"recession probability 6M={recession_prob:.1%}, "
        f"equity risk score={equity_score:.1f}, "
        f"credit stress score={credit_score:.1f}."
    )
    gov_rec = register_model('xgb_direction_model', 'v5', 'TradFi direction forecasting and portfolio signal generation', metrics=metrics)
    validation = validation_check({'sharpe':0.0,'max_drawdown':1.0,'hit_rate':metrics.get('accuracy',0.0)})

    v5_weights = pd.concat([
        weights_rp.rename('risk_parity'), weights_mvo.rename('mean_variance'),
        weights_hrp.rename('hrp_lite'), weights_overlay.rename('credit_macro_overlay')
    ], axis=1).fillna(0).reset_index().rename(columns={'index':'symbol'})
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    v5_weights.to_csv(DATA_PROCESSED/'v5_portfolio_construction.csv', index=False)
    factors.to_csv(DATA_PROCESSED/'v5_factor_scores.csv')
    exposures.to_frame('exposure').to_csv(DATA_PROCESSED/'v5_factor_exposure.csv')
    overlay.reset_index().rename(columns={'index':'date'}).to_csv(DATA_PROCESSED/'v5_credit_macro_overlay.csv', index=False)
    inst_risk.to_csv(DATA_PROCESSED/'v5_institutional_risk.csv', index=False)
    stress.to_csv(DATA_PROCESSED/'v5_stress_scenarios.csv', index=False)
    alt.to_csv(DATA_PROCESSED/'v5_alternative_sentiment.csv')
    events.to_csv(DATA_PROCESSED/'v5_macro_event_calendar.csv', index=False)
    macro_credit.to_csv(DATA_PROCESSED/'v55_macro_credit_dashboard.csv', index=False)
    econ_regime.to_csv(DATA_PROCESSED/'v55_economic_regime.csv', index=False)
    cross_asset.to_csv(DATA_PROCESSED/'v55_cross_asset_intelligence.csv', index=False)
    earnings.to_csv(DATA_PROCESSED/'v55_earnings_intelligence.csv', index=False)
    strategic_alloc.to_csv(DATA_PROCESSED/'v55_dynamic_asset_allocation.csv', index=False)
    (DATA_PROCESSED/'v5_ai_research_note.txt').write_text(research_note, encoding='utf-8')
    pd.DataFrame([gov_rec]).to_csv(DATA_PROCESSED/'v5_model_governance_latest.csv', index=False)
    pd.DataFrame([validation]).to_csv(DATA_PROCESSED/'v5_model_validation.csv', index=False)

    signals['run_id'] = run_id
    signals['strategy'] = 'ml_cost_adjusted'
    signals.to_csv(DATA_PROCESSED/'latest_signals.csv', index=False)
    ensemble.to_csv(DATA_PROCESSED/'ensemble_signals.csv', index=False)
    risks.to_csv(DATA_PROCESSED/'risk_metrics.csv', index=False)
    regimes.to_csv(DATA_PROCESSED/'market_regimes.csv', index=False)
    portfolio.to_csv(DATA_PROCESSED/'target_portfolio.csv', index=False)
    strategy_alloc.to_csv(DATA_PROCESSED/'strategy_allocation.csv', index=False)
    pd.DataFrame([kill]).to_json(DATA_PROCESSED/'kill_switch.json', orient='records', indent=2)
    pd.DataFrame([monitoring]).to_csv(DATA_PROCESSED/'model_monitoring.csv', index=False)

    try:
        replace_dataframe(signals[['run_id','date','symbol','close','prob_up','signal','rsi_14','atr_14','ret_20d','strategy']], 'signals')
        upsert_dataframe(pd.DataFrame([{**monitoring, 'ts': now_utc()}]), 'model_monitoring')
    except Exception as e:
        print(f'Database write skipped: {e}')

    if not kill.get('allow_trading', True) or monitoring.get('status') == 'ALERT':
        send_risk_alert(f'Risk/model alert: kill={kill}, monitoring={monitoring}')

    bt_summary = {}
    if run_walk_forward:
        bt = walk_forward_backtest(ds, MODELS, train_days=756, test_days=63, threshold=0.60)
        bt.to_csv(DATA_PROCESSED/'walk_forward_backtest.csv', index=False)
        bt_summary = summarize_backtest(bt)
        pd.DataFrame([bt_summary]).to_csv(DATA_PROCESSED/'walk_forward_summary.csv', index=False)

    log_event('RUN', 'END', f'metrics={metrics}, kill={kill}, monitoring={monitoring}')
    return metrics, signals, risks, regimes, portfolio, kill, bt_summary


if __name__ == '__main__':
    print(run_all())
