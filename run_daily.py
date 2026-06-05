from src.pipeline import run_all

if __name__ == '__main__':
    metrics, signals, risks, regimes, portfolio, kill, bt_summary = run_all(
        start='2018-01-01',
        prefer='yahoo',
        nav=100000.0,
        run_walk_forward=False,
    )
    print('Model metrics:', metrics)
    print('Kill switch:', kill)
    print('Top signals:')
    print(signals.sort_values('prob_up', ascending=False).head(10))
    print('Target portfolio:')
    print(portfolio.head(10))
