import pandas as pd


def evaluate_kill_switch(risk_df: pd.DataFrame | None = None, trades_df: pd.DataFrame | None = None,
                         daily_loss_pct: float = 0.02, max_drawdown_pct: float = 0.10,
                         var_limit_pct: float = 0.03, api_error_count: int = 0,
                         api_error_limit: int = 3, slippage_pct: float = 0.01) -> dict:
    """Central safety gate. Returns allow_trading=False if any hard limit is breached."""
    breaches = []
    if risk_df is not None and not risk_df.empty:
        worst_dd = abs(float(risk_df['max_drawdown'].min())) if 'max_drawdown' in risk_df else 0
        worst_var = abs(float(risk_df['var_95_1d'].min())) if 'var_95_1d' in risk_df else 0
        if worst_dd >= max_drawdown_pct:
            breaches.append(f'Max drawdown breach: {worst_dd:.2%} >= {max_drawdown_pct:.2%}')
        if worst_var >= var_limit_pct:
            breaches.append(f'VaR breach: {worst_var:.2%} >= {var_limit_pct:.2%}')

    if trades_df is not None and not trades_df.empty and 'realized_pnl_pct' in trades_df.columns:
        today_loss = trades_df['realized_pnl_pct'].dropna().tail(20).sum()
        if today_loss <= -abs(daily_loss_pct):
            breaches.append(f'Daily/trailing loss breach: {today_loss:.2%} <= -{daily_loss_pct:.2%}')

    if api_error_count >= api_error_limit:
        breaches.append(f'API error breach: {api_error_count} >= {api_error_limit}')
    if slippage_pct >= 0.02:
        breaches.append(f'Slippage breach: {slippage_pct:.2%}')

    return {
        'allow_trading': len(breaches) == 0,
        'status': 'OK' if len(breaches) == 0 else 'BLOCKED',
        'breaches': breaches,
    }
