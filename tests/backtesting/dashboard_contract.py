"""Test-only mirror of the dashboard's Settings/MetricSet field names -- see the
docstring in tests/backtesting/test_report.py's Task description for why this exists
and how to keep it in sync. Source: trading_agent_framework/dashboard/models.py.
"""

from __future__ import annotations

REQUIRED_SETTINGS_FIELDS = frozenset({
    "name", "backtesting_start", "backtesting_end", "budget", "risk_free_rate",
    "backtesting_data_sources", "backtest_time_seconds", "parameters",
})

METRIC_SET_FIELDS = frozenset({
    "total_return_strategy", "total_return_benchmark", "cagr_strategy", "cagr_benchmark",
    "sharpe_strategy", "sharpe_benchmark", "sortino_strategy", "sortino_benchmark",
    "calmar_strategy", "calmar_benchmark", "omega_strategy", "omega_benchmark",
    "max_drawdown_strategy", "max_drawdown_benchmark", "volatility_strategy",
    "volatility_benchmark", "beta", "alpha", "correlation", "treynor_ratio",
    "information_ratio_strategy", "information_ratio_benchmark", "r_squared_strategy",
    "r_squared_benchmark", "skew_strategy", "skew_benchmark", "kurtosis_strategy",
    "kurtosis_benchmark", "win_days_pct_strategy", "win_days_pct_benchmark",
    "win_month_pct_strategy", "win_month_pct_benchmark", "longest_dd_days_strategy",
    "longest_dd_days_benchmark", "avg_drawdown_strategy", "avg_drawdown_benchmark",
    "recovery_factor_strategy", "recovery_factor_benchmark", "raw",
})
