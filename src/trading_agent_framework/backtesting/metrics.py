"""vectorbt-based performance metrics: the dashboard's full `MetricSet` (design spec,
section 6.5) computed from a returns series. The only module importing `vectorbt`, and
only inside `compute_metrics` -- importing `backtesting.metrics` at module level must
never pull vectorbt in.

vectorbt's `returns` accessor computes Sharpe/Sortino/Calmar/Omega/max-drawdown/
annualised-return/annualised-volatility -- the ratios it is built for. Alpha/Beta/
correlation/R^2/Treynor/information-ratio/skew/kurtosis/win-rates/recovery-factor are
plain closed-form statistics (linear regression, pandas' own `.skew()`/`.kurt()`),
computed directly rather than guessed through an uncertain vectorbt method name.
`raw.summary_tables` carries the yearly-returns and drawdown tables `metrics.json`
promises the dashboard (design spec, section 6.4).

vectorbt note (verified against the installed 1.x package, not just its docs): the
`returns` accessor's `year_freq` defaults to 365 calendar days, which annualises daily
returns with sqrt(365)/365 rather than the standard sqrt(252)/252 trading-day
convention the rest of this codebase (and the golden-value tests) use. Passing
`year_freq="{periods}{unit}"` alongside a matching per-bar `freq="1{unit}"` pins
`accessor.ann_factor` to exactly `periods` (252 for "day"), which is what makes
`sharpe_ratio`/`annualized`/`annualized_volatility`/`calmar_ratio`/`sortino_ratio`
match the hand-computed formulas. Also: `sharpe_ratio(risk_free=...)`,
`omega_ratio(risk_free=...)`, and `sortino_ratio(required_return=...)` all treat their
threshold argument as a *per-period* rate (subtracted directly from each return, per
`vectorbt.returns.nb.sortino_ratio_1d_nb`/`downside_risk_1d_nb`), not an annual one --
so the annual `risk_free_rate` argument is divided by `periods` before being passed to
any of them. `sortino_ratio()` with no argument silently defaults `required_return` to
0.0, which is why an earlier version of this module that omitted the argument computed
Sortino against a zero threshold regardless of `risk_free_rate` -- invisible in tests
that only ever passed `risk_free_rate=0.0`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

PERIODS_PER_YEAR = {"day": 252, "minute": 252 * 390}
TIMESTEP_UNIT = {"day": "D", "minute": "min"}


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    timestep: str,
    risk_free_rate: float,
) -> dict[str, Any]:
    """`returns`/`benchmark_returns`: daily fractional returns, float64, no NaNs (the
    caller drops the first row of a pct_change() series). Returns a flat dict keyed
    exactly by the dashboard's `MetricSet` field names, plus `raw.summary_tables`."""
    import vectorbt as vbt  # noqa: F401 -- registers the .vbt accessor on pd.Series

    periods = PERIODS_PER_YEAR.get(timestep, 252)
    unit = TIMESTEP_UNIT.get(timestep, "D")
    freq = f"1{unit}"
    year_freq = f"{periods}{unit}"
    daily_rf = risk_free_rate / periods

    accessor = returns.vbt.returns(freq=freq, year_freq=year_freq)

    metrics: dict[str, Any] = {
        "total_return_strategy": float(accessor.total()),
        "cagr_strategy": float(accessor.annualized()),
        "sharpe_strategy": float(accessor.sharpe_ratio(risk_free=daily_rf)),
        "sortino_strategy": float(accessor.sortino_ratio(required_return=daily_rf)),
        "calmar_strategy": float(accessor.calmar_ratio()),
        "omega_strategy": float(accessor.omega_ratio(risk_free=daily_rf)),
        "max_drawdown_strategy": float(accessor.max_drawdown()),
        "volatility_strategy": float(accessor.annualized_volatility()),
        "skew_strategy": float(returns.skew()),
        "kurtosis_strategy": float(returns.kurt()),
        "win_days_pct_strategy": float((returns > 0).mean()),
    }
    metrics.update(_drawdown_stats(returns, suffix="strategy"))
    metrics.update(_monthly_win_pct(returns, suffix="strategy"))

    if benchmark_returns is not None and not benchmark_returns.empty:
        bm_accessor = benchmark_returns.vbt.returns(freq=freq, year_freq=year_freq)
        metrics.update(
            {
                "total_return_benchmark": float(bm_accessor.total()),
                "cagr_benchmark": float(bm_accessor.annualized()),
                "sharpe_benchmark": float(bm_accessor.sharpe_ratio(risk_free=daily_rf)),
                "sortino_benchmark": float(bm_accessor.sortino_ratio(required_return=daily_rf)),
                "calmar_benchmark": float(bm_accessor.calmar_ratio()),
                "omega_benchmark": float(bm_accessor.omega_ratio(risk_free=daily_rf)),
                "max_drawdown_benchmark": float(bm_accessor.max_drawdown()),
                "volatility_benchmark": float(bm_accessor.annualized_volatility()),
                "skew_benchmark": float(benchmark_returns.skew()),
                "kurtosis_benchmark": float(benchmark_returns.kurt()),
                "win_days_pct_benchmark": float((benchmark_returns > 0).mean()),
            }
        )
        metrics.update(_drawdown_stats(benchmark_returns, suffix="benchmark"))
        metrics.update(_monthly_win_pct(benchmark_returns, suffix="benchmark"))
        metrics.update(_relative_stats(returns, benchmark_returns, periods, risk_free_rate))

    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": _yearly_table(returns, benchmark_returns),
            "drawdowns": _drawdown_table(returns),
        }
    }
    return metrics


def _drawdown_stats(returns: pd.Series, *, suffix: str) -> dict[str, float]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    underwater = drawdown < 0
    longest = _longest_run(underwater)
    avg_dd = float(drawdown[underwater].mean()) if underwater.any() else 0.0
    max_dd = float(drawdown.min())
    recovery_factor = float(curve.iloc[-1] - 1) / abs(max_dd) if max_dd != 0 else 0.0
    return {
        f"longest_dd_days_{suffix}": float(longest),
        f"avg_drawdown_{suffix}": avg_dd,
        f"recovery_factor_{suffix}": recovery_factor,
    }


def _longest_run(mask: pd.Series) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _monthly_win_pct(returns: pd.Series, *, suffix: str) -> dict[str, float]:
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {f"win_month_pct_{suffix}": float((monthly > 0).mean()) if not monthly.empty else 0.0}


def _relative_stats(returns: pd.Series, benchmark_returns: pd.Series, periods: int, risk_free_rate: float) -> dict[str, float]:
    import numpy as np

    aligned = returns.align(benchmark_returns, join="inner")
    strat, bench = aligned[0].to_numpy(), aligned[1].to_numpy()
    if len(strat) < 2 or np.std(bench) == 0:
        return {
            "beta": 0.0,
            "alpha": 0.0,
            "correlation": 0.0,
            "r_squared_strategy": 0.0,
            "r_squared_benchmark": 0.0,
            "treynor_ratio": 0.0,
            "information_ratio_strategy": 0.0,
            "information_ratio_benchmark": 0.0,
        }
    covariance = np.cov(strat, bench)[0, 1]
    beta = covariance / np.var(bench)
    daily_rf = risk_free_rate / periods
    alpha_daily = (np.mean(strat) - daily_rf) - beta * (np.mean(bench) - daily_rf)
    correlation = float(np.corrcoef(strat, bench)[0, 1])
    r_squared = correlation**2
    excess = strat - bench
    tracking_error = np.std(excess, ddof=1)
    information_ratio = float(np.mean(excess) / tracking_error * np.sqrt(periods)) if tracking_error else 0.0
    mean_excess_return = np.mean(strat) - daily_rf
    treynor = float(mean_excess_return * periods / beta) if beta != 0 else 0.0
    return {
        "beta": float(beta),
        "alpha": float(alpha_daily * periods),
        "correlation": correlation,
        "r_squared_strategy": float(r_squared),
        "r_squared_benchmark": float(r_squared),
        "treynor_ratio": treynor,
        "information_ratio_strategy": information_ratio,
        "information_ratio_benchmark": information_ratio,
    }


def _yearly_table(returns: pd.Series, benchmark_returns: pd.Series | None) -> list[dict[str, Any]]:
    years = sorted({ts.year for ts in returns.index})
    rows: list[dict[str, Any]] = []
    for year in years:
        mask = returns.index.year == year
        strat_ret = float((1 + returns[mask]).prod() - 1)
        bench_ret = None
        won = False
        if benchmark_returns is not None:
            b_year = benchmark_returns[benchmark_returns.index.year == year]
            if not b_year.empty:
                bench_ret = float((1 + b_year).prod() - 1)
                won = strat_ret > bench_ret
        rows.append(
            {
                "year": year,
                "strategy": round(strat_ret, 6),
                "benchmark": round(bench_ret, 6) if bench_ret is not None else None,
                "won": won,
            }
        )
    return rows


def _drawdown_table(returns: pd.Series) -> list[dict[str, Any]]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    underwater = drawdown < 0
    rows: list[dict[str, Any]] = []
    start = None
    for time, is_under in underwater.items():
        if is_under and start is None:
            start = time
        elif not is_under and start is not None:
            window = drawdown[start:time]
            rows.append(
                {
                    "start": str(start.date()),
                    "end": str(time.date()),
                    "max_drawdown": round(float(window.min()), 6),
                    "days": int(len(window)),
                }
            )
            start = None
    if start is not None:
        window = drawdown[start:]
        rows.append(
            {
                "start": str(start.date()),
                "end": str(drawdown.index[-1].date()),
                "max_drawdown": round(float(window.min()), 6),
                "days": int(len(window)),
            }
        )
    return rows
