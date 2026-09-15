from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent_framework.backtesting.metrics import compute_metrics


def _returns() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, -0.03, 0.015, 0.005], index=dates)


def test_compute_metrics_sharpe_matches_the_standard_annualised_formula() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    expected_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert result["sharpe_strategy"] == pytest.approx(expected_sharpe, rel=1e-3)


def test_compute_metrics_max_drawdown_matches_the_cumulative_curve_formula() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    curve = (1 + returns).cumprod()
    expected_max_dd = float((curve / curve.cummax() - 1).min())
    assert result["max_drawdown_strategy"] == pytest.approx(expected_max_dd, rel=1e-6)


def test_compute_metrics_total_return_matches_the_compounded_return() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    expected_total = float((1 + returns).prod() - 1)
    assert result["total_return_strategy"] == pytest.approx(expected_total, rel=1e-6)


def test_compute_metrics_without_a_benchmark_omits_relative_fields_but_not_the_others() -> None:
    result = compute_metrics(_returns(), None, timestep="day", risk_free_rate=0.0)
    assert "sharpe_strategy" in result
    assert "beta" not in result  # no benchmark given -> no relative stats block


def test_compute_metrics_with_a_benchmark_computes_beta_alpha_and_correlation() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    strategy = pd.Series([0.02, -0.01, 0.03, 0.0, 0.02, -0.02, 0.03, -0.01, 0.02, 0.01], index=dates)
    benchmark = pd.Series([0.01, -0.005, 0.015, 0.0, 0.01, -0.01, 0.015, -0.005, 0.01, 0.005], index=dates)

    result = compute_metrics(strategy, benchmark, timestep="day", risk_free_rate=0.0)

    expected_beta = float(np.cov(strategy, benchmark)[0, 1] / np.var(benchmark))
    assert result["beta"] == pytest.approx(expected_beta, rel=1e-3)
    assert result["correlation"] == pytest.approx(float(np.corrcoef(strategy, benchmark)[0, 1]), rel=1e-3)
    assert "sharpe_benchmark" in result


def test_compute_metrics_sortino_incorporates_a_nonzero_risk_free_rate() -> None:
    """Regression test for a review finding: compute_metrics called sortino_ratio()
    with no arguments, so risk_free_rate never reached the Sortino threshold even
    though the design spec (section 6.5) requires it for Sharpe/Sortino/Treynor. Every
    other golden-value test in this file uses risk_free_rate=0.0, where the bug is
    invisible (the zero threshold is correct either way) -- this test uses a nonzero
    rate and hand-computes the expected ratio independently of vectorbt."""
    returns = _returns()
    risk_free_rate = 0.02
    periods = 252
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=risk_free_rate)

    daily_rf = risk_free_rate / periods
    adjusted = returns - daily_rf
    average_annualized_return = adjusted.mean() * periods
    downside = adjusted.clip(upper=0.0)
    downside_risk = float(np.sqrt((downside**2).mean()) * np.sqrt(periods))
    expected_sortino = float(average_annualized_return / downside_risk)

    assert result["sortino_strategy"] == pytest.approx(expected_sortino, rel=1e-6)

    # Sanity check: the buggy zero-threshold value must differ from the fixed one,
    # otherwise this test wouldn't actually be exercising the fix.
    zero_threshold_downside = float(np.sqrt((returns.clip(upper=0.0) ** 2).mean()) * np.sqrt(periods))
    zero_threshold_sortino = float((returns.mean() * periods) / zero_threshold_downside)
    assert expected_sortino != pytest.approx(zero_threshold_sortino, rel=1e-6)


def test_compute_metrics_skew_and_kurtosis_match_pandas() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)
    assert result["skew_strategy"] == pytest.approx(float(returns.skew()), rel=1e-6)
    assert result["kurtosis_strategy"] == pytest.approx(float(returns.kurt()), rel=1e-6)


def test_compute_metrics_includes_summary_tables_in_raw() -> None:
    result = compute_metrics(_returns(), None, timestep="day", risk_free_rate=0.0)
    assert "summary_tables" in result["raw"]
    assert "eoy_returns_vs_benchmark" in result["raw"]["summary_tables"]
    assert "drawdowns" in result["raw"]["summary_tables"]
    assert result["raw"]["summary_tables"]["eoy_returns_vs_benchmark"][0]["year"] == 2024


def test_compute_metrics_never_raises_on_a_constant_return_series() -> None:
    """A flat return series has zero std/variance -- must degrade gracefully, not NaN-crash."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    flat = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates)
    result = compute_metrics(flat, flat, timestep="day", risk_free_rate=0.0)
    assert result["beta"] == 0.0
