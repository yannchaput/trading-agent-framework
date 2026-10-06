"""cross_momentum's portfolio risk overlay aligns series by session date, never by position.

On 2026-10-06 about half of the target series ended a session before SPY's; aligned from the end by position,
every return was paired with the wrong day, and the overlay read NORMAL (beta 1.35) instead of CRITICAL (2.8).
"""

from datetime import datetime, time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.portfolio_risk_overlay import (
    RiskState,
    classify_risk_state,
    compute_risk_overlay,
)
from trading_agent_framework.strategies.cross_momentum.utils import close_series
from trading_agent_framework.utils.clock import MARKET_TZ

DATES = [d.date() for d in pd.bdate_range("2026-01-02", periods=120)]


def _market(n_stocks: int = 10, beta: float = 2.5, seed: int = 0) -> tuple[dict[str, pd.Series], pd.Series]:
    """Stocks moving `beta` times SPY plus noise, one close per date in DATES."""
    rng = np.random.default_rng(seed)
    spy_returns = rng.normal(0.0, 0.01, len(DATES) - 1)
    spy = pd.Series(100 * np.cumprod(np.r_[1.0, 1 + spy_returns]), index=DATES)
    stocks = {}
    for i in range(n_stocks):
        returns = beta * spy_returns + rng.normal(0.0, 0.01, len(spy_returns))
        stocks[f"S{i}"] = pd.Series(50 * np.cumprod(np.r_[1.0, 1 + returns]), index=DATES)
    return stocks, spy


def _equal(stocks: dict[str, pd.Series]) -> dict[str, float]:
    return {symbol: 1 / len(stocks) for symbol in stocks}


def test_series_one_day_short_are_aligned_by_date_not_by_position():
    stocks, spy = _market()
    short = {symbol: (series.iloc[:-1] if i % 2 else series) for i, (symbol, series) in enumerate(stocks.items())}

    _, _, full = compute_risk_overlay(stocks, _equal(stocks), spy)
    state, exposure, metrics = compute_risk_overlay(short, _equal(short), spy)

    assert metrics["beta_63d"] == pytest.approx(full["beta_63d"], abs=0.05)
    assert metrics["corr_20d"] == pytest.approx(full["corr_20d"], abs=0.1)
    assert (state, exposure) == ("critical", 0.4)


def test_complete_series_give_one_return_per_date():
    stocks, spy = _market()

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), spy)

    assert metrics["observations"] == len(DATES) - 1


def test_a_date_missing_from_one_series_is_dropped_for_all():
    stocks, spy = _market()
    stocks["S0"] = stocks["S0"].drop(DATES[60])

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), spy)

    assert metrics["observations"] == len(DATES) - 2


def test_fewer_than_min_obs_aligned_returns_is_normal():
    stocks, spy = _market()
    recent = {symbol: series.iloc[-30:] for symbol, series in stocks.items()}

    state, exposure, metrics = compute_risk_overlay(recent, _equal(recent), spy)

    assert (state, exposure) == ("normal", 1.0)
    assert metrics["beta_63d"] is None
    assert metrics["observations"] == 29


def test_series_sharing_no_date_give_normal_with_zero_observations():
    stocks, spy = _market()
    stale = {symbol: series.iloc[:50] for symbol, series in stocks.items()}

    state, exposure, metrics = compute_risk_overlay(stale, _equal(stale), spy.iloc[60:])

    assert (state, exposure) == ("normal", 1.0)
    assert metrics["observations"] == 0


def test_without_a_benchmark_vol_and_corr_are_still_computed():
    stocks, _ = _market()

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), None)

    assert metrics["beta_63d"] is None
    assert metrics["vol_20d"] > 0
    assert metrics["corr_20d"] > 0.5


@pytest.mark.parametrize(
    ("beta", "vol", "corr", "expected"),
    [
        (2.0, None, None, RiskState.CRITICAL),
        (1.99, None, None, RiskState.ELEVATED),
        (1.7, None, None, RiskState.ELEVATED),
        (1.69, None, None, RiskState.NORMAL),
        (None, 0.35, 0.35, RiskState.CRITICAL),
        (None, 0.30, 0.30, RiskState.ELEVATED),
        (None, 0.29, 0.35, RiskState.NORMAL),
    ],
)
def test_the_classification_thresholds_are_unchanged(beta, vol, corr, expected):
    assert classify_risk_state(beta, vol, corr) is expected


# --- close_series -----------------------------------------------------------------------


def _frame(dates, closes, hour: int) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime.combine(d, time(hour), tzinfo=MARKET_TZ) for d in dates])
    return pd.DataFrame({"close": closes}, index=index)


@pytest.mark.parametrize("hour", [0, 16])  # live Alpaca bars are stamped at midnight, backtest bars at the close
def test_close_series_is_indexed_by_market_date(hour):
    series = close_series(_frame(DATES[:3], [1.0, 2.0, 3.0], hour))

    assert list(series.index) == DATES[:3]
    assert list(series) == [1.0, 2.0, 3.0]


def test_close_series_keeps_one_value_per_date():
    frame = pd.concat([_frame(DATES[:2], [1.0, 2.0], 0), _frame(DATES[1:2], [2.5], 16)]).sort_index()

    series = close_series(frame)

    assert list(series.index) == DATES[:2]
    assert list(series) == [1.0, 2.5]


# --- CrossMomentumStrategy._risk_exposure ---------------------------------------------------


class FakeStrategy:
    """Just enough of `Strategy` for `_risk_exposure`."""

    def __init__(self, target_closes, spy):
        self.vars = SimpleNamespace(target_closes=target_closes, alpaca_rate_limiter=SimpleNamespace(wait=lambda: None))
        self._spy = spy
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def get_historical_prices(self, ticker, length, timestep):
        if self._spy is None:
            return None
        return Bars(Asset("SPY"), "day", _frame(list(self._spy.index), list(self._spy), 0))

    def log_info(self, message, *args, **kwargs):
        self.infos.append(message)

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)


def _target(stocks):
    return [{"symbol": symbol, "target_weight": 1 / len(stocks)} for symbol in stocks]


def test_risk_exposure_aligns_by_date_and_names_the_series_a_session_short():
    stocks, spy = _market()
    short = {symbol: (series.iloc[:-1] if i % 2 else series) for i, (symbol, series) in enumerate(stocks.items())}
    fake = FakeStrategy(short, spy)

    exposure = CrossMomentumStrategy._risk_exposure(fake, _target(short))

    assert exposure == 0.4
    assert len(fake.warnings) == 1
    assert "S1" in fake.warnings[0] and "S0" not in fake.warnings[0]
    assert any("CRITICAL" in line and "obs=" in line for line in fake.infos)


def test_risk_exposure_without_spy_data_is_neutral():
    stocks, _ = _market()
    fake = FakeStrategy(stocks, None)

    assert CrossMomentumStrategy._risk_exposure(fake, _target(stocks)) == 1.0
    assert any("SPY" in message for message in fake.warnings)


def test_risk_exposure_without_targets_is_neutral():
    fake = FakeStrategy({}, None)

    assert CrossMomentumStrategy._risk_exposure(fake, []) == 1.0
