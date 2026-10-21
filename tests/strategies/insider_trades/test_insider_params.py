from __future__ import annotations

import dataclasses
import math
import re

import pytest

from trading_agent_framework.strategies.common.trade_desk import DeskParams
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.utils.errors import InsiderDataError, TradingFrameworkError


def test_defaults_are_the_spec_values() -> None:
    p = InsiderParams()

    assert p.watchlist == ("AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX", "PFE", "INTC", "F", "KO")
    assert (p.min_weight, p.max_position_weight, p.max_total_weight) == (0.03, 0.20, 0.95)
    assert (p.min_buy_holding_fraction, p.min_sell_holding_fraction) == (0.01, 0.10)
    assert p.signal_window_days == 30 and p.history_years == 3
    assert p.weight_tolerance == 0.001
    assert (p.rebalance_band, p.min_trade_pct) == (0.01, 0.005)
    assert (p.max_trade_retries, p.max_consecutive_abandoned, p.reason_max_chars) == (3, 3, 300)
    assert (p.agent_temperature, p.order_wait_seconds) == (0.3, 60.0)
    assert p.even_weight == pytest.approx(0.095)


def test_the_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        InsiderParams().min_weight = 0.5  # type: ignore[misc]


def test_the_params_satisfy_the_desk_protocol() -> None:
    desk_params: DeskParams = InsiderParams()

    assert (desk_params.rebalance_band, desk_params.min_trade_pct, desk_params.order_wait_seconds) == (0.01, 0.005, 60.0)


def test_insider_data_error_is_a_framework_error() -> None:
    assert issubclass(InsiderDataError, TradingFrameworkError)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"watchlist": ()}, "watchlist must not be empty"),
        ({"watchlist": ("AAPL", "aapl")}, "watchlist must not repeat a ticker"),
        ({"watchlist": ("AAPL", " ")}, "watchlist tickers must not be blank"),
        ({"min_weight": 0.10}, "watchlist x min_weight must fit in max_total_weight"),
        ({"max_position_weight": 0.05}, "the even weight must lie between min_weight and max_position_weight"),
        ({"min_sell_holding_fraction": 0.0}, "holding fractions must be in (0, 1]"),
        ({"min_buy_holding_fraction": 1.5}, "holding fractions must be in (0, 1]"),
        ({"signal_window_days": 0}, "signal_window_days must be at least 1"),
        ({"history_years": 0}, "history_years must be at least 1"),
        ({"min_weight": 0.005}, "min_weight must be at least rebalance_band"),
        ({"max_total_weight": 1.5}, "max_total_weight must be in (0, 1]"),
        ({"rebalance_band": 1.0}, "rebalance_band must be in (0, 1)"),
        ({"min_trade_pct": 1.0}, "min_trade_pct must be in [0, 1)"),
        ({"weight_tolerance": -0.1}, "weight_tolerance must be in [0, 1)"),
        ({"order_wait_seconds": 0.0}, "order_wait_seconds must be above 0"),
        ({"max_trade_retries": -1}, "max_trade_retries must be at least 0"),
        ({"max_consecutive_abandoned": 0}, "max_consecutive_abandoned must be at least 1"),
        ({"reason_max_chars": 0}, "reason_max_chars must be at least 1"),
        ({"agent_temperature": 2.5}, "agent_temperature must be None or in [0, 2]"),
        ({"agent_temperature": math.nan}, "agent_temperature must be None or in [0, 2]"),
        ({"min_weight": math.nan}, "must be finite"),
    ],
)
def test_invalid_settings_are_refused(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        InsiderParams(**kwargs)


def test_an_empty_watchlist_reports_only_its_own_problem() -> None:
    with pytest.raises(ValueError) as error:
        InsiderParams(watchlist=())

    assert str(error.value) == "InsiderParams: watchlist must not be empty"


def test_every_problem_is_reported_in_one_error() -> None:
    with pytest.raises(ValueError) as error:
        InsiderParams(signal_window_days=0, history_years=0)

    assert "signal_window_days must be at least 1" in str(error.value)
    assert "history_years must be at least 1" in str(error.value)


def test_a_watchlist_is_stored_as_given() -> None:
    watchlist = ("aapl", "msft", "jpm", "bac", "xom")

    assert InsiderParams(watchlist=watchlist).watchlist == watchlist
