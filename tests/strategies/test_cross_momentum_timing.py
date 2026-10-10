"""cross_momentum decides on completed sessions only: a bar dated today is partial while the session is open."""

from datetime import date, datetime, time

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config import TradingMode
from trading_agent_framework.strategies.common.sessions import completed_bars, parse_rebalance_time
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

MONDAY, TUESDAY = date(2026, 10, 5), date(2026, 10, 6)


def _frame(dates, hour: int) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime.combine(d, time(hour), tzinfo=MARKET_TZ) for d in dates])
    return pd.DataFrame({"close": [float(i) for i in range(len(dates))]}, index=index)


@pytest.mark.parametrize("hour", [0, 16])  # live Alpaca bars are stamped at midnight, backtest bars at the close
def test_completed_bars_drops_the_bar_dated_today(hour):
    df = completed_bars(_frame([date(2026, 10, 2), MONDAY, TUESDAY], hour), TUESDAY)

    assert [ts.date() for ts in df.index] == [date(2026, 10, 2), MONDAY]


def test_completed_bars_without_a_bar_for_today_changes_nothing():
    frame = _frame([date(2026, 10, 2), MONDAY], 16)

    pd.testing.assert_frame_equal(completed_bars(frame, TUESDAY), frame)


def test_the_default_rebalance_time_is_noon():
    assert CONFIG["rebalance_time"] == "12:00"


def test_parse_rebalance_time_reads_hh_mm():
    assert parse_rebalance_time("12:00") == time(12, 0)
    assert parse_rebalance_time("10:30") == time(10, 30)


@pytest.mark.parametrize("value", ["09:30", "10:30", "12:00", "15:58"])
def test_parse_rebalance_time_accepts_a_time_in_the_regular_session(value):
    assert parse_rebalance_time(value) == time.fromisoformat(value)


@pytest.mark.parametrize(
    "value",
    [
        "noon", "25:00", "12:60", "24:00", "12:00+02:00", "12:00:30", "12:00:00.5", "1200", "12", "T12:00", "", " 12:00",
        None, 1200, time(12, 0),
        "08:00", "09:29", "15:59", "16:30",
    ],
)  # fmt: skip
def test_parse_rebalance_time_rejects_anything_else(value):
    with pytest.raises(ConfigurationError, match="rebalance_time"):
        parse_rebalance_time(value)


def _strategy(**parameters):
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 7), []))
    return CrossMomentumStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], parameters=parameters)


def test_the_strategy_iterates_at_its_rebalance_time():
    assert _strategy().iteration_start_time == time(12, 0)
    assert _strategy(rebalance_time="10:30").iteration_start_time == time(10, 30)


def test_a_malformed_rebalance_time_fails_at_construction():
    with pytest.raises(ConfigurationError, match="rebalance_time"):
        _strategy(rebalance_time="noon")


def test_a_rebalance_time_of_24_00_fails_at_construction():
    with pytest.raises(ConfigurationError, match="rebalance_time"):
        _strategy(rebalance_time="24:00")
