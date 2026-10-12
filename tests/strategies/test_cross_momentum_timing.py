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


def _initialized(tmp_path, **parameters):
    """A strategy through `initialize()`, with the two state files it reads pointed at `tmp_path` (never the repo's `data/`)."""
    strategy = _strategy(**parameters)
    strategy.vars.history_file_path = tmp_path / "history.json"
    strategy.vars.breadth_file_path = tmp_path / "breadth.json"
    strategy.initialize()
    return strategy


def test_initialize_sets_the_iteration_start_time_with_the_sleeptime(tmp_path):
    strategy = _strategy()
    assert strategy.iteration_start_time is None  # a run configuration, set by initialize() next to the sleeptime

    strategy.vars.history_file_path = tmp_path / "history.json"
    strategy.vars.breadth_file_path = tmp_path / "breadth.json"
    strategy.initialize()

    assert (strategy.sleeptime, strategy.iteration_start_time) == ("1D", time(12, 0))


def test_the_strategy_iterates_at_its_rebalance_time(tmp_path):
    assert _initialized(tmp_path).iteration_start_time == time(12, 0)
    assert _initialized(tmp_path, rebalance_time="10:30").iteration_start_time == time(10, 30)


@pytest.mark.parametrize("value", ["noon", "24:00"])
def test_a_malformed_rebalance_time_fails_at_initialize(tmp_path, value):
    strategy = _strategy(rebalance_time=value)  # construction accepts it: the value is read when the run is configured
    strategy.vars.history_file_path = tmp_path / "history.json"
    strategy.vars.breadth_file_path = tmp_path / "breadth.json"

    with pytest.raises(ConfigurationError, match="rebalance_time"):
        strategy.initialize()
