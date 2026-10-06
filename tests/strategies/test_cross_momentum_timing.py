"""cross_momentum decides on completed sessions only: a bar dated today is partial while the session is open."""

from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent_framework.strategies.cross_momentum.utils import completed_bars
from trading_agent_framework.utils.clock import MARKET_TZ

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
