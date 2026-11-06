"""`completed_bars`: the strategy never reads today's partial daily bar."""

from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent_framework.strategies.common.sessions import completed_bars
from trading_agent_framework.utils.clock import MARKET_TZ

TODAY = date(2026, 10, 6)


@pytest.mark.parametrize("hour", [0, 16])  # live Alpaca daily bars are stamped at midnight market time, backtest bars at the close
def test_completed_bars_drops_only_today(hour: int) -> None:
    days = [date(2026, 10, 2), date(2026, 10, 5), TODAY]
    index = pd.DatetimeIndex([datetime.combine(d, time(hour), tzinfo=MARKET_TZ) for d in days]).tz_convert("UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)

    result = completed_bars(df, TODAY)

    assert result["close"].tolist() == [1.0, 2.0]
    assert result.index.tz is not None
    pd.testing.assert_frame_equal(result, df.iloc[:2])
