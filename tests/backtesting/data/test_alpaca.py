from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from tests.fakes import FakeStockHistoricalDataClient, FakeTradingClient, bar_payload, make_alpaca_calendar

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData, reindex_to_bar_close
from trading_agent_framework.brokers.alpaca.market_data import MARKET_TZ
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

AAPL = Asset("AAPL")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def _sessions() -> list[MarketSession]:
    return [
        MarketSession(
            open=datetime(2026, 1, 5, 9, 30, tzinfo=MARKET_TZ),
            close=datetime(2026, 1, 5, 16, 0, tzinfo=MARKET_TZ),
        ),
        MarketSession(
            open=datetime(2026, 1, 6, 9, 30, tzinfo=MARKET_TZ),
            close=datetime(2026, 1, 6, 13, 0, tzinfo=MARKET_TZ),  # early close
        ),
    ]


def test_reindex_to_bar_close_maps_daily_bars_to_their_sessions_close() -> None:
    df = pd.DataFrame(
        {"open": [150.0, 151.0], "high": [151.0, 152.0], "low": [149.0, 150.0],
         "close": [150.5, 151.5], "volume": [1000.0, 1100.0]},
        index=pd.DatetimeIndex(
            [datetime(2026, 1, 5, tzinfo=MARKET_TZ), datetime(2026, 1, 6, tzinfo=MARKET_TZ)]
        ),
    )
    result = reindex_to_bar_close(df, "day", _sessions())
    assert list(result.index) == [_sessions()[0].close, _sessions()[1].close]  # early close respected


def test_reindex_to_bar_close_shifts_minute_bars_by_one_minute() -> None:
    df = pd.DataFrame(
        {"open": [150.0], "high": [151.0], "low": [149.0], "close": [150.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([datetime(2026, 1, 5, 9, 30, tzinfo=MARKET_TZ)]),
    )
    result = reindex_to_bar_close(df, "minute", [])
    assert result.index[0] == datetime(2026, 1, 5, 9, 31, tzinfo=MARKET_TZ)


def test_reindex_to_bar_close_handles_an_empty_frame() -> None:
    assert reindex_to_bar_close(pd.DataFrame(), "day", []).empty


def test_reindex_to_bar_close_raises_when_a_daily_bars_date_has_no_session() -> None:
    """Important whole-branch review finding. This used to fall back to the bar's
    ORIGINAL (bar-open, midnight-ET) index, which is a real no-look-ahead violation:
    the Jan-7 bar below would have been indexed at 2026-01-07 00:00 ET and so become
    visible through `bars(..., cutoff)` more than nine hours before its session even
    opens. There is no correct close time to re-index to, so it must fail loudly.
    """
    df = pd.DataFrame(
        {"open": [150.0, 151.0], "high": [151.0, 152.0], "low": [149.0, 150.0],
         "close": [150.5, 151.5], "volume": [1000.0, 1100.0]},
        index=pd.DatetimeIndex(
            # Jan 7 is absent from `_sessions()` -- exactly what a window filter that
            # drops a date the bars fetch kept produces.
            [datetime(2026, 1, 5, tzinfo=MARKET_TZ), datetime(2026, 1, 7, tzinfo=MARKET_TZ)]
        ),
    )
    with pytest.raises(BacktestDataError, match="2026-01-07"):
        reindex_to_bar_close(df, "day", _sessions(), symbol="AAPL")


def test_a_daily_bar_outside_the_session_window_raises_through_the_public_bars_method() -> None:
    """The same defect reached the way it actually happens in a run: `end` lands
    mid-session, so `sessions()`'s `s.close <= end` filter drops that day while the
    bars fetch still returns its daily bar."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [
        make_alpaca_calendar("2026-01-05"), make_alpaca_calendar("2026-01-06"),
    ]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [
        bar_payload("2026-01-05T05:00:00Z", 150.0), bar_payload("2026-01-06T05:00:00Z", 151.0),
    ]
    mid_session_end = datetime(2026, 1, 6, 12, 0, tzinfo=MARKET_TZ)  # before the 16:00 close
    source = AlpacaBacktestData(data_client, trading_client, START, mid_session_end)

    with pytest.raises(BacktestDataError, match="2026-01-06"):
        source.bars(AAPL, mid_session_end, 2, "day")


def test_bars_fetches_and_reindexes_via_the_injected_clients() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [
        make_alpaca_calendar("2026-01-05"), make_alpaca_calendar("2026-01-06", close_at="13:00"),
    ]
    data_client = FakeStockHistoricalDataClient()
    # Daily bars, realistically timestamped at midnight *market time* (EST here:
    # UTC-5, so 05:00Z), not a fixed 00:00:00Z -- see market_data.py:146.
    data_client.bars["AAPL"] = [
        bar_payload("2026-01-05T05:00:00Z", 150.0), bar_payload("2026-01-06T05:00:00Z", 151.0),
    ]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    result = source.bars(AAPL, END, 2, "day")

    assert result is not None
    assert list(result.df["close"]) == [150.0, 151.0]
    # Each bar lands on its own session's ET calendar date and close time --
    # the Jan-5 bar on the regular 16:00 close, the Jan-6 bar (early close) on
    # 13:00 -- proving the date alignment, not just a shared "16:00" hour.
    assert (result.df.index[0].date(), result.df.index[0].hour) == (date(2026, 1, 5), 16)
    assert (result.df.index[1].date(), result.df.index[1].hour) == (date(2026, 1, 6), 13)


def test_bars_returns_none_for_an_asset_with_no_data() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    assert source.bars(Asset("MISSING"), END, 1, "day") is None


def test_a_bars_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.raises["get_stock_bars"] = RuntimeError("API down")
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_a_calendar_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.raises["get_calendar"] = RuntimeError("API down")
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    with pytest.raises(BacktestDataError, match="calendar"):
        source.bars(AAPL, END, 1, "day")


def test_a_malformed_bars_response_raises_backtest_data_error() -> None:
    """`parse_bars` runs in the same guarded block as the fetch: a malformed
    response must not escape as a raw exception either."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [{"t": "2026-01-05T05:00:00Z"}]  # missing o/h/l/c/v
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_sessions_are_exact_including_early_closes() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-06", close_at="13:00")]
    data_client = FakeStockHistoricalDataClient()
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    [session] = source.sessions(START, END)
    assert session.close.hour == 13


def test_name_is_alpaca() -> None:
    source = AlpacaBacktestData(FakeStockHistoricalDataClient(), FakeTradingClient(), START, END)
    assert source.name == "alpaca"


def test_load_honors_its_own_start_and_end_arguments() -> None:
    """load()'s start/end must reach the built bars request, not the constructor's
    fixed window -- CachedDataSource trusts load()'s own window when it derives
    a cache file's path."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-02-01")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-02-01T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    other_start = datetime(2026, 2, 1, tzinfo=UTC)
    other_end = datetime(2026, 2, 10, tzinfo=UTC)
    source.load([AAPL], other_start, other_end, "day")

    assert len(data_client.bars_requests) == 1
    request = data_client.bars_requests[0]
    # `StockBarsRequest` stores naive datetimes internally.
    assert request.start.replace(tzinfo=UTC) == other_start
    assert request.end.replace(tzinfo=UTC) == other_end


def test_bars_lazy_fetch_uses_the_constructor_fixed_window() -> None:
    """Unlike `load()`, `bars()`'s own lazy-fetch path has no start/end parameters
    of its own -- it must fall back to the constructor's fixed window."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    source.bars(AAPL, END, 1, "day")

    assert len(data_client.bars_requests) == 1
    request = data_client.bars_requests[0]
    assert request.start.replace(tzinfo=UTC) == START
    assert request.end.replace(tzinfo=UTC) == END
