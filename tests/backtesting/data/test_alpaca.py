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


def test_bars_are_reindexed_through_the_end_of_the_end_date_even_with_a_midnight_end() -> None:
    """Bug report (2026-09-17 cross_momentum backtest, empty log file):
    `Strategy.backtesting_end`/`BACKTESTING_PARAMS` are conventionally a bare
    midnight `datetime` (e.g. `datetime(2026, 4, 24, tzinfo=MARKET_TZ)`), read by
    `YahooBacktestData` as "through the end of that day". `sessions()` used to
    compare exact clock time instead of calendar date, so that same midnight `end`
    excluded its own date's session while the bars fetch (a separate, date-only
    request) still returned that day's bar -- `reindex_to_bar_close` then raised
    `BacktestDataError` before a single log line was written. `sessions()` now
    filters by calendar date like `YahooBacktestData`, so a midnight `end` still
    covers its whole day."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [
        make_alpaca_calendar("2026-01-05"), make_alpaca_calendar("2026-01-06"),
    ]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [
        bar_payload("2026-01-05T05:00:00Z", 150.0), bar_payload("2026-01-06T05:00:00Z", 151.0),
    ]
    midnight_end = datetime(2026, 1, 6, tzinfo=MARKET_TZ)  # outside market hours, not the 16:00 close
    source = AlpacaBacktestData(START, midnight_end, client=data_client, trading_client=trading_client)

    # Fetching/reindexing against a midnight `end` must not raise -- this is the bug
    # itself (it used to blow up here, before a single bar was ever inspected).
    source.load([AAPL], START, midnight_end, "day")

    # And once the clock actually reaches the session's real close, both days'
    # (correctly reindexed) bars are visible -- the midnight `end` widened the
    # covered calendar dates, it didn't corrupt each bar's own close timestamp.
    session_close = datetime(2026, 1, 6, 16, 0, tzinfo=MARKET_TZ)
    result = source.bars(AAPL, session_close, 2, "day")
    assert result is not None
    assert list(result.df["close"]) == [150.0, 151.0]
    assert result.df.index[-1] == session_close


def test_a_daily_bar_dated_outside_the_known_calendar_raises_through_the_public_bars_method() -> None:
    """`reindex_to_bar_close`'s defensive raise still guards a genuine calendar/data
    mismatch: a date the bars endpoint returned a bar for but the calendar has no
    session on at all (e.g. Alpaca's calendar and bars endpoints disagreeing, or a
    stale cached calendar) -- not merely an `end` that lands mid-session or at
    midnight, which `sessions()`'s calendar-date filter now tolerates."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]  # 2026-01-06 missing
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [
        bar_payload("2026-01-05T05:00:00Z", 150.0), bar_payload("2026-01-06T05:00:00Z", 151.0),
    ]
    end = datetime(2026, 1, 6, 16, 0, tzinfo=MARKET_TZ)
    source = AlpacaBacktestData(START, end, client=data_client, trading_client=trading_client)

    with pytest.raises(BacktestDataError, match="2026-01-06"):
        source.bars(AAPL, end, 2, "day")


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
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

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
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    assert source.bars(Asset("MISSING"), END, 1, "day") is None


def test_a_bars_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.raises["get_stock_bars"] = RuntimeError("API down")
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_a_calendar_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.raises["get_calendar"] = RuntimeError("API down")
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    with pytest.raises(BacktestDataError, match="calendar"):
        source.bars(AAPL, END, 1, "day")


def test_a_malformed_bars_response_raises_backtest_data_error() -> None:
    """`parse_bars` runs in the same guarded block as the fetch: a malformed
    response must not escape as a raw exception either."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [{"t": "2026-01-05T05:00:00Z"}]  # missing o/h/l/c/v
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_sessions_are_exact_including_early_closes() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-06", close_at="13:00")]
    data_client = FakeStockHistoricalDataClient()
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    [session] = source.sessions(START, END)
    assert session.close.hour == 13


def test_name_is_alpaca() -> None:
    source = AlpacaBacktestData(START, END, client=FakeStockHistoricalDataClient(), trading_client=FakeTradingClient())
    assert source.name == "alpaca"


def test_load_honors_its_own_start_and_end_arguments() -> None:
    """load()'s start/end must reach the built bars request, not the constructor's
    fixed window -- CachedDataSource trusts load()'s own window when it derives
    a cache file's path."""
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-02-01")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-02-01T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

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
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    source.bars(AAPL, END, 1, "day")

    assert len(data_client.bars_requests) == 1
    request = data_client.bars_requests[0]
    assert request.start.replace(tzinfo=UTC) == START
    assert request.end.replace(tzinfo=UTC) == END


def test_load_widened_for_warmup_reaches_bars_before_the_narrow_start() -> None:
    """Task 6 (review-fix pass) belt-and-suspenders test: proves the exact
    `load()`-then-`bars()` interaction `backtesting/runner.py`'s `_run` relies on for
    `warmup_trading_days` actually works against `AlpacaBacktestData`, not just against
    `tests/backtesting/fakes.FakeBacktestDataSource` (which `test_runner.py`'s own
    warmup test uses).

    Mirrors `runner._run`'s shape: `AlpacaBacktestData` is constructed with the narrow
    simulation window (per `Strategy.run_backtesting`'s own docstring, an explicit
    `data_source` is used as given and is not itself widened), then `load()` is called
    with a start earlier than that narrow window -- exactly what `_run`'s widened
    `warmup_start` does for the benchmark asset. `bars()` is then asked for history as
    of the very first simulated session's open, before that session's own bar has
    closed: the only bars it can legitimately return are the warm-up ones fetched by
    the widened `load()` call, so seeing them here (and not the not-yet-closed narrow-
    window bar) proves warm-up history genuinely reaches `bars()` rather than merely
    being requested over HTTP (already covered by
    `test_load_honors_its_own_start_and_end_arguments`).
    """
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [
        make_alpaca_calendar("2026-01-02"),  # warm-up-only session, before the narrow start
        make_alpaca_calendar("2026-01-05"),  # warm-up-only session, before the narrow start
        make_alpaca_calendar("2026-01-06"),  # the narrow simulation's first session
    ]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["SPY"] = [
        bar_payload("2026-01-02T05:00:00Z", 400.0),
        bar_payload("2026-01-05T05:00:00Z", 402.0),
        bar_payload("2026-01-06T05:00:00Z", 405.0),
    ]
    narrow_start = datetime(2026, 1, 6, 9, 30, tzinfo=MARKET_TZ)
    end = datetime(2026, 1, 6, 16, 0, tzinfo=MARKET_TZ)
    # Constructed with the NARROW window -- an explicit data_source is never widened
    # by `Strategy.run_backtesting` itself, only `runner._run`'s own eager benchmark
    # `load()` call is.
    source = AlpacaBacktestData(narrow_start, end, client=data_client, trading_client=trading_client)
    spy = Asset("SPY")

    warmup_start = datetime(2026, 1, 2, 9, 30, tzinfo=MARKET_TZ)
    source.load([spy], warmup_start, end, "day")

    # Cutoff at the narrow start's session OPEN, before that session's own bar has
    # closed -- only warm-up history can legitimately be visible yet.
    result = source.bars(spy, narrow_start, 3, "day")

    assert result is not None
    assert list(result.df["close"]) == [400.0, 402.0]  # Jan-2 and Jan-5, not the un-closed Jan-6 bar
    # And the fetch actually reached back to the widened start, not the narrow one.
    assert data_client.bars_requests[0].start.replace(tzinfo=UTC) == warmup_start.astimezone(UTC)


def test_load_batches_multiple_assets_into_one_alpaca_call() -> None:
    """The whole point: filtering a large universe must not fire one `get_stock_bars`
    call per ticker -- `load()` fetches every requested asset in a single request,
    mirroring `YahooBacktestData.load()`'s single batched `download` call."""
    MSFT = Asset("MSFT")
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    data_client.bars["MSFT"] = [bar_payload("2026-01-05T05:00:00Z", 300.0)]
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    source.load([AAPL, MSFT], START, END, "day")

    assert len(data_client.bars_requests) == 1
    assert sorted(data_client.bars_requests[0].symbol_or_symbols) == ["AAPL", "MSFT"]
    aapl_bars = source.bars(AAPL, END, 1, "day")
    msft_bars = source.bars(MSFT, END, 1, "day")
    assert aapl_bars is not None and list(aapl_bars.df["close"]) == [150.0]
    assert msft_bars is not None and list(msft_bars.df["close"]) == [300.0]


def test_load_already_cached_assets_makes_no_extra_alpaca_call() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    source.load([AAPL], START, END, "day")
    source.load([AAPL], START, END, "day")

    assert len(data_client.bars_requests) == 1


def test_concurrent_fetches_are_serialized() -> None:
    """A strategy filtering a large universe fetches per ticker from a thread pool
    when a position falls outside a preloaded batch. Firing many `get_stock_bars`
    calls at once races the shared client's session state, mirroring the exact
    concern `YahooBacktestData._download_lock` addresses for `yf.download`.
    `AlpacaBacktestData` must serialize its fetches so only one is ever in flight."""
    import threading as threading_module
    import time

    active = 0
    max_active = 0
    lock = threading_module.Lock()

    class SlowStockDataClient(FakeStockHistoricalDataClient):
        def get_stock_bars(self, request_params):  # type: ignore[override]
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return super().get_stock_bars(request_params)

    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = SlowStockDataClient()
    tickers = [Asset(f"T{i}") for i in range(8)]
    for ticker in tickers:
        data_client.bars[ticker.symbol] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
    source = AlpacaBacktestData(START, END, client=data_client, trading_client=trading_client)

    threads = [
        threading_module.Thread(target=source.bars, args=(t, END, 1, "day")) for t in tickers
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert max_active == 1


def test_default_clients_are_built_from_env_credentials_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors `YahooBacktestData`'s lazy `_real_download()`: an omitted `client`/
    `trading_client` is not required up front -- it's built from
    `AlpacaCredentials.from_env()` only the first time it's actually needed, which
    is what lets `AlpacaBacktestData` be passed as a bare `data_source` class, the
    same way `YahooBacktestData` already can be."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret")
    built: list[str] = []

    def fake_build_stock_data_client(creds):
        built.append("data")
        data_client = FakeStockHistoricalDataClient()
        data_client.bars["AAPL"] = [bar_payload("2026-01-05T05:00:00Z", 150.0)]
        return data_client

    def fake_build_trading_client(creds):
        built.append("trading")
        trading_client = FakeTradingClient()
        trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
        return trading_client

    monkeypatch.setattr(
        "trading_agent_framework.brokers.alpaca.client.build_stock_data_client",
        fake_build_stock_data_client,
    )
    monkeypatch.setattr(
        "trading_agent_framework.brokers.alpaca.client.build_trading_client",
        fake_build_trading_client,
    )
    source = AlpacaBacktestData(START, END)

    result = source.bars(AAPL, END, 1, "day")

    assert result is not None and list(result.df["close"]) == [150.0]
    assert built == ["data", "trading"]
