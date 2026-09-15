from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData, parse_yahoo_frame
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BacktestDataError

AAPL = Asset("AAPL")
ET = ZoneInfo("America/New_York")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def _raw_yahoo_frame() -> pd.DataFrame:
    """Shaped like a real single-ticker yfinance.download(...) result: capitalised
    columns, a naive DatetimeIndex of session dates."""
    index = pd.date_range("2026-01-05", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0, 152.0], "High": [151.0, 152.0, 153.0],
            "Low": [149.0, 150.0, 151.0], "Close": [150.5, 151.5, 152.5],
            "Volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


def test_parse_yahoo_frame_lowercases_columns_and_indexes_by_session_close() -> None:
    df = parse_yahoo_frame(_raw_yahoo_frame())

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 150.5
    assert df.index[0] == datetime(2026, 1, 5, 16, 0, tzinfo=ET)
    assert df.index[0].tzinfo is not None


def test_parse_yahoo_frame_handles_an_empty_frame() -> None:
    assert parse_yahoo_frame(pd.DataFrame()).empty


def test_parse_yahoo_frame_drops_a_multiindex_ticker_level() -> None:
    raw = _raw_yahoo_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
    df = parse_yahoo_frame(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_parse_yahoo_frame_drops_nan_ohlcv_rows_and_keeps_the_valid_ones() -> None:
    """yfinance legitimately emits NaN rows (halts, delistings, multi-ticker gaps).
    Left in, `BacktestBroker._latest_bar_with_time` turns one into `Decimal('NaN')`
    prices: a MARKET order filling against it poisons the broker's cash with NaN
    permanently (visible only much later, as blank metrics), and a LIMIT/STOP order
    raises `decimal.InvalidOperation` out of `fills.py` instead (Important
    whole-branch review finding).
    """
    raw = _raw_yahoo_frame()
    raw.loc[raw.index[1], ["Open", "High", "Low", "Close"]] = float("nan")

    df = parse_yahoo_frame(raw)

    assert len(df) == 2  # the middle row is gone, the other two survive
    assert not df.isna().to_numpy().any()
    assert list(df["close"]) == [150.5, 152.5]
    assert df.index[0] == datetime(2026, 1, 5, 16, 0, tzinfo=ET)
    assert datetime(2026, 1, 6, 16, 0, tzinfo=ET) not in list(df.index)


def test_parse_yahoo_frame_with_every_row_nan_produces_an_empty_frame() -> None:
    """The degenerate case must still come back well-formed (the columns downstream
    expects, zero rows), not raise -- `bars()` then simply returns None."""
    raw = _raw_yahoo_frame()
    raw[["Open", "High", "Low", "Close", "Volume"]] = float("nan")

    df = parse_yahoo_frame(raw)

    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_bars_never_returns_a_nan_row_to_the_broker() -> None:
    """End-to-end through the public `bars()` path -- the shape the broker actually
    sees, where a NaN OHLC row would become `Decimal('NaN')` prices."""
    raw = _raw_yahoo_frame()
    raw.loc[raw.index[1], ["Open", "High", "Low", "Close"]] = float("nan")

    source = YahooBacktestData(START, END, download=lambda *a, **k: raw)
    result = source.bars(AAPL, END, 10, "day")

    assert result is not None
    assert not result.df.isna().to_numpy().any()
    assert list(result.df["close"]) == [150.5, 152.5]


def test_load_and_bars_use_the_injected_download_function() -> None:
    calls: list[tuple] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append((symbol, kwargs))
        return _raw_yahoo_frame()

    source = YahooBacktestData(START, END, download=fake_download)
    source.load([AAPL], START, END, "day")

    assert len(calls) == 1
    assert calls[0][0] == "AAPL"

    result = source.bars(AAPL, END, 2, "day")
    assert result is not None
    assert list(result.df["close"]) == [151.5, 152.5]


def test_bars_fetches_lazily_when_load_was_never_called() -> None:
    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        return _raw_yahoo_frame()

    source = YahooBacktestData(START, END, download=fake_download)
    result = source.bars(AAPL, END, 1, "day")
    assert result is not None


def test_a_download_failure_raises_backtest_data_error() -> None:
    def failing_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("network is down")

    source = YahooBacktestData(START, END, download=failing_download)
    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_an_injected_download_raising_module_not_found_is_wrapped() -> None:
    """The real `_real_download()` path's `import yfinance` must not leak a raw
    ModuleNotFoundError out of the public `bars()` method (see repo note in
    CLAUDE.md about lazy yfinance import) -- reproduced via an injected
    `download` so the test doesn't depend on yfinance actually being absent
    from the environment (it may be installed for an unrelated extra, e.g.
    `batch-universe`)."""

    def missing_yfinance(symbol: str, **kwargs: object) -> pd.DataFrame:
        raise ModuleNotFoundError("No module named 'yfinance'")

    source = YahooBacktestData(START, END, download=missing_yfinance)
    with pytest.raises(BacktestDataError, match="yfinance"):
        source.bars(AAPL, END, 1, "day")


def test_a_malformed_raw_frame_raises_backtest_data_error() -> None:
    def bad_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        # Missing "close"/"volume" columns -- parse_yahoo_frame's column
        # selection must raise a KeyError here.
        return pd.DataFrame({"Open": [150.0]}, index=pd.date_range("2026-01-05", periods=1))

    source = YahooBacktestData(START, END, download=bad_download)
    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_load_honors_its_own_start_and_end_arguments() -> None:
    """load()'s start/end must reach the download call, not the constructor's
    fixed window -- CachedDataSource trusts load()'s own window when it derives
    a cache file's path."""
    calls: list[dict] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return _raw_yahoo_frame()

    other_start = datetime(2026, 2, 1, tzinfo=UTC)
    other_end = datetime(2026, 2, 10, tzinfo=UTC)
    source = YahooBacktestData(START, END, download=fake_download)
    source.load([AAPL], other_start, other_end, "day")

    assert calls[0]["start"] == other_start.date().isoformat()
    assert calls[0]["end"] == (other_end.date() + timedelta(days=1)).isoformat()


def test_minute_timestep_is_not_supported() -> None:
    source = YahooBacktestData(START, END, download=lambda *a, **k: _raw_yahoo_frame())
    with pytest.raises(BacktestDataError, match="day"):
        source.bars(AAPL, END, 1, "minute")


def _frame_for_dates(dates: list[str]) -> pd.DataFrame:
    """A raw yfinance-shaped frame with one daily row per given session date."""
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [150.0 + i for i in range(n)], "High": [151.0 + i for i in range(n)],
            "Low": [149.0 + i for i in range(n)], "Close": [150.5 + i for i in range(n)],
            "Volume": [1000.0] * n,
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]),
    )


def test_sessions_are_one_per_daily_bar_930_to_1600_et() -> None:
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    source = YahooBacktestData(START, END, download=lambda *a, **k: _frame_for_dates(dates))

    sessions = source.sessions(datetime(2026, 1, 5, tzinfo=ET), datetime(2026, 1, 9, tzinfo=ET))

    assert len(sessions) == 5  # Mon-Fri, all traded
    assert sessions[0].open == datetime(2026, 1, 5, 9, 30, tzinfo=ET)
    assert sessions[0].close == datetime(2026, 1, 5, 16, 0, tzinfo=ET)


def test_sessions_skip_a_market_holiday_that_has_no_daily_bar() -> None:
    """Design spec section 4.1: "One 9:30-16:00 ET session per daily bar." `sessions()`
    used to iterate calendar WEEKDAYS instead, so the ~9 market holidays a year each
    became a session -- a full, pointless `before_market_opens` ->
    `on_trading_iteration` -> `after_market_closes` lifecycle plus a spurious equity
    sample on a day with no bar and no price movement (Important whole-branch review
    finding).

    Monday 2026-01-19 is Martin Luther King Jr. Day: a real weekday, a real US market
    holiday, and therefore absent from the fetched bar index.
    """
    traded = ["2026-01-16", "2026-01-20", "2026-01-21"]  # Fri, Tue, Wed -- no Mon the 19th
    calls: list[str] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append(symbol)
        return _frame_for_dates(traded)

    source = YahooBacktestData(START, END, download=fake_download)
    sessions = source.sessions(datetime(2026, 1, 16, tzinfo=ET), datetime(2026, 1, 21, tzinfo=ET))

    session_dates = [s.open.date() for s in sessions]
    assert session_dates == [date(2026, 1, 16), date(2026, 1, 20), date(2026, 1, 21)]
    assert date(2026, 1, 19) not in session_dates  # the holiday, explicitly
    # The calendar comes from SPY's bars, and is fetched once and cached.
    assert calls == ["SPY"]
    source.sessions(datetime(2026, 1, 16, tzinfo=ET), datetime(2026, 1, 21, tzinfo=ET))
    assert calls == ["SPY"]


def test_sessions_reuse_an_already_loaded_calendar_frame_without_refetching() -> None:
    """`sessions()` must not introduce a redundant fetch: the runner pre-loads the
    benchmark (SPY by default) before asking for sessions, so the calendar frame is
    already cached by then."""
    calls: list[str] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append(symbol)
        return _frame_for_dates(["2026-01-05", "2026-01-06"])

    source = YahooBacktestData(START, END, download=fake_download)
    source.load([Asset("SPY")], START, END, "day")
    assert calls == ["SPY"]

    sessions = source.sessions(START, END)

    assert len(sessions) == 2
    assert calls == ["SPY"]  # no second download


def test_constructing_the_source_fetches_nothing() -> None:
    """The lazy contract: neither `__init__` nor the calendar derivation may fetch
    until something is actually asked for."""
    calls: list[str] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append(symbol)
        return _frame_for_dates(["2026-01-05"])

    YahooBacktestData(START, END, download=fake_download)
    assert calls == []


def test_name_is_yahoo() -> None:
    assert YahooBacktestData(START, END).name == "yahoo"
