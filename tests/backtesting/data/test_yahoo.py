from __future__ import annotations

from datetime import UTC, datetime
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


def test_minute_timestep_is_not_supported() -> None:
    source = YahooBacktestData(START, END, download=lambda *a, **k: _raw_yahoo_frame())
    with pytest.raises(BacktestDataError, match="day"):
        source.bars(AAPL, END, 1, "minute")


def test_sessions_are_one_per_weekday_930_to_1600_et() -> None:
    source = YahooBacktestData(START, END)
    sessions = source.sessions(datetime(2026, 1, 5, tzinfo=ET), datetime(2026, 1, 9, tzinfo=ET))
    assert len(sessions) == 5  # Mon-Fri
    assert sessions[0].open == datetime(2026, 1, 5, 9, 30, tzinfo=ET)
    assert sessions[0].close == datetime(2026, 1, 5, 16, 0, tzinfo=ET)


def test_name_is_yahoo() -> None:
    assert YahooBacktestData(START, END).name == "yahoo"
