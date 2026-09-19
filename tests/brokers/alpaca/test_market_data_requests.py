from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from tests.fakes import ET, et, weekday_sessions

from trading_agent_framework.brokers.alpaca.market_data import (
    MAX_NEWS_LIMIT,
    MAX_SYMBOLS_PER_REQUEST,
    bars_start,
    build_bars_request,
    build_latest_quote_request,
    build_latest_trade_request,
    build_news_request,
    calendar_lookback_start,
    chunk_assets,
    parse_timestep,
    sessions_needed,
)
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BrokerError

_END = et(2026, 9, 10, 9, 31)  # Thursday, one minute after the open
_SESSIONS = weekday_sessions(date(2026, 9, 8), 3)  # Tue 8, Wed 9, Thu 10


@pytest.mark.parametrize(("timestep", "value"), [("minute", "1Min"), ("day", "1Day")])
def test_parse_timestep(timestep: str, value: str) -> None:
    assert parse_timestep(timestep).value == value


@pytest.mark.parametrize("timestep", ["hour", "5minute", "Day", ""])
def test_parse_timestep_rejects_everything_else(timestep: str) -> None:
    with pytest.raises(ValueError, match="Unsupported timestep"):
        parse_timestep(timestep)


@pytest.mark.parametrize(
    ("length", "timestep", "expected"),
    [(20, "day", 21), (1, "day", 2), (30, "minute", 2), (390, "minute", 2), (391, "minute", 3)],
)
def test_sessions_needed_adds_one_for_a_partial_current_session(
    length: int, timestep: str, expected: int
) -> None:
    assert sessions_needed(length, timestep) == expected


def test_sessions_needed_rejects_a_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length"):
        sessions_needed(0, "day")


def test_sessions_needed_rejects_an_unknown_timestep() -> None:
    with pytest.raises(ValueError, match="Unsupported timestep"):
        sessions_needed(5, "hour")


@pytest.mark.parametrize(
    ("length", "timestep", "expected"),
    [(30, "minute", date(2026, 8, 28)), (20, "day", date(2026, 7, 30))],
)
def test_calendar_lookback_leaves_room_for_weekends_and_holidays(
    length: int, timestep: str, expected: date
) -> None:
    # ceil(sessions * 1.5) + 10 calendar days: 2 sessions -> 13 days, 21 sessions -> 42 days
    assert calendar_lookback_start(_END, length, timestep) == expected


def test_bars_start_is_midnight_of_the_earliest_session_needed() -> None:
    assert bars_start(_END, 30, "minute", _SESSIONS) == datetime(2026, 9, 9, tzinfo=ET)


def test_bars_start_falls_back_to_the_earliest_session_available() -> None:
    assert bars_start(_END, 10, "day", _SESSIONS) == datetime(2026, 9, 8, tzinfo=ET)


def test_bars_start_ignores_sessions_after_end() -> None:
    sessions = weekday_sessions(date(2026, 9, 8), 5)  # through Monday the 14th
    assert bars_start(_END, 1, "day", sessions) == datetime(2026, 9, 9, tzinfo=ET)


def test_bars_start_without_any_past_session_raises() -> None:
    with pytest.raises(BrokerError, match="no session"):
        bars_start(_END, 1, "day", weekday_sessions(date(2026, 9, 14), 2))


def test_chunk_assets_dedupes_and_batches() -> None:
    assets = [Asset(f"S{i}") for i in range(MAX_SYMBOLS_PER_REQUEST * 2 + 1)] + [Asset("S0")]

    chunks = list(chunk_assets(assets))

    assert [len(chunk) for chunk in chunks] == [150, 150, 1]
    assert chunks[0][0] == Asset("S0")


def test_bars_request_asks_for_adjusted_iex_bars_of_every_symbol() -> None:
    start = datetime(2026, 9, 9, tzinfo=ET)

    request = build_bars_request([Asset("AAPL"), Asset("MSFT")], "minute", start, _END)

    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.timeframe.value == "1Min"
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.ALL
    # alpaca-py normalises start/end to naive UTC
    assert request.start == start.astimezone(UTC).replace(tzinfo=None)
    assert request.end == _END.astimezone(UTC).replace(tzinfo=None)


@pytest.mark.parametrize("build", [build_latest_trade_request, build_latest_quote_request])
def test_latest_requests_use_the_iex_feed(build) -> None:
    request = build([Asset("AAPL"), Asset("MSFT")])

    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.feed == DataFeed.IEX


def test_build_news_request_joins_symbols_and_clamps_limit() -> None:
    start = et(2026, 9, 8)
    end = et(2026, 9, 10)

    request = build_news_request(["SPY", "QQQ"], start=start, end=end, limit=999, include_content=True)

    assert request.symbols == "SPY,QQQ"
    assert request.limit == MAX_NEWS_LIMIT
    assert request.include_content is True
    assert request.start == start.astimezone(UTC).replace(tzinfo=None)
    assert request.end == end.astimezone(UTC).replace(tzinfo=None)


def test_build_news_request_with_no_symbols_omits_them() -> None:
    request = build_news_request([], start=None, end=_END, limit=5, include_content=False)

    assert request.symbols is None
    assert request.start is None
