from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal

from tests.fakes import (
    bar_payload,
    et,
    make_alpaca_barset,
    make_alpaca_quote,
    make_alpaca_trade,
    make_session,
)

from trading_agent_framework.brokers.alpaca.market_data import (
    parse_bars,
    parse_latest_trades,
    parse_quote,
)
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars

AAPL = Asset("AAPL")
MSFT = Asset("MSFT")

# Thursday 2026-09-10, daylight saving time: 13:30Z is 09:30 in New York.
_MINUTES = [
    bar_payload("2026-09-10T13:29:00Z", 99.0),  # 09:29, pre-market
    bar_payload("2026-09-10T13:30:00Z", 100.0),  # 09:30
    bar_payload("2026-09-10T13:31:00Z", 101.0),  # 09:31
    bar_payload("2026-09-10T20:00:00Z", 102.0),  # 16:00, after hours
]
_SESSION = [make_session(date(2026, 9, 10))]


def test_day_bars_become_a_float_frame_indexed_in_market_time() -> None:
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-09-09T04:00:00Z", 10.0), bar_payload("2026-09-10T04:00:00Z", 11.0)]}
    )

    bars = parse_bars(barset, [AAPL], "day", 5)[AAPL]

    assert isinstance(bars, Bars)
    assert (bars.asset, bars.timestep) == (AAPL, "day")
    assert list(bars.df.columns) == ["open", "high", "low", "close", "volume"]
    assert all(str(dtype) == "float64" for dtype in bars.df.dtypes)
    assert list(bars.df["close"]) == [10.0, 11.0]
    assert bars.df.index[0] == et(2026, 9, 9)
    assert str(bars.df.index.tz) == "America/New_York"  # ty: ignore[unresolved-attribute]


def test_only_the_last_length_bars_are_kept_oldest_first() -> None:
    barset = make_alpaca_barset({"AAPL": list(reversed(_MINUTES))})

    bars = parse_bars(barset, [AAPL], "minute", 2)[AAPL]

    assert list(bars.df["close"]) == [101.0, 102.0]


def test_duplicate_timestamps_keep_the_first_bar() -> None:
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-09-10T13:30:00Z", 100.0), bar_payload("2026-09-10T13:30:00Z", 555.0)]}
    )

    assert list(parse_bars(barset, [AAPL], "minute", 5)[AAPL].df["close"]) == [100.0]


def test_sessions_filter_extended_hours_before_truncating() -> None:
    barset = make_alpaca_barset({"AAPL": _MINUTES})

    bars = parse_bars(barset, [AAPL], "minute", 2, sessions=_SESSION)[AAPL]

    # Truncating first would keep [101, 102] and then filter down to [101].
    assert list(bars.df["close"]) == [100.0, 101.0]


def test_without_sessions_extended_hours_bars_are_kept() -> None:
    bars = parse_bars(make_alpaca_barset({"AAPL": _MINUTES}), [AAPL], "minute", 10)[AAPL]

    assert len(bars.df) == 4


def test_early_close_sessions_drop_bars_after_the_early_close() -> None:
    # Friday 2026-11-27 closes at 13:00 in New York (standard time: 18:00Z).
    early = [make_session(date(2026, 11, 27), close_at=time(13, 0))]
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-11-27T17:59:00Z", 1.0), bar_payload("2026-11-27T18:05:00Z", 2.0)]}
    )

    bars = parse_bars(barset, [AAPL], "minute", 5, sessions=early)[AAPL]

    assert list(bars.df["close"]) == [1.0]


def test_assets_without_bars_are_left_out() -> None:
    barset = make_alpaca_barset({"AAPL": _MINUTES, "MSFT": []})

    assert set(parse_bars(barset, [AAPL, MSFT, Asset("TSLA")], "minute", 5)) == {AAPL}


def test_assets_filtered_down_to_nothing_are_left_out() -> None:
    barset = make_alpaca_barset({"AAPL": [_MINUTES[0]]})  # pre-market only

    assert parse_bars(barset, [AAPL], "minute", 5, sessions=_SESSION) == {}


def test_latest_trades_become_decimal_prices_and_missing_symbols_none() -> None:
    response = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    assert parse_latest_trades(response, [AAPL, MSFT]) == {AAPL: Decimal("100.15"), MSFT: None}


def test_quote_maps_bid_ask_sizes_and_timestamp() -> None:
    quote = parse_quote({"AAPL": make_alpaca_quote(bid=100.1, ask=100.2)}, AAPL)

    assert quote is not None
    assert quote.asset == AAPL
    assert (quote.bid, quote.ask) == (Decimal("100.1"), Decimal("100.2"))
    assert (quote.bid_size, quote.ask_size) == (Decimal(3), Decimal(4))
    assert quote.timestamp == datetime(2026, 9, 10, 13, 30, tzinfo=UTC)
    assert str(quote.timestamp.tzinfo) == "America/New_York"


def test_an_empty_book_side_is_none_not_zero() -> None:
    quote = parse_quote({"AAPL": make_alpaca_quote(bid=0.0, ask=100.2)}, AAPL)

    assert quote is not None
    assert quote.bid is None
    assert quote.ask == Decimal("100.2")
    assert quote.mid is None


def test_no_quote_for_the_symbol_is_none() -> None:
    assert parse_quote({}, AAPL) is None
