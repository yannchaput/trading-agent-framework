from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote

_TS = datetime(2026, 9, 10, 13, 30, tzinfo=UTC)


def _quote(bid: Decimal | None, ask: Decimal | None) -> Quote:
    return Quote(
        asset=Asset("AAPL"),
        bid=bid,
        ask=ask,
        bid_size=Decimal(3),
        ask_size=Decimal(4),
        timestamp=_TS,
    )


def test_mid_is_the_average_of_bid_and_ask() -> None:
    assert _quote(Decimal("100.10"), Decimal("100.20")).mid == Decimal("100.15")


@pytest.mark.parametrize(
    ("bid", "ask"),
    [(None, Decimal("100.20")), (Decimal("100.10"), None), (None, None)],
)
def test_mid_is_none_without_both_sides(bid: Decimal | None, ask: Decimal | None) -> None:
    assert _quote(bid, ask).mid is None


def test_quote_is_frozen() -> None:
    quote = _quote(Decimal(1), Decimal(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        quote.bid = Decimal(5)  # ty: ignore[invalid-assignment]
