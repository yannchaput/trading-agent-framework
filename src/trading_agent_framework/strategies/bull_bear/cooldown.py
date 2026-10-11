"""The re-entry cooldown: a stock the judge sold stays out of the candidates for a number of completed reviews (pure).

The audit of 73 sells (spec 2026-10-11-bull-bear-debate-realignment-design.md) found half of them bought back, most
within two weeks, and those re-entries then lagged the other candidates by 4.2 points over 4 weeks.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping


def advance_cooldowns(current: Mapping[str, int], *, started: Collection[str], length: int) -> dict[str, int]:
    """The cooldowns after a completed review: each running one counts down and is dropped after its last review,
    and each stock in `started` begins one of `length` reviews (none when `length` is 0)."""
    running = {symbol: left - 1 for symbol, left in current.items() if left > 1}
    return {**running, **({symbol: length for symbol in started} if length > 0 else {})}
