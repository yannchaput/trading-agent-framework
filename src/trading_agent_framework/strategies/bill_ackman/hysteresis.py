"""Fail counters, forced exits and the trader's allowed set (pure).

A holding that gets the verdict `fail` has its counter raised; at `forced_exit_fails` consecutive fails it is a
forced exit (code sells it whatever the trader submits). A `survive` resets the counter. A new candidate that
fails is simply not allowed and has no counter. A forced exit then stays out of the candidates and the allowed set
for a few completed reviews (`advance_cooldowns`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SURVIVE = "survive"
FAIL = "fail"


@dataclass(frozen=True, slots=True)
class HysteresisOutcome:
    fail_counts: dict[str, int]  # after this review, for holdings that have failed and not recovered (forced exits included)
    forced_exits: list[str]  # holdings to sell in full, in `holdings` order
    pending: list[str]  # holdings that failed but are below the threshold, in `holdings` order
    allowed: list[str]  # what the trader may choose from: ranked survivors, other surviving holdings, pending fails


def apply_verdicts(
    *,
    holdings: Sequence[str],
    ranking: Sequence[str],
    verdicts: Mapping[str, str],
    fail_counts: Mapping[str, int],
    forced_exit_fails: int,
) -> HysteresisOutcome:
    """Update the counters from today's `verdicts` and derive the forced exits and the allowed set.

    `verdicts` must hold a verdict for every holding and every ranked symbol (a missing one is a pipeline bug and
    raises `ValueError`, as does a verdict other than `survive`/`fail`). `fail_counts` entries for symbols that
    are no longer held are dropped.
    """
    for symbol in [*holdings, *ranking]:
        verdict = verdicts.get(symbol)
        if verdict is None:
            raise ValueError(f"no verdict for {symbol}")
        if verdict not in (SURVIVE, FAIL):
            raise ValueError(f"verdict for {symbol} must be 'survive' or 'fail', got {verdict!r}")

    counts: dict[str, int] = {}
    forced_exits: list[str] = []
    pending: list[str] = []
    for symbol in holdings:
        if verdicts[symbol] == SURVIVE:
            continue
        count = fail_counts.get(symbol, 0) + 1
        counts[symbol] = count
        (forced_exits if count >= forced_exit_fails else pending).append(symbol)

    allowed: list[str] = []
    for symbol in ranking:
        if verdicts[symbol] == SURVIVE and symbol not in allowed:
            allowed.append(symbol)
    for symbol in holdings:
        if verdicts[symbol] == SURVIVE and symbol not in allowed:
            allowed.append(symbol)
    allowed.extend(symbol for symbol in pending if symbol not in allowed)
    return HysteresisOutcome(fail_counts=counts, forced_exits=forced_exits, pending=pending, allowed=allowed)


def advance_cooldowns(cooldowns: Mapping[str, int], *, forced_exits: Sequence[str], reviews: int) -> dict[str, int]:
    """The cooldowns after a completed review: each one counts down (dropped at 0), then each forced exit starts at `reviews`.

    The pipeline keeps a symbol with a cooldown out of the researcher's candidates and the trader's allowed set, so
    a forced exit at review R stays out from R+1 through R+`reviews`. `reviews` = 0 disables cooldowns.
    """
    after = {symbol: left - 1 for symbol, left in cooldowns.items() if left > 1}
    if reviews > 0:
        after.update(dict.fromkeys(forced_exits, reviews))
    return after
