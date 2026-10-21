"""PURE routine / opportunistic labelling of an insider's open-market trades (Cohen, Malloy and Pomorski, "Decoding Inside Information").

An insider is *routine* when they traded in the same calendar month in each of the `years` calendar years before the trade's own
year: the trade is part of a habit and carries little information. They are *opportunistic* when they traded in each of those years
but never in one month common to all of them. They are *unclassified* when at least one of those years has no trade, so there is no
pattern to judge. A trade filed under a 10b5-1 plan is routine whatever the history: the plan fixed it in advance.

Only `history` counts (every known attributed trade of the issuer, keyed on `owner_cik`, buys and sales alike); the trades being
labelled are not part of it. No I/O, no clock.
"""

from __future__ import annotations

import calendar
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Trade


class Label(StrEnum):
    ROUTINE = "routine"
    OPPORTUNISTIC = "opportunistic"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True, slots=True)
class Labelled:
    trade: Form4Trade
    label: Label
    rule: str
    months_by_year: dict[int, tuple[int, ...]]  # each look-back year -> the sorted distinct months the insider traded in


def label_trades(trades: Sequence[Form4Trade], history: Sequence[Form4Trade], *, years: int) -> list[Labelled]:
    """One `Labelled` per trade, in order. The look-back is the `years` calendar years before the trade's own year."""
    months: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for h in history:
        months[h.owner_cik][h.trade_date.year].add(h.trade_date.month)
    return [_label(t, months.get(t.owner_cik, {}), years) for t in trades]


def _label(trade: Form4Trade, owner_months: Mapping[int, set[int]], years: int) -> Labelled:
    first = trade.trade_date.year - years
    months_by_year = {y: tuple(sorted(owner_months.get(y, ()))) for y in range(first, trade.trade_date.year)}
    if trade.plan:
        return Labelled(trade, Label.ROUTINE, "plan: the 10b5-1 box is checked", months_by_year)
    listed = ", ".join(str(y) for y in months_by_year)
    missing = [str(y) for y, m in months_by_year.items() if not m]
    if missing:
        return Labelled(trade, Label.UNCLASSIFIED, f"unclassified: no trade in {', '.join(missing)}", months_by_year)
    shared = set.intersection(*(set(m) for m in months_by_year.values()))
    if shared:
        month = calendar.month_name[min(shared)]
        return Labelled(trade, Label.ROUTINE, f"routine: traded in {month} in each of {listed}", months_by_year)
    rule = f"opportunistic: traded in each of {listed}, never in the same month every year"
    return Labelled(trade, Label.OPPORTUNISTIC, rule, months_by_year)
