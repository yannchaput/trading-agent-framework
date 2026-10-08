"""PURE reconstruction of what a member owns today, from a yearly report plus the trades since.

The yearly report gives each asset as a value band as of its period end (Dec 31). Every PTR trade made AFTER that
date is then applied, whatever its filing date: a yearly report is filed months after its period end, so PTRs filed
in between describe trades the report does not contain. A trade dated on or before the period end is already inside
the report and is ignored.

Values are intervals (`value_low`, `value_high`) because disclosures only give ranges: a buy adds its range, a
partial sale subtracts it (each bound floored at zero), a full sale zeroes that owner's slice of the ticker. The
point estimate is the interval's midpoint and the tier is the value band that midpoint falls in. The caller passes
only filings known on the strategy's date; nothing here reads a clock.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_DOWN, Decimal

from trading_agent_framework.congress.annual import AssetHolding, tier_of
from trading_agent_framework.congress.ptr import Transaction

_ZERO = Decimal(0)
_WEIGHT_PLACES = Decimal("0.0001")
_SIDE_ORDER = {"buy": 0, "sell_partial": 1, "sell": 1}  # same-day trades: buys first, so a sale never hits a holding that day's buy would have filled


@dataclass(frozen=True, slots=True)
class Holding:
    ticker: str
    asset_name: str
    value_low: Decimal
    value_high: Decimal
    tier: int
    sources: tuple[str, ...]  # DocIDs of the filings behind the estimate

    @property
    def midpoint(self) -> Decimal:
        return (self.value_low + self.value_high) / 2


@dataclass(slots=True)
class _Slice:
    """One owner's stake in one ticker while the trades are applied."""

    low: Decimal
    high: Decimal


def reconstruct(assets: Sequence[AssetHolding], transactions: Iterable[Transaction], *, period_end: date) -> list[Holding]:
    """The holdings after applying every trade dated after `period_end` to the yearly report's `assets`, sorted by ticker."""
    slices: dict[tuple[str, str], _Slice] = {}
    names: dict[str, str] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    for item in assets:
        stake = slices.setdefault((item.ticker, item.owner), _Slice(_ZERO, _ZERO))
        stake.low += item.value_low
        stake.high += item.value_high
        names.setdefault(item.ticker, item.asset_name)
        sources[item.ticker].add(item.doc_id)
    later = [t for t in transactions if t.transaction_date > period_end]
    for trade in sorted(later, key=lambda t: (t.transaction_date, _SIDE_ORDER.get(t.side, 2), t.doc_id)):
        key = (trade.ticker, trade.owner)
        stake = slices.get(key)
        if trade.side == "buy":
            stake = slices.setdefault(key, _Slice(_ZERO, _ZERO))
            stake.low += trade.amount_low
            stake.high += trade.amount_high
        elif stake is None:
            continue  # a sale of something this owner is not known to hold
        elif trade.side == "sell":
            stake.low = stake.high = _ZERO
        elif trade.side == "sell_partial":
            stake.low = max(_ZERO, stake.low - trade.amount_high)
            stake.high = max(_ZERO, stake.high - trade.amount_low)
        else:
            continue
        names.setdefault(trade.ticker, trade.asset_name)
        sources[trade.ticker].add(trade.doc_id)
    totals: dict[str, _Slice] = {}
    for (ticker, _owner), stake in slices.items():
        total = totals.setdefault(ticker, _Slice(_ZERO, _ZERO))
        total.low += stake.low
        total.high += stake.high
    holdings = []
    for ticker in sorted(totals):
        total = totals[ticker]
        if total.high <= 0:
            continue
        mid = (total.low + total.high) / 2
        holdings.append(Holding(ticker=ticker, asset_name=names[ticker], value_low=total.low, value_high=total.high, tier=tier_of(mid), sources=tuple(sorted(sources[ticker]))))
    return holdings


def baseline_weights(
    holdings: Sequence[Holding],
    *,
    max_total: Decimal,
    max_position: Decimal,
    min_weight: Decimal,
    max_positions: int,
    tier_base: Decimal,
) -> dict[str, Decimal]:
    """A suggested weight per ticker, ordered by value tier: a holding's raw weight is `tier_base ** tier`.

    Dollar midpoints are too coarse and too far apart to size a book on (a $15M name would get 60 times a $250K one), so
    the tier decides: with the default base of 1.5 a holding one tier up gets 1.5 times the weight. Only the
    `max_positions` highest holdings (by tier, then midpoint) are considered; the weights are scaled to sum to `max_total`
    and capped at `max_position` (a cap is not redistributed: the spare stays cash); the lowest holding is dropped while its
    weight is under `min_weight`. Holdings left out get no entry. Rounded down to four places, so the sum never exceeds
    `max_total`. A higher tier never gets a smaller weight than a lower one.
    """
    ranked = sorted(holdings, key=lambda h: (h.tier, h.midpoint, h.ticker), reverse=True)[:max_positions]
    while ranked:
        raw = {h.ticker: tier_base**h.tier for h in ranked}
        total = sum(raw.values(), _ZERO)
        weights = {ticker: min(value / total * max_total, max_position).quantize(_WEIGHT_PLACES, rounding=ROUND_DOWN) for ticker, value in raw.items()}
        if weights[ranked[-1].ticker] >= min_weight:
            return weights
        ranked = ranked[:-1]
    return {}
