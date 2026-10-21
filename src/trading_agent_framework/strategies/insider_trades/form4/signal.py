"""PURE signal window, overrides and a stock's signal class from its labelled insider trades.

Only OPPORTUNISTIC trades move the class: routine and unclassified dollars are summarised but never counted. A stock is BUYING when
any opportunistic purchase moved at least `min_buy_fraction` of the insider's holding, else SELLING when opportunistic insiders sold
more dollars than they bought and one of them sold at least `min_sell_fraction`, else NONE. BUYING wins when both hold.

`InsiderSummary.sell_fraction` adds up an insider's opportunistic sales fractions, each measured against the holding before that
sale: an approximation (10 of 100 then 9 of 90 reads 0.20, the true figure being 0.19). No I/O, no clock: `today` is passed in.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from trading_agent_framework.strategies.insider_trades.form4.classify import Label, Labelled


class SignalClass(StrEnum):
    BUYING = "buying"
    SELLING = "selling"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Override:
    insider_id: str  # the owner CIK
    from_label: Label
    to_label: Label
    reason: str


@dataclass(frozen=True, slots=True)
class InsiderSummary:
    insider_id: str
    name: str
    role: str
    bought: dict[Label, Decimal]  # dollars per label, only the labels that have trades
    sold: dict[Label, Decimal]
    max_buy_fraction: Decimal  # the largest single opportunistic purchase's holding fraction, 0 if none
    sell_fraction: Decimal  # the sum of the opportunistic sales' holding fractions
    labelled: tuple[Labelled, ...]

    @property
    def dollars(self) -> Decimal:
        return sum(self.bought.values(), Decimal(0)) + sum(self.sold.values(), Decimal(0))


@dataclass(frozen=True, slots=True)
class StockFacts:
    ticker: str
    today: date
    window_start: date
    insiders: tuple[InsiderSummary, ...]  # by total dollars, largest first

    @property
    def accessions(self) -> frozenset[str]:
        return frozenset(lt.trade.accession for i in self.insiders for lt in i.labelled)


def window_trades(labelled: Sequence[Labelled], *, today: date, window_days: int) -> list[Labelled]:
    """The trades filed in `[today - window_days, today)`: today's own filings are not out yet in a backtest, so they never count."""
    start = today - timedelta(window_days)
    return [lt for lt in labelled if start <= lt.trade.filed < today]


def apply_overrides(labelled: Sequence[Labelled], overrides: Sequence[Override]) -> list[Labelled]:
    """Apply the overrides in order. One relabels its insider's trades of `from_label`, never a plan trade (the plan fixed it in advance)."""
    result = list(labelled)
    for o in overrides:
        result = [
            replace(lt, label=o.to_label, rule=f"override: {o.reason}")
            if lt.trade.owner_cik == o.insider_id and lt.label is o.from_label and not lt.trade.plan
            else lt
            for lt in result
        ]
    return result


def summarise(ticker: str, window: Sequence[Labelled], *, today: date, window_start: date) -> StockFacts:
    by_insider: dict[str, list[Labelled]] = defaultdict(list)
    for lt in window:
        by_insider[lt.trade.owner_cik].append(lt)
    insiders = sorted((_summarise_insider(cik, trades) for cik, trades in by_insider.items()), key=lambda i: (-i.dollars, i.name))
    return StockFacts(ticker, today, window_start, tuple(insiders))


def _summarise_insider(insider_id: str, trades: list[Labelled]) -> InsiderSummary:
    bought: dict[Label, Decimal] = defaultdict(Decimal)
    sold: dict[Label, Decimal] = defaultdict(Decimal)
    for lt in trades:
        (bought if lt.trade.code == "P" else sold)[lt.label] += lt.trade.dollars
    opportunistic = [lt.trade for lt in trades if lt.label is Label.OPPORTUNISTIC]
    first = trades[0].trade
    return InsiderSummary(
        insider_id=insider_id,
        name=first.owner_name,
        role=first.role,
        bought=dict(bought),
        sold=dict(sold),
        max_buy_fraction=max((t.holding_fraction for t in opportunistic if t.code == "P"), default=Decimal(0)),
        sell_fraction=sum((t.holding_fraction for t in opportunistic if t.code == "S"), Decimal(0)),
        labelled=tuple(trades),
    )


def signal_class(facts: StockFacts, *, min_buy_fraction: Decimal, min_sell_fraction: Decimal) -> SignalClass:
    if any(i.max_buy_fraction >= min_buy_fraction for i in facts.insiders):
        return SignalClass.BUYING
    bought = sum((i.bought.get(Label.OPPORTUNISTIC, Decimal(0)) for i in facts.insiders), Decimal(0))
    sold = sum((i.sold.get(Label.OPPORTUNISTIC, Decimal(0)) for i in facts.insiders), Decimal(0))
    if sold > bought and any(i.sell_fraction >= min_sell_fraction for i in facts.insiders):
        return SignalClass.SELLING
    return SignalClass.NONE
