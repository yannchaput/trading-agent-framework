"""Pure quality screen: which companies were simple, predictable, cash-generative, lightly indebted and
reasonably priced on a date, from annual SEC figures known on that date.

No I/O, no clock, no state (same rules as `annual_figures.py`). `assess` applies the numeric gates to one company's
reduced annual figures (`annual_figures.annual_figures`); `rank` scores the companies that were also priced.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise
from typing import Any

from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import MAX_FISCAL_YEAR_DAYS, MIN_FISCAL_YEAR_DAYS

_DAYS_PER_MONTH = 30.4375
_REQUIRED_FLOWS = ("revenue", "operating_income", "operating_cash_flow", "capex")


@dataclass(frozen=True, slots=True)
class ScreenParams:
    years: int = 5
    min_growth_years: int = 3
    max_filing_age_months: int = 18
    max_net_debt_to_operating_income: float = 4.0
    excluded_sic_ranges: tuple[tuple[int, int], ...] = ((4900, 4999), (6000, 6799))  # utilities; finance, insurance, real estate
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)  # fcf_yield, fcf_margin, operating-margin stability
    top_n: int = 15
    max_age_days: int = 30  # annual SEC figures, by as_of
    split_max_age_days: int = 1  # split history, by the wall clock (it must match today's adjusted prices)
    max_fetch_failure_ratio: float = 0.2
    hollow_min_sample: int = 5  # symbols that must reach the SIC or split gate before its failure ratio can abort a screen

    def __post_init__(self) -> None:
        problems = {
            "years": self.years < 2,
            "min_growth_years": not 0 <= self.min_growth_years <= self.years - 1,
            "max_filing_age_months": self.max_filing_age_months <= 0,
            "max_net_debt_to_operating_income": not math.isfinite(self.max_net_debt_to_operating_income),
            "top_n": self.top_n < 0,
            "weights": len(self.weights) != 3 or any(not math.isfinite(weight) or weight < 0 for weight in self.weights),
            "max_age_days": self.max_age_days < 0,
            "split_max_age_days": self.split_max_age_days < 0,
            "max_fetch_failure_ratio": not 0 <= self.max_fetch_failure_ratio <= 1,  # also false for NaN
            "hollow_min_sample": self.hollow_min_sample < 1,
        }
        invalid = [name for name, bad in problems.items() if bad]
        if invalid:
            raise ValueError(f"invalid ScreenParams: {', '.join(f'{name}={getattr(self, name)!r}' for name in invalid)}")


@dataclass(frozen=True, slots=True)
class Survivor:
    """A company that passed the numeric gates; the sector, price and split gates come after."""

    symbol: str
    fiscal_year_end: date
    filed: date  # filing date of the latest fiscal year's revenue figure
    free_cash_flow: int  # latest fiscal year
    fcf_margin: float  # mean over the window
    operating_margin: float  # latest fiscal year
    operating_margin_stdev: float
    revenue_growth: float  # compound annual rate over the window
    net_debt_to_operating_income: float
    debt_reported: bool
    shares: int | None
    counted_on: date | None


def _known(rows: Sequence[Mapping[str, Any]], cutoff: date) -> list[Mapping[str, Any]]:
    """Rows filed strictly before `cutoff`: SEC gives no filing time, so a row filed today is known tomorrow."""
    return [row for row in rows if date.fromisoformat(row["filed"]) < cutoff]


def _latest_versions(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, date], Mapping[str, Any]]:
    """(field, period end) -> the most recently filed version."""
    latest: dict[tuple[str, date], Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda row: row["filed"]):
        latest[(row["field"], date.fromisoformat(row["end"]))] = row
    return latest


def _consecutive(year_ends: Sequence[date]) -> bool:
    return all(MIN_FISCAL_YEAR_DAYS <= (later - earlier).days <= MAX_FISCAL_YEAR_DAYS for earlier, later in pairwise(year_ends))


def _counted_on(share_row: Mapping[str, Any]) -> date:
    """The day a share count is true on, for restating it across later splits.

    A cover-page count is as of its period `end`. A weighted-average count is restated by ASC 260 for
    splits that happen after the period end but before the report is issued, so it is true as of its
    `filed` date: applying a split dated between `end` and `filed` would count it twice.
    """
    return date.fromisoformat(share_row["filed"] if share_row["kind"] == "weighted" else share_row["end"])


def assess(symbol: str, figures: Mapping[str, Any] | None, *, as_of: datetime, params: ScreenParams) -> Survivor | str:
    """Apply the numeric gates to one company; the rejection reason is the first gate that fails."""
    if figures is None or figures.get("status") == "absent":
        return "no_data"
    cutoff = as_of.date()
    flows = _latest_versions(_known(figures.get("flows", []), cutoff))
    if not flows:
        return "no_data"

    year_ends = sorted(end for field, end in flows if field == "revenue")
    if not year_ends:
        return "insufficient_history"
    window = year_ends[-params.years :]
    latest_end = window[-1]
    if (cutoff - latest_end).days > params.max_filing_age_months * _DAYS_PER_MONTH:
        return "stale_filing"
    if len(window) < params.years or not _consecutive(window):
        return "insufficient_history"

    revenues: list[int] = []
    operating_incomes: list[int] = []
    free_cash_flows: list[int] = []
    for end in window:
        rows = [flows.get((field, end)) for field in _REQUIRED_FLOWS]
        if any(row is None for row in rows):
            return "insufficient_history"
        revenue, operating_income, operating_cash_flow, capex = (row["value"] for row in rows)  # ty: ignore[not-subscriptable]
        if revenue <= 0:
            return "insufficient_history"
        revenues.append(revenue)
        operating_incomes.append(operating_income)
        free_cash_flows.append(operating_cash_flow - capex)

    # Implausible figures (a partial revenue tag, a tagging error): not a judgement on the company.
    if any(income > revenue or fcf > revenue for income, fcf, revenue in zip(operating_incomes, free_cash_flows, revenues, strict=True)):
        return "insufficient_history"
    if any(value <= 0 for value in operating_incomes):
        return "operating_loss"
    if any(value <= 0 for value in free_cash_flows):
        return "negative_fcf"
    increases = sum(later > earlier for earlier, later in pairwise(revenues))
    if increases < params.min_growth_years or revenues[-1] < revenues[0]:
        return "shrinking_revenue"

    balances = _latest_versions(_known(figures.get("balances", []), cutoff))
    debt_row, cash_row = balances.get(("debt", latest_end)), balances.get(("cash", latest_end))
    if debt_row is None and any(("debt", end) in balances for end in window[:-1]):
        return "debt_unknown"  # a changed or missing tag for the latest year must not read as zero debt
    net_debt = (debt_row["value"] if debt_row else 0) - (cash_row["value"] if cash_row else 0)
    if net_debt > params.max_net_debt_to_operating_income * operating_incomes[-1]:
        return "too_much_debt"

    margins = [income / revenue for income, revenue in zip(operating_incomes, revenues, strict=True)]
    share_rows = _known(figures.get("shares", []), cutoff)
    # The latest period end wins, then the latest filing, then the cover-page count over a weighted one.
    share_row = max(share_rows, key=lambda row: (row["end"], row["filed"], row["kind"] == "cover")) if share_rows else None
    return Survivor(
        symbol=symbol,
        fiscal_year_end=latest_end,
        filed=date.fromisoformat(flows[("revenue", latest_end)]["filed"]),
        free_cash_flow=free_cash_flows[-1],
        fcf_margin=statistics.fmean(fcf / revenue for fcf, revenue in zip(free_cash_flows, revenues, strict=True)),
        operating_margin=margins[-1],
        operating_margin_stdev=statistics.pstdev(margins),
        revenue_growth=(revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1,
        net_debt_to_operating_income=net_debt / operating_incomes[-1],
        debt_reported=debt_row is not None,
        shares=share_row["value"] if share_row else None,
        counted_on=_counted_on(share_row) if share_row else None,
    )


@dataclass(frozen=True, slots=True)
class Priced:
    """A survivor of every gate, with its sector code and its split-restated market cap (above zero)."""

    survivor: Survivor
    sic: int | None
    market_cap: Decimal


@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    rank: int
    score: float
    sic: int | None
    market_cap: Decimal
    fcf_yield: float
    fcf_margin: float
    operating_margin: float
    operating_margin_stdev: float
    revenue_growth: float
    net_debt_to_operating_income: float
    debt_reported: bool
    fiscal_year_end: date
    filed: date


@dataclass(frozen=True, slots=True)
class ScreenResult:
    candidates: list[Candidate]
    rejections: dict[str, str]  # symbol -> reason of the first failed gate


def sector_excluded(sic: int | None, params: ScreenParams) -> bool:
    """Whether the SIC code falls in an excluded range; a company with no code passes."""
    return sic is not None and any(low <= sic <= high for low, high in params.excluded_sic_ranges)


def _percentiles(values: Sequence[float]) -> list[float]:
    """Each value's percentile rank in `values`: (rank - 1) / (n - 1), ties sharing their average rank."""
    count = len(values)
    if count == 1:
        return [1.0]
    order = sorted(range(count), key=values.__getitem__)
    ranks = [0.0] * count
    start = 0
    while start < count:
        stop = start
        while stop + 1 < count and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            ranks[order[position]] = average_rank
        start = stop + 1
    return [(rank - 1) / (count - 1) for rank in ranks]


def rank(priced: Sequence[Priced], params: ScreenParams) -> list[Candidate]:
    """Score the priced survivors against each other and return the best `params.top_n`, best first."""
    if not priced:
        return []
    yields = [float(Decimal(item.survivor.free_cash_flow) / item.market_cap) for item in priced]
    yield_pct = _percentiles(yields)
    margin_pct = _percentiles([item.survivor.fcf_margin for item in priced])
    stdev_pct = _percentiles([item.survivor.operating_margin_stdev for item in priced])
    yield_weight, margin_weight, stability_weight = params.weights
    scored = [
        (yield_weight * yield_pct[index] + margin_weight * margin_pct[index] + stability_weight * (1 - stdev_pct[index]), yields[index], item)
        for index, item in enumerate(priced)
    ]
    scored.sort(key=lambda entry: (-entry[0], -entry[2].market_cap, entry[2].survivor.symbol))
    return [
        Candidate(
            symbol=item.survivor.symbol,
            rank=position,
            score=score,
            sic=item.sic,
            market_cap=item.market_cap,
            fcf_yield=fcf_yield,
            fcf_margin=item.survivor.fcf_margin,
            operating_margin=item.survivor.operating_margin,
            operating_margin_stdev=item.survivor.operating_margin_stdev,
            revenue_growth=item.survivor.revenue_growth,
            net_debt_to_operating_income=item.survivor.net_debt_to_operating_income,
            debt_reported=item.survivor.debt_reported,
            fiscal_year_end=item.survivor.fiscal_year_end,
            filed=item.survivor.filed,
        )
        for position, (score, fcf_yield, item) in enumerate(scored[: params.top_n], start=1)
    ]
