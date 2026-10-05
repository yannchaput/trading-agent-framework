"""Benzinga's earnings headline, parsed into numbers (spec §3.3). Pure: no I/O, no clock.

BEAT/MISS is computed from the numbers, never read from the headline's verb: the agent gets facts it does not
have to derive (a local model has misread the sign of "actual vs estimate" prints before).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

_NUMBER = r"\$?\(?\$?-?[\d,]*\.?\d+\)?"
_VERB = r"(?:Beats?|Miss(?:es)?|In-?Line\s+With|Inline\s+With|Meets?)"
_EPS = re.compile(
    rf"\b(?:Q[1-4]|FY)(?:\s*'?\d{{2,4}})?\s+(?:Adj(?:usted|\.)?\s+)?EPS\s+(?P<actual>{_NUMBER})\s+{_VERB}\s+(?P<estimate>{_NUMBER})\s+Estimate",
    re.IGNORECASE,
)
_SALES = re.compile(
    rf"\b(?:Sales|Revenue)\s+(?P<actual>{_NUMBER})(?P<actual_unit>[KMB])?\s+{_VERB}\s+(?P<estimate>{_NUMBER})(?P<estimate_unit>[KMB])?\s+Estimate",
    re.IGNORECASE,
)
_UNITS = {"": Decimal(1), "K": Decimal(1_000), "M": Decimal(1_000_000), "B": Decimal(1_000_000_000)}
_CORRECTION = "CORRECTION"


def _result(actual: Decimal, estimate: Decimal) -> str:
    if actual > estimate:
        return "BEAT"
    return "MISS" if actual < estimate else "IN-LINE"


def _surprise_pct(actual: Decimal, estimate: Decimal) -> float | None:
    if estimate == 0:
        return None
    return float((actual - estimate) / abs(estimate))


@dataclass(frozen=True, slots=True)
class Surprise:
    eps_actual: Decimal
    eps_estimate: Decimal
    sales_actual: Decimal | None = None
    sales_estimate: Decimal | None = None

    @property
    def eps_beat(self) -> bool:
        return self.eps_actual > self.eps_estimate

    @property
    def eps_result(self) -> str:
        return _result(self.eps_actual, self.eps_estimate)

    @property
    def eps_surprise_pct(self) -> float | None:
        return _surprise_pct(self.eps_actual, self.eps_estimate)

    @property
    def sales_result(self) -> str | None:
        if self.sales_actual is None or self.sales_estimate is None:
            return None
        return _result(self.sales_actual, self.sales_estimate)

    @property
    def sales_surprise_pct(self) -> float | None:
        if self.sales_actual is None or self.sales_estimate is None:
            return None
        return _surprise_pct(self.sales_actual, self.sales_estimate)


@dataclass(frozen=True, slots=True)
class PickedSurprise:
    surprise: Surprise
    headline: str
    created_at: datetime


def _number(text: str) -> Decimal | None:
    negative = "(" in text or "-" in text
    digits = re.sub(r"[^\d.]", "", text)
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


def parse_surprise(headline: str) -> Surprise | None:
    """The EPS (and sales, when present) actual and estimate in a Benzinga headline; None without an EPS estimate."""
    eps = _EPS.search(headline or "")
    if eps is None:
        return None
    actual, estimate = _number(eps["actual"]), _number(eps["estimate"])
    if actual is None or estimate is None:
        return None
    sales_actual = sales_estimate = None
    sales = _SALES.search(headline, eps.end())
    if sales is not None:
        raw_actual, raw_estimate = _number(sales["actual"]), _number(sales["estimate"])
        if raw_actual is not None and raw_estimate is not None:
            sales_actual = raw_actual * _UNITS[(sales["actual_unit"] or "").upper()]
            sales_estimate = raw_estimate * _UNITS[(sales["estimate_unit"] or "").upper()]
    return Surprise(actual, estimate, sales_actual, sales_estimate)


def article_time(article: Mapping[str, Any]) -> datetime | None:
    """An article's aware `created_at`; None when missing or unparsable."""
    try:
        created = datetime.fromisoformat(str(article.get("created_at") or ""))
    except ValueError:
        return None
    return created if created.tzinfo is not None else None


def articles_for(articles: Sequence[Mapping[str, Any]], symbol: str, *, start: datetime, end: datetime) -> list[Mapping[str, Any]]:
    """The articles tagged with `symbol` and created in `[start, end]`, oldest first."""
    wanted = symbol.upper()
    kept: list[tuple[datetime, Mapping[str, Any]]] = []
    for article in articles:
        created = article_time(article)
        symbols = [str(s).upper() for s in article.get("symbols") or []]
        if created is not None and wanted in symbols and start <= created <= end:
            kept.append((created, article))
    return [article for _, article in sorted(kept, key=lambda pair: pair[0])]


def pick_surprise(articles: Sequence[Mapping[str, Any]]) -> PickedSurprise | None:
    """The surprise among `articles` (one symbol, oldest first): the latest CORRECTION, else the earliest parsable headline."""
    parsed: list[PickedSurprise] = []
    for article in articles:
        headline = str(article.get("headline") or "")
        surprise = parse_surprise(headline)
        created = article_time(article)
        if surprise is not None and created is not None:
            parsed.append(PickedSurprise(surprise, headline, created))
    corrections = [p for p in parsed if p.headline.strip().upper().startswith(_CORRECTION)]
    if corrections:
        return corrections[-1]
    return parsed[0] if parsed else None
