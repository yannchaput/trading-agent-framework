"""`QualityScreen`: wires the annual-figures store and the split history to the pure quality gates.

Knows no `Strategy`, broker or LLM: the caller passes the date (`as_of`, its own clock) and a price
lookup. Every per-symbol problem becomes a rejection with a named reason; only a screen hollowed out
by transport failures raises.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from trading_agent_framework.fundamentals.annual_store import AnnualFiguresStore
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.quality import Priced, ScreenParams, ScreenResult, assess, rank, sector_excluded
from trading_agent_framework.fundamentals.splits import Split, SplitHistory, restate_shares
from trading_agent_framework.utils.errors import FundamentalsError, TradingFrameworkError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "QualityScreen")


class FiguresSource(Protocol):
    """What the screen needs from `AnnualFiguresStore`."""

    def cik(self, symbol: str) -> str | None: ...

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, Any] | None: ...

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None: ...


class SplitSource(Protocol):
    """What the screen needs from `SplitHistory`."""

    def splits(self, symbol: str, *, max_age_days: int) -> list[Split]: ...


class QualityScreen:
    def __init__(self, store: FiguresSource, splits: SplitSource, *, params: ScreenParams | None = None) -> None:
        self._store = store
        self._splits = splits
        self.params = params or ScreenParams()

    def run(self, symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None]) -> ScreenResult:
        """Rank the companies among `symbols` that pass every gate on `as_of`, best first.

        `as_of` is expected in market-local time (New York): the screen uses `as_of.date()`, and a
        filing counts as known only when filed strictly before that date, so a UTC `as_of` late in the
        New York evening reads one day ahead and would see filings the market has not yet seen.
        `price_of` is called only for companies that passed every gate before the price gate.
        Raises `FundamentalsError` when more than `params.max_fetch_failure_ratio` of the symbols
        could not be fetched: a ranking of what happened to download is worse than none.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        params = self.params
        max_age_days = params.max_age_days
        unique = list(dict.fromkeys(symbols))
        rejections: dict[str, str] = {}
        priced: list[Priced] = []
        accepted_ciks: set[str | None] = set()
        transport_failures = 0

        for symbol in unique:
            try:
                outcome = assess(symbol, self._store.figures(symbol, as_of=as_of, max_age_days=max_age_days), as_of=as_of, params=params)
                if isinstance(outcome, str):
                    rejections[symbol] = outcome
                    continue
                sic = self._store.sic(symbol, as_of=as_of, max_age_days=max_age_days)
            except FundamentalsError as exc:
                transport_failures += 1
                rejections[symbol] = "no_data"
                logger.log_debug(f"{symbol}: {exc}")
                continue
            if sector_excluded(sic, params):
                rejections[symbol] = "excluded_sector"
                continue
            cik = self._store.cik(symbol)
            if cik in accepted_ciks:
                rejections[symbol] = "duplicate_listing"
                continue
            price = _price(price_of, symbol)
            if price is None or not outcome.shares or outcome.counted_on is None:
                rejections[symbol] = "no_price"
                continue
            try:
                splits = self._splits.splits(symbol, max_age_days=params.split_max_age_days)
            except FundamentalsError as exc:
                rejections[symbol] = "no_split_data"
                logger.log_debug(f"{symbol}: {exc}")
                continue
            market_cap = restate_shares(outcome.shares, outcome.counted_on, splits) * price
            if not market_cap.is_finite() or market_cap <= 0:  # a corrupt split ratio (0, negative, NaN, inf)
                rejections[symbol] = "no_split_data"
                continue
            accepted_ciks.add(cik)  # only now: a first listing with no price must not block the second
            priced.append(Priced(survivor=outcome, sic=sic, market_cap=market_cap))

        if unique and transport_failures / len(unique) > params.max_fetch_failure_ratio:
            raise FundamentalsError(f"quality screen aborted: {transport_failures} of {len(unique)} symbols could not be fetched from SEC")

        candidates = rank(priced, params)
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(Counter(rejections.values()).items())) or "none"
        logger.log_info(f"screened {len(unique)} symbols as of {as_of.date()}: {len(candidates)} candidates from {len(priced)} survivors; rejected: {reasons}")
        return ScreenResult(candidates=candidates, rejections=rejections)


def _price(price_of: Callable[[str], Decimal | None], symbol: str) -> Decimal | None:
    """`price_of(symbol)` as a `Decimal` when it is a usable price; a failed, non-numeric or non-positive quote is no price.

    A float or int price (a caller's lookup may return one) is converted through `str`, so `10.5` is
    exactly 10.5, not the nearest binary fraction.
    """
    try:
        price = price_of(symbol)
    except TradingFrameworkError as exc:
        logger.log_debug(f"{symbol}: no price: {exc}")
        return None
    if isinstance(price, bool) or not isinstance(price, int | float | Decimal):
        return None
    price = price if isinstance(price, Decimal) else Decimal(str(price))
    return price if price.is_finite() and price > 0 else None


def build_quality_screen(project_root: Path, *, params: ScreenParams | None = None) -> QualityScreen:
    """The real screen: SEC annual figures and Yahoo split history, cached under `<project_root>/cache/`.

    Raises `ConfigurationError` when `SEC_EDGAR_USER_AGENT` is missing or blank.
    """
    sec_cache = project_root / "cache" / "sec"
    client = SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), sec_cache)
    store = AnnualFiguresStore(client, sec_cache / "annual")
    return QualityScreen(store, SplitHistory(project_root / "cache" / "splits.json"), params=params)
