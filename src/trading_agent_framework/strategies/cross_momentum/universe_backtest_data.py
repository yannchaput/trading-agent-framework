"""The preload of a point-in-time backtest (cross_momentum, bull_bear): Yahoo bars for every symbol any month lists.

`YahooBacktestData.load` fetches its assets in ONE `yf.download`, and yfinance reports a rate limit or a failed ticker as
an empty column, which `load` only warns about; `bars()` then returns None for that symbol for the whole run. With the
point-in-time universe's thousands of tickers that silently shrinks the universe. `UniverseBacktestData` loads in
batches through the history build's rate-limit gate and retries, asks once more for the symbols still empty, and
raises `UniverseHistoryError` when more than `_MAX_ABSENT_FRACTION` of what it loaded stays empty: every symbol in the
union had bars when it was ranked, so an empty frame is a failed download. The few left (a ticker Yahoo has since
dropped) are counted in `missing` and, when the strategy passes `record`, in `record["preload_missing"]`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, MutableMapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import (
    CALENDAR_SYMBOL,
    DownloadFn,
    YahooBacktestData,
    _trim_to_window,
    extract_ticker_frame,
    parse_yahoo_frame,
)
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.common.universe_history import UniverseHistory
from trading_agent_framework.strategies.cross_momentum import batch_us_stock_universe as batch
from trading_agent_framework.utils.errors import UniverseHistoryError

logger = logging.getLogger(__name__)


def universe_parameters(history: UniverseHistory) -> dict[str, Any]:
    """What a point-in-time backtest records in `parameters["universe"]` (so `settings.json` shows it): the history's
    build and pool counts, plus `preload_missing`, which the data source updates once the bars are loaded."""
    return {
        "mode": "point_in_time",
        "history_built": history.built.isoformat(),
        "pool_date": history.pool_date.isoformat(),
        "top_n": history.top_n,
        "pool_size": history.pool_size,
        "unanswered_count": len(history.unanswered),
        "bars_absent_count": len(history.bars_absent),
        "preload_missing": 0,
    }


class UniverseBacktestData(YahooBacktestData):
    """`YahooBacktestData` whose `load` downloads in batches, retries and checks coverage (see the module docstring)."""

    def __init__(
        self,
        start: datetime,
        end: datetime,
        *,
        record: MutableMapping[str, Any] | None = None,
        download: DownloadFn | None = None,
        calendar_symbol: str = CALENDAR_SYMBOL,
        today: Callable[[], date] = date.today,
    ) -> None:
        super().__init__(start, end, download=download, calendar_symbol=calendar_symbol, today=today)
        self._record = record
        self._missing: set[str] = set()

    @property
    def missing(self) -> list[str]:
        """The symbols loaded with no data, sorted (within the 1% tolerance, else `load` raised)."""
        return sorted(self._missing)

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        todo = [asset.symbol for asset in assets if asset not in self._frames]
        if not todo:
            return
        try:
            download = self._download if self._download is not None else self._real_download()
        except ImportError as exc:
            raise UniverseHistoryError("yfinance is required for the point-in-time backtest preload") from exc
        loaded: dict[str, pd.DataFrame] = {}
        with self._download_lock:
            empty = self._pass(todo, download, start, end, loaded, empty_is_failure=True)
            if empty:
                batch.console.print(f"  {len(empty)} symbol(s) without bars: asking for them once more after a pause")
                gate = batch._get_rate_limit_gate()
                gate.on_rate_limited()
                gate.wait()
                empty = self._pass(empty, download, start, end, loaded, empty_is_failure=False)
        if len(empty) > batch._MAX_ABSENT_FRACTION * len(todo):
            raise UniverseHistoryError(
                f"Yahoo returned no bars for {len(empty)} of {len(todo)} backtest symbols, above the {batch._MAX_ABSENT_FRACTION:.0%} limit: "
                f"Yahoo is failing, the backtest would trade a shrunken universe (e.g. {', '.join(empty[:20])})"
            )
        for symbol in todo:  # an empty frame is kept, so `bars()` never falls back to a lazy per-ticker download
            self._frames[Asset(symbol=symbol)] = loaded.get(symbol, pd.DataFrame())
        if empty:
            logger.warning("Yahoo has no bars for %d backtest symbols, left out of the run: %s", len(empty), ", ".join(empty))
        self._missing.update(empty)
        if self._record is not None:
            self._record["preload_missing"] = len(self._missing)

    def _pass(
        self,
        symbols: list[str],
        download: DownloadFn,
        start: datetime,
        end: datetime,
        loaded: dict[str, pd.DataFrame],
        *,
        empty_is_failure: bool,
    ) -> list[str]:
        """Download `symbols` in batches into `loaded`; returns those that came back with no rows."""
        empty: list[str] = []
        for offset in range(0, len(symbols), batch._HISTORY_BATCH_SIZE):
            group = symbols[offset : offset + batch._HISTORY_BATCH_SIZE]
            raw = batch._download_batch(download, group, start.date(), self._today(), empty_is_failure=empty_is_failure)
            for symbol in group:
                frame = _trim_to_window(parse_yahoo_frame(extract_ticker_frame(raw, symbol)), end)
                if frame.empty:
                    empty.append(symbol)
                else:
                    loaded[symbol] = frame
        return empty

