"""Yahoo as the source of daily bars in paper/live (cross_momentum, bull_bear).

Alpaca's IEX bars carry ~5% of consolidated volume, skip a thin stock's quiet days, and close at the last IEX
trade rather than the official close, so the strategy's filter and ranks ran on other data than the backtest's
(whose bars are Yahoo's). One batched download per scan gives paper/live the backtest's own bars: the same
download (`auto_adjust=False, actions=True`, so the parse can rebuild the split- and dividend-adjusted columns and
add the traded close and volume) and the same parse (`backtesting/data/yahoo.py`), stamped at the session close.
`yfinance` is imported lazily.
"""

import logging
from collections.abc import Callable, Collection, Sequence
from datetime import date, timedelta
from typing import Protocol

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import extract_ticker_frame, parse_yahoo_frame, yfinance_download
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.errors import YahooDataError

logger = logging.getLogger(__name__)

HISTORY_BARS = 301  # the scan's 300 completed sessions plus the bar still forming
_CALENDAR_DAYS_PER_BAR = 1.5  # 301 sessions span ~440 days; the slack covers holidays and missing bars

DownloadFn = Callable[..., pd.DataFrame]


class BarsSource(Protocol):
    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]: ...


class UnusableBatch(Exception):
    """The last batch was still unusable after every retry; the message is its problem."""


class YahooDailyBars:
    """Daily `Bars` per symbol from one batched Yahoo download."""

    def __init__(self, *, download: DownloadFn | None = None, history: int = HISTORY_BARS) -> None:
        self._download = download  # injected in tests; yfinance.download otherwise
        self._history = history

    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]:
        """{symbol: bars} for the symbols Yahoo has data for, oldest first, each stamped at its session's close. The
        bar dated `today` is included while the session is open: the caller drops it.

        Raises YahooDataError when the download fails or returns nothing at all (an outage, not a missing symbol). A
        symbol whose data cannot be parsed is left out with a warning, like one Yahoo has nothing for.

        The `traded_*` columns are right only when `today` is the real download date: Yahoo adjusts to the day its
        download ends, so a past `today` (as `scores_cli` passes) misses the splits after it. Only `traded_*` are affected.
        """
        symbol_list = list(symbols)
        try:
            download = self._download if self._download is not None else yfinance_download()
            raw = download(
                symbol_list,
                start=(today - timedelta(days=round(self._history * _CALENDAR_DAYS_PER_BAR))).isoformat(),
                end=(today + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=False,
                actions=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            raise YahooDataError(f"Yahoo download failed for {len(symbol_list)} symbols: {exc}") from exc
        if raw is None or raw.empty:
            raise YahooDataError(f"Yahoo returned no data for {len(symbol_list)} symbols")

        result: dict[str, Bars] = {}
        for symbol in symbol_list:
            try:
                frame = parse_yahoo_frame(extract_ticker_frame(raw, symbol))
            except Exception as exc:  # an unexpected layout for one ticker must not abort the whole scan
                logger.warning("Unreadable Yahoo data for %s: %s", symbol, exc)
                continue
            if not frame.empty:
                result[symbol] = Bars(asset=Asset(symbol), timestep="day", df=frame)
        return result


def fetch_usable_batch(
    source: BarsSource,
    universe: Sequence[str],
    *,
    extras: Sequence[str] = (),
    today: Callable[[], date],
    held: Callable[[], Collection[str]],
    min_coverage: float,
    retry_delays: Sequence[float],
    sleep: Callable[[float], None],
    on_retry: Callable[[str, float], None],
) -> dict[str, Bars]:
    """One batched download of `universe` plus `extras`, fetched again after each delay while it is unusable.

    Unusable when Yahoo fails, covers less than `min_coverage` of the universe (extras are not counted), or misses a
    held stock that is in the universe (it would be forced out as unranked; a holding outside the universe was never
    requested and is sold as "not ranked" by design). Any other stock Yahoo lacks is simply absent from the result.
    `on_retry(problem, delay)` is called before each `sleep(delay)`; `UnusableBatch(problem)` after the last attempt.
    """
    in_universe = set(universe)
    problem = "no attempt made"
    for delay in (*retry_delays, None):
        try:
            bars = source.bars([*universe, *extras], today())
        except YahooDataError as exc:
            problem = f"Yahoo lookup failed ({exc})"
        else:
            covered = sum(1 for symbol in universe if symbol in bars)
            if covered < len(universe) * min_coverage:
                problem = f"Yahoo covers only {covered}/{len(universe)} symbols"
            elif missing_held := sorted((set(held()) & in_universe) - bars.keys()):  # read only once coverage passes
                problem = f"Yahoo has no bars for held {', '.join(missing_held)}"
            else:
                return bars
        if delay is not None:
            on_retry(problem, delay)
            sleep(delay)
    raise UnusableBatch(problem)
