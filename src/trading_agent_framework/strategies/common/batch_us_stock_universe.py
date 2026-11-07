"""Batch Stock Universe Builder
=============================
Runs once per month to compute the investable US stock universe.

Sources:
  - NASDAQ public screener API — all stocks listed on NASDAQ, NYSE, and AMEX.
  - iShares Russell 1000 (IWB) and Russell 2000 (IWM) holdings CSVs.
  - Vanguard Total Stock Market (VTI) holdings API.
  All four are unioned to fill gaps any single source misses; NASDAQ
  supplies the market-cap pre-filter data, ETF-only symbols are kept with
  a $0 placeholder and verified by yfinance. Free, no API keys required.

Screening: yfinance metadata (quote type, price, dollar volume).

Filters:
  - Common stocks only (quoteType == EQUITY)
  - Market cap > $2B
  - Price > $10
  - Average daily dollar volume > $20M
  - Top 1200 by market cap

Output: JSON file with date, source metadata, and sorted symbol list.

History mode (`--history`): the point-in-time universe of strategies/common/universe_history.py, one top-1200 list
per month ranked only on data dated before that month, for backtests (`ensure_universe_history`). Its Yahoo bars
are cached in `cache/universe_history/` (git-ignored), laid out as:
  - `pool.json`: the candidate pool and its date, reused while it is dated in or after the last month needed;
  - `bars/<SYMBOL>.parquet`: one symbol's raw `auto_adjust=False, actions=True` download (SPY included, for the
    market calendar);
  - `bars/index.json`: per symbol, the window [start, end] its file was downloaded over.
One file per symbol (not per batch) so a changed pool, or a rerun after a crash, downloads only the symbols not
yet cached. A symbol is cached for a request when its window covers it; a larger window (earlier start, later
end) downloads that symbol again over the whole new window instead of appending, because Yahoo restates `Close`
for every split up to the download date and a spliced frame would mix two adjustment bases. An index entry whose
file is gone counts as not cached, and a batch's entries are dropped from the index before its files are replaced
(written to a temp file, then renamed), so a crash can never leave an entry claiming a window its file lacks.
yfinance reports a rate limit as missing data, not an error: an all-empty batch is retried, symbols still without
data are asked for once more after a pause, and the build stops (nothing saved) when more than 1% of the pool
remain without data. Symbols without data are not cached. The history file records the pool size, the candidates
whose common-stock check got no answer and the pool symbols still without bars. A build holds an exclusive lock
(`<cache>/.lock`), so two backtests starting at once build once: the second waits, then finds the first one's history.

Usage:
  uv run batch-universe
  uv run batch-universe --output-dir /custom/path
  uv run batch-universe --history --start 2023-01 --end 2024-12
"""

import argparse
import csv
import fcntl
import io
import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import requests
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from yfinance.exceptions import YFRateLimitError

from trading_agent_framework.backtesting.data.yahoo import (
    CALENDAR_SYMBOL,
    DownloadFn,
    extract_ticker_frame,
    parse_yahoo_frame,
    yfinance_download,
)
from trading_agent_framework.strategies.common.universe_history import UniverseHistory, _month_keys, monthly_lists
from trading_agent_framework.strategies.common.universe_params import MIN_DOLLAR_VOLUME, MIN_MARKET_CAP, MIN_PRICE
from trading_agent_framework.utils.errors import UniverseHistoryError

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "universe"

logger = logging.getLogger(__name__)
console = Console()

# ── NASDAQ screener API ─────────────────────────────────────────────────────

# NASDAQ's API only honours the LAST repeated `exchange=` param, so listing
# nasdaq/nyse/amex as three separate params (the old URL) silently returned
# NASDAQ-only results. Omitting the param entirely returns all US exchanges.
_NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ── ETF holdings sources (IWB/IWM/VTI) ──────────────────────────────────────
# Fill the NASDAQ screener's coverage gaps with the constituent lists of three
# widely tracked ETFs, fetched straight from the fund manager's own holdings
# export. Each is looked up independently and a failure only drops that one
# source (see get_extra_universe_symbols).

_ISHARES_HOLDINGS_URLS = {
    "IWB (Russell 1000)": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv",
    "IWM (Russell 2000)": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/latest-holdings.csv",
}
_VANGUARD_VTI_HOLDINGS_URL = "https://investor.vanguard.com/vmf/api/VTI/portfolio-holding/stock?start=1&count=5000"

# ── Rate-limiting for yfinance ───────────────────────────────────────────────
# `Ticker.info` costs 2+ real HTTP requests (quoteSummary plus a second
# internal call), and yfinance itself silently retries once more on any 4xx
# before giving up — so one "attempt" below can cost up to ~4 requests to
# Yahoo. With ~4,500+ tickers now (NASDAQ + IWB/IWM/VTI, vs ~1,200 before),
# sustained pressure trips Yahoo's rate limiter. RateLimitGate below makes
# recovery global instead of per-ticket, so a 429 actually pauses every
# worker instead of only the thread that hit it.

_MAX_CONCURRENT_YF = 2  # max parallel yfinance calls
_INTER_REQUEST_DELAY = 0.5  # seconds between requests (per thread)
_RATE_LIMIT_INITIAL_BACKOFF = 30.0  # seconds ALL workers pause after a 429
_RATE_LIMIT_MAX_BACKOFF = 300.0  # cap on doubling backoff for consecutive 429s


def is_rate_limit_error(exc: Exception) -> bool:
    """True if `exc` represents Yahoo's HTTP 429, however it surfaced.

    Usually yfinance's own `YFRateLimitError`, but matched on message too
    in case a 429 ever reaches us wrapped in a different exception type.
    """
    if isinstance(exc, YFRateLimitError):
        return True
    message = str(exc)
    return "429" in message or "Too Many Requests" in message


class RateLimitGate:
    """Shared cooldown so one rate-limited yfinance call pauses every worker.

    Per-ticker retry backoff alone only pauses the thread that hit the
    429 — the other concurrent threads keep hammering Yahoo, so the
    aggregate request rate never actually drops during a rate-limit
    event. This gate makes the cooldown global: any worker can extend it,
    and every worker waits on it before its next request.

    `now`/`sleep` are injectable so tests can drive a fake clock instead
    of blocking on real time.
    """

    def __init__(
        self,
        initial_backoff: float,
        max_backoff: float,
        now=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff = initial_backoff
        self._cooldown_until = 0.0
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until any active cooldown has elapsed."""
        while True:
            with self._lock:
                remaining = self._cooldown_until - self._now()
            if remaining <= 0:
                return
            self._sleep(remaining)

    def on_rate_limited(self) -> None:
        """Extend the shared cooldown; doubles on each consecutive hit."""
        with self._lock:
            self._cooldown_until = self._now() + self._backoff
            self._backoff = min(self._backoff * 2, self._max_backoff)

    def on_success(self) -> None:
        """Reset backoff to its initial value after a clean call."""
        with self._lock:
            self._backoff = self._initial_backoff


_semaphore = threading.Semaphore(_MAX_CONCURRENT_YF)
rate_limit_gate = RateLimitGate(_RATE_LIMIT_INITIAL_BACKOFF, _RATE_LIMIT_MAX_BACKOFF)


# ── Thresholds ──────────────────────────────────────────────────────────────

# Gets the thresholds from strategy central config
THRESHOLDS = {
    "market_cap": MIN_MARKET_CAP,
    "price": MIN_PRICE,
    "dollar_volume": MIN_DOLLAR_VOLUME,
}

# ── Ticker source ───────────────────────────────────────────────────────────


def normalize_symbol(raw: str) -> str | None:
    """Normalize a ticker to yfinance's dash-separated share-class form.

    Uppercases and strips whitespace, then maps the space/dot/slash
    share-class separators used by iShares, Vanguard and NASDAQ
    (``"BRK B"`` / ``"BRK.B"`` / ``"BRK/B"``) onto yfinance's ``"BRK-B"``.
    Returns None for anything that isn't left with only letters and dashes
    (cash/futures placeholders like ``"-"`` or ``"USD"`` fall out at the
    caller via asset-class filtering, but this still guards against junk).
    """
    if not raw:
        return None
    symbol = raw.strip().upper().replace(" ", "-").replace(".", "-").replace("/", "-")
    if not symbol or not symbol.replace("-", "").isalpha():
        return None
    return symbol


def parse_ishares_csv(text: str) -> set[str]:
    """Extract equity ticker symbols from an iShares ``latest-holdings.csv`` export.

    The file leads with several ``key,"value"`` metadata rows before the
    real header row (``Ticker,Name,Sector,Asset Class,...``) and mixes in
    non-equity rows (cash, money-market, futures) that must be excluded.
    """
    header_index = text.find("Ticker,Name,")
    if header_index == -1:
        return set()
    reader = csv.DictReader(io.StringIO(text[header_index:]))
    symbols: set[str] = set()
    for row in reader:
        if row.get("Asset Class") != "Equity":
            continue
        symbol = normalize_symbol(row.get("Ticker", ""))
        if symbol is not None:
            symbols.add(symbol)
    return symbols


def parse_vanguard_holdings(payload: dict) -> set[str]:
    """Extract equity ticker symbols from a Vanguard portfolio-holding API response.

    A handful of holdings (e.g. escrow claims on delisted stocks) carry no
    ``ticker`` at all and are skipped rather than normalized.
    """
    symbols: set[str] = set()
    for entity in payload.get("fund", {}).get("entity", []):
        raw = entity.get("ticker", "")
        if not raw:
            continue
        symbol = normalize_symbol(raw)
        if symbol is not None:
            symbols.add(symbol)
    return symbols


def _fetch_ishares_csv(url: str) -> str:
    r = requests.get(url, headers=_NASDAQ_HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_vanguard_holdings() -> dict:
    r = requests.get(_VANGUARD_VTI_HOLDINGS_URL, headers=_NASDAQ_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def _parse_nasdaq_market_cap(value: str) -> float:
    """Parse a NASDAQ market-cap string like '$3.45B' or '$150M' into a float.

    Returns 0.0 for unparseable values (which will fail the >$2B threshold).
    """
    if not value or not isinstance(value, str):
        return 0.0
    value = value.strip().upper().replace("$", "")
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            try:
                return float(value[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_nasdaq_rows(rows: list[dict]) -> list[tuple[str, float]]:
    """Pre-filter NASDAQ screener rows into (symbol, market_cap) candidates.

    Uses NASDAQ data as a fast-path skip only when we can positively
    determine a stock fails: market cap or price that's stale, empty, or
    $0.00 keeps the symbol and lets yfinance re-check with live data.

    Returns candidates sorted by market cap descending.
    """
    candidates: list[tuple[str, float]] = []

    for row in rows:
        symbol = normalize_symbol(row.get("symbol", ""))
        if symbol is None:
            continue

        market_cap = _parse_nasdaq_market_cap(row.get("marketCap", ""))
        if market_cap > 0 and market_cap < THRESHOLDS["market_cap"]:
            continue

        price_str = row.get("lastsale", "").replace("$", "")
        try:
            price = float(price_str)
        except ValueError, TypeError:
            price = 0.0
        if price > 0 and price < THRESHOLDS["price"]:
            continue

        candidates.append((symbol, market_cap))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def merge_ticker_candidates(nasdaq_candidates: list[tuple[str, float]], extra_symbols: set[str]) -> list[tuple[str, float]]:
    """Union NASDAQ (symbol, market_cap) candidates with ETF-holdings-only symbols.

    Symbols only an ETF source knows about (not in the NASDAQ screener
    results) are added with market_cap=0.0 — the same "keep, let yfinance
    verify" placeholder NASDAQ's own $0.00/unparseable rows already use.
    """
    known = {symbol for symbol, _ in nasdaq_candidates}
    merged = list(nasdaq_candidates)
    for symbol in extra_symbols - known:
        merged.append((symbol, 0.0))
    return merged


def _fetch_nasdaq_rows() -> list[dict]:
    """The NASDAQ screener's rows for every US exchange; raises on any HTTP or parse failure.

    Shared by the live batch (`get_ticker_universe`, which logs and returns [] on failure) and the history pool
    (`build_history_pool`, which raises).
    """
    r = requests.get(_NASDAQ_SCREENER_URL, headers=_NASDAQ_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json().get("data", {}).get("rows", [])


def get_ticker_universe() -> list[tuple[str, float]]:
    """Fetch all US exchange-listed stock symbols from the NASDAQ screener API.

    A single API call returns every stock on NASDAQ, NYSE, and AMEX
    (~7,000 symbols) — equivalent in coverage to VTI's CRSP US Total
    Market Index.

    Pre-filters by market cap (>$2B) and price (>$10) using the NASDAQ
    data to reduce the yfinance workload.

    Returns:
        List of (symbol, market_cap) candidates sorted by market cap descending.
    """
    console.print("[bold]Fetching all US exchange-listed stocks from NASDAQ screener...[/bold]")

    try:
        rows = _fetch_nasdaq_rows()
    except Exception as e:
        console.print(f"[red]Failed to fetch NASDAQ screener: {e}[/red]")
        return []

    if not rows:
        console.print("[red]NASDAQ screener returned no rows.[/red]")
        return []

    candidates = parse_nasdaq_rows(rows)
    console.print(f"  Got {len(rows)} total listed stocks → [green]{len(candidates)} candidates[/green] after pre-filter")
    return candidates


# ── yfinance screening ──────────────────────────────────────────────────────


def _fetch_yf_info(ticker: str) -> dict | None:
    """Fetch `yf.Ticker(ticker).info`, or None if every attempt fails.

    Rate-limited via a semaphore (max `_MAX_CONCURRENT_YF` concurrent calls),
    a small inter-request delay, and a shared `RateLimitGate`: a 429 on any
    thread pauses every worker, not just the one that hit it. Retries up to
    3 times, with a short per-attempt backoff on top for non-rate-limit
    transient errors.
    """
    sem = _semaphore
    gate = rate_limit_gate

    info = None
    for attempt in range(3):
        gate.wait()
        acquired = sem.acquire(timeout=15.0)
        if not acquired:
            logger.warning("Semaphore timeout for %s — skipping", ticker)
            return None
        try:
            time.sleep(_INTER_REQUEST_DELAY)
            info = yf.Ticker(ticker).info
        except Exception as exc:
            logger.warning("yfinance error for %s (attempt %d): %s", ticker, attempt + 1, exc)
            info = None
            if is_rate_limit_error(exc):
                gate.on_rate_limited()
        else:
            gate.on_success()
        finally:
            sem.release()

        if info is not None:
            break

        if attempt < 2:
            wait = 1.0 * (2**attempt)  # 1 s, then 2 s
            time.sleep(wait)

    return info


def _screen_single_ticker(ticker: str) -> dict | None:
    """Fetch yfinance info for a single ticker (see `_fetch_yf_info`) and return screening data.

    Returns a dict with keys: symbol, market_cap, price, avg_volume,
    dollar_volume, quote_type, or None if the ticker fails all retries.
    """
    info = _fetch_yf_info(ticker)
    if not info or (info.get("trailingPegRatio") is None and info.get("marketCap") is None):
        return None

    quote_type = info.get("quoteType", "")
    if quote_type != "EQUITY":
        return None

    market_cap = info.get("marketCap")
    if market_cap is None or market_cap < THRESHOLDS["market_cap"]:
        # We should never go there as this pre condition is checked in step 1.
        return None

    # Try multiple price fields (yfinance is inconsistent)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if price is None or price <= 0:
        return None
    if price < THRESHOLDS["price"]:
        return None

    avg_volume = info.get("averageVolume") or info.get("volume")
    if avg_volume is None or avg_volume <= 0:
        return None

    dollar_volume = price * avg_volume
    if dollar_volume < THRESHOLDS["dollar_volume"]:
        return None

    return {
        "symbol": ticker,
        "market_cap": market_cap,
        "price": price,
        "avg_volume": avg_volume,
        "dollar_volume": dollar_volume,
        "quote_type": quote_type,
    }


def screen_tickers(tickers: list[str], max_workers: int = 3) -> list[dict]:
    """Screen a list of tickers using yfinance in parallel.

    Args:
        tickers: List of ticker symbols.
        max_workers: Number of parallel threads for yfinance calls.

    Returns:
        List of dicts for tickers that passed all filters.
    """
    passed: list[dict] = []
    total = len(tickers)
    failed = 0

    # [progress.description] = style
    # {task.description} = dynamic text
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Screening {total} tickers...", total=total)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_screen_single_ticker, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is not None:
                    passed.append(result)
                else:
                    failed += 1
                progress.update(task, advance=1, description=f"Screening... {len(passed)} passed, {failed} failed")

    console.print(f"[bold green]Screening complete: {len(passed)} passed, {failed} rejected[/bold green]")
    return passed


# ── History pool ────────────────────────────────────────────────────────────
# The candidate symbols for the point-in-time universe history (strategies/common/universe_history.py). Same
# sources as the live batch, but no market-cap or price pre-filter (a stock that is large in 2024 may have been
# small, or cheap, years before; the history ranks on traded dollar volume instead) and no tolerance for a failing
# source (a silently smaller pool would bias every month of the history).


# The most symbols (a share of the pool, or of the candidates) that may be unreadable before the history stops. yfinance
# swallows rate limits and per-ticker failures instead of raising, so a partial outage looks like missing data; a
# handful of symbols with no data are normal (a ticker delisted since the screener ran, a warrant), more means Yahoo is
# failing. 1% is about 60 of 6,000: the floor is explicit, and the history file records the names and counts that fell
# under it (`UniverseHistory.unanswered`, `bars_absent`), because a real stock left out of every month is a bias.
MAX_ABSENT_FRACTION = 0.01


class HistoryPool(NamedTuple):
    """The pool's common stocks, sorted, and the candidates whose common-stock check got no answer (left out), sorted."""

    symbols: list[str]
    unanswered: list[str]


def parse_nasdaq_symbols(rows: list[dict]) -> list[str]:
    """Every normalized NASDAQ screener symbol, with no market-cap or price pre-filter, deduplicated and sorted."""
    symbols = {symbol for row in rows if (symbol := normalize_symbol(row.get("symbol", ""))) is not None}
    return sorted(symbols)


def _is_common_stock(symbol: str) -> bool | None:
    """True if yfinance reports `quoteType == "EQUITY"` for `symbol`, False for any other quote type, None when it gave none.

    Reuses the live batch's `info` fetch (rate-limit gate, semaphore, retries). A fetch that fails after the retries,
    an empty `info` and an `info` with no `quoteType` (what a throttled lookup returns, as the live batch also
    treats it) are all None (logged): no answer, counted apart from a genuine "not common" by `build_history_pool`,
    which asks again once and stops when too many are still unanswered.
    """
    info = _fetch_yf_info(symbol)
    if not info:
        logger.warning("No yfinance info for %s after retries: no answer", symbol)
        return None
    quote_type = info.get("quoteType")
    if not quote_type:
        logger.warning("yfinance info for %s has no quoteType: no answer", symbol)
        return None
    return quote_type == "EQUITY"


def _etf_symbols(
    fetch_ishares_csv: Callable[[str], str],
    fetch_vanguard_holdings: Callable[[], dict],
    *,
    strict: bool,
) -> set[str]:
    """Union Russell 1000 (IWB), Russell 2000 (IWM) and total-market (VTI) holdings, each source fetched independently.

    Lenient (the live batch): a failing source is logged and skipped rather than aborting the others. Strict (the history
    pool, where a silently smaller pool would bias every month): a source that fails or returns nothing raises
    `UniverseHistoryError`.
    """
    symbols: set[str] = set()

    def take(name: str, load: Callable[[], set[str]], label: str | None = None) -> None:
        # `name` is in the failure texts; `label` (default `name`) only in the count line
        try:
            source_symbols = load()
        except Exception as e:
            if strict:
                raise UniverseHistoryError(f"{name} holdings failed: {e}") from e
            console.print(f"[yellow]Failed to fetch {name} holdings: {e}[/yellow]")
            return
        if strict and not source_symbols:
            raise UniverseHistoryError(f"{name} holdings returned no equity symbols")
        console.print(f"  {label or name}: [green]{len(source_symbols)}[/green] equity holdings")
        symbols.update(source_symbols)

    for name, url in _ISHARES_HOLDINGS_URLS.items():
        take(name, lambda url=url: parse_ishares_csv(fetch_ishares_csv(url)))
    take("VTI", lambda: parse_vanguard_holdings(fetch_vanguard_holdings()), label="VTI (total market)")
    return symbols


def get_extra_universe_symbols(
    fetch_ishares_csv=_fetch_ishares_csv,
    fetch_vanguard_holdings=_fetch_vanguard_holdings,
) -> set[str]:
    """The ETF holdings union for the live batch (lenient); `fetch_*` are injectable so tests need no network."""
    return _etf_symbols(fetch_ishares_csv, fetch_vanguard_holdings, strict=False)


def _filter_common_stocks(symbols: list[str], is_common_stock: Callable[[str], bool | None]) -> tuple[list[str], list[str]]:
    """The symbols `is_common_stock` accepts, and those it could not answer (None), checked over `_MAX_CONCURRENT_YF`
    workers like `screen_tickers`.
    """
    kept: list[str] = []
    failed: list[str] = []
    rejected = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Checking {len(symbols)} symbols...", total=len(symbols))
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_YF) as executor:
            futures = {executor.submit(is_common_stock, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                answer = future.result()
                if answer is None:
                    failed.append(futures[future])
                elif answer:
                    kept.append(futures[future])
                else:
                    rejected += 1
                progress.update(task, advance=1, description=f"Checking... {len(kept)} common, {rejected} not, {len(failed)} unanswered")
    return kept, sorted(failed)


def build_history_pool(
    *,
    fetch_nasdaq_rows: Callable[[], list[dict]] = _fetch_nasdaq_rows,
    fetch_ishares_csv: Callable[[str], str] = _fetch_ishares_csv,
    fetch_vanguard_holdings: Callable[[], dict] = _fetch_vanguard_holdings,
    is_common_stock: Callable[[str], bool | None] = _is_common_stock,
) -> HistoryPool:
    """The candidate symbols for the universe history (NASDAQ screener plus IWB/IWM/VTI, common stocks only) and the
    candidates left unanswered.

    The candidates `is_common_stock` could not answer (None) are asked once more after a pause through the shared
    rate-limit gate (a throttled lookup looks like no answer). Raises `UniverseHistoryError` if any source fails or
    returns nothing, if more than `MAX_ABSENT_FRACTION` of the candidates are still unanswered after that, or if no
    common stock is left. The few unanswered are left out, logged and returned. All the fetchers and
    `is_common_stock` are injectable so tests supply fakes without touching the network.
    """
    try:
        rows = fetch_nasdaq_rows()
    except Exception as e:
        raise UniverseHistoryError(f"NASDAQ screener failed: {e}") from e
    nasdaq_symbols = parse_nasdaq_symbols(rows)
    if not nasdaq_symbols:
        raise UniverseHistoryError("NASDAQ screener returned no symbols")
    console.print(f"  NASDAQ screener: [green]{len(nasdaq_symbols)}[/green] listed symbols")

    etf_symbols = _etf_symbols(fetch_ishares_csv, fetch_vanguard_holdings, strict=True)
    candidates = sorted(set(nasdaq_symbols) | etf_symbols)
    console.print(f"  Union: [green]{len(candidates)}[/green] candidates")

    kept, unanswered = _filter_common_stocks(candidates, is_common_stock)
    if unanswered:
        console.print(f"  {len(unanswered)} candidate(s) with no answer: asking for them once more after a pause")
        gate = rate_limit_gate
        gate.on_rate_limited()
        gate.wait()
        recovered, unanswered = _filter_common_stocks(unanswered, is_common_stock)
        kept = [*kept, *recovered]
    if len(unanswered) > MAX_ABSENT_FRACTION * len(candidates):
        raise UniverseHistoryError(
            f"yfinance gave no answer for {len(unanswered)} of {len(candidates)} candidates, above the {MAX_ABSENT_FRACTION:.0%} limit: "
            f"Yahoo is failing (e.g. {', '.join(unanswered[:20])})"
        )
    if unanswered:
        logger.warning("yfinance gave no answer for %d candidates, left out of the history pool: %s", len(unanswered), ", ".join(unanswered))
        console.print(f"  [yellow]{len(unanswered)} candidate(s) with no yfinance answer[/yellow] left out of the pool")
    pool = sorted(kept)
    if not pool:
        raise UniverseHistoryError("the history pool is empty: no candidate is a common stock")
    console.print(f"  History pool: [green]{len(pool)}[/green] common stocks")
    return HistoryPool(pool, unanswered)


# ── History build ───────────────────────────────────────────────────────────
# `ensure_universe_history`: the backtest pre-step. Cache layout: see the module docstring.

_HISTORY_TOP_N = 1200
_HISTORY_LOOKBACK = 63
HISTORY_BATCH_SIZE = 100  # symbols per Yahoo download
# The bars start this many calendar days before the first needed month: 63 sessions are about 90 calendar days.
_HISTORY_WARMUP_DAYS = 120
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_RETRY_BACKOFF = 2.0  # seconds, doubled per attempt, on a non-rate-limit error
_DEFAULT_HISTORY_PATH = DEFAULT_OUTPUT_DIR / "us_stock_universe_history.json"
_DEFAULT_HISTORY_CACHE = PROJECT_ROOT / "cache" / "universe_history"


def _same_parameters(history: UniverseHistory) -> bool:
    """True when `history` was built with the parameters this build would use (else its months are not comparable)."""
    return (
        history.top_n == _HISTORY_TOP_N
        and history.lookback == _HISTORY_LOOKBACK
        and history.min_price == float(MIN_PRICE)
        and history.min_dollar_volume == float(MIN_DOLLAR_VOLUME)
    )


def _load_pool(cache_dir: Path) -> tuple[date, HistoryPool] | None:
    path = cache_dir / "pool.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return date.fromisoformat(data["date"]), HistoryPool(list(data["symbols"]), list(data.get("unanswered", ())))


def _save_pool(cache_dir: Path, pool_date: date, pool: HistoryPool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "pool.json").write_text(json.dumps({"date": pool_date.isoformat(), "symbols": pool.symbols, "unanswered": pool.unanswered}))


def _resolve_pool(cache_dir: Path, last_month_start: date, today: date, pool: Callable[[], HistoryPool] | None) -> tuple[date, HistoryPool]:
    """The saved pool when it is dated in or after the last month needed; else a fresh one, saved."""
    try:
        saved = _load_pool(cache_dir)
        if saved is not None and saved[0] >= last_month_start:
            console.print(f"  Reusing the saved pool of {len(saved[1].symbols)} symbols (dated {saved[0]})")
            return saved
        fresh = build_history_pool() if pool is None else pool()
        resolved = HistoryPool(sorted(set(fresh.symbols)), sorted(set(fresh.unanswered)))
    except UniverseHistoryError:
        raise
    except Exception as exc:
        raise UniverseHistoryError(f"the history pool failed: {exc}") from exc
    if not resolved.symbols:
        raise UniverseHistoryError("the history pool is empty")
    _save_pool(cache_dir, today, resolved)
    return today, resolved


def download_batch(download: DownloadFn, symbols: list[str], start: date, end: date, *, empty_is_failure: bool) -> pd.DataFrame:
    """One batched Yahoo download, through the live batch's rate-limit gate, retried.

    `end` is inclusive here; Yahoo's is exclusive, so the call passes the day after. yfinance swallows a rate limit or
    a per-ticker failure into an empty frame instead of raising, so with `empty_is_failure` a batch that came back
    with no data at all counts as a failed attempt (retried with a backoff); when the last attempt is still empty
    the empty frame is returned, and the caller treats its symbols as absent. An exception on every attempt raises
    `UniverseHistoryError`.
    """
    gate = rate_limit_gate
    failure: Exception | None = None
    for attempt in range(_DOWNLOAD_ATTEMPTS):
        gate.wait()
        last = attempt == _DOWNLOAD_ATTEMPTS - 1
        try:
            time.sleep(_INTER_REQUEST_DELAY)
            raw = download(
                symbols,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                actions=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            failure = exc
            logger.warning("Yahoo download of %d symbols (from %s) failed, attempt %d: %s", len(symbols), symbols[0], attempt + 1, exc)
            if is_rate_limit_error(exc):
                gate.on_rate_limited()  # the next gate.wait() sleeps the shared cooldown
            elif not last:
                time.sleep(_DOWNLOAD_RETRY_BACKOFF * 2**attempt)
            continue
        if empty_is_failure and raw.dropna(how="all").empty:
            failure = None
            logger.warning("Yahoo returned no data for any of %d symbols (from %s), attempt %d", len(symbols), symbols[0], attempt + 1)
            if not last:
                time.sleep(_DOWNLOAD_RETRY_BACKOFF * 2**attempt)
            continue
        gate.on_success()
        return raw
    if failure is None:
        return raw  # still empty on the last attempt: the symbols are absent
    raise UniverseHistoryError(f"the Yahoo download of {len(symbols)} symbols (from {symbols[0]}) failed after {_DOWNLOAD_ATTEMPTS} attempts: {failure}") from failure


def _load_bars_index(bars_dir: Path) -> dict[str, tuple[date, date]]:
    path = bars_dir / "index.json"
    if not path.exists():
        return {}
    return {symbol: (date.fromisoformat(window[0]), date.fromisoformat(window[1])) for symbol, window in json.loads(path.read_text()).items()}


def _save_bars_index(bars_dir: Path, index: dict[str, tuple[date, date]]) -> None:
    temporary = bars_dir / "index.json.tmp"
    temporary.write_text(json.dumps({symbol: [window[0].isoformat(), window[1].isoformat()] for symbol, window in sorted(index.items())}))
    temporary.replace(bars_dir / "index.json")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write `frame` to a temp file beside `path` and rename it over, so `path` is never half-written."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _download_symbols(
    todo: list[str],
    window_start: date,
    download_end: date,
    bars_dir: Path,
    index: dict[str, tuple[date, date]],
    download: DownloadFn,
    *,
    empty_is_failure: bool,
) -> list[str]:
    """Download `todo` in batches over [window_start, download_end] into the cache; returns the symbols with no data.

    Per batch, the index entries of its symbols are dropped and saved BEFORE any file is overwritten or removed,
    and re-added after the files are written: a crash leaves a file without an entry (downloaded again), never an
    entry claiming a window its file does not hold.
    """
    absent: list[str] = []
    for offset in range(0, len(todo), HISTORY_BATCH_SIZE):
        batch_symbols = todo[offset : offset + HISTORY_BATCH_SIZE]
        for symbol in batch_symbols:
            index.pop(symbol, None)
        _save_bars_index(bars_dir, index)
        raw = download_batch(download, batch_symbols, window_start, download_end, empty_is_failure=empty_is_failure)
        for symbol in batch_symbols:
            frame = extract_ticker_frame(raw, symbol)
            frame = frame.dropna(how="all") if not frame.empty else frame
            path = bars_dir / f"{symbol}.parquet"
            if frame.empty:
                absent.append(symbol)
                path.unlink(missing_ok=True)  # never rank on an older download of it
                continue
            _write_frame(frame, path)
            index[symbol] = (window_start, download_end)
        _save_bars_index(bars_dir, index)
        console.print(f"  Downloaded {min(offset + HISTORY_BATCH_SIZE, len(todo))}/{len(todo)}")
    return absent


def _ensure_bars(
    symbols: list[str],
    window_start: date,
    window_end: date,
    download_end: date,
    cache_dir: Path,
    download: DownloadFn,
) -> list[str]:
    """Make every symbol's raw bars cached over [window_start, window_end]; returns the symbols Yahoo has no data for.

    A symbol already cached over a window covering the request, with its file present, is skipped; the rest are
    downloaded in batches over [window_start, download_end] (`download_end` is today, so the split data covers every
    split since the window). A first pass counts an all-empty batch as a failed attempt; every symbol still without
    data is then requested once more after a pause through the shared rate-limit gate, because yfinance reports a
    rate limit as missing data. The index is saved after each batch, so a crash keeps the batches already paid for.
    """
    bars_dir = cache_dir / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    index = _load_bars_index(bars_dir)
    todo = [
        symbol
        for symbol in symbols
        if symbol not in index or index[symbol][0] > window_start or index[symbol][1] < window_end or not (bars_dir / f"{symbol}.parquet").exists()
    ]
    console.print(f"  Bars: {len(symbols) - len(todo)} symbols cached, {len(todo)} to download")
    absent = _download_symbols(todo, window_start, download_end, bars_dir, index, download, empty_is_failure=True)
    if absent:
        console.print(f"  {len(absent)} symbol(s) without data: asking for them once more after a pause")
        gate = rate_limit_gate
        gate.on_rate_limited()
        gate.wait()
        absent = _download_symbols(absent, window_start, download_end, bars_dir, index, download, empty_is_failure=False)
    return sorted(absent)


def _load_traded_frames(symbols: list[str], bars_dir: Path) -> dict[str, pd.DataFrame]:
    """Each symbol's `traded_close` and `traded_volume` (the only columns the ranking reads); a missing file is an error."""
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = bars_dir / f"{symbol}.parquet"
        if not path.exists():
            raise UniverseHistoryError(f"the cached bars of {symbol} are missing: {path}")
        frames[symbol] = parse_yahoo_frame(pd.read_parquet(path))[["traded_close", "traded_volume"]]
    return frames


@contextmanager
def _build_lock(cache_dir: Path) -> Iterator[None]:
    """An exclusive `flock` on `<cache_dir>/.lock`, held for the block: two processes building at once would share the
    cache's temp files and each save its own bars index over the other's files. Released when the process exits.
    A caller that finds it held logs a warning and prints a line saying it waits, then blocks until the other build ends.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A first build can take an hour: say so instead of looking hung, then wait for it.
            message = f"another universe-history build holds the lock on {cache_dir}: waiting for it (a first build can take tens of minutes to over an hour)"
            logger.warning(message)
            console.print(f"[yellow]{message}[/yellow]")
            waited_from = time.monotonic()
            fcntl.flock(handle, fcntl.LOCK_EX)
            console.print(f"[green]Lock acquired after {time.monotonic() - waited_from:.0f}s[/green]")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_usable(path: Path) -> UniverseHistory | None:
    """The history saved at `path` when it was built with the current parameters, else None."""
    try:
        saved = UniverseHistory.load(path)
    except Exception as exc:
        raise UniverseHistoryError(f"the universe history at {path} is unreadable: {exc}") from exc
    return saved if saved is not None and _same_parameters(saved) else None


def ensure_universe_history(
    start: date,
    end: date,
    *,
    path: Path = _DEFAULT_HISTORY_PATH,
    cache_dir: Path = _DEFAULT_HISTORY_CACHE,
    today: Callable[[], date] = date.today,
    download: DownloadFn | None = None,
    pool: Callable[[], HistoryPool] | None = None,
) -> UniverseHistory:
    """The point-in-time universe history covering every month from `start`'s to `end`'s, built or extended if needed.

    A history file at `path` that covers the window (and was built with the current parameters) is returned as is,
    with no download. Otherwise the missing months are computed from the pool's Yahoo daily bars (cached in
    `cache_dir`) plus SPY's sessions, merged with the months already saved, and saved. The build holds an exclusive
    lock on `cache_dir` and re-reads the history once it has it, so a second process starting at the same time waits and
    then returns the first one's result without downloading. `download` (default `yfinance.download`) and `pool`
    (default `build_history_pool`) are injectable for tests. Every failure raises
    `UniverseHistoryError`: the pool, a download, a month with no session or no symbol, missing SPY sessions.
    """
    existing = _load_usable(path)
    if existing is not None and existing.covers(start, end):
        return existing
    with _build_lock(cache_dir):
        existing = _load_usable(path)  # another process may have finished while this one waited
        if existing is not None and existing.covers(start, end):
            return existing
        return _build_history(start, end, existing, path, cache_dir, today, download, pool)


def _build_history(
    start: date,
    end: date,
    existing: UniverseHistory | None,
    path: Path,
    cache_dir: Path,
    today: Callable[[], date],
    download: DownloadFn | None,
    pool: Callable[[], HistoryPool] | None,
) -> UniverseHistory:
    """The build path of `ensure_universe_history`, run under the cache lock: compute the missing months and save."""
    needed = _month_keys(start, end)
    missing = [month for month in needed if existing is None or month not in existing.months]
    console.print(f"[bold]Building the universe history for {', '.join(missing)}[/bold]")
    first_missing = date.fromisoformat(f"{missing[0]}-01")
    last_month_start = date(end.year, end.month, 1)
    now = today()
    window_start = first_missing - timedelta(days=_HISTORY_WARMUP_DAYS)
    window_end = min(end, now)  # nothing later than today exists to cache

    pool_date, history_pool = _resolve_pool(cache_dir, last_month_start, now, pool)
    symbols = history_pool.symbols
    try:
        fetch = download if download is not None else yfinance_download()
    except ImportError as exc:
        raise UniverseHistoryError("yfinance is required to build the universe history") from exc
    try:
        absent = _ensure_bars([*symbols, CALENDAR_SYMBOL] if CALENDAR_SYMBOL not in symbols else symbols, window_start, window_end, now, cache_dir, fetch)
        if CALENDAR_SYMBOL in absent:
            raise UniverseHistoryError(f"Yahoo has no {CALENDAR_SYMBOL} bars: the market calendar for the universe history is missing")
        spy = parse_yahoo_frame(pd.read_parquet(cache_dir / "bars" / f"{CALENDAR_SYMBOL}.parquet"))
        sessions = [ts.date() for ts in spy.index]
        pool_absent = [symbol for symbol in absent if symbol != CALENDAR_SYMBOL]
        if len(pool_absent) > MAX_ABSENT_FRACTION * len(symbols):
            raise UniverseHistoryError(
                f"Yahoo has no data for {len(pool_absent)} of {len(symbols)} pool symbols, above the {MAX_ABSENT_FRACTION:.0%} limit: "
                f"Yahoo is failing, no history saved (e.g. {', '.join(pool_absent[:20])})"
            )
        frames = _load_traded_frames([symbol for symbol in symbols if symbol != CALENDAR_SYMBOL and symbol not in pool_absent], cache_dir / "bars")
    except UniverseHistoryError:
        raise
    except Exception as exc:
        raise UniverseHistoryError(f"the universe history bars failed: {exc}") from exc
    if pool_absent:
        logger.warning("Yahoo has no data for %d pool symbols: %s", len(pool_absent), ", ".join(pool_absent))
        console.print(f"  [yellow]{len(pool_absent)} symbol(s) with no Yahoo data[/yellow] left out of the history")

    computed = monthly_lists(
        frames,
        sessions,
        missing,
        min_price=float(MIN_PRICE),
        min_dollar_volume=float(MIN_DOLLAR_VOLUME),
        top_n=_HISTORY_TOP_N,
        lookback=_HISTORY_LOOKBACK,
    )
    for month, listed in computed.items():
        if not listed:
            raise UniverseHistoryError(f"the universe of {month} is empty: no symbol passes the price and dollar-volume screens")

    history = UniverseHistory(
        built=now,
        pool_date=pool_date,
        top_n=_HISTORY_TOP_N,
        min_price=float(MIN_PRICE),
        min_dollar_volume=float(MIN_DOLLAR_VOLUME),
        lookback=_HISTORY_LOOKBACK,
        months=dict(sorted({**(existing.months if existing is not None else {}), **computed}.items())),
        pool_size=len(symbols),
        unanswered=tuple(sorted(history_pool.unanswered)),
        bars_absent=tuple(sorted(pool_absent)),
    )
    history.save(path)
    return history


# ── Output ──────────────────────────────────────────────────────────────────


def build_universe_json(results: list[dict], source: str) -> dict:
    """Sort results by market cap descending, take top 1200, build output dict."""
    results.sort(key=lambda x: x["market_cap"], reverse=True)

    top = results[:_HISTORY_TOP_N]

    return {
        "date": date.today().isoformat(),
        "source": source,
        "total_screened": len(results),
        "total_passed": len(top),
        "symbols": [r["symbol"] for r in top],
    }


def save_universe(data: dict, output_dir: Path) -> Path:
    """Save universe JSON to output_dir.

    Writes two files:
      - us_stock_universe.json (latest, overwritten)
      - us_stock_universe_YYYY-MM-DD.json (dated archive)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    content = json.dumps(data, indent=2)

    latest_path = output_dir / "us_stock_universe.json"
    latest_path.write_text(content)
    console.print(f"[green]Saved: {latest_path}[/green]")

    dated_path = output_dir / f"us_stock_universe_{data['date']}.json"
    dated_path.write_text(content)
    console.print(f"[green]Saved: {dated_path}[/green]")

    return latest_path


# ── Main ────────────────────────────────────────────────────────────────────


def _parse_month(text: str) -> date:
    """The first day of a "YYYY-MM" month."""
    try:
        return date.fromisoformat(f"{text}-01")
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a month: expected YYYY-MM") from None


def _main_history(start: date, end: date, output_dir: Path) -> None:
    """The `--history` CLI: ensure the point-in-time history covers [start, end] and print the months and sizes."""
    try:
        history = ensure_universe_history(start, end, path=output_dir / _DEFAULT_HISTORY_PATH.name)
    except UniverseHistoryError as exc:
        console.print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)
    console.print(f"\n[bold green]Universe history: {len(history.months)} months (pool dated {history.pool_date}, built {history.built})[/bold green]")
    for month, symbols in history.months.items():
        console.print(f"  {month}: {len(symbols)} symbols")


def main():
    """Entry point for the batch-universe CLI."""
    parser = argparse.ArgumentParser(
        description="Build the investable US stock universe for cross-sectional momentum.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for universe JSON (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Build or extend the point-in-time universe history (one list per month) instead of the live universe",
    )
    parser.add_argument("--start", type=_parse_month, help="--history: first month, YYYY-MM")
    parser.add_argument("--end", type=_parse_month, help="--history: last month, YYYY-MM (default: this month)")
    args = parser.parse_args()

    if args.history:
        if args.start is None:
            parser.error("--history needs --start YYYY-MM")
        end = args.end or date.today().replace(day=1)
        if end < args.start:
            parser.error("--end is before --start")
        _main_history(args.start, end, Path(args.output_dir))
        return

    console.print("[bold cyan]══════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Batch Stock Universe Builder[/bold cyan]")
    console.print(f"[bold cyan]  Date: {date.today().isoformat()}[/bold cyan]")
    console.print("[bold cyan]══════════════════════════════════════════[/bold cyan]")

    output_dir = Path(args.output_dir)

    # Step 1: Get ticker list from NASDAQ screener (all US exchanges)
    console.print("\n[bold]Step 1: Fetching all US-listed stocks from NASDAQ screener...[/bold]")
    nasdaq_candidates = get_ticker_universe()
    if not nasdaq_candidates:
        console.print("[red]ERROR: No tickers retrieved from NASDAQ screener. Aborting.[/red]")
        sys.exit(1)

    # Step 2: Fill coverage gaps from Russell 1000/2000 and total-market ETF holdings
    console.print("\n[bold]Step 2: Fetching IWB/IWM/VTI holdings to fill NASDAQ screener coverage gaps...[/bold]")
    extra_symbols = get_extra_universe_symbols()
    candidates = merge_ticker_candidates(nasdaq_candidates, extra_symbols)
    added = len(candidates) - len(nasdaq_candidates)
    console.print(f"  Merged universe: [green]{len(candidates)} candidates[/green] ({added} added by ETF holdings)")

    tickers = [symbol for symbol, _ in candidates]

    # Step 3: Screen with yfinance
    console.print(f"\n[bold]Step 3: Screening {len(tickers)} tickers with yfinance...[/bold]")
    console.print("  (This may take several minutes — monthly batch, acceptable runtime)")
    start = time.monotonic()
    # The max concurrent threads is set to _MAX_CONCURRENT_YF (3) to avoid yfinance rate-limiting issues.
    results = screen_tickers(tickers, max_workers=_MAX_CONCURRENT_YF)
    elapsed = time.monotonic() - start
    console.print(f"  Screening took {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if not results:
        console.print("[red]ERROR: No tickers passed screening. Aborting without overwriting existing files.[/red]")
        sys.exit(1)

    # Step 4: Build and save
    console.print("\n[bold]Step 4: Building universe JSON...[/bold]")
    data = build_universe_json(results, source="nasdaq_screener+ishares_iwb+ishares_iwm+vanguard_vti")
    save_universe(data, output_dir)

    console.print(f"\n[bold green]Done! Universe: {data['total_passed']} symbols saved.[/bold green]")
    console.print(f"[bold green]Top 10: {', '.join(data['symbols'][:10])}[/bold green]")


if __name__ == "__main__":
    main()
