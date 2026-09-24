"""
Batch Stock Universe Builder
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

Usage:
  uv run batch-universe
  uv run batch-universe --output-dir /custom/path
"""

import argparse
import csv
import io
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import requests
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

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

_MAX_CONCURRENT_YF = 3  # max parallel yfinance calls
_INTER_REQUEST_DELAY = 0.3  # seconds between requests (per thread)

_semaphore: threading.Semaphore | None = None


def _get_semaphore() -> threading.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = threading.Semaphore(_MAX_CONCURRENT_YF)
    return _semaphore


# ── Thresholds ──────────────────────────────────────────────────────────────

# Gets the thresholds from strategy central config
THRESHOLDS = {
    "market_cap": CONFIG["min_market_cap"],
    "price": CONFIG["min_price"],
    "dollar_volume": CONFIG["min_dollar_volume"],
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


def get_extra_universe_symbols(
    fetch_ishares_csv=_fetch_ishares_csv,
    fetch_vanguard_holdings=_fetch_vanguard_holdings,
) -> set[str]:
    """Union Russell 1000 (IWB), Russell 2000 (IWM), and total-market (VTI) holdings.

    Each source is fetched independently; a failing source is logged and
    skipped rather than aborting the others. `fetch_ishares_csv` and
    `fetch_vanguard_holdings` are injectable so tests can supply fixtures
    without touching the network.
    """
    symbols: set[str] = set()

    for name, url in _ISHARES_HOLDINGS_URLS.items():
        try:
            text = fetch_ishares_csv(url)
            source_symbols = parse_ishares_csv(text)
        except Exception as e:
            console.print(f"[yellow]Failed to fetch {name} holdings: {e}[/yellow]")
            continue
        console.print(f"  {name}: [green]{len(source_symbols)}[/green] equity holdings")
        symbols |= source_symbols

    try:
        payload = fetch_vanguard_holdings()
        vti_symbols = parse_vanguard_holdings(payload)
    except Exception as e:
        console.print(f"[yellow]Failed to fetch VTI holdings: {e}[/yellow]")
    else:
        console.print(f"  VTI (total market): [green]{len(vti_symbols)}[/green] equity holdings")
        symbols |= vti_symbols

    return symbols


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
        except (ValueError, TypeError):
            price = 0.0
        if price > 0 and price < THRESHOLDS["price"]:
            continue

        candidates.append((symbol, market_cap))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def merge_ticker_candidates(
    nasdaq_candidates: list[tuple[str, float]], extra_symbols: set[str]
) -> list[tuple[str, float]]:
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
        r = requests.get(_NASDAQ_SCREENER_URL, headers=_NASDAQ_HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        console.print(f"[red]Failed to fetch NASDAQ screener: {e}[/red]")
        return []

    rows = data.get("data", {}).get("rows", [])
    if not rows:
        console.print("[red]NASDAQ screener returned no rows.[/red]")
        return []

    candidates = parse_nasdaq_rows(rows)
    console.print(
        f"  Got {len(rows)} total listed stocks → [green]{len(candidates)} candidates[/green] after pre-filter"
    )
    return candidates


# ── yfinance screening ──────────────────────────────────────────────────────


def _screen_single_ticker(ticker: str) -> dict | None:
    """Fetch yfinance info for a single ticker and return screening data.

    Rate-limited via a semaphore (max 3 concurrent calls) and a small
    inter-request delay.  Retries up to 3 times with exponential backoff
    on rate-limit / transient errors.

    Returns a dict with keys: symbol, market_cap, price, avg_volume,
    dollar_volume, quote_type, or None if the ticker fails all retries.
    """
    sem = _get_semaphore()

    for attempt in range(3):
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
        finally:
            sem.release()

        if info is not None:
            break

        if attempt < 2:
            wait = 1.0 * (2**attempt)  # 1 s, then 2 s
            time.sleep(wait)

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


# ── Output ──────────────────────────────────────────────────────────────────


def build_universe_json(results: list[dict], source: str) -> dict:
    """Sort results by market cap descending, take top 1200, build output dict."""
    results.sort(key=lambda x: x["market_cap"], reverse=True)

    top = results[:1200]

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
    args = parser.parse_args()

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
