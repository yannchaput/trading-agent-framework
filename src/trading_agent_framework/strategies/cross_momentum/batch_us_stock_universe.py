"""
Batch Stock Universe Builder
=============================
Runs once per month to compute the investable US stock universe.

Source: NASDAQ public screener API — all stocks listed on NASDAQ, NYSE,
        and AMEX (equivalent coverage to VTI / CRSP US Total Market Index).
        Free, no API keys required.

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

_NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true&exchange=nasdaq&exchange=nyse&exchange=amex"
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

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


def get_ticker_universe() -> list[str]:
    """Fetch all US exchange-listed stock symbols from the NASDAQ screener API.

    A single API call returns every stock on NASDAQ, NYSE, and AMEX
    (~4,100 symbols) — equivalent in coverage to VTI's CRSP US Total
    Market Index.

    Pre-filters by market cap (>$2B) and price (>$10) using the NASDAQ
    data to reduce the yfinance workload (cuts ~4,100 → ~1,200-1,500).

    Returns:
        List of ticker symbols sorted by market cap descending.
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

    # Pre-filter: use NASDAQ data as a fast-path skip only when we can
    # positively determine a stock fails.  NASDAQ's marketCap is often
    # stale or $0.00 for mid/small-caps — in those cases we keep the
    # symbol and let yfinance re-check with live data.
    candidates: list[tuple[str, float]] = []  # (symbol, market_cap)
    skipped_mc = 0
    skipped_price = 0

    for row in rows:
        symbol = row.get("symbol", "").strip().upper()
        if not symbol or not symbol.replace("-", "").isalpha():
            continue

        # Market cap: only skip if we can parse it AND it's clearly < $2B.
        # $0.00 / empty / unparseable → keep (let yfinance verify).
        mc_str = row.get("marketCap", "")
        market_cap = _parse_nasdaq_market_cap(mc_str)
        if market_cap > 0 and market_cap < THRESHOLDS["market_cap"]:
            skipped_mc += 1
            continue

        # Price: only skip if it's parseable AND < $10.
        price_str = row.get("lastsale", "").replace("$", "")
        try:
            price = float(price_str)
        except ValueError, TypeError:
            price = 0.0
        if price > 0 and price < THRESHOLDS["price"]:
            skipped_price += 1
            continue

        candidates.append((symbol, market_cap))

    # Sort by market cap descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    symbols = [s for s, _ in candidates]
    console.print(f"  Got {len(rows)} total listed stocks → [green]{len(symbols)} candidates[/green] after pre-filter ({skipped_mc} skipped on market cap, {skipped_price} on price)")
    return symbols


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
    tickers = get_ticker_universe()
    if not tickers:
        console.print("[red]ERROR: No tickers retrieved. Aborting.[/red]")
        sys.exit(1)

    # Step 2: Screen with yfinance
    console.print(f"\n[bold]Step 2: Screening {len(tickers)} tickers with yfinance...[/bold]")
    console.print("  (This may take several minutes — monthly batch, acceptable runtime)")
    start = time.monotonic()
    # The max concurrent threads is set to _MAX_CONCURRENT_YF (3) to avoid yfinance rate-limiting issues.
    results = screen_tickers(tickers, max_workers=_MAX_CONCURRENT_YF)
    elapsed = time.monotonic() - start
    console.print(f"  Screening took {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if not results:
        console.print("[red]ERROR: No tickers passed screening. Aborting without overwriting existing files.[/red]")
        sys.exit(1)

    # Step 3: Build and save
    console.print("\n[bold]Step 3: Building universe JSON...[/bold]")
    data = build_universe_json(results, source="nasdaq_screener_all_us")
    save_universe(data, output_dir)

    console.print(f"\n[bold green]Done! Universe: {data['total_passed']} symbols saved.[/bold green]")
    console.print(f"[bold green]Top 10: {', '.join(data['symbols'][:10])}[/bold green]")


if __name__ == "__main__":
    main()
