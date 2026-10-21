#!/usr/bin/env python3
"""Manual check of insider_trades' Form 4 reading against real SEC EDGAR.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_insider_trades.py

`SEC_EDGAR_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts (or from the
environment); without it the script prints `SKIP:` and exits 0, so the aggregate runner `scripts/tests/run_smoke_tests.sh`
reports it as skipped. The script is read-only (it fills the SEC cache under `cache/sec/`). For AAPL and JPM it loads the Form 4
filings since 1 January three years back, then prints the load report, the open-market purchases and sales per year, five labelled
trades of the last 90 days with their rule, and the path of the first cached raw XML file. It fails when a ticker does not load,
no sale parses with a price, or the cached file is not an `ownershipDocument`.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.insider_trades.form4.classify import label_trades
from trading_agent_framework.strategies.insider_trades.form4.source import Form4Source
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.utils.errors import TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
CACHE_DIR = PROJECT_ROOT / "cache" / "sec"
TICKERS = ["AAPL", "JPM"]
HISTORY_YEARS = InsiderParams().history_years
RECENT_DAYS = 90
SHOWN = 5


def main() -> int:
    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=True)
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        print("SKIP: SEC_EDGAR_USER_AGENT is not set")
        return 0

    now = datetime.now(UTC)
    since = date(now.year - HISTORY_YEARS, 1, 1)
    source = Form4Source(SecEdgarClient(user_agent, CACHE_DIR), TICKERS)
    report = source.load(as_of=now, since=since)
    print(f"SEC: since {since}, loaded {list(report.loaded)}, failed: {report.failed or 'none'}, ignored forms: {report.ignored_forms}, other issuer: {report.other_issuer}")
    known = source.known(now + timedelta(days=1))  # every filing up to today
    if set(known.filings) != set(TICKERS):
        print(f"FAIL: not every ticker loaded ({known.failed})")
        return 1

    priced_sales = 0
    first_accession = ""
    for ticker in TICKERS:
        filings = known.filings[ticker]
        trades = [trade for filing in filings for trade in filing.trades]
        per_year = Counter((trade.trade_date.year, trade.code) for trade in trades)
        counts = ", ".join(f"{year}: {per_year[year, 'P']}P/{per_year[year, 'S']}S" for year in sorted({year for year, _ in per_year}))
        print(f"\n{ticker}: {len(filings)} Form 4 filings, {len(trades)} open-market trades ({counts})")
        priced_sales += sum(1 for trade in trades if trade.code == "S" and trade.price > 0)
        first_accession = first_accession or next((filing.accession for filing in filings), "")
        recent = [trade for trade in trades if trade.filed >= now.date() - timedelta(days=RECENT_DAYS)]
        for labelled in label_trades(recent[-SHOWN:], trades, years=HISTORY_YEARS):
            t = labelled.trade
            print(f"  {t.filed} {t.owner_name} ({t.role}) {t.code} {t.shares} @ {t.price} -> {labelled.label}: {labelled.rule}")

    if priced_sales == 0:
        print("FAIL: no sale parsed with a price")
        return 1
    cached = next(CACHE_DIR.glob(f"filings/*/{first_accession}/*"), None)
    if cached is None:
        print(f"FAIL: no cached file for {first_accession}")
        return 1
    head = cached.read_text(encoding="utf-8", errors="replace")[:400]
    print(f"\ncached raw XML: {cached}")
    if "<ownershipDocument" not in head:
        print(f"FAIL: the cached file is not an ownershipDocument: {head[:120]!r}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
