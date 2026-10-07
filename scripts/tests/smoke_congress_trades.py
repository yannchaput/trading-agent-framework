#!/usr/bin/env python3
"""Manual run of the congress_trades data layer against the real House Clerk site.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_congress_trades.py [politician]

`CONGRESS_USER_AGENT` comes from env/.env.alpaca.integration-tests, as in the other smoke scripts (or from the
environment); without it the script prints `SKIP:` and exits 0, so `scripts/tests/run_smoke_tests.sh` reports it as
skipped. The script is read-only. It downloads the yearly filing indexes and the member's newest yearly report and
the PTRs since (cached under cache/house_clerk/ for the next run), prints what it parsed, and writes each filing's raw
extracted text to cache/house_clerk/smoke/<doc_id>.txt.

THIS IS THE FORMAT CHECKPOINT of the congress_trades plan (Task 5, step 7): the parsers in congress/ptr.py and
congress/annual.py were written against an assumed layout. Read the output and the raw text files, copy one yearly
report and one PTR (trimmed) into tests/congress/fixtures/{annual_real,ptr_real}.txt, and fix the parsers until they
pass on both. It fails only on things that must always hold: a yearly report is found, at least one stock holding
parses from it, and no filing known today is dated today or later.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.congress import holdings
from trading_agent_framework.congress.annual import VALUE_BANDS
from trading_agent_framework.congress.clerk_client import ClerkClient
from trading_agent_framework.congress.source import CongressSource
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
CACHE_DIR = PROJECT_ROOT / "cache" / "house_clerk"
DEFAULT_POLITICIAN = "Nancy Pelosi"


class SmokeTestFailure(Exception):
    pass


def main() -> int:
    load_dotenv(ENV_FILE)
    user_agent = os.environ.get("CONGRESS_USER_AGENT", "")
    if not user_agent.strip():
        # `run_smoke_tests.sh` reports a script whose output has a `SKIP:` line as skipped, not failed.
        print(f"SKIP: CONGRESS_USER_AGENT is not set (define it in {ENV_FILE} or in the environment)")
        return 0
    politician = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_POLITICIAN
    now = datetime.now(MARKET_TZ)
    print(f"Politician: {politician}; as of {now.isoformat(timespec='minutes')}")
    try:
        with ClerkClient(user_agent, CACHE_DIR) as client:
            known = CongressSource(client, politician).known(now)
            smoke_dir = CACHE_DIR / "smoke"
            smoke_dir.mkdir(parents=True, exist_ok=True)
            for ref in known.refs:
                (smoke_dir / f"{ref.doc_id}.txt").write_text(client.filing_text(ref), encoding="utf-8")
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}")
        return 1

    annual = known.annual_ref
    print(f"\nYearly report {annual.doc_id}: reporting year {annual.year}, filed {annual.filed}, period end {known.period_end}")
    print(f"  {len(known.assets)} stock holdings; PTRs since: {len(known.refs) - 1}; stock trades after the period end: {sum(1 for t in known.transactions if t.transaction_date > known.period_end)}")
    print(f"  unparsed (image-only) filings: {known.unparsed_filings}; rows skipped (options, bonds, unreadable): {known.skipped_non_stock}")
    print(f"  raw text of every known filing: {CACHE_DIR / 'smoke'}")

    print("\nNewest filings (newest first):")
    for ref in known.refs[:8]:
        print(f"  {ref.filed}  {ref.kind:6}  {ref.doc_id}")

    print("\nReconstructed holdings (ticker, tier band, estimated range):")
    current = holdings.reconstruct(known.assets, known.transactions, period_end=known.period_end)
    weights = holdings.baseline_weights(current, max_total=Decimal("0.95"), max_position=Decimal("0.15"))
    for holding in sorted(current, key=lambda h: h.midpoint, reverse=True):
        band = VALUE_BANDS[holding.tier].label
        weight = weights.get(holding.ticker, Decimal(0))
        print(f"  {holding.ticker:8} tier {holding.tier:2} ({band:28}) ${holding.value_low:>13,.0f} - ${holding.value_high:>13,.0f}  baseline weight {weight:.4f}")
    if not current:
        print("  (none)")

    problems = []
    if not known.assets:
        problems.append("the yearly report parsed to no stock holding: check congress/annual.py against the raw text")
    today = now.date()
    problems += [f"filing {ref.doc_id} is dated {ref.filed}, not before today" for ref in known.refs if ref.filed >= today]
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("\nOK (now compare the printed rows with the raw text files, row by row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
