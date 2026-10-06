#!/usr/bin/env python3
"""Manual check of earnings_drift's event detection against real SEC EDGAR and Alpaca news.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_earnings_events.py

`SEC_EDGAR_USER_AGENT` and `ALPACA_NEWS_API_*` come from env/.env.alpaca.integration-tests, as in the other smoke
scripts (or from the environment); without either the script prints `SKIP:` and exits 0, so the aggregate runner
`scripts/tests/run_smoke_tests.sh` reports it as skipped. The script is read-only. For a few large caps over the last 120
days it prints each 8-K item 2.02 event, its reaction date and the parsed Benzinga surprise, then the share of events
with a parsable headline (news is read as the scanner does: the reaction session, oldest first). It fails only when no
event is found at all or no surprise parses.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.event_source import EventSource
from trading_agent_framework.strategies.earnings_drift.events import news_window, reaction_date
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.surprise import articles_for, pick_surprise
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError, TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"
SYMBOLS = ["OMC", "AAPL", "MSFT", "JPM", "NFLX", "CAT", "KO", "UNH"]
LOOKBACK_DAYS = 120
NEWS_LIMIT = DriftParams().news_limit  # the scanner's per-query limit; it reads oldest first, over the reaction session (`news_window`)


def main() -> int:
    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=True)
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        print("SKIP: SEC_EDGAR_USER_AGENT is not set")
        return 0
    try:
        news_credentials = AlpacaCredentials.for_news()
    except ConfigurationError as exc:
        print(f"SKIP: no Alpaca news credentials ({exc})")
        return 0
    news = AlpacaNewsProvider.from_credentials(news_credentials)

    now = datetime.now(UTC)
    since = (now - timedelta(days=LOOKBACK_DAYS)).date()
    source = EventSource(SecEdgarClient(user_agent, PROJECT_ROOT / "cache" / "sec"), reload_every_cycle=True)
    report = source.load(SYMBOLS, as_of=now, since=since)
    print(f"SEC: {report.loaded} symbols loaded, failed: {report.failed or 'none'}")
    events = [event for event in source.all_events() if event.accepted_at.date() >= since]
    if not events:
        print(f"FAIL: no 8-K item 2.02 event in the last {LOOKBACK_DAYS} days")
        return 1

    # Weekdays stand in for the trading dates: holidays are ignored, the reaction dates are printed for a reader, not asserted.
    weekdays = [day for day in (since + timedelta(days=i) for i in range((now.date() - since).days + 2)) if day.weekday() < 5]
    parsed = 0
    for event in sorted(events, key=lambda e: e.accepted_at):
        when = event.accepted_at.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M")
        reacts = reaction_date(event.accepted_at, weekdays)
        window = news_window(reacts, weekdays, now) if reacts is not None else None
        if window is None:
            print(f"{event.symbol:5} {when} -> reacts {reacts}: no news window")
            continue
        start, end = window  # as the scanner: the reaction session's, not anchored on the 8-K's (late) acceptance time
        articles = news.get_news([event.symbol], start=start, end=end, limit=NEWS_LIMIT, sort="asc")
        picked = pick_surprise(articles_for(articles, event.symbol, start=start, end=end))
        parsed += picked is not None
        if picked is None:
            print(f"{event.symbol:5} {when} -> reacts {reacts}: no surprise headline")
        else:
            s = picked.surprise
            print(f"{event.symbol:5} {when} -> reacts {reacts}: EPS {s.eps_actual} vs {s.eps_estimate} {s.eps_result}, sales {s.sales_result} | {picked.headline}")
    print(f"\n{parsed}/{len(events)} events with a parsed surprise ({parsed / len(events):.0%})")
    if parsed == 0:
        print("FAIL: no surprise headline parsed")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
