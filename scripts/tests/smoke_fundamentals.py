#!/usr/bin/env python3
"""Manual check of the SEC fundamentals tools against the live SEC EDGAR API.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_fundamentals.py

Needs SEC_EDGAR_USER_AGENT in env/.env.alpaca.integration-tests (or another loaded env file) and
a paper-trading Alpaca broker (only used to build a Strategy/clock; no orders are placed). First
run writes to <project_root>/cache/sec/; delete that directory to force a fresh fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-fundamentals"
SYMBOL = "AAPL"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)
    tools = {tool.__name__: tool for tool in fundamentals_tools(strategy)}  # type: ignore[unresolved-attribute]

    facts = tools["get_company_facts"](SYMBOL)
    print(f"get_company_facts({SYMBOL}) -> cik={facts.get('cik')} fact_count={facts.get('fact_count')}")
    _check("error" not in facts, f"get_company_facts returned an error: {facts.get('error')}")

    income = tools["get_income_statement"](SYMBOL)
    print(f"get_income_statement({SYMBOL}) -> {list(income.get('values', {}))}")
    _check("error" not in income, f"get_income_statement returned an error: {income.get('error')}")

    filings = tools["get_filings"](SYMBOL, form="10-K", limit=1)
    print(f"get_filings({SYMBOL}, 10-K) -> {filings.get('filings')}")
    _check("error" not in filings, f"get_filings returned an error: {filings.get('error')}")
    _check(len(filings["filings"]) > 0, "no 10-K filings found for AAPL")

    accession = filings["filings"][0]["accession_number"]
    document = tools["get_filing_document"](SYMBOL, accession, max_chars=500)
    print(f"get_filing_document({SYMBOL}, {accession}) -> {len(document.get('text', ''))} chars")
    _check("error" not in document, f"get_filing_document returned an error: {document.get('error')}")

    print("\nPASS: company facts, income statement, filings and a filing document all returned data.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
