"""Plain typed SEC fundamentals tools for a LangChain agent, gated on the strategy clock."""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_FILINGS_LIMIT = 1
MAX_FILINGS_LIMIT = 25


def _default_client(strategy: "Strategy") -> SecEdgarClient:  # noqa: UP037
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        raise ConfigurationError("Missing or blank SEC_EDGAR_USER_AGENT environment variable")
    return SecEdgarClient(user_agent, strategy.project_root / "cache" / "sec")


def fundamentals_tools(
    strategy: "Strategy", *, client: SecEdgarClient | None = None  # noqa: UP037
) -> list[Callable[..., dict[str, Any]]]:
    """SEC fundamentals tools bound to `strategy`, backed by one shared `SecEdgarClient`."""
    edgar = client if client is not None else _default_client(strategy)

    def _statement(symbol: str, tag_map: dict[str, list[str]]) -> dict[str, Any]:
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            payload = edgar.get_company_facts_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        values = sec.statement_values(payload, tag_map, as_of=as_of)
        return {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "values": values}

    def get_company_facts(symbol: str, max_facts: int = 40) -> dict[str, Any]:
        """Get a company's latest SEC XBRL facts (revenue, assets, EPS, ...) as of now."""
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            payload = edgar.get_company_facts_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        compact = sec.compact_company_facts(payload, as_of=as_of, max_facts=max_facts)
        return {"symbol": symbol.upper(), "cik": cik, "as_of": as_of.isoformat(), **compact}

    def get_income_statement(symbol: str) -> dict[str, Any]:
        """Get a company's income statement (revenue, net income, EPS, ...) as of now."""
        return _statement(symbol, sec.INCOME_STATEMENT_TAGS)

    def get_balance_sheet(symbol: str) -> dict[str, Any]:
        """Get a company's balance sheet (assets, liabilities, equity, ...) as of now."""
        return _statement(symbol, sec.BALANCE_SHEET_TAGS)

    def get_filings(symbol: str, form: str | None = None, limit: int = 10) -> dict[str, Any]:
        """List a company's recent SEC filings (10-K, 10-Q, 8-K, ...) as of now."""
        clamped_limit = min(max(int(limit), MIN_FILINGS_LIMIT), MAX_FILINGS_LIMIT)
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            submissions = edgar.get_submissions_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        rows = sec.parse_filings(submissions, cik=cik, as_of=as_of, form=form, limit=clamped_limit)
        filings = [{k: v for k, v in row.items() if k != "primary_document"} for row in rows]
        return {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "filings": filings}

    def get_filing_document(symbol: str, accession_number: str, max_chars: int = 8000) -> dict[str, Any]:
        """Fetch the readable text of one SEC filing document."""
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            submissions = edgar.get_submissions_payload(cik)
            rows = sec.parse_filings(submissions, cik=cik, as_of=as_of, limit=1000)
            match = next((row for row in rows if row["accession_number"] == accession_number), None)
            if match is None:
                return {"error": f"unknown accession_number {accession_number!r}"}
            raw = edgar.get_filing_text(cik, accession_number, match["primary_document"])
        except FundamentalsError as exc:
            return {"error": str(exc)}
        text = sec.strip_html(raw)
        truncated = len(text) > max_chars
        return {
            "symbol": symbol.upper(),
            "accession_number": accession_number,
            "document_url": match["document_url"],
            "text": text[:max_chars],
            "truncated": truncated,
        }

    return [get_company_facts, get_income_statement, get_balance_sheet, get_filings, get_filing_document]
