from __future__ import annotations

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import FundamentalsError

_CIK = "0000320193"


class _FakeEdgarClient:
    def __init__(self) -> None:
        self.company_facts: dict[str, object] = {"facts": {"us-gaap": {}}}
        self.submissions: dict[str, object] = {"filings": {"recent": {}}}
        self.filing_text = "<p>Annual report text</p>"
        self.raises: FundamentalsError | None = None

    def ticker_to_cik(self, symbol: str) -> str:
        if self.raises is not None:
            raise self.raises
        return _CIK

    def get_company_facts_payload(self, cik: str) -> dict[str, object]:
        return self.company_facts

    def get_submissions_payload(self, cik: str) -> dict[str, object]:
        return self.submissions

    def get_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        return self.filing_text


def _strategy() -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))


def _tools(client: _FakeEdgarClient) -> dict[str, object]:
    return {tool.__name__: tool for tool in fundamentals_tools(_strategy(), client=client)}  # type: ignore


def test_returns_five_tools_with_one_line_docstrings() -> None:
    tools = fundamentals_tools(_strategy(), client=_FakeEdgarClient())  # type: ignore
    assert [t.__name__ for t in tools] == [  # type: ignore
        "get_company_facts", "get_income_statement", "get_balance_sheet", "get_filings", "get_filing_document",
    ]
    for tool in tools:
        assert len(tool.__doc__.splitlines()) == 1  # type: ignore


def test_get_company_facts_returns_symbol_cik_and_compact_facts() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_company_facts"]("aapl")  # type: ignore

    assert result["symbol"] == "AAPL"
    assert result["cik"] == _CIK
    assert "NetIncomeLoss" in result["facts"]


def test_get_company_facts_returns_error_on_unknown_ticker() -> None:
    client = _FakeEdgarClient()
    client.raises = FundamentalsError("No SEC CIK found for ticker 'ZZZZ'")
    tools = _tools(client)

    assert tools["get_company_facts"]("ZZZZ") == {"error": "No SEC CIK found for ticker 'ZZZZ'"}  # type: ignore


def test_get_income_statement_returns_values_for_the_symbol() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_income_statement"]("AAPL")  # type: ignore

    assert result["symbol"] == "AAPL"
    assert result["values"]["net_income"]["value"] == 10


def test_get_balance_sheet_returns_values_for_the_symbol() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"Assets": {"units": {"USD": [{"val": 999, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_balance_sheet"]("AAPL")  # type: ignore

    assert result["values"]["assets"]["value"] == 999


def test_get_filings_omits_primary_document_from_the_returned_rows() -> None:
    client = _FakeEdgarClient()
    client.submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-01"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm"],
            }
        }
    }
    tools = _tools(client)

    result = tools["get_filings"]("AAPL", form="10-K")  # type: ignore

    [filing] = result["filings"]
    assert "primary_document" not in filing
    assert filing["accession_number"] == "0000320193-26-000001"


def test_get_filing_document_strips_html_and_truncates() -> None:
    client = _FakeEdgarClient()
    client.submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["acc-1"],
                "filingDate": ["2026-01-01"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm"],
            }
        }
    }
    client.filing_text = "<p>" + "x" * 20 + "</p>"
    tools = _tools(client)

    result = tools["get_filing_document"]("AAPL", "acc-1", max_chars=5)  # type: ignore

    assert result["text"] == "xxxxx"
    assert result["truncated"] is True


def test_get_filing_document_unknown_accession_number_returns_error() -> None:
    tools = _tools(_FakeEdgarClient())

    result = tools["get_filing_document"]("AAPL", "does-not-exist")  # type: ignore

    assert "error" in result
