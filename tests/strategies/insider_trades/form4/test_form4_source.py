from __future__ import annotations

from datetime import date
from pathlib import Path

from tests.fakes import et

from trading_agent_framework.strategies.insider_trades.form4.source import Form4Source, KnownInsiderData, LoadReport
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError

FIXTURES = Path(__file__).parent / "fixtures"
SALE = (FIXTURES / "sale_ceo.xml").read_text()  # issuer 0000320193 (AAPL)
PURCHASE = (FIXTURES / "purchase_director.xml").read_text()
AS_OF = et(2025, 10, 14, 10)
SINCE = date(2022, 1, 1)
CIKS = {"AAPL": "320193", "JPM": "19617"}


def _recent(rows: list[tuple[str, str, str, str]], files: list[dict[str, str]] | None = None) -> dict:
    """rows: (form, filingDate, accession, primaryDocument)."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "accessionNumber": [r[2] for r in rows],
                "primaryDocument": [r[3] for r in rows],
            },
            "files": files or [],
        }
    }


def _page(rows: list[tuple[str, str, str, str]]) -> dict:
    return _recent(rows)["filings"]["recent"]


class FakeClient:
    """The slice of `SecEdgarClient` the source uses; every call is recorded."""

    def __init__(self, submissions: dict[str, dict], pages: dict[str, dict] | None = None, xml: dict[str, str] | None = None, errors: dict[str, Exception] | None = None) -> None:
        self.submissions = submissions
        self.pages = pages or {}
        self.xml = xml or {}
        self.errors = errors or {}
        self.text_requests: list[tuple[str, str, str]] = []
        self.page_requests: list[str] = []
        self.payload_requests: list[tuple[str, object, object]] = []

    def ticker_to_cik(self, symbol: str) -> str:
        if symbol not in CIKS:
            raise FundamentalsNotFoundError(f"No CIK for {symbol}")
        return CIKS[symbol]

    def get_submissions_payload(self, cik: str, *, as_of=None, max_age_days=None) -> dict:
        self.payload_requests.append((cik, as_of, max_age_days))
        if cik in self.errors:
            raise self.errors[cik]
        return self.submissions[cik]

    def get_submissions_page(self, name: str) -> dict:
        self.page_requests.append(name)
        return self.pages[name]

    def get_filing_text(self, cik: str, accession: str, primary_document: str) -> str:
        self.text_requests.append((cik, accession, primary_document))
        return self.xml.get(accession, SALE)  # any filing not given is the AAPL sale


def _aapl(rows, *, xml=None, **kwargs) -> tuple[Form4Source, FakeClient]:
    client = FakeClient({"320193": _recent(rows, kwargs.pop("files", None))}, xml=xml, **kwargs)
    return Form4Source(client, ["AAPL"]), client  # type: ignore[arg-type]


def test_only_form_4_filed_since_the_history_start_is_read_across_recent_and_older_pages():
    files = [
        {"name": "CIK0000320193-submissions-001.json", "filingFrom": "2015-01-01", "filingTo": "2024-12-31"},
        {"name": "CIK0000320193-submissions-002.json", "filingFrom": "2005-01-01", "filingTo": "2021-12-31"},
    ]
    source, client = _aapl(
        [("4", "2025-10-03", "0001-25-000003", "xslF345X05/wk-form4_3.xml"), ("8-K", "2025-10-02", "0001-25-000002", "x.htm"), ("4", "2021-12-30", "0001-21-000001", "wk-form4_0.xml")],
        files=files,
        pages={"CIK0000320193-submissions-001.json": _page([("4", "2023-05-02", "0001-23-000002", "xslF345X05/wk-form4_2.xml"), ("4", "2022-01-03", "0001-22-000001", "wk-form4_1.xml")])},
    )
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report == LoadReport(loaded=("AAPL",), failed={}, ignored_forms=0, other_issuer=0)
    assert client.page_requests == ["CIK0000320193-submissions-001.json"]  # page 002 ends before `since`
    assert [accession for _, accession, _ in client.text_requests] == ["0001-25-000003", "0001-23-000002", "0001-22-000001"]  # the 2021 row and the 8-K are never fetched
    assert client.payload_requests == [("320193", AS_OF, 0)]
    filings = source.known(et(2025, 10, 14, 10)).filings["AAPL"]
    assert [(f.accession, f.filed, f.ticker) for f in filings] == [
        ("0001-22-000001", date(2022, 1, 3), "AAPL"),
        ("0001-23-000002", date(2023, 5, 2), "AAPL"),
        ("0001-25-000003", date(2025, 10, 3), "AAPL"),
    ]


def test_amendments_and_form_5_are_counted_not_read():
    source, client = _aapl(
        [("4", "2025-10-03", "a1", "f1.xml"), ("4/A", "2025-10-04", "a2", "f2.xml"), ("5", "2025-10-05", "a3", "f3.xml"), ("5/A", "2025-10-06", "a4", "f4.xml"), ("5", "2020-01-02", "a5", "f5.xml")]
    )
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.ignored_forms == 3  # the 2020 form 5 predates `since`: out of scope, not counted
    assert [accession for _, accession, _ in client.text_requests] == ["a1"]


def test_known_keeps_filings_filed_strictly_before_the_market_date():
    source, _ = _aapl([("4", "2025-10-13", "a1", "f1.xml"), ("4", "2025-10-14", "a2", "f2.xml")])
    source.load(as_of=et(2025, 10, 20, 10), since=SINCE)
    assert [f.accession for f in source.known(et(2025, 10, 14, 10)).filings["AAPL"]] == ["a1"]
    assert [f.accession for f in source.known(et(2025, 10, 14, 23, 59)).filings["AAPL"]] == ["a1"]  # the market date, not the UTC one
    assert [f.accession for f in source.known(et(2025, 10, 15, 0, 1)).filings["AAPL"]] == ["a1", "a2"]
    assert source.known(et(2025, 10, 13, 10)).filings["AAPL"] == []


def test_known_is_a_filter_of_the_last_load_and_loaded_on_is_its_market_date():
    source, client = _aapl([("4", "2025-10-13", "a1", "f1.xml")])
    assert source.loaded_on is None
    assert source.known(AS_OF) == KnownInsiderData(filings={}, failed={})
    source.load(as_of=et(2025, 10, 14, 23, 30), since=SINCE)
    assert source.loaded_on == date(2025, 10, 14)
    calls = len(client.text_requests)
    source.known(AS_OF)
    source.known(et(2025, 12, 1))
    assert len(client.text_requests) == calls and len(client.payload_requests) == 1


def test_the_raw_xml_is_fetched_not_the_xsl_rendering():
    source, client = _aapl([("4", "2025-10-03", "0001-25-000001", "xslF345X05/wk-form4_1.xml")])
    source.load(as_of=AS_OF, since=SINCE)
    assert client.text_requests == [("320193", "0001-25-000001", "wk-form4_1.xml")]


def test_a_ticker_that_fails_is_reported_and_absent_while_the_others_load():
    client = FakeClient(
        {"320193": _recent([("4", "2025-10-03", "a1", "f1.xml")]), "19617": _recent([])},
        xml={"a1": SALE},
        errors={"19617": FundamentalsError("403")},
    )
    source = Form4Source(client, ["AAPL", "JPM"])  # type: ignore[arg-type]
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.loaded == ("AAPL",) and report.failed == {"JPM": "403"}
    known = source.known(et(2025, 10, 14, 10))
    assert list(known.filings) == ["AAPL"] and known.failed == {"JPM": "403"}


def test_an_unknown_ticker_fails_that_ticker_only():
    client = FakeClient({"320193": _recent([("4", "2025-10-03", "a1", "f1.xml")])}, xml={"a1": SALE})
    report = Form4Source(client, ["NOPE", "AAPL"]).load(as_of=AS_OF, since=SINCE)  # type: ignore[arg-type]
    assert report.loaded == ("AAPL",) and report.failed == {"NOPE": "No CIK for NOPE"}


def test_a_ticker_with_one_bad_filing_loses_all_its_filings():
    source, _ = _aapl([("4", "2025-10-03", "a1", "f1.xml"), ("4", "2025-10-02", "a2", "f2.xml")], xml={"a1": SALE, "a2": "<html>broken</html>"})
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.loaded == () and list(report.failed) == ["AAPL"]
    assert source.known(et(2025, 10, 14, 10)).filings == {}


def test_a_throttle_page_instead_of_xml_fails_the_ticker_not_the_load():
    client = FakeClient(
        {"320193": _recent([("4", "2025-10-03", "a1", "f1.xml")]), "19617": _recent([("4", "2025-10-03", "a2", "f2.xml")])},
        xml={"a1": "<html><body>Request Rate Threshold Exceeded</body></html>", "a2": SALE},
    )
    source = Form4Source(client, ["AAPL", "JPM"])  # type: ignore[arg-type]
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.loaded == ("JPM",) and list(report.failed) == ["AAPL"]
    assert "ownershipDocument" in report.failed["AAPL"]


def test_a_filing_of_another_issuer_is_counted_and_not_known():
    # the watchlist company is only a reporting owner of the other issuer's shares
    other = SALE.replace("<issuerCik>0000320193</issuerCik>", "<issuerCik>0000789019</issuerCik>")
    source, _ = _aapl([("4", "2025-10-03", "a1", "f1.xml"), ("4", "2025-10-02", "a2", "f2.xml")], xml={"a1": other, "a2": SALE})
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.other_issuer == 1 and report.loaded == ("AAPL",)
    assert [f.accession for f in source.known(et(2025, 10, 14, 10)).filings["AAPL"]] == ["a2"]


def test_the_issuer_cik_is_compared_as_a_number_not_as_text():
    source, _ = _aapl([("4", "2025-10-03", "a1", "f1.xml")], xml={"a1": SALE.replace("0000320193", "320193")})
    report = source.load(as_of=AS_OF, since=SINCE)
    assert report.other_issuer == 0 and len(source.known(et(2025, 10, 14, 10)).filings["AAPL"]) == 1


def test_a_later_load_replaces_the_earlier_one():
    source, client = _aapl([("4", "2025-10-03", "a1", "f1.xml")])
    source.load(as_of=AS_OF, since=SINCE)
    client.errors["320193"] = FundamentalsError("429")
    source.load(as_of=et(2025, 10, 15, 10), since=SINCE)
    known = source.known(et(2025, 10, 16, 10))
    assert known.filings == {} and known.failed == {"AAPL": "429"} and source.loaded_on == date(2025, 10, 15)
