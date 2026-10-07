from __future__ import annotations

import inspect
from datetime import date, datetime
from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.congress import congress_research_tools
from trading_agent_framework.congress.annual import AssetHolding
from trading_agent_framework.congress.ptr import FilingRef, Transaction
from trading_agent_framework.congress.source import KnownFilings
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.utils.errors import CongressDataError

ANNUAL = FilingRef("A1", "Nancy Pelosi", "annual", date(2026, 5, 15), 2025)
PTR_OLD = FilingRef("P1", "Nancy Pelosi", "ptr", date(2026, 6, 23), 2026)
PTR_NEW = FilingRef("P2", "Nancy Pelosi", "ptr", date(2026, 9, 10), 2026)
PTR_EMPTY = FilingRef("P3", "Nancy Pelosi", "ptr", date(2026, 9, 11), 2026)
PTR_TODAY = FilingRef("P4", "Nancy Pelosi", "ptr", date(2026, 9, 14), 2026)  # filed on the strategy's date: not known yet


def _asset(ticker: str, low: int, high: int, owner: str = "spouse") -> AssetHolding:
    return AssetHolding("A1", owner, ticker, f"{ticker} Inc.", Decimal(low), Decimal(high), 8)


def _trade(doc_id: str, ticker: str, side: str, filed: date) -> Transaction:
    return Transaction(doc_id, "spouse", ticker, f"{ticker} Inc.", side, date(2026, 6, 20), date(2026, 6, 21), Decimal(250001), Decimal(500000), filed)


def _known(*, extra_refs: tuple[FilingRef, ...] = ()) -> KnownFilings:
    return KnownFilings(
        annual_ref=ANNUAL,
        period_end=date(2025, 12, 31),
        assets=[_asset("AAPL", 5_000_001, 25_000_000), _asset("NVDA", 1_000_001, 5_000_000, owner="joint")],
        transactions=[_trade("P1", "BE", "buy", PTR_OLD.filed), _trade("P2", "AAPL", "sell_partial", PTR_NEW.filed)],
        refs=[*extra_refs, PTR_EMPTY, PTR_NEW, PTR_OLD, ANNUAL],
        unparsed_filings=1,
        skipped_non_stock=79,
    )


class FakeSource:
    def __init__(self, known: KnownFilings | Exception | None = None) -> None:
        self.result = known if known is not None else _known()
        self.calls: list[datetime] = []

    def known(self, as_of: datetime) -> KnownFilings:
        self.calls.append(as_of)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _tools(source: FakeSource, clock: FakeClock | None = None):
    clock = clock or FakeClock(et(2026, 9, 14, 10))
    list_filings, read_filing = congress_research_tools(Strategy(FakeBroker(clock)), source)
    return list_filings, read_filing, clock


# --- list_filings ---------------------------------------------------------------------------------


def test_list_filings_returns_rows_newest_first_with_the_base_report() -> None:
    list_filings, _, _ = _tools(FakeSource())

    result = list_filings()

    assert result["as_of"] == "2026-09-14"
    assert result["base_report"] == "A1"
    assert result["period_end"] == "2025-12-31"
    assert [f["doc_id"] for f in result["filings"]] == ["P3", "P2", "P1", "A1"]
    assert result["count"] == 4
    assert result["filings"][3] == {"doc_id": "A1", "kind": "annual", "filed": "2026-05-15", "year": 2025}
    assert (result["unparsed_filings"], result["skipped_non_stock"]) == (1, 79)


def test_list_filings_drops_a_filing_dated_today_or_later_even_if_the_source_returns_it() -> None:
    list_filings, _, _ = _tools(FakeSource(_known(extra_refs=(PTR_TODAY,))))

    result = list_filings()

    assert "P4" not in {f["doc_id"] for f in result["filings"]}
    assert result["count"] == 4


def test_list_filings_asks_the_source_as_of_the_strategy_clock() -> None:
    source = FakeSource()
    list_filings, _, clock = _tools(source)

    list_filings()

    assert source.calls == [clock.now()]


def test_a_source_failure_is_an_error_payload_not_an_exception() -> None:
    list_filings, read_filing, _ = _tools(FakeSource(CongressDataError("No readable yearly report")))

    assert list_filings() == {"error": "No readable yearly report"}
    assert read_filing("A1") == {"error": "No readable yearly report"}


# --- read_filing ----------------------------------------------------------------------------------


def test_read_filing_of_the_yearly_report_returns_lean_holdings() -> None:
    _, read_filing, _ = _tools(FakeSource())

    result = read_filing("A1")

    assert result["kind"] == "annual"
    assert result["period_end"] == "2025-12-31"
    assert result["count"] == 2
    assert result["holdings"][0] == {"ticker": "AAPL", "owner": "spouse", "value_low": 5_000_001, "value_high": 25_000_000, "tier": 8}
    assert result["holdings"][1]["owner"] == "joint"
    assert all("name" not in h for h in result["holdings"])  # asset names are left out: tokens


def test_read_filing_of_a_trade_report_returns_its_stock_trades() -> None:
    _, read_filing, _ = _tools(FakeSource())

    result = read_filing("P2")

    assert result["kind"] == "ptr"
    assert result["filed"] == "2026-09-10"
    assert result["trades"] == [{"ticker": "AAPL", "side": "sell_partial", "owner": "spouse", "traded": "2026-06-20", "notified": "2026-06-21", "amount_low": 250001, "amount_high": 500000}]


def test_a_trade_report_with_no_stock_trade_is_an_empty_list_not_an_error() -> None:
    _, read_filing, _ = _tools(FakeSource())

    result = read_filing("P3")

    assert result["count"] == 0
    assert result["trades"] == []


def test_read_filing_refuses_a_doc_id_filed_today_or_later() -> None:
    _, read_filing, _ = _tools(FakeSource(_known(extra_refs=(PTR_TODAY,))))

    result = read_filing("P4")

    assert "error" in result
    assert "2026-09-14" in result["error"]


def test_read_filing_refuses_an_unknown_doc_id() -> None:
    _, read_filing, _ = _tools(FakeSource())

    assert "error" in read_filing("nope")
    assert "error" in read_filing("")


def test_read_filing_tolerates_surrounding_whitespace_in_the_doc_id() -> None:
    _, read_filing, _ = _tools(FakeSource())

    assert read_filing(" P2 ")["doc_id"] == "P2"


# --- caching and shape ----------------------------------------------------------------------------


def test_both_tools_share_one_lookup_per_market_date() -> None:
    source = FakeSource()
    list_filings, read_filing, clock = _tools(source)

    list_filings()
    read_filing("A1")
    read_filing("P2")
    assert len(source.calls) == 1

    clock.advance(24 * 3600)
    list_filings()
    assert len(source.calls) == 2


def test_an_identical_call_in_one_run_reuses_the_first_result() -> None:
    source = FakeSource()
    list_filings, read_filing, _ = _tools(source)

    with agent_call_context(run_id="run-1"):
        first = read_filing("A1")
        again = read_filing("A1")
        list_filings()

    assert again == first
    assert len(source.calls) == 1


def test_a_failed_lookup_is_retried_on_the_next_call_even_within_a_run() -> None:
    source = FakeSource(CongressDataError("boom"))
    list_filings, _, _ = _tools(source)

    with agent_call_context(run_id="run-1"):
        assert "error" in list_filings()
        source.result = _known()
        assert "error" not in list_filings()


def test_tools_have_one_line_docstrings_and_real_annotations() -> None:
    list_filings, read_filing, _ = _tools(FakeSource())

    for tool in (list_filings, read_filing):
        assert tool.__doc__ is not None
        assert "\n" not in tool.__doc__.strip()
    assert inspect.signature(read_filing).parameters["doc_id"].annotation is str  # not the string "str": no `from __future__ import annotations`
    assert [t.__name__ for t in (list_filings, read_filing)] == ["list_filings", "read_filing"]
