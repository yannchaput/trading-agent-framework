from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing, Form4Trade, Owner
from trading_agent_framework.strategies.insider_trades.form4.source import KnownInsiderData
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.strategies.insider_trades.tools import insider_research_tools
from trading_agent_framework.utils.errors import InsiderDataError

TODAY = date(2025, 10, 14)
COOK = Owner("0001", "COOK TIMOTHY D", False, True, "CEO", False, False)


def _trade(
    accession: str,
    *,
    filed: date,
    traded: date | None = None,
    code: str = "S",
    shares: int = 100,
    price: str = "10",
    owned_after: int = 900,
    plan: bool = False,
    cik: str = "0001",
    name: str = "COOK TIMOTHY D",
) -> Form4Trade:
    return Form4Trade(accession, filed, "AAPL", cik, name, "CEO", code, traded or filed, Decimal(shares), Decimal(price), Decimal(owned_after), plan, ())


def _filing(accession: str, trades: tuple[Form4Trade, ...], *, filed: date, footnotes: dict[str, str] | None = None, plan: bool = False) -> Form4Filing:
    return Form4Filing(
        accession=accession,
        filed=filed,
        ticker="AAPL",
        issuer_cik="0000320193",
        issuer_symbol="AAPL",
        owners=(COOK, Owner("0002", "ARA FUND LP", False, False, "", True, False)),
        attributed_to=COOK,
        co_filers=("ARA FUND LP",),
        trades=trades,
        plan=plan,
        other_codes=0,
        unpriced=0,
        non_insider=0,
        footnotes=footnotes or {},
    )


def _history() -> list[Form4Filing]:
    """Cook sold in March 2022, June 2023, September 2024 and March 2025: every year, never the same month."""
    return [
        _filing("H22", (_trade("H22", filed=date(2022, 3, 4)),), filed=date(2022, 3, 4)),
        _filing("H23", (_trade("H23", filed=date(2023, 6, 6)),), filed=date(2023, 6, 6)),
        _filing("H24", (_trade("H24", filed=date(2024, 9, 5)),), filed=date(2024, 9, 5)),
        _filing("H25", (_trade("H25", filed=date(2025, 3, 5)),), filed=date(2025, 3, 5)),
    ]


class FakeSource:
    def __init__(self, filings: dict[str, list[Form4Filing]], failed: dict[str, str] | None = None) -> None:
        self.data = KnownInsiderData(filings, failed or {})
        self.calls: list[datetime] = []

    def known(self, as_of: datetime) -> KnownInsiderData:
        self.calls.append(as_of)
        return self.data


def _tools(source: FakeSource, *, clock: FakeClock | None = None, fetched: list[str] | None = None, params: InsiderParams | None = None):
    clock = clock or FakeClock(et(2025, 10, 14, 10))
    strategy = Strategy(FakeBroker(clock))
    facts = InsiderFacts(strategy, source, params or InsiderParams())
    log = fetched if fetched is not None else []
    get_insider_trades, read_form4 = insider_research_tools(strategy, facts, log.append)
    return get_insider_trades, read_form4, facts


def test_get_insider_trades_returns_code_labels_with_their_evidence() -> None:
    recent = _filing(
        "R1",
        (
            _trade("R1", filed=date(2025, 10, 6), traded=date(2025, 10, 2), code="P", shares=50, price="12.5", owned_after=550),
            _trade("R1", filed=date(2025, 10, 6), traded=date(2025, 10, 2), code="S", shares=30, price="12.5", owned_after=500, plan=True),
        ),
        filed=date(2025, 10, 6),
    )
    get_insider_trades, _, _ = _tools(FakeSource({"AAPL": [*_history(), recent]}))

    result = get_insider_trades("aapl")

    assert result["ticker"] == "AAPL"
    assert result["as_of"] == "2025-10-14"
    assert result["window_start"] == "2025-09-14"
    assert result["count"] == 1
    cook = result["insiders"][0]
    assert cook["insider_id"] == "0001"
    assert cook["name"] == "COOK TIMOTHY D"
    assert cook["role"] == "CEO"
    # only the October 2025 trades fall in the window; the older filings are history. Cook traded in each of 2022-2024 but never in a
    # month common to all three, so the purchase is opportunistic; the sale is routine because its 10b5-1 box is checked
    assert cook["bought"] == {"opportunistic": 625}
    assert cook["sold"] == {"routine": 375}
    assert cook["max_buy_fraction"] == round(50 / 550, 4)
    assert cook["sell_fraction"] == 0.0
    assert cook["months_by_year"] == {"2022": [3], "2023": [6], "2024": [9], "2025": [3, 10]}
    purchase, sale = cook["trades"]
    assert purchase == {
        "accession": "R1",
        "filed": "2025-10-06",
        "traded": "2025-10-02",
        "code": "P",
        "shares": 50,
        "price": 12.5,
        "label": "opportunistic",
        "rule": "opportunistic: traded in each of 2022, 2023, 2024, never in the same month every year",
    }
    assert sale["label"] == "routine"
    assert sale["rule"] == "plan: the 10b5-1 box is checked"
    assert "more_trades" not in cook


def test_a_filing_dated_today_is_invisible_even_if_the_source_leaks_it() -> None:
    recent = _filing("R1", (_trade("R1", filed=date(2025, 10, 6), traded=date(2025, 10, 2), code="P", shares=50, price="12.5", owned_after=550),), filed=date(2025, 10, 6))
    # a late filing: filed today for a July trade. `window_trades` would drop it from the window anyway, but unguarded it would still
    # reach the insider's months and the labelling history
    leaked = _filing("T1", (_trade("T1", filed=TODAY, traded=date(2025, 7, 1), code="P"),), filed=TODAY)
    get_insider_trades, read_form4, _ = _tools(FakeSource({"AAPL": [*_history(), recent, leaked]}))

    cook = get_insider_trades("AAPL")["insiders"][0]

    assert cook["months_by_year"] == {"2022": [3], "2023": [6], "2024": [9], "2025": [3, 10]}
    assert [t["accession"] for t in cook["trades"]] == ["R1"]
    assert cook["trades"][0]["label"] == "opportunistic"
    assert cook["trades"][0]["rule"] == "opportunistic: traded in each of 2022, 2023, 2024, never in the same month every year"
    assert "error" in read_form4("T1")


def test_read_form4_returns_the_known_filing() -> None:
    filing = _filing(
        "R1",
        (_trade("R1", filed=date(2025, 10, 6), traded=date(2025, 10, 2), code="P", shares=50, price="12.5", owned_after=550),),
        filed=date(2025, 10, 6),
        footnotes={"F1": "bought on the open market"},
        plan=True,
    )
    _, read_form4, _ = _tools(FakeSource({"AAPL": [filing]}))

    assert read_form4(" R1 ") == {
        "accession": "R1",
        "filed": "2025-10-06",
        "owners": [{"name": "COOK TIMOTHY D", "role": "CEO"}, {"name": "ARA FUND LP", "role": "10% owner"}],
        "attributed_to": "COOK TIMOTHY D",
        "co_filers": ["ARA FUND LP"],
        "plan": True,
        "rows": [{"code": "P", "traded": "2025-10-02", "shares": 50, "price": 12.5, "owned_after": 550}],
        "footnotes": {"F1": "bought on the open market"},
    }


def test_read_form4_refuses_an_accession_filed_today() -> None:
    leaked = _filing("T1", (_trade("T1", filed=TODAY),), filed=TODAY)
    _, read_form4, _ = _tools(FakeSource({"AAPL": [leaked]}))

    assert read_form4("T1") == {"error": "T1 is not a known Form 4: call get_insider_trades for the filings filed before 2025-10-14"}
    assert read_form4("nope")["error"].startswith("nope is not a known Form 4")


def test_footnotes_are_capped_at_500_characters() -> None:
    filing = _filing("R1", (_trade("R1", filed=date(2025, 10, 6)),), filed=date(2025, 10, 6), footnotes={"F1": "x" * 900})
    _, read_form4, _ = _tools(FakeSource({"AAPL": [filing]}))

    assert read_form4("R1")["footnotes"] == {"F1": "x" * 500}


def test_more_than_ten_trades_per_insider_are_cut_with_a_count() -> None:
    filings = [_filing(f"R{i}", (_trade(f"R{i}", filed=date(2025, 10, 1 + i % 10), shares=1, owned_after=999),), filed=date(2025, 10, 1 + i % 10)) for i in range(13)]
    get_insider_trades, _, _ = _tools(FakeSource({"AAPL": filings}))

    cook = get_insider_trades("AAPL")["insiders"][0]

    assert len(cook["trades"]) == 10
    assert cook["more_trades"] == 3
    assert cook["sold"] == {"unclassified": 130}  # the cut hides rows, never dollars


def test_a_ticker_off_the_watchlist_is_refused() -> None:
    get_insider_trades, _, _ = _tools(FakeSource({}), params=InsiderParams(watchlist=("AAPL", "MSFT"), max_total_weight=0.4))

    assert get_insider_trades("TSLA") == {"error": "TSLA is not on the watchlist: AAPL, MSFT"}


def test_a_failed_ticker_answers_with_an_error_not_an_exception() -> None:
    fetched: list[str] = []
    get_insider_trades, _, facts = _tools(FakeSource({}, {"AAPL": "SEC throttled"}), fetched=fetched)

    result = get_insider_trades("AAPL")

    assert "AAPL" in result["error"] and "SEC throttled" in result["error"]
    assert fetched == []
    for method in (facts.labelled, facts.window, facts.facts):
        with pytest.raises(InsiderDataError, match="AAPL.*SEC throttled"):
            method("AAPL")


def test_on_fetch_is_called_with_the_ticker_only_on_success() -> None:
    fetched: list[str] = []
    get_insider_trades, _, _ = _tools(FakeSource({"AAPL": _history()}), fetched=fetched)

    get_insider_trades("TSLA")
    assert fetched == []
    get_insider_trades("aapl")
    assert fetched == ["AAPL"]


def test_a_watchlist_ticker_absent_from_the_source_has_no_filings_and_the_source_is_read_once_a_day() -> None:
    source = FakeSource({})
    clock = FakeClock(et(2025, 10, 14, 10))
    get_insider_trades, _, facts = _tools(source, clock=clock)

    assert get_insider_trades("AAPL")["insiders"] == []
    assert facts.labelled("AAPL") == []
    assert len(source.calls) == 1

    clock.advance(24 * 3600)
    assert facts.today() == date(2025, 10, 15)
    facts.known()
    assert len(source.calls) == 2
