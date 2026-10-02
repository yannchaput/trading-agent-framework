"""The screen end to end on real-shaped data: a company-facts payload served through `httpx.MockTransport`
into the real `SecEdgarClient`, `AnnualFiguresStore`, `SplitHistory` and `QualityScreen`. Only the network
(SEC transport, the Yahoo fetch function) and the clocks are fakes."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.bill_ackman.screen import QualityScreen, ScreenParams
from trading_agent_framework.strategies.bill_ackman.screen.annual_store import AnnualFiguresStore
from trading_agent_framework.strategies.bill_ackman.screen.splits import Split, SplitHistory

NEW_YORK = ZoneInfo("America/New_York")
M = 1_000_000
CIK = "0000001234"
TICKERS = {"0": {"cik_str": 1234, "ticker": "ACME"}}
PRICE = Decimal("250")
WALL_CLOCK = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)

# Fiscal years 2018..2025, calendar years; each 10-K for FY f is filed on 15 February f+1 and repeats FY f-2..f.
YEARS = range(2018, 2026)
REVENUE = {2018: 80, 2019: 90, 2020: 100, 2021: 110, 2022: 120, 2023: 130, 2024: 140, 2025: 150}  # millions
RESTATED_2023_REVENUE = 135  # the FY2025 10-K restates FY2023, under another tag
OPERATING_INCOME = {year: REVENUE[year] // 5 for year in YEARS}  # 16 .. 30
OPERATING_CASH_FLOW = {year: 25 + 2 * (year - 2018) for year in YEARS}  # 25 .. 39
CAPEX = 5
NONCURRENT_DEBT = {year: 5 * (year - 2017) for year in YEARS}  # FY2024 35, FY2025 40
CURRENT_DEBT = {year: 10 if year == 2025 else 5 for year in YEARS}
CASH = {year: 10 if year == 2025 else 8 for year in YEARS}


def _filed(fiscal_year: int) -> str:
    return f"{fiscal_year + 1}-02-15"


def _flow(value: int, year: int, filed: str, *, form: str = "10-K") -> dict[str, Any]:
    return {"val": value * M, "start": f"{year}-01-01", "end": f"{year}-12-31", "filed": filed, "form": form}


def _instant(value: int, year: int, filed: str) -> dict[str, Any]:
    return {"val": value * M, "end": f"{year}-12-31", "filed": filed, "form": "10-K"}


def _usd(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"units": {"USD": rows}}


def _count(value: int, end: str, filed: str, *, form: str, start: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"val": value, "end": end, "filed": filed, "form": form}
    if start:
        row["start"] = start
    return row


def company_facts(*, cover_rows: list[dict[str, Any]], weighted_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Six 10-Ks (FY2020..FY2025), 10-Q noise, debt as noncurrent + current, and the given share rows."""
    revenues, restated, op_income, op_cash_flow, capex = [], [], [], [], []
    for fiscal_year in range(2020, 2026):
        filed = _filed(fiscal_year)
        for year in (fiscal_year - 2, fiscal_year - 1, fiscal_year):
            if year == 2023 and fiscal_year == 2025:
                restated.append(_flow(RESTATED_2023_REVENUE, year, filed))
            else:
                revenues.append(_flow(REVENUE[year], year, filed))
            op_income.append(_flow(OPERATING_INCOME[year], year, filed))
            op_cash_flow.append(_flow(OPERATING_CASH_FLOW[year], year, filed))
            capex.append(_flow(CAPEX, year, filed))
    noncurrent, current, cash = [], [], []
    for fiscal_year in range(2020, 2026):
        filed = _filed(fiscal_year)
        for year in (fiscal_year - 1, fiscal_year):
            noncurrent.append(_instant(NONCURRENT_DEBT[year], year, filed))
            current.append(_instant(CURRENT_DEBT[year], year, filed))
            cash.append(_instant(CASH[year], year, filed))
    # Quarterly reports repeat the same tags: a 3-month figure and a year-to-date one. None may be read as a year.
    quarter = {"start": "2025-07-01", "end": "2025-09-30", "filed": "2025-11-05", "form": "10-Q"}
    year_to_date = {"start": "2025-01-01", "end": "2025-09-30", "filed": "2025-11-05", "form": "10-Q"}
    for rows, value in ((revenues, 9_999), (op_income, 9_999), (op_cash_flow, 9_999), (capex, 9_999)):
        rows.extend([{**quarter, "val": value * M}, {**year_to_date, "val": value * 3 * M}])
    return {
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover_rows}}},
            "us-gaap": {
                "Revenues": _usd(revenues),
                "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(restated),
                "OperatingIncomeLoss": _usd(op_income),
                "NetCashProvidedByUsedInOperatingActivities": _usd(op_cash_flow),
                "PaymentsToAcquirePropertyPlantAndEquipment": _usd(capex),
                "LongTermDebtNoncurrent": _usd(noncurrent),
                "LongTermDebtCurrent": _usd(current),
                "CashAndCashEquivalentsAtCarryingValue": _usd(cash),
                "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": weighted_rows}},
            },
        }
    }


# The FY2024 and FY2025 10-K cover pages; the Q3 2025 10-Q reports only weighted-average counts, both the
# 3-month one (kept: the shortest period of a filing) and the 9-month year-to-date one.
COVER_ROWS = [
    _count(900_000, "2025-01-31", "2025-02-15", form="10-K"),
    _count(1_000_000, "2026-01-31", "2026-02-15", form="10-K"),
]
WEIGHTED_ROWS = [
    _count(910_000, "2025-03-31", "2025-05-05", form="10-Q", start="2025-01-01"),
    _count(940_000, "2025-09-30", "2025-11-05", form="10-Q", start="2025-07-01"),
    _count(930_000, "2025-09-30", "2025-11-05", form="10-Q", start="2025-01-01"),
]


class FakeSec:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "/companyfacts/" in url:
            return httpx.Response(200, json=self.payload)
        if "/submissions/" in url:
            return httpx.Response(200, json={"sic": "3571"})  # electronic computers: not an excluded sector
        return httpx.Response(404)

    def count(self, fragment: str) -> int:
        return sum(fragment in url for url in self.requests)


class Clock:
    def __init__(self, now: datetime = WALL_CLOCK) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class YahooSplits:
    """The Yahoo fetch function: counts calls, returns a fixed split list."""

    def __init__(self, splits: list[Split]) -> None:
        self.splits = splits
        self.calls: list[str] = []

    def __call__(self, symbol: str) -> list[Split]:
        self.calls.append(symbol)
        return list(self.splits)


def _build(tmp_path: Path, payload: dict[str, Any], yahoo: YahooSplits, clock: Clock | None = None) -> tuple[QualityScreen, FakeSec]:
    clock = clock or Clock()
    sec = FakeSec(payload)
    client = SecEdgarClient("TestApp test@example.com", tmp_path / "sec", min_request_interval_seconds=0.0, transport=httpx.MockTransport(sec))
    store = AnnualFiguresStore(client, tmp_path / "sec" / "annual", wall_clock=clock)
    splits = SplitHistory(tmp_path / "splits.json", fetch=yahoo, wall_clock=clock)
    return QualityScreen(store, splits, params=ScreenParams()), sec


def _run(screen: QualityScreen, as_of: datetime) -> Any:
    result = screen.run(["ACME"], as_of=as_of, price_of=lambda symbol: PRICE)
    assert result.rejections == {}
    (candidate,) = result.candidates
    return candidate


DAY_OF_FILING = datetime(2026, 2, 15, 18, 0, tzinfo=NEW_YORK)  # the FY2025 10-K is filed today: not yet known
DAY_AFTER = datetime(2026, 2, 16, 9, 0, tzinfo=NEW_YORK)

# An old split, and one between the Q3 2025 quarter end (2025-09-30) and the 10-Q's filing (2025-11-05).
SPLITS = [(date(2019, 6, 3), 2.0), (date(2025, 10, 15), 2.0)]


def test_a_real_shaped_company_is_screened_on_the_day_of_its_filing_and_the_day_after(tmp_path: Path) -> None:
    screen, sec = _build(tmp_path, company_facts(cover_rows=COVER_ROWS, weighted_rows=WEIGHTED_ROWS), YahooSplits(SPLITS))

    before = _run(screen, DAY_OF_FILING)
    after = _run(screen, DAY_AFTER)

    # Day of filing: the FY2025 10-K is not known yet, so FY2020..FY2024 is the window, FY2023 revenue is the
    # original 130, and the latest count is the Q3 weighted average (the 3-month row), counted on its filing date
    # 2025-11-05: the 2025-10-15 split is already in it and is not applied again.
    assert before.fiscal_year_end == date(2024, 12, 31)
    assert before.filed == date(2025, 2, 15)
    assert before.market_cap == Decimal(940_000) * PRICE == Decimal("235000000")
    assert before.fcf_yield == pytest.approx(32 / 235)
    assert before.net_debt_to_operating_income == pytest.approx((40 - 8) / 28)
    assert before.debt_reported is True
    assert before.fcf_margin == pytest.approx(sum(fcf / rev for fcf, rev in zip((24, 26, 28, 30, 32), (100, 110, 120, 130, 140), strict=True)) / 5)
    assert before.sic == 3571

    # Day after: FY2021..FY2025, FY2023 revenue restated to 135, and the FY2025 cover count (2026-01-31).
    assert after.fiscal_year_end == date(2025, 12, 31)
    assert after.filed == date(2026, 2, 15)
    assert after.market_cap == Decimal(1_000_000) * PRICE == Decimal("250000000")
    assert after.fcf_yield == pytest.approx(34 / 250)
    assert after.net_debt_to_operating_income == pytest.approx((50 - 10) / 30)
    assert after.debt_reported is True
    assert after.fcf_margin == pytest.approx(sum(fcf / rev for fcf, rev in zip((26, 28, 30, 32, 34), (110, 120, RESTATED_2023_REVENUE, 140, 150), strict=True)) / 5)

    assert sec.count("/companyfacts/") == 1  # one fetch answers both dates
    reduced = json.loads((tmp_path / "sec" / "annual" / f"CIK{CIK}.json").read_text(encoding="utf-8"))
    assert all(row["value"] != 9_999 * M for row in reduced["flows"])  # no 10-Q figure got in
    assert {row["kind"] for row in reduced["shares"]} == {"cover", "weighted"}
    assert [row["value"] for row in reduced["shares"] if row["end"] == "2025-09-30"] == [940_000]  # one row per (end, filed)


@pytest.mark.parametrize(
    ("split_date", "expected_cap"),
    [
        (date(2025, 10, 15), Decimal("235000000")),  # between the quarter end and the filing: already in the count
        (date(2025, 11, 5), Decimal("235000000")),  # on the filing date itself
        (date(2025, 12, 1), Decimal("470000000")),  # after the filing: the count predates it
    ],
)
def test_a_weighted_only_company_is_restated_only_for_splits_after_its_filing(tmp_path: Path, split_date: date, expected_cap: Decimal) -> None:
    payload = company_facts(cover_rows=[], weighted_rows=WEIGHTED_ROWS)
    screen, _ = _build(tmp_path, payload, YahooSplits([(split_date, 2.0)]))

    assert _run(screen, DAY_OF_FILING).market_cap == expected_cap


def test_a_split_entry_older_than_the_max_age_by_the_wall_clock_is_refetched_for_a_past_as_of(tmp_path: Path) -> None:
    payload = company_facts(cover_rows=COVER_ROWS, weighted_rows=WEIGHTED_ROWS)
    yahoo = YahooSplits(SPLITS)
    clock = Clock(WALL_CLOCK)
    screen, _ = _build(tmp_path, payload, yahoo, clock)

    _run(screen, DAY_AFTER)
    _run(screen, DAY_AFTER)
    assert yahoo.calls == ["ACME"]  # fetched today: fresh, whatever the simulated date

    clock.now = WALL_CLOCK + timedelta(days=2)  # past split_max_age_days (1) by the wall clock
    yahoo.splits = [*SPLITS, (date(2026, 6, 1), 3.0)]  # a split that happened after the entry was fetched
    after_new_split = _run(screen, DAY_AFTER)

    assert yahoo.calls == ["ACME", "ACME"]
    assert after_new_split.market_cap == Decimal(1_000_000) * 3 * PRICE  # the cover count (2026-01-31) predates it


def test_the_same_past_date_never_refetches_split_history_by_the_simulated_date_alone(tmp_path: Path) -> None:
    # Simulated dates a year apart, same wall clock: the entry is fresh both times.
    payload = company_facts(cover_rows=COVER_ROWS, weighted_rows=WEIGHTED_ROWS)
    yahoo = YahooSplits(SPLITS)
    screen, _ = _build(tmp_path, payload, yahoo)

    _run(screen, DAY_AFTER)
    screen.run(["ACME"], as_of=datetime(2025, 3, 1, tzinfo=NEW_YORK), price_of=lambda symbol: PRICE)

    assert yahoo.calls == ["ACME"]
