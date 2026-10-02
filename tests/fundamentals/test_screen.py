from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from tests.fundamentals.annual_fixtures import healthy_figures

from trading_agent_framework.fundamentals import Candidate, QualityScreen, ScreenParams, ScreenResult, build_quality_screen
from trading_agent_framework.fundamentals.splits import Split
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, FundamentalsError

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


class FakeStore:
    """Annual figures and SIC codes from dicts; symbols in `failing` raise like a transport error."""

    def __init__(
        self,
        records: dict[str, dict[str, object] | None],
        *,
        ciks: dict[str, str] | None = None,
        sics: dict[str, int] | None = None,
        failing: Collection[str] = (),
        sic_failing: Collection[str] = (),
    ) -> None:
        self.records = records
        self.ciks = ciks or {}
        self.sics = sics or {}
        self.failing = failing
        self.sic_failing = sic_failing

    def cik(self, symbol: str) -> str | None:
        return self.ciks.get(symbol, f"cik-{symbol}")

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, object] | None:
        if symbol in self.failing:
            raise FundamentalsError(f"network down for {symbol}")
        return self.records.get(symbol)

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None:
        if symbol in self.sic_failing:
            raise FundamentalsError(f"network down for {symbol}")
        return self.sics.get(symbol)


class FakeSplits:
    def __init__(self, splits: dict[str, list[Split]] | None = None, *, failing: Collection[str] = ()) -> None:
        self._splits = splits or {}
        self.failing = failing
        self.calls: list[str] = []
        self.max_ages: list[int] = []

    def splits(self, symbol: str, *, max_age_days: int) -> list[Split]:
        self.calls.append(symbol)
        self.max_ages.append(max_age_days)
        if symbol in self.failing:
            raise FundamentalsError(f"yahoo is down for {symbol}")
        return self._splits.get(symbol, [])


class Prices:
    """A `price_of` that records which symbols it was asked about."""

    def __init__(self, prices: dict[str, Decimal | None] | None = None, *, default: Decimal | None = Decimal("10")) -> None:
        self._prices = prices or {}
        self._default = default
        self.calls: list[str] = []

    def __call__(self, symbol: str) -> Decimal | None:
        self.calls.append(symbol)
        return self._prices.get(symbol, self._default)


def _run(
    store: FakeStore,
    symbols: list[str],
    *,
    splits: FakeSplits | None = None,
    price_of: Callable[[str], Decimal | None] | None = None,
    params: ScreenParams | None = None,
    as_of: datetime = AS_OF,
) -> ScreenResult:
    screen = QualityScreen(store, splits or FakeSplits(), params=params)
    return screen.run(symbols, as_of=as_of, price_of=price_of or Prices())


def _symbols(result: ScreenResult) -> list[str]:
    return [candidate.symbol for candidate in result.candidates]


def test_a_healthy_company_becomes_a_candidate_with_its_market_cap() -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}, sics={"AAA": 5812}), ["AAA"])

    (candidate,) = result.candidates
    assert isinstance(candidate, Candidate)
    assert (candidate.symbol, candidate.rank, candidate.sic) == ("AAA", 1, 5812)
    assert candidate.market_cap == Decimal("10000")  # 1000 shares x 10
    assert candidate.fcf_yield == pytest.approx(28 / 10000)
    assert result.rejections == {}


def test_candidates_are_ranked_against_each_other() -> None:
    store = FakeStore({"AAA": healthy_figures(), "BBB": healthy_figures()})

    result = _run(store, ["AAA", "BBB"], price_of=Prices({"AAA": Decimal("20"), "BBB": Decimal("10")}))

    assert _symbols(result) == ["BBB", "AAA"]  # same company figures, BBB is cheaper


def test_a_split_after_the_share_count_restates_the_market_cap() -> None:
    # healthy_figures counts 1000 shares on 2026-01-31; a 2-for-1 follows.
    splits = FakeSplits({"AAA": [(date(2026, 3, 2), 2.0)]})

    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], splits=splits)

    assert result.candidates[0].market_cap == Decimal("20000")


def test_a_numeric_gate_failure_is_reported_with_its_reason() -> None:
    store = FakeStore({"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28)), "GONE": None})

    result = _run(store, ["AAA", "LOSS", "GONE"])

    assert _symbols(result) == ["AAA"]
    assert result.rejections == {"LOSS": "operating_loss", "GONE": "no_data"}


def test_an_excluded_sector_is_rejected() -> None:
    result = _run(FakeStore({"BANK": healthy_figures()}, sics={"BANK": 6021}), ["BANK"])

    assert result.rejections == {"BANK": "excluded_sector"}


def test_two_listings_of_one_company_take_one_slot() -> None:
    store = FakeStore({"GOOGL": healthy_figures(), "GOOG": healthy_figures()}, ciks={"GOOGL": "0001652044", "GOOG": "0001652044"})

    result = _run(store, ["GOOGL", "GOOG"])

    assert _symbols(result) == ["GOOGL"]
    assert result.rejections == {"GOOG": "duplicate_listing"}


def test_the_second_listing_gets_its_chance_when_the_first_has_no_price() -> None:
    store = FakeStore({"GOOGL": healthy_figures(), "GOOG": healthy_figures()}, ciks={"GOOGL": "0001652044", "GOOG": "0001652044"})

    result = _run(store, ["GOOGL", "GOOG"], price_of=Prices({"GOOGL": None}))

    assert _symbols(result) == ["GOOG"]
    assert result.rejections == {"GOOGL": "no_price"}


def test_prices_and_splits_are_asked_only_for_survivors_of_the_earlier_gates() -> None:
    store = FakeStore(
        {"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28)), "BANK": healthy_figures(), "AAA2": healthy_figures()},
        ciks={"AAA": "1", "AAA2": "1"},
        sics={"BANK": 6021},
    )
    prices, splits = Prices(), FakeSplits()

    _run(store, ["AAA", "LOSS", "GONE", "BANK", "AAA2"], splits=splits, price_of=prices)

    assert prices.calls == ["AAA"]
    assert splits.calls == ["AAA"]


@pytest.mark.parametrize("price", [None, Decimal("0"), Decimal("-1")])
def test_a_missing_or_non_positive_price_is_no_price(price: Decimal | None) -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], price_of=Prices({"AAA": price}))

    assert result.rejections == {"AAA": "no_price"}


def test_a_price_lookup_that_raises_a_framework_error_is_no_price_and_the_run_continues() -> None:
    def price_of(symbol: str) -> Decimal | None:
        if symbol == "AAA":
            raise BrokerError("no quote")
        return Decimal("10")

    result = _run(FakeStore({"AAA": healthy_figures(), "BBB": healthy_figures()}), ["AAA", "BBB"], price_of=price_of)

    assert _symbols(result) == ["BBB"]
    assert result.rejections == {"AAA": "no_price"}


@pytest.mark.parametrize("shares", [None, 0])
def test_a_missing_or_zero_share_count_is_no_price(shares: int | None) -> None:
    result = _run(FakeStore({"AAA": healthy_figures(shares=shares)}), ["AAA"])

    assert result.rejections == {"AAA": "no_price"}


def test_a_failed_split_lookup_is_no_split_data() -> None:
    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], splits=FakeSplits(failing={"AAA"}))

    assert result.rejections == {"AAA": "no_split_data"}


def test_a_transport_failure_rejects_the_symbol_as_no_data() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, failing={"EEE"})

    result = _run(store, names)  # 1 of 5 = 20%, not above the 20% limit

    assert result.rejections == {"EEE": "no_data"}
    assert len(result.candidates) == 4


def test_a_failed_sic_lookup_counts_as_a_transport_failure() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, sic_failing={"EEE"})

    assert _run(store, names).rejections == {"EEE": "no_data"}


def test_a_hollow_screen_raises_instead_of_ranking_what_happened_to_download() -> None:
    names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    store = FakeStore({name: healthy_figures() for name in names}, failing={"DDD", "EEE"})

    with pytest.raises(FundamentalsError, match="2 of 5"):
        _run(store, names)


def test_a_naive_as_of_is_refused_up_front() -> None:
    screen = QualityScreen(FakeStore({"AAA": healthy_figures()}), FakeSplits())

    with pytest.raises(ValueError, match="timezone-aware"):
        screen.run(["AAA"], as_of=datetime(2026, 6, 1), price_of=Prices())


def test_an_empty_symbol_list_gives_an_empty_result() -> None:
    assert _run(FakeStore({}), []) == ScreenResult(candidates=[], rejections={})


def test_a_list_where_everything_is_rejected_gives_no_candidates() -> None:
    result = _run(FakeStore({}), ["GONE", "ALSO"])

    assert result.candidates == []
    assert result.rejections == {"GONE": "no_data", "ALSO": "no_data"}


def test_a_symbol_listed_twice_is_processed_once() -> None:
    prices = Prices()

    result = _run(FakeStore({"AAA": healthy_figures()}), ["AAA", "AAA"], price_of=prices)

    assert _symbols(result) == ["AAA"]
    assert prices.calls == ["AAA"]
    assert result.rejections == {}


def test_top_n_limits_the_candidates() -> None:
    names = ["AAA", "BBB", "CCC"]

    result = _run(FakeStore({name: healthy_figures() for name in names}), names, params=ScreenParams(top_n=2))

    assert len(result.candidates) == 2


def test_each_run_logs_one_summary_line(caplog: pytest.LogCaptureFixture) -> None:
    store = FakeStore({"AAA": healthy_figures(), "LOSS": healthy_figures(operating_income=(20, 22, -1, 26, 28))})

    with caplog.at_level(logging.INFO):
        _run(store, ["AAA", "LOSS", "GONE"])

    assert "screened 3 symbols" in caplog.text
    assert "1 candidates" in caplog.text
    assert "no_data=1" in caplog.text
    assert "operating_loss=1" in caplog.text


def test_as_of_is_read_as_a_market_local_date() -> None:
    # healthy_figures files FY2025 on 2026-02-15: known only strictly after that date.
    evening_in_new_york = datetime(2026, 2, 15, 23, 0, tzinfo=ZoneInfo("America/New_York"))
    same_instant_in_utc = evening_in_new_york.astimezone(UTC)  # already 2026-02-16
    store = FakeStore({"AAA": healthy_figures()})

    assert _run(store, ["AAA"], as_of=evening_in_new_york).rejections == {"AAA": "insufficient_history"}
    assert _symbols(_run(store, ["AAA"], as_of=same_instant_in_utc)) == ["AAA"]


def test_build_quality_screen_needs_a_sec_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    with pytest.raises(ConfigurationError, match="SEC_EDGAR_USER_AGENT"):
        build_quality_screen(tmp_path)


def test_build_quality_screen_builds_a_screen_under_the_project_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")

    screen = build_quality_screen(tmp_path, params=ScreenParams(top_n=3))

    assert isinstance(screen, QualityScreen)
    assert screen.params.top_n == 3
    assert (tmp_path / "cache" / "sec").is_dir()


def test_a_company_whose_latest_debt_figure_is_missing_is_rejected_as_debt_unknown() -> None:
    store = FakeStore({"AAA": healthy_figures(), "BBB": healthy_figures(debt_by_year={2022: 50}, cash=10)})

    result = _run(store, ["AAA", "BBB"])

    assert result.rejections == {"BBB": "debt_unknown"}
    assert _symbols(result) == ["AAA"]


def test_the_split_lookup_gets_the_split_max_age_not_the_annual_one() -> None:
    splits = FakeSplits()

    _run(FakeStore({"AAA": healthy_figures()}), ["AAA"], splits=splits, params=ScreenParams(max_age_days=30, split_max_age_days=3))

    assert splits.max_ages == [3]
