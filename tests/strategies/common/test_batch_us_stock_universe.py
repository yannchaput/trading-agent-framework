from __future__ import annotations

import fcntl
import json
import logging
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from trading_agent_framework.backtesting.data.yahoo import parse_yahoo_frame
from trading_agent_framework.strategies.common import batch_us_stock_universe as batch
from trading_agent_framework.strategies.common.batch_us_stock_universe import (
    RateLimitGate,
    build_history_pool,
    get_extra_universe_symbols,
    is_rate_limit_error,
    merge_ticker_candidates,
    normalize_symbol,
    parse_ishares_csv,
    parse_nasdaq_rows,
    parse_nasdaq_symbols,
    parse_vanguard_holdings,
)
from trading_agent_framework.strategies.common.universe_history import UniverseHistory, monthly_lists
from trading_agent_framework.utils.errors import UniverseHistoryError


class FakeClock:
    """Deterministic now()/sleep() pair: sleep() advances the fake clock instead of blocking."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.t += seconds


_ISHARES_CSV = (
    "iShares Russell 1000 ETF\n"
    'Fund Holdings as of,"Sep 23, 2026"\n'
    'Inception Date,"May 15, 2000"\n'
    'Shares Outstanding,"116,750,000.00"\n'
    "\n"
    "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,"
    "Exchange,Currency,FX Rate,Market Currency,Accrual Date\n"
    '"NVDA","NVIDIA","Information Technology","Equity","3,628,087,416.66","7.41","3,628,087,416.66",'
    '"16,088,366.00","225.51","United States","NASDAQ","USD","1.00","USD","-"\n'
    '"MOG A","MOOG INC CLASS A","Industrials","Equity","273,002,314.10","0.36","273,002,314.10",'
    '"728,045.00","374.98","United States","NYSE","USD","1.00","USD","-"\n'
    '"XTSLA","BLK CSH FND TREASURY SL AGENCY","Cash and/or Derivatives","Money Market","26,009,026.49",'
    '"0.05","26,009,026.49","26,009,026.00","1.00","United States","-","USD","1.00","USD","-"\n'
    '"ESZ6","S&P500 EMINI DEC 26","Cash and/or Derivatives","Futures","0.00","0.00","62,568,625.00",'
    '"161.00","7,772.50","-","Index And Options Market","USD","1.00","USD","-"\n'
)


def test_normalize_symbol_uppercases_and_strips() -> None:
    assert normalize_symbol("  aapl  ") == "AAPL"


def test_normalize_symbol_converts_share_class_separators() -> None:
    assert normalize_symbol("BRK B") == "BRK-B"  # iShares
    assert normalize_symbol("BRK.B") == "BRK-B"  # Vanguard
    assert normalize_symbol("BRK/B") == "BRK-B"  # NASDAQ


def test_normalize_symbol_rejects_non_letter_content() -> None:
    assert normalize_symbol("-") is None
    assert normalize_symbol("USD") is not None  # letters-only still passes here
    assert normalize_symbol("") is None
    assert normalize_symbol("S&P500") is None


def test_parse_ishares_csv_keeps_only_equity_rows() -> None:
    assert parse_ishares_csv(_ISHARES_CSV) == {"NVDA", "MOG-A"}


_VANGUARD_PAYLOAD = {
    "fund": {
        "entity": [
            {"ticker": "NVDA", "longName": "NVIDIA Corp."},
            {"ticker": "BRK.B", "longName": "Berkshire Hathaway Class B"},
            {"ticker": "", "longName": "BOWL AMERICA INC ESCOW LINE - NOV 2025"},
        ]
    }
}


def test_parse_vanguard_holdings_extracts_and_normalizes_tickers() -> None:
    assert parse_vanguard_holdings(_VANGUARD_PAYLOAD) == {"NVDA", "BRK-B"}


def _nasdaq_row(symbol: str, market_cap: str, price: str) -> dict:
    return {"symbol": symbol, "marketCap": market_cap, "lastsale": price}


def test_parse_nasdaq_rows_normalizes_slash_share_classes() -> None:
    rows = [_nasdaq_row("BRK/B", "$1.05T", "$450.00")]

    candidates = parse_nasdaq_rows(rows)

    assert candidates == [("BRK-B", 1.05e12)]


def test_parse_nasdaq_rows_drops_rows_below_market_cap_threshold() -> None:
    rows = [_nasdaq_row("TINY", "$500M", "$50.00")]

    assert parse_nasdaq_rows(rows) == []


def test_parse_nasdaq_rows_keeps_rows_with_unparseable_market_cap() -> None:
    rows = [_nasdaq_row("NEWCO", "$0.00", "$50.00")]

    assert parse_nasdaq_rows(rows) == [("NEWCO", 0.0)]


def test_merge_ticker_candidates_keeps_nasdaq_market_cap_for_overlapping_symbols() -> None:
    nasdaq_candidates = [("AAPL", 3.5e12)]

    merged = merge_ticker_candidates(nasdaq_candidates, {"AAPL", "IWM-ONLY"})

    assert set(merged) == {("AAPL", 3.5e12), ("IWM-ONLY", 0.0)}


def test_merge_ticker_candidates_with_no_extra_symbols_returns_nasdaq_candidates() -> None:
    nasdaq_candidates = [("AAPL", 3.5e12)]

    assert merge_ticker_candidates(nasdaq_candidates, set()) == [("AAPL", 3.5e12)]


def test_rate_limit_gate_wait_returns_immediately_with_no_active_cooldown() -> None:
    clock = FakeClock()
    gate = RateLimitGate(initial_backoff=30.0, max_backoff=300.0, now=clock.now, sleep=clock.sleep)

    gate.wait()

    assert clock.sleep_calls == []


def test_rate_limit_gate_wait_sleeps_out_the_full_cooldown_after_a_hit() -> None:
    clock = FakeClock()
    gate = RateLimitGate(initial_backoff=30.0, max_backoff=300.0, now=clock.now, sleep=clock.sleep)

    gate.on_rate_limited()
    gate.wait()

    assert clock.sleep_calls == [30.0]
    assert clock.t == 30.0


def test_rate_limit_gate_backoff_doubles_on_consecutive_hits_without_success() -> None:
    clock = FakeClock()
    gate = RateLimitGate(initial_backoff=30.0, max_backoff=300.0, now=clock.now, sleep=clock.sleep)

    gate.on_rate_limited()
    gate.wait()
    gate.on_rate_limited()
    gate.wait()

    assert clock.sleep_calls == [30.0, 60.0]


def test_rate_limit_gate_backoff_caps_at_max() -> None:
    clock = FakeClock()
    gate = RateLimitGate(initial_backoff=30.0, max_backoff=90.0, now=clock.now, sleep=clock.sleep)

    for _ in range(5):
        gate.on_rate_limited()
        gate.wait()

    assert clock.sleep_calls == [30.0, 60.0, 90.0, 90.0, 90.0]


def test_rate_limit_gate_on_success_resets_backoff_to_initial() -> None:
    clock = FakeClock()
    gate = RateLimitGate(initial_backoff=30.0, max_backoff=300.0, now=clock.now, sleep=clock.sleep)

    gate.on_rate_limited()
    gate.wait()
    gate.on_success()
    gate.on_rate_limited()
    gate.wait()

    assert clock.sleep_calls == [30.0, 30.0]


def test_is_rate_limit_error_detects_yfinance_rate_limit_error() -> None:
    assert is_rate_limit_error(YFRateLimitError()) is True


def test_is_rate_limit_error_detects_generic_429_message() -> None:
    assert is_rate_limit_error(Exception("HTTPError: 429 Client Error: Too Many Requests")) is True


def test_is_rate_limit_error_returns_false_for_unrelated_errors() -> None:
    assert is_rate_limit_error(ValueError("bad ticker")) is False


def test_get_extra_universe_symbols_unions_sources_and_tolerates_one_failure() -> None:
    def fetch_ishares_csv(url: str) -> str:
        if "russell-1000" in url:
            return _ISHARES_CSV
        raise RuntimeError("IWM endpoint down")

    def fetch_vanguard_holdings() -> dict:
        return _VANGUARD_PAYLOAD

    symbols = get_extra_universe_symbols(
        fetch_ishares_csv=fetch_ishares_csv,
        fetch_vanguard_holdings=fetch_vanguard_holdings,
    )

    assert symbols == {"NVDA", "MOG-A", "BRK-B"}


def test_the_live_batch_still_skips_a_failing_etf_source() -> None:
    def fetch_ishares_csv(url: str) -> str:
        return _ISHARES_CSV

    def fetch_vanguard_holdings() -> dict:
        raise RuntimeError("VTI endpoint down")

    symbols = get_extra_universe_symbols(
        fetch_ishares_csv=fetch_ishares_csv,
        fetch_vanguard_holdings=fetch_vanguard_holdings,
    )

    assert symbols == {"NVDA", "MOG-A"}


def _vti_fetchers(fetch_vanguard_holdings):
    return {"fetch_ishares_csv": lambda url: _ISHARES_CSV, "fetch_vanguard_holdings": fetch_vanguard_holdings}


def test_the_vti_texts_are_pinned(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> dict:
        raise RuntimeError("down")

    batch._etf_symbols(**_vti_fetchers(boom), strict=False)
    out = " ".join(capsys.readouterr().out.split())
    assert "Failed to fetch VTI holdings: down" in out

    with pytest.raises(UniverseHistoryError) as failed:
        batch._etf_symbols(**_vti_fetchers(boom), strict=True)
    assert str(failed.value) == "VTI holdings failed: down"

    with pytest.raises(UniverseHistoryError) as empty:
        batch._etf_symbols(**_vti_fetchers(lambda: {"data": []}), strict=True)
    assert str(empty.value) == "VTI holdings returned no equity symbols"

    batch._etf_symbols(**_vti_fetchers(lambda: _VANGUARD_PAYLOAD), strict=True)
    assert "VTI (total market): 2 equity holdings" in " ".join(capsys.readouterr().out.split())


# ── history pool ────────────────────────────────────────────────────────────


def _ishares_csv(*tickers: str) -> str:
    rows = "".join(f'"{t}","{t} INC","Industrials","Equity","1.00","1.00","1.00","1.00","1.00","United States","NYSE","USD","1.00","USD","-"\n' for t in tickers)
    return (
        'Fund Holdings as of,"Sep 23, 2026"\n\n'
        "Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Quantity,Price,Location,"
        "Exchange,Currency,FX Rate,Market Currency,Accrual Date\n" + rows
    )


def _vanguard_payload(*tickers: str) -> dict:
    return {"fund": {"entity": [{"ticker": t} for t in tickers]}}


def _pool_fetchers(
    nasdaq: list[str] = ("A", "B"),
    ishares: dict[str, tuple[str, ...]] | None = None,
    vanguard: tuple[str, ...] = ("D",),
) -> dict:
    by_name = ishares if ishares is not None else {"russell-1000": ("B", "C"), "russell-2000": ("B",)}

    def fetch_nasdaq_rows() -> list[dict]:
        return [_nasdaq_row(symbol, "$1.00B", "$50.00") for symbol in nasdaq]

    def fetch_ishares_csv(url: str) -> str:
        for key, tickers in by_name.items():
            if key in url:
                return _ishares_csv(*tickers)
        raise AssertionError(url)

    def fetch_vanguard_holdings() -> dict:
        return _vanguard_payload(*vanguard)

    return {
        "fetch_nasdaq_rows": fetch_nasdaq_rows,
        "fetch_ishares_csv": fetch_ishares_csv,
        "fetch_vanguard_holdings": fetch_vanguard_holdings,
    }


def test_parse_nasdaq_symbols_has_no_cap_or_price_filter_and_dedups() -> None:
    rows = [
        _nasdaq_row("TINY", "$500M", "$4.00"),
        _nasdaq_row("brk/b", "$1.05T", "$450.00"),
        _nasdaq_row("BRK.B", "", ""),
        _nasdaq_row("12345", "$1B", "$5.00"),
    ]

    assert parse_nasdaq_symbols(rows) == ["BRK-B", "TINY"]


def test_history_pool_keeps_small_and_cheap_listings() -> None:
    rows = [_nasdaq_row("TINY", "$500M", "$4.00")]
    fetchers = _pool_fetchers()
    fetchers["fetch_nasdaq_rows"] = lambda: rows

    pool = build_history_pool(**fetchers, is_common_stock=lambda symbol: True).symbols

    assert "TINY" in pool
    assert parse_nasdaq_rows(rows) == []


def test_history_pool_unions_the_sources_and_keeps_common_stocks_only() -> None:
    pool = build_history_pool(**_pool_fetchers(), is_common_stock=lambda symbol: symbol != "C").symbols

    assert pool == ["A", "B", "D"]


def _names(count: int) -> list[str]:
    """`count` letters-only symbols: AA, AB, ..."""
    return [chr(65 + index // 26) + chr(65 + index % 26) for index in range(count)]


def _two_hundred_candidates() -> dict:
    only_aa = {"russell-1000": ("AA",), "russell-2000": ("AA",)}
    return _pool_fetchers(nasdaq=_names(200), ishares=only_aa, vanguard=("AA",))


@pytest.mark.usefixtures("no_sleep")
def test_history_pool_stops_when_over_1_percent_of_the_info_fetches_fail() -> None:
    failed = set(_names(3))  # 3 of 200 candidates = 1.5%

    with pytest.raises(UniverseHistoryError, match="3 of 200"):
        build_history_pool(**_two_hundred_candidates(), is_common_stock=lambda symbol: None if symbol in failed else True)


@pytest.mark.usefixtures("no_sleep")
def test_history_pool_tolerates_exactly_the_limit_of_failed_info_fetches() -> None:
    failed = set(_names(2))  # 2 of 200 candidates = 1%, not above the limit

    pool = build_history_pool(**_two_hundred_candidates(), is_common_stock=lambda symbol: None if symbol in failed else True).symbols

    assert pool == sorted(set(_names(200)) - failed)


@pytest.mark.usefixtures("no_sleep")
def test_history_pool_counts_a_genuine_non_equity_answer_apart_from_a_failure() -> None:
    not_equity = set(_names(50))  # 25% answered "not common": legitimate, never an error

    pool = build_history_pool(**_two_hundred_candidates(), is_common_stock=lambda symbol: symbol not in not_equity).symbols

    assert len(pool) == 150


def test_unanswered_candidates_are_asked_once_more_after_a_pause(no_sleep: FakeClock) -> None:
    asked: list[str] = []

    def is_common_stock(symbol: str) -> bool | None:
        asked.append(symbol)
        return None if symbol in {"AA", "AB", "AC"} and asked.count(symbol) == 1 else True  # 1.5% fail the first time only

    details = build_history_pool(**_two_hundred_candidates(), is_common_stock=is_common_stock)

    assert details.symbols == sorted(_names(200))  # all three recovered on the second pass
    assert details.unanswered == []
    assert sorted(asked[-3:]) == ["AA", "AB", "AC"] and len(asked) == 203  # only the three, only once more
    assert no_sleep.sleep_calls == [30.0]  # after a pause through the shared rate-limit gate


def test_no_pause_when_every_candidate_answered(no_sleep: FakeClock) -> None:
    build_history_pool(**_pool_fetchers(), is_common_stock=lambda symbol: True)

    assert no_sleep.sleep_calls == []


@pytest.mark.usefixtures("no_sleep")
def test_the_candidates_still_unanswered_after_the_retry_are_returned_sorted() -> None:
    failed = set(_names(2))

    details = build_history_pool(**_two_hundred_candidates(), is_common_stock=lambda symbol: None if symbol in failed else True)

    assert details.unanswered == ["AA", "AB"]
    assert len(details.symbols) == 198


@pytest.mark.usefixtures("no_sleep")
def test_the_retry_pass_counts_only_the_still_unanswered_against_the_limit() -> None:
    asked: list[str] = []

    def is_common_stock(symbol: str) -> bool | None:
        asked.append(symbol)
        # 10 of 200 (5%) fail the first time; four of them keep failing: 4 > 2 still raises, and not 10.
        return None if symbol in set(_names(10)) and (asked.count(symbol) == 1 or symbol in set(_names(4))) else True

    with pytest.raises(UniverseHistoryError, match="4 of 200"):
        build_history_pool(**_two_hundred_candidates(), is_common_stock=is_common_stock)


@pytest.mark.parametrize(
    ("failing", "named"),
    [
        ("nasdaq", "NASDAQ"),
        ("iwb", "IWB"),
        ("iwm", "IWM"),
        ("vti", "VTI"),
    ],
)
def test_a_failing_source_stops_the_history_pool(failing: str, named: str) -> None:
    fetchers = _pool_fetchers()
    good_ishares = fetchers["fetch_ishares_csv"]

    def boom(*args: object) -> None:
        raise RuntimeError("source down")

    if failing == "nasdaq":
        fetchers["fetch_nasdaq_rows"] = boom
    elif failing == "vti":
        fetchers["fetch_vanguard_holdings"] = boom
    else:
        marker = "russell-1000" if failing == "iwb" else "russell-2000"

        def fetch_ishares_csv(url: str) -> str:
            if marker in url:
                raise RuntimeError("source down")
            return good_ishares(url)

        fetchers["fetch_ishares_csv"] = fetch_ishares_csv

    with pytest.raises(UniverseHistoryError, match=named):
        build_history_pool(**fetchers, is_common_stock=lambda symbol: True)


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"nasdaq": []}, "NASDAQ"),
        ({"ishares": {"russell-1000": (), "russell-2000": ("B",)}}, "IWB"),
        ({"ishares": {"russell-1000": ("B",), "russell-2000": ()}}, "IWM"),
        ({"vanguard": ()}, "VTI"),
    ],
)
def test_a_source_returning_nothing_stops_the_history_pool(kwargs: dict, named: str) -> None:
    with pytest.raises(UniverseHistoryError, match=named):
        build_history_pool(**_pool_fetchers(**kwargs), is_common_stock=lambda symbol: True)


def test_an_empty_history_pool_raises() -> None:
    with pytest.raises(UniverseHistoryError, match="empty"):
        build_history_pool(**_pool_fetchers(), is_common_stock=lambda symbol: False)


class _FakeTicker:
    """yf.Ticker stand-in: `infos[symbol]` is the info dict, or an exception to raise."""

    infos: dict[str, object] = {}
    calls: list[str] = []

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def info(self) -> dict:
        _FakeTicker.calls.append(self.symbol)
        result = _FakeTicker.infos[self.symbol]
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def fake_yf(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTicker]:
    _FakeTicker.infos = {}
    _FakeTicker.calls = []
    monkeypatch.setattr(batch.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    return _FakeTicker


def test_the_real_common_stock_check_reads_the_quote_type(fake_yf: type[_FakeTicker]) -> None:
    fake_yf.infos = {
        "AAPL": {"quoteType": "EQUITY"},
        "SPY": {"quoteType": "ETF"},
        "ODD": {"marketCap": 5},
    }

    assert batch._is_common_stock("AAPL") is True
    assert batch._is_common_stock("SPY") is False
    assert batch._is_common_stock("ODD") is None  # an info with no quoteType is no answer, like a failed fetch


def test_the_real_common_stock_check_reports_an_empty_or_quote_type_less_info_as_unknown(fake_yf: type[_FakeTicker], caplog: pytest.LogCaptureFixture) -> None:
    fake_yf.infos = {"EMPTY": {}, "BLANK": {"quoteType": ""}, "NONE": {"quoteType": None}}

    with caplog.at_level(logging.WARNING):
        answers = [batch._is_common_stock(symbol) for symbol in ("EMPTY", "BLANK", "NONE")]

    assert answers == [None, None, None]
    assert "EMPTY" in caplog.text


def test_the_real_common_stock_check_reports_a_failed_fetch_as_unknown(fake_yf: type[_FakeTicker], caplog: pytest.LogCaptureFixture) -> None:
    fake_yf.infos = {"BAD": RuntimeError("yahoo down")}

    with caplog.at_level(logging.WARNING):
        assert batch._is_common_stock("BAD") is None  # not False: a failed fetch is no answer

    assert fake_yf.calls == ["BAD", "BAD", "BAD"]
    assert "BAD" in caplog.text


def test_the_history_pool_default_check_runs_the_real_filter(fake_yf: type[_FakeTicker]) -> None:
    fake_yf.infos = {"A": {"quoteType": "EQUITY"}, "B": {"quoteType": "ETF"}, "D": {"quoteType": "EQUITY"}, "C": {"quoteType": "EQUITY"}}

    assert build_history_pool(**_pool_fetchers()).symbols == ["A", "C", "D"]


# ── ensure_universe_history ─────────────────────────────────────────────────

_TODAY = date(2024, 4, 30)
# Weekdays from 2023-06-01 to 2024-06-28 are the fake market's sessions (SPY prints on all of them).
_DATA_START = date(2023, 6, 1)
_DATA_END = date(2024, 6, 28)
_FEB_FIRST = date(2024, 2, 1)

# symbol -> (price, volume before 2024-02-01, volume from 2024-02-01); the dollar volume is price x volume.
_FAKE_MARKET = {
    "AAA": (50.0, 1_000_000, 1_000_000),  # $50M every day
    "BBB": (20.0, 2_000_000, 2_000_000),  # $40M every day
    "CCC": (5.0, 20_000_000, 20_000_000),  # $100M a day but a $5 price: under min_price, never listed
    "DDD": (100.0, 100_000, 100_000),  # $10M a day: under min_dollar_volume, never listed
    "FFF": (50.0, 100_000, 5_000_000),  # $5M until January, then $250M a day from 2024-02-01
}
# 195 fillers (ZF000...) that never pass the price screen, to make a pool of 200 for the absent-share tests (1% = 2).
_FILLERS = [f"ZF{index:03d}" for index in range(195)]
_FAKE_MARKET.update({symbol: (5.0, 1_000_000, 1_000_000) for symbol in _FILLERS})
_CORE_POOL = ["AAA", "BBB", "CCC", "DDD", "FFF"]
# February's list ranks on the 63 sessions before 2024-02-01: FFF averages $5M, so it is out -> AAA, BBB.
_FEB_LIST = ("AAA", "BBB")
# March's list ranks on the 63 sessions before 2024-03-01, the last 21 of which are February's: FFF averages
# (42 x $5M + 21 x $250M) / 63 = $86.67M, the top -> FFF, AAA, BBB.
_MAR_LIST = ("FFF", "AAA", "BBB")


def _fake_raw(symbol: str) -> pd.DataFrame:
    """One symbol's `auto_adjust=False, actions=True` download over the whole fake market."""
    index = pd.bdate_range(_DATA_START, _DATA_END)
    if symbol == "SPY":
        price, volumes = 400.0, [1_000_000.0] * len(index)
    else:
        price, before, after = _FAKE_MARKET[symbol]
        volumes = [float(before if day < pd.Timestamp(_FEB_FIRST) else after) for day in index]
    return pd.DataFrame(
        {
            "Open": price,
            "High": price,
            "Low": price,
            "Close": price,
            "Adj Close": price,
            "Volume": volumes,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=index,
    )


class _FakeDownload:
    """Stands in for `yfinance.download`: batched frames for the requested window, symbols it does not know absent."""

    def __init__(self, *, fail_with: Exception | None = None, missing: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self.fail_with = fail_with
        self.missing = missing

    def __call__(self, symbols: list[str], **kwargs: object) -> pd.DataFrame:
        self.calls.append({"symbols": list(symbols), **kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        frames = {}
        for symbol in symbols:
            if symbol in self.missing or (symbol != "SPY" and symbol not in _FAKE_MARKET):
                continue
            raw = _fake_raw(symbol)
            frames[symbol] = raw[(raw.index >= pd.Timestamp(kwargs["start"])) & (raw.index < pd.Timestamp(kwargs["end"]))]
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _pool() -> list[str]:
    return sorted(_CORE_POOL)


def _big_pool() -> list[str]:
    """200 symbols: two missing symbols are exactly 1% of it, three are 1.5%."""
    return sorted([*_CORE_POOL, *_FILLERS])


def _big_history_pool() -> batch.HistoryPool:
    return batch.HistoryPool(_big_pool(), [])


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """No real waiting: time.sleep does nothing and the shared rate-limit gate runs on a fake clock."""
    clock = FakeClock()
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(batch, "rate_limit_gate", RateLimitGate(30.0, 300.0, now=clock.now, sleep=clock.sleep))
    return clock


def _ensure(tmp_path: Path, start: date, end: date, download: _FakeDownload, **kwargs: object) -> UniverseHistory:
    options = {
        "path": tmp_path / "history.json",
        "cache_dir": tmp_path / "cache",
        "today": lambda: _TODAY,
        "download": download,
        "pool": lambda: batch.HistoryPool(_pool(), []),
    }
    return batch.ensure_universe_history(start, end, **{**options, **kwargs})


def _saved_history(months: dict[str, tuple[str, ...]], **overrides: object) -> UniverseHistory:
    fields = {
        "built": date(2024, 3, 1),
        "pool_date": date(2024, 3, 1),
        "top_n": 1200,
        "min_price": 10.0,
        "min_dollar_volume": 20_000_000.0,
        "lookback": 63,
        "months": months,
    }
    return UniverseHistory(**{**fields, **overrides})


@pytest.mark.usefixtures("no_sleep")
def test_a_covering_history_is_reused_without_downloading(tmp_path: Path) -> None:
    saved = _saved_history({"2024-02": ("ZZZ",), "2024-03": ("ZZZ", "YYY")})
    saved.save(tmp_path / "history.json")
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), download, pool=lambda: pytest.fail("pool fetched"))

    assert history == saved
    assert download.calls == []


@pytest.mark.usefixtures("no_sleep")
def test_a_missing_history_is_built_and_saved(tmp_path: Path) -> None:
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), download)

    assert history.months == {"2024-02": _FEB_LIST, "2024-03": _MAR_LIST}
    assert (history.top_n, history.min_price, history.min_dollar_volume, history.lookback) == (1200, 10.0, 20_000_000.0, 63)
    assert (history.built, history.pool_date) == (_TODAY, _TODAY)
    assert UniverseHistory.load(tmp_path / "history.json") == history
    # The download starts 120 calendar days before 2024-02-01 and ends the day after today (Yahoo's end is exclusive).
    assert {(call["start"], call["end"]) for call in download.calls} == {("2023-10-04", "2024-05-01")}
    assert all(call["auto_adjust"] is False and call["actions"] is True and call["group_by"] == "ticker" for call in download.calls)
    assert sorted(symbol for call in download.calls for symbol in call["symbols"]) == ["AAA", "BBB", "CCC", "DDD", "FFF", "SPY"]


@pytest.mark.usefixtures("no_sleep")
def test_each_month_matches_monthly_lists_on_the_same_data(tmp_path: Path) -> None:
    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), _FakeDownload())

    frames = {symbol: parse_yahoo_frame(_fake_raw(symbol)) for symbol in _pool()}
    sessions = [ts.date() for ts in parse_yahoo_frame(_fake_raw("SPY")).index]
    expected = monthly_lists(frames, sessions, ["2024-02", "2024-03"], min_price=10.0, min_dollar_volume=20_000_000, top_n=1200)
    assert history.months == expected


@pytest.mark.usefixtures("no_sleep")
def test_the_pool_is_downloaded_in_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "HISTORY_BATCH_SIZE", 2)
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download)

    assert history.months == {"2024-02": _FEB_LIST}
    assert [len(call["symbols"]) for call in download.calls] == [2, 2, 2]  # five pool symbols and SPY


@pytest.mark.usefixtures("no_sleep")
def test_a_history_is_extended_with_only_the_missing_months(tmp_path: Path) -> None:
    # The saved February list is a sentinel the data would never produce: it must come back untouched.
    _saved_history({"2024-02": ("ZZZ",)}).save(tmp_path / "history.json")
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), download)

    assert history.months == {"2024-02": ("ZZZ",), "2024-03": _MAR_LIST}
    assert UniverseHistory.load(tmp_path / "history.json") == history
    # Only March is computed, so the download starts 120 days before 2024-03-01.
    assert {call["start"] for call in download.calls} == {"2023-11-02"}


@pytest.mark.usefixtures("no_sleep")
def test_a_history_built_with_other_parameters_is_rebuilt(tmp_path: Path) -> None:
    _saved_history({"2024-02": ("ZZZ",)}, top_n=500).save(tmp_path / "history.json")

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload())

    assert history.months == {"2024-02": _FEB_LIST}
    assert history.top_n == 1200


@pytest.mark.usefixtures("no_sleep")
def test_cached_bars_are_not_downloaded_again(tmp_path: Path) -> None:
    _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), _FakeDownload())
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), download, path=tmp_path / "other.json", pool=lambda: pytest.fail("pool fetched"))

    assert download.calls == []
    assert history.months == {"2024-02": _FEB_LIST, "2024-03": _MAR_LIST}


@pytest.mark.usefixtures("no_sleep")
def test_a_pool_older_than_the_last_month_needed_is_fetched_again(tmp_path: Path) -> None:
    pools: list[str] = []

    def pool() -> batch.HistoryPool:
        pools.append("fetched")
        return batch.HistoryPool(_pool(), [])

    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=pool)
    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), path=tmp_path / "b.json", pool=pool)  # pool dated 2024-04-30, reused
    assert pools == ["fetched"]

    _ensure(tmp_path, date(2024, 5, 6), date(2024, 5, 28), _FakeDownload(), today=lambda: date(2024, 5, 30), path=tmp_path / "c.json", pool=pool)
    assert pools == ["fetched", "fetched"]  # a May request needs a pool dated in May or later


@pytest.mark.usefixtures("no_sleep")
def test_a_larger_window_than_the_cached_one_is_downloaded_again(tmp_path: Path) -> None:
    _ensure(tmp_path, date(2024, 3, 4), date(2024, 3, 28), _FakeDownload())
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), download, path=tmp_path / "other.json")

    assert history.months == {"2024-02": _FEB_LIST, "2024-03": _MAR_LIST}
    # The cache starts at 2023-11-02, too late for February: every symbol is fetched again from 2023-10-04.
    assert {call["start"] for call in download.calls} == {"2023-10-04"}
    assert sorted(symbol for call in download.calls for symbol in call["symbols"]) == ["AAA", "BBB", "CCC", "DDD", "FFF", "SPY"]


@pytest.mark.usefixtures("no_sleep")
def test_a_few_absent_symbols_are_left_out_and_counted(tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    download = _FakeDownload(missing=("BBB",))  # 1 of 200 pool symbols = 0.5%, under the 1% limit

    with caplog.at_level(logging.WARNING):
        history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)

    assert history.months == {"2024-02": ("AAA",)}
    assert history.bars_absent == ("BBB",)
    assert "BBB" in caplog.text
    assert "1 symbol(s) with no Yahoo data" in capsys.readouterr().out


@pytest.mark.usefixtures("no_sleep")
def test_exactly_the_limit_of_absent_symbols_proceeds_and_is_recorded(tmp_path: Path) -> None:
    download = _FakeDownload(missing=("ZF000", "BBB"))  # 2 of 200 = exactly 1%, not above it

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)

    assert history.bars_absent == ("BBB", "ZF000")  # sorted
    assert UniverseHistory.load(tmp_path / "history.json").bars_absent == ("BBB", "ZF000")


@pytest.mark.usefixtures("no_sleep")
def test_the_history_records_the_pool_size_and_the_unanswered_candidates(tmp_path: Path) -> None:
    pool = lambda: batch.HistoryPool(_big_pool(), ["UN1", "UN2"])  # noqa: E731

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=pool)

    assert (history.pool_size, history.unanswered, history.bars_absent) == (200, ("UN1", "UN2"), ())
    assert UniverseHistory.load(tmp_path / "history.json") == history


@pytest.mark.usefixtures("no_sleep")
def test_the_unanswered_candidates_survive_a_reused_pool(tmp_path: Path) -> None:
    pool = lambda: batch.HistoryPool(_pool(), ["UN1"])  # noqa: E731
    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=pool)

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), _FakeDownload(), pool=lambda: pytest.fail("pool fetched"))

    assert (history.pool_size, history.unanswered) == (5, ("UN1",))  # an extension keeps the latest build's values


@pytest.mark.usefixtures("no_sleep")
def test_a_plain_list_pool_has_no_unanswered_candidates(tmp_path: Path) -> None:
    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload())

    assert (history.pool_size, history.unanswered) == (5, ())


@pytest.mark.usefixtures("no_sleep")
def test_nothing_is_printed_about_absent_symbols_when_there_are_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload())

    assert "no Yahoo data" not in capsys.readouterr().out


@pytest.mark.usefixtures("no_sleep")
def test_too_many_absent_symbols_raise_and_save_no_history(tmp_path: Path) -> None:
    download = _FakeDownload(missing=("AAA", "BBB", "ZF000"))  # 3 of 200 = 1.5%, above the 1% limit

    with pytest.raises(UniverseHistoryError, match=r"3 of 200.*AAA.*BBB.*ZF000"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)

    assert not (tmp_path / "history.json").exists()


@pytest.mark.usefixtures("no_sleep")
def test_the_error_samples_at_most_20_absent_symbols(tmp_path: Path) -> None:
    download = _FakeDownload(missing=tuple(_FILLERS))  # 195 of 200 absent

    with pytest.raises(UniverseHistoryError) as raised:
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)

    assert "195 of 200" in str(raised.value)
    assert "ZF019" in str(raised.value)
    assert "ZF020" not in str(raised.value)


def test_an_all_empty_batch_is_retried_and_the_second_attempt_succeeds(tmp_path: Path, no_sleep: FakeClock) -> None:
    real = _FakeDownload()
    attempts: list[int] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        attempts.append(len(symbols))
        if len(attempts) == 1:
            return pd.DataFrame()  # Yahoo swallowed a rate limit: nothing for any symbol
        return real(symbols, **kwargs)

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download)  # type: ignore[arg-type]

    assert history.months == {"2024-02": _FEB_LIST}
    assert attempts == [6, 6]  # the whole batch (five pool symbols and SPY), twice
    assert no_sleep.sleep_calls == []  # an empty batch backs off with time.sleep, not the gate cooldown


def test_a_batch_whose_every_symbol_comes_back_empty_is_retried(tmp_path: Path, no_sleep: FakeClock) -> None:
    real = _FakeDownload()
    attempts: list[int] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        attempts.append(len(symbols))
        if len(attempts) == 1:  # a frame with every requested column present but all NaN
            return pd.concat({symbol: _fake_raw("AAA") * float("nan") for symbol in symbols}, axis=1)
        return real(symbols, **kwargs)

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download)  # type: ignore[arg-type]

    assert history.months == {"2024-02": _FEB_LIST}
    assert attempts == [6, 6]


def test_the_second_pass_recovers_a_symbol_that_was_absent_the_first_time(tmp_path: Path, no_sleep: FakeClock) -> None:
    real = _FakeDownload()
    asked: list[list[str]] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        asked.append(list(symbols))
        if len(asked) == 1:  # the first pass: Yahoo drops BBB from the batch
            return real([symbol for symbol in symbols if symbol != "BBB"], **kwargs)
        return real(symbols, **kwargs)

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)  # type: ignore[arg-type]

    assert history.months == {"2024-02": _FEB_LIST}  # BBB is back
    assert len(asked) == 4  # three batches of the 201 symbols, then BBB once more, alone
    assert asked[-1] == ["BBB"]
    assert no_sleep.sleep_calls == [30.0]  # after a pause through the shared gate


@pytest.mark.usefixtures("no_sleep")
def test_a_symbol_still_absent_is_asked_for_only_once_more(tmp_path: Path) -> None:
    download = _FakeDownload(missing=("BBB",))

    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=_big_history_pool)

    assert [call["symbols"] for call in download.calls].count(["BBB"]) == 1


@pytest.mark.usefixtures("no_sleep")
def test_a_symbol_that_loses_its_data_on_a_new_download_is_not_ranked_on_the_old_one(tmp_path: Path) -> None:
    _ensure(tmp_path, date(2024, 3, 4), date(2024, 3, 28), _FakeDownload(), pool=_big_history_pool)

    # February needs an earlier window, so every symbol is downloaded again; Yahoo now has nothing for BBB.
    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(missing=("BBB",)), path=tmp_path / "other.json", pool=_big_history_pool)

    assert history.months == {"2024-02": ("AAA",)}


@pytest.mark.usefixtures("no_sleep")
def test_an_index_entry_without_its_file_is_downloaded_again(tmp_path: Path) -> None:
    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload())
    (tmp_path / "cache" / "bars" / "BBB.parquet").unlink()  # a pruned cache
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, path=tmp_path / "other.json")

    assert history.months == {"2024-02": _FEB_LIST}  # BBB is still ranked
    assert [call["symbols"] for call in download.calls] == [["BBB"]]
    assert (tmp_path / "cache" / "bars" / "BBB.parquet").exists()


@pytest.mark.usefixtures("no_sleep")
def test_a_crash_while_overwriting_a_file_leaves_no_index_entry_for_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure(tmp_path, date(2024, 3, 4), date(2024, 3, 28), _FakeDownload())
    real_to_parquet = pd.DataFrame.to_parquet

    def to_parquet(self: pd.DataFrame, path: object, *args: object, **kwargs: object) -> None:
        if "BBB" in str(path):
            raise OSError("disk full")
        real_to_parquet(self, path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)

    # An earlier window re-downloads every symbol; the write of BBB's new file crashes.
    with pytest.raises(UniverseHistoryError, match="disk full"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), _FakeDownload(), path=tmp_path / "other.json")

    index = json.loads((tmp_path / "cache" / "bars" / "index.json").read_text())
    assert "BBB" not in index  # never claims a window its file does not hold
    assert sorted(path.name for path in (tmp_path / "cache" / "bars").glob("*.tmp")) == []
    monkeypatch.undo()
    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 3, 28), _FakeDownload(), path=tmp_path / "other.json")
    assert history.months == {"2024-02": _FEB_LIST, "2024-03": _MAR_LIST}


@pytest.mark.usefixtures("no_sleep")
def test_a_download_failure_raises_universe_history_error(tmp_path: Path) -> None:
    download = _FakeDownload(fail_with=RuntimeError("yahoo down"))

    with pytest.raises(UniverseHistoryError, match="yahoo down"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download)

    assert len(download.calls) == 3  # three attempts
    assert not (tmp_path / "history.json").exists()


def test_a_rate_limited_download_waits_on_the_shared_gate_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(batch, "rate_limit_gate", RateLimitGate(30.0, 300.0, now=clock.now, sleep=clock.sleep))
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    real = _FakeDownload()
    attempts: list[int] = []

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        attempts.append(len(symbols))
        if len(attempts) == 1:
            raise YFRateLimitError()
        return real(symbols, **kwargs)

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download)  # type: ignore[arg-type]

    assert history.months == {"2024-02": _FEB_LIST}
    assert clock.sleep_calls == [30.0]  # the retry waited out the cooldown


@pytest.mark.usefixtures("no_sleep")
def test_a_pool_failure_raises_universe_history_error(tmp_path: Path) -> None:
    def pool() -> batch.HistoryPool:
        raise RuntimeError("nasdaq down")

    with pytest.raises(UniverseHistoryError, match="nasdaq down"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=pool)


@pytest.mark.usefixtures("no_sleep")
def test_an_empty_month_raises_universe_history_error(tmp_path: Path) -> None:
    # CCC trades under $10 and DDD under $20M a day: nothing is listable.
    with pytest.raises(UniverseHistoryError, match="2024-02"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=lambda: batch.HistoryPool(["CCC", "DDD"], []))

    assert not (tmp_path / "history.json").exists()


@pytest.mark.usefixtures("no_sleep")
def test_missing_spy_sessions_raise_universe_history_error(tmp_path: Path) -> None:
    with pytest.raises(UniverseHistoryError, match="SPY"):
        _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(missing=("SPY",)))


@pytest.mark.usefixtures("no_sleep")
def test_a_window_without_enough_prior_sessions_raises_universe_history_error(tmp_path: Path) -> None:
    # The fake market starts 2023-06-01, so a request for 2023-07 has under 63 prior sessions.
    with pytest.raises(UniverseHistoryError, match="2023-07"):
        _ensure(tmp_path, date(2023, 7, 3), date(2023, 7, 31), _FakeDownload())


# ── concurrent builds ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("no_sleep")
def test_a_build_waits_for_the_lock_then_finds_the_other_builds_history(tmp_path: Path) -> None:
    finished = _saved_history({"2024-02": ("ZZZ",)})
    cache = tmp_path / "cache"
    locked = threading.Event()

    def other_process() -> None:
        with batch._build_lock(cache):
            locked.set()
            threading.Event().wait(0.3)  # the other build takes its time...
            finished.save(tmp_path / "history.json")  # ...and saves its history before releasing the lock

    thread = threading.Thread(target=other_process)
    thread.start()
    assert locked.wait(5)
    download = _FakeDownload()

    history = _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), download, pool=lambda: pytest.fail("pool fetched"))
    thread.join()

    assert history == finished  # re-read once the lock was held: nothing was built
    assert download.calls == []


@pytest.mark.usefixtures("no_sleep")
def test_a_build_runs_under_an_exclusive_lock_on_the_cache_directory(tmp_path: Path) -> None:
    held: list[bool] = []

    def pool() -> batch.HistoryPool:
        with (tmp_path / "cache" / ".lock").open("w") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(True)
            else:
                held.append(False)
        return batch.HistoryPool(_pool(), [])

    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload(), pool=pool)

    assert held == [True]
    # Released afterwards: a later build can take it.
    with batch._build_lock(tmp_path / "cache"):
        pass


def test_a_build_that_has_to_wait_for_the_lock_says_so(tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    cache = tmp_path / "cache"
    locked = threading.Event()
    release = threading.Event()

    def other_process() -> None:
        with batch._build_lock(cache):
            locked.set()
            release.wait(5)

    thread = threading.Thread(target=other_process)
    thread.start()
    assert locked.wait(5)
    waited: list[bool] = []

    def waiter() -> None:
        with batch._build_lock(cache):
            waited.append(True)

    with caplog.at_level(logging.WARNING, logger=batch.logger.name):
        second = threading.Thread(target=waiter)
        second.start()
        for _ in range(100):  # the waiter says so before it blocks; give it a moment, then release the first build
            if "waiting" in caplog.text:
                break
            threading.Event().wait(0.05)
        release.set()
        second.join(5)
        thread.join(5)

    assert waited == [True]  # it did get the lock once the first build ended
    assert "another universe-history build holds" in caplog.text
    assert "another universe-history build holds" in capsys.readouterr().out


def test_an_uncontended_build_lock_prints_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    with caplog.at_level(logging.WARNING, logger=batch.logger.name):
        with batch._build_lock(tmp_path / "cache"):
            pass

    assert caplog.text == ""
    assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("no_sleep")
def test_a_covering_history_is_returned_without_taking_the_lock(tmp_path: Path) -> None:
    _saved_history({"2024-02": ("ZZZ",)}).save(tmp_path / "history.json")

    _ensure(tmp_path, date(2024, 2, 5), date(2024, 2, 28), _FakeDownload())

    assert not (tmp_path / "cache" / ".lock").exists()
