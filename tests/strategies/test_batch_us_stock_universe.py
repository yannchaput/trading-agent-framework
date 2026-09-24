from __future__ import annotations

from yfinance.exceptions import YFRateLimitError

from trading_agent_framework.strategies.cross_momentum.batch_us_stock_universe import (
    RateLimitGate,
    get_extra_universe_symbols,
    is_rate_limit_error,
    merge_ticker_candidates,
    normalize_symbol,
    parse_ishares_csv,
    parse_nasdaq_rows,
    parse_vanguard_holdings,
)


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
