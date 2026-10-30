from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame
from tests.fakes import FakeStockHistoricalDataClient, bar_payload, make_alpaca_position

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.dashboard.pnl import Dividend, Fee
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.errors import ConfigurationError

KEYS = "ALPACA_API_KEY=trade-key\nALPACA_API_SECRET=trade-secret\nALPACA_DATA_API_KEY=data-key\nALPACA_DATA_API_SECRET=data-secret\n"
SHOWN = "DASHBOARD_DISPLAY=true\n"


def _env(env_dir: Path, name: str, body: str) -> None:
    env_dir.mkdir(exist_ok=True)
    (env_dir / name).write_text(body, encoding="utf-8")


def test_strategies_are_the_alpaca_env_files_of_the_mode(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER=alpaca\n" + SHOWN + KEYS)
    _env(tmp_path, ".env.beta.paper", SHOWN + KEYS)  # no BROKER: alpaca
    _env(tmp_path, ".env.gamma.paper", "BROKER=ibkr\n" + SHOWN)
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\n" + SHOWN + KEYS)
    _env(tmp_path, ".env.alpha.backtesting", "BROKER=alpaca\n" + SHOWN + KEYS)
    _env(tmp_path, ".env.example", SHOWN + KEYS)

    assert account_reader.strategies_for_mode("paper", tmp_path) == ["alpha", "beta"]
    assert account_reader.strategies_for_mode("live", tmp_path) == ["alpha"]


def test_a_strategy_without_dashboard_display_is_hidden(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.live", SHOWN + KEYS)
    _env(tmp_path, ".env.beta.live", KEYS)  # no DASHBOARD_DISPLAY: hidden
    _env(tmp_path, ".env.gamma.live", "DASHBOARD_DISPLAY=false\n" + KEYS)

    assert account_reader.strategies_for_mode("live", tmp_path) == ["alpha"]


@pytest.mark.parametrize(("value", "shown"), [("true", True), ("TRUE", True), ("1", True), ("Yes", True), (" yes ", True), ("no", False), ("0", False), ("", False), ("ture", False)])
def test_dashboard_display_is_on_only_for_true_1_or_yes(tmp_path: Path, value: str, shown: bool) -> None:
    _env(tmp_path, ".env.alpha.live", f"DASHBOARD_DISPLAY={value}\n" + KEYS)

    assert account_reader.strategies_for_mode("live", tmp_path) == (["alpha"] if shown else [])


def test_no_env_dir_lists_no_strategy(tmp_path: Path) -> None:
    assert account_reader.strategies_for_mode("paper", tmp_path / "missing") == []


def test_credentials_come_from_the_file_without_touching_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_DATA_API_KEY", "ALPACA_DATA_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\nBROKER_API_IS_PAPER=false\n" + KEYS)

    trading, data = account_reader.load_account_credentials("alpha", "live", tmp_path)

    assert (trading.api_key, trading.api_secret, trading.is_paper) == ("trade-key", "trade-secret", False)
    assert (data.api_key, data.api_secret) == ("data-key", "data-secret")
    assert "ALPACA_API_KEY" not in os.environ


def test_the_endpoint_is_the_one_the_keys_file_selects_whatever_the_file_name(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER_API_IS_PAPER=false\n" + KEYS)
    _env(tmp_path, ".env.alpha.live", KEYS)
    _env(tmp_path, ".env.beta.paper", KEYS)

    assert account_reader.account_is_paper("alpha", "paper", tmp_path) is False
    assert account_reader.account_is_paper("alpha", "live", tmp_path) is True
    assert account_reader.account_is_paper("beta", "paper", tmp_path) is True


def test_blank_keys_raise_a_configuration_error_without_the_values(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "ALPACA_API_KEY=\nALPACA_API_SECRET=s3cret\n")

    with pytest.raises(ConfigurationError, match="ALPACA_API_KEY") as excinfo:
        account_reader.load_account_credentials("alpha", "paper", tmp_path)

    assert "s3cret" not in str(excinfo.value)


def test_a_missing_env_file_raises_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"\.env\.alpha\.paper"):
        account_reader.load_account_credentials("alpha", "paper", tmp_path)


def test_load_strategy_pnl_builds_the_clients_from_the_strategys_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER_API_IS_PAPER=true\n" + KEYS)
    built: dict[str, object] = {}
    monkeypatch.setattr(account_reader, "build_trading_client", lambda creds: built.setdefault("trading", creds))
    monkeypatch.setattr(account_reader, "build_stock_data_client", lambda creds: built.setdefault("data", creds))
    monkeypatch.setattr(account_reader, "load_account_pnl", lambda trading, data, today: (trading, data, today))

    result = account_reader.load_strategy_pnl("alpha", "paper", tmp_path, date(2026, 10, 13))

    trading, data, today = result  # ty: ignore[not-iterable]
    assert trading is built["trading"] and data is built["data"]
    assert trading.api_key == "trade-key" and trading.is_paper is True
    assert data.api_key == "data-key"
    assert today == date(2026, 10, 13)


def _fill_activity(n: int, symbol: str = "AAA", side: str = "buy", qty: str = "10", price: str = "100", time: str = "2026-10-05T15:00:00.123456789Z") -> dict[str, str]:
    return {
        "id": f"20261005150000000::{n:08d}",
        "activity_type": "FILL",
        "transaction_time": time,
        "type": "fill",
        "price": price,
        "qty": qty,
        "side": side,
        "symbol": symbol,
        "leaves_qty": "0",
        "order_id": "b5abe576-6a8a-49f3-a353-46b72c1ccae9",
        "cum_qty": qty,
        "order_status": "filled",
    }


def _fee_activity(n: int, day: str, amount: str, status: str = "executed") -> dict[str, str]:
    return {"id": f"20261006000000000::{n:08d}", "activity_type": "FEE", "activity_sub_type": "TAF", "date": f"{day}T00:00:00Z", "net_amount": amount, "status": status, "symbol": "AAA"}


def _dividend_activity(n: int, day: str, amount: str) -> dict[str, str]:
    return {"id": f"20261006000000000::{n:08d}", "activity_type": "DIV", "date": day, "net_amount": amount, "description": "Cash DIV @ 0.22, Pos QTY: 10", "status": "executed", "symbol": "AAA"}


def _withholding_activity(n: int, day: str, amount: str) -> dict[str, str]:
    description = "DIV tax withholding on $0.43 at 15% for tax country FRA; w8w9: w8"
    return {"id": f"20261006000000000::{n:08d}", "activity_type": "DIVNRA", "date": day, "net_amount": amount, "description": description, "status": "executed"}


class FakeAccountClient:
    """Answers `GET /account/activities` from `activities` like Alpaca: `page_size` items after the `page_token` id."""

    def __init__(self, activities: list[dict[str, str]], positions: list[object] | None = None) -> None:
        self.activities = activities
        self.positions = positions or []
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, path: str, data: dict[str, object] | None = None) -> list[dict[str, str]]:
        params = dict(data or {})
        self.requests.append((path, params))
        start = 0
        if "page_token" in params:
            start = next(i for i, item in enumerate(self.activities) if item["id"] == params["page_token"]) + 1
        return self.activities[start : start + int(params["page_size"])]  # ty: ignore[invalid-argument-type]

    def get_all_positions(self) -> list[object]:
        return self.positions


def test_activities_are_fills_and_fees_oldest_first() -> None:
    client = FakeAccountClient([_fill_activity(1), _fill_activity(2, side="sell", price="110"), _fee_activity(3, "2026-10-06", "-0.02")])

    fills, fees, _ = account_reader.fetch_activities(client)

    assert client.requests == [("/account/activities", {"activity_types": "FILL,FEE,DIV,DIVNRA", "direction": "asc", "page_size": 100})]
    assert [(f.symbol, f.side, f.qty, f.price) for f in fills] == [("AAA", "buy", 10.0, 100.0), ("AAA", "sell", 10.0, 110.0)]
    assert fills[0].time.isoformat() == "2026-10-05T15:00:00.123456+00:00"
    assert fees == [Fee(date(2026, 10, 6), -0.02)]


def test_activities_follow_the_page_token_through_a_full_page_to_an_empty_one() -> None:
    client = FakeAccountClient([_fill_activity(n) for n in range(100)])

    fills, _, _ = account_reader.fetch_activities(client)

    assert len(fills) == 100
    assert len(client.requests) == 2
    assert client.requests[1][1]["page_token"] == "20261005150000000::00000099"


def test_activities_over_several_pages_are_all_read() -> None:
    client = FakeAccountClient([_fill_activity(n) for n in range(150)])

    fills, _, _ = account_reader.fetch_activities(client)

    assert len(fills) == 150
    assert len(client.requests) == 2


def test_a_cancelled_fee_is_left_out() -> None:
    client = FakeAccountClient([_fee_activity(1, "2026-10-06", "-0.02", status="canceled")])

    assert account_reader.fetch_activities(client) == ([], [], [])


def test_dividends_and_their_tax_withholding_are_read() -> None:
    client = FakeAccountClient([_dividend_activity(1, "2026-10-06", "0.43"), _withholding_activity(2, "2026-10-06", "-0.06")])

    _, _, dividends = account_reader.fetch_activities(client)

    assert dividends == [Dividend(date(2026, 10, 6), 0.43), Dividend(date(2026, 10, 6), -0.06)]


def test_daily_closes_are_raw_iex_day_bars_keyed_by_new_york_date() -> None:
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAA": [bar_payload("2026-10-05T04:00:00Z", 101.0), bar_payload("2026-10-06T04:00:00Z", 102.0)]}

    closes = account_reader.fetch_daily_closes(data, ["AAA", "BBB"], date(2026, 10, 5), date(2026, 10, 7))

    assert closes == {"AAA": {date(2026, 10, 5): 101.0, date(2026, 10, 6): 102.0}}
    (request,) = data.bars_requests
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.RAW
    assert request.timeframe.value == TimeFrame.Day.value
    # alpaca-py stores the bounds as naive UTC: midnight in New York is 04:00 UTC in October. Ending there leaves
    # today's partial bar out.
    assert request.start == datetime(2026, 10, 5, 4, 0)
    assert request.end == datetime(2026, 10, 7, 4, 0)


def test_no_symbol_or_a_first_day_of_today_fetches_no_bars() -> None:
    data = FakeStockHistoricalDataClient()

    assert account_reader.fetch_daily_closes(data, [], date(2026, 10, 5), date(2026, 10, 7)) == {}
    assert account_reader.fetch_daily_closes(data, ["AAA"], date(2026, 10, 7), date(2026, 10, 7)) == {}
    assert data.bars_requests == []


def test_load_account_pnl_rebuilds_the_days_from_the_three_endpoints() -> None:
    trading = FakeAccountClient(
        [_fill_activity(1, time="2026-10-05T15:00:00Z")],
        positions=[make_alpaca_position(symbol="AAA", qty="10", unrealized_pl="30.00", unrealized_plpc="0.03", current_price="103.00", market_value="1030.00")],
    )
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAA": [bar_payload("2026-10-05T04:00:00Z", 101.0), bar_payload("2026-10-06T04:00:00Z", 102.0)]}

    result = account_reader.load_account_pnl(trading, data, date(2026, 10, 7))

    assert result.days == [
        {"date": "2026-10-05", "realized": 0.0, "unrealized": 10.0, "change": 10.0},
        {"date": "2026-10-06", "realized": 0.0, "unrealized": 20.0, "change": 10.0},
        {"date": "2026-10-07", "realized": 0.0, "unrealized": 30.0, "change": 10.0},
    ]
    (position,) = result.positions
    assert (position.symbol, position.qty, position.current_price, position.unrealized_plpc) == ("AAA", 10.0, 103.0, 0.03)
    assert result.mismatches == []


# --- daily returns: portfolio history and benchmark closes ------------------------------------------


class FakeHistoryClient:
    """Answers `GET /account/portfolio/history` with a canned payload and records the request."""

    def __init__(self, history: dict[str, list[object]]) -> None:
        self.history = history
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, path: str, data: dict[str, object] | None = None) -> dict[str, list[object]]:
        self.requests.append((path, dict(data or {})))
        return self.history


def _stamp(day: str) -> int:
    return int(datetime.fromisoformat(f"{day}T00:00:00+00:00").timestamp())


def test_equity_closes_move_alpacas_next_day_labels_back_to_the_day_of_the_close() -> None:
    # Alpaca labels Monday's close Tuesday and Friday's close Saturday.
    client = FakeHistoryClient({"timestamp": [_stamp("2026-10-06"), _stamp("2026-10-10"), _stamp("2026-10-13")], "equity": [10000.0, 9957.42, None]})

    closes = account_reader.fetch_equity_closes(client, date(2026, 10, 12), date(2026, 10, 13))  # ty: ignore[invalid-argument-type]

    assert closes == {date(2026, 10, 5): 10000.0, date(2026, 10, 9): 9957.42}  # the null point is dropped
    ((path, params),) = client.requests
    assert path == "/account/portfolio/history"
    assert params == {"timeframe": "1D", "start": "2026-10-05", "end": "2026-10-14"}  # a week of base before the first day


def test_an_empty_history_gives_no_equity() -> None:
    assert account_reader.fetch_equity_closes(FakeHistoryClient({}), date(2026, 10, 12), date(2026, 10, 13)) == {}  # ty: ignore[invalid-argument-type]


class FakeBenchmarkBars:
    """Stands in for `YahooDailyBars`: daily closes per symbol, each stamped at its session's close (16:00 New York)."""

    def __init__(self, closes: dict[str, dict[str, float]]) -> None:
        self.closes = closes
        self.requests: list[tuple[list[str], date]] = []

    def bars(self, symbols: list[str], today: date) -> dict[str, Bars]:
        self.requests.append((list(symbols), today))
        frames = {}
        for symbol in symbols:
            if symbol in self.closes:
                index = pd.DatetimeIndex([datetime.fromisoformat(f"{day}T16:00:00-04:00") for day in self.closes[symbol]], name="timestamp")
                frames[symbol] = Bars(Asset(symbol), "day", pd.DataFrame({"close": list(self.closes[symbol].values())}, index=index))
        return frames


def test_benchmark_closes_are_the_yahoo_closes_by_session_day_from_a_week_before_the_first_day() -> None:
    source = FakeBenchmarkBars({"SPY": {"2026-10-01": 98.0, "2026-10-09": 100.0, "2026-10-12": 101.0, "2026-10-13": 100.5}})

    closes = account_reader.fetch_benchmark_closes(source, date(2026, 10, 12), date(2026, 10, 13))

    assert closes == {date(2026, 10, 9): 100.0, date(2026, 10, 12): 101.0, date(2026, 10, 13): 100.5}  # the 1st is before the week of base
    assert source.requests == [(["SPY"], date(2026, 10, 13))]


def test_benchmark_closes_are_empty_when_yahoo_has_no_spy() -> None:
    assert account_reader.fetch_benchmark_closes(FakeBenchmarkBars({}), date(2026, 10, 12), date(2026, 10, 13)) == {}


def test_load_return_bases_reads_the_equity_history_and_spys_closes() -> None:
    trading = FakeHistoryClient({"timestamp": [_stamp("2026-10-13")], "equity": [1000.0]})
    source = FakeBenchmarkBars({"SPY": {"2026-10-09": 100.0, "2026-10-12": 101.0}})

    bases = account_reader.load_return_bases(trading, source, date(2026, 10, 12), date(2026, 10, 12))  # ty: ignore[invalid-argument-type]

    assert dict(bases.equity) == {date(2026, 10, 12): 1000.0}
    assert dict(bases.benchmark) == {date(2026, 10, 9): 100.0, date(2026, 10, 12): 101.0}
