"""End to end: a daily backtest with the real executor, the real backtest broker, fake agents (they call the real tools) and a fake dated Form 4 source."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.insider_trades import InsiderTradesStrategy
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing, Form4Trade
from trading_agent_framework.strategies.insider_trades.form4.source import KnownInsiderData, LoadReport
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.utils.errors import AgentError, BacktestError

PRIOR = weekday_sessions(date(2026, 9, 9), 3)  # history before the window: the first tick needs a last price
SESSIONS = weekday_sessions(date(2026, 9, 14), 14)  # Mon 14 Sep .. Thu 1 Oct
PRICES = {"AAA": 50.0, "BBB": 25.0, "CCC": 20.0}
PARAMS = InsiderParams(watchlist=("AAA", "BBB", "CCC"), max_total_weight=0.90, min_weight=0.03, max_position_weight=0.40)  # even weight 0.30


def _trade(accession: str, ticker: str, cik: str, *, filed: date, traded: date, code: str, shares: int, owned_after: int) -> Form4Trade:
    return Form4Trade(accession, filed, ticker, cik, f"OWNER {cik}", "CEO", code, traded, Decimal(shares), Decimal(10), Decimal(owned_after), False, ())


def _filing(trade: Form4Trade) -> Form4Filing:
    return Form4Filing(trade.accession, trade.filed, trade.ticker, "0000000001", trade.ticker, (), None, (), (trade,), False, 0, 0, 0, {})


def _history(ticker: str, cik: str) -> list[Form4Filing]:
    """One trade in each of the three years before 2026, in three different months: the insider has no habit, so a trade of his is opportunistic."""
    return [
        _filing(_trade(f"{ticker}-H{year}", ticker, cik, filed=date(year, month, 5), traded=date(year, month, 3), code="S", shares=100, owned_after=900))
        for year, month in zip((2023, 2024, 2025), (3, 6, 9), strict=True)
    ]


F1 = _filing(_trade("AAA-F1", "AAA", "I1", filed=date(2026, 8, 20), traded=date(2026, 8, 18), code="P", shares=1000, owned_after=9000))  # known from the 1st session; leaves the window on the 6th
F2 = _filing(_trade("BBB-F2", "BBB", "I2", filed=date(2026, 9, 16), traded=date(2026, 9, 14), code="S", shares=1000, owned_after=4000))  # filed on the 3rd session: known from the 4th
F3 = _filing(_trade("CCC-F3", "CCC", "I3", filed=date(2026, 9, 22), traded=date(2026, 9, 21), code="P", shares=500, owned_after=4500))  # filed on the 7th session: known from the 8th
F4 = _filing(_trade("CCC-F4", "CCC", "I3", filed=date(2026, 9, 23), traded=date(2026, 9, 22), code="P", shares=200, owned_after=4700))  # filed on the 8th session: known from the 9th
BASE = (F1, F2, F3)


def _data(*filings: Form4Filing) -> dict[str, list[Form4Filing]]:
    rows: dict[str, list[Form4Filing]] = {ticker: _history(ticker, f"I{index}") for index, ticker in enumerate(PARAMS.watchlist, start=1)}
    for filing in filings:
        rows[filing.ticker].append(filing)
    return rows


class DatedSource:
    """What EDGAR would have published by each simulated day: `known(as_of)` keeps the filings filed strictly before `as_of`'s date.

    `leaky` returns every filing whatever the day (a broken source): `InsiderFacts` must hide the future from the agents anyway.
    """

    def __init__(self, data: dict[str, list[Form4Filing]], *, leaky: bool = False) -> None:
        self.data, self.leaky = data, leaky
        self.loaded_on: date | None = None
        self.loads: list[tuple[Any, date]] = []

    def load(self, *, as_of: Any, since: date) -> LoadReport:
        self.loads.append((as_of, since))
        self.loaded_on = as_of.date()
        return LoadReport(loaded=tuple(self.data), failed={})

    def known(self, as_of: Any) -> KnownInsiderData:
        today = as_of.date()
        return KnownInsiderData({ticker: [f for f in rows if self.leaky or f.filed < today] for ticker, rows in self.data.items()}, {})


Tools = dict[str, Callable[..., dict[str, Any]]]


class _Handle:
    def __init__(self, tools: Tools, script: Callable[[Tools, Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


class _Manager:
    """The three agents, scripted: each calls the real tools, and the manager keeps what they saw and sent, by simulated day."""

    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": self._research, "trader": self._build_book, "trader_retry": self._finish_trade}
        self.read: list[tuple[str, str, dict[str, Any]]] = []  # (day, ticker, what get_insider_trades answered)
        self.sent: list[tuple[str, str, str, int]] = []  # (day, symbol, side, quantity)

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run

    def runs(self) -> dict[str, int]:
        return {name: handle.runs for name, handle in self.handles.items()}

    def _research(self, tools: Tools, context: Any) -> None:
        ticker = context["ticker"]
        self.read.append((context["current_datetime"][:10], ticker, tools["get_insider_trades"](ticker)))
        answer = tools["submit_note"](f"{ticker}: insider activity read", [])
        assert "error" not in answer, answer

    @staticmethod
    def _weights(classes: dict[str, str]) -> dict[str, float]:
        """Tilted toward the buying stock and away from the selling one (the checkpoint's rules: buying >= none >= selling)."""
        return {ticker: {"buying": 0.40, "none": 0.25, "selling": 0.10}[signal] for ticker, signal in classes.items()}

    def _build_book(self, tools: Tools, context: Any) -> None:
        classes = {stock["ticker"]: stock["class"] for stock in context["stocks"]}
        weights = self._weights(classes)
        answer = tools["submit_target"]([{"ticker": t, "weight": w, "reason": "tilt to the signal"} for t, w in weights.items()])
        assert "error" not in answer, answer
        self._place(tools, context["current_datetime"][:10], answer["shortfalls"])

    def _finish_trade(self, tools: Tools, context: Any) -> None:
        self._place(tools, context["current_datetime"][:10], context["shortfalls"])

    def _place(self, tools: Tools, day: str, shortfalls: list[dict[str, Any]]) -> None:
        for shortfall in sorted(shortfalls, key=lambda s: s["kind"] != "sell"):
            quantity = math.floor(shortfall["dollars"] / PRICES[shortfall["symbol"]])
            if quantity > 0:
                answer = tools["place_order"](shortfall["symbol"], shortfall["kind"], quantity)
                assert "error" not in answer, answer
                self.sent.append((day, shortfall["symbol"], shortfall["kind"], quantity))
        checked = tools["check_orders"]()
        tools["submit_trade_report"]([{"order_id": o["order_id"], **({} if o["status"] == "filled" else {"reason": "still working at the check"})} for o in checked["orders"]])


def _bars(closes: list[float]) -> pd.DataFrame:
    assert len(closes) == len(PRIOR) + len(SESSIONS)
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in [*PRIOR, *SESSIONS]], name="timestamp"),
    )


def _run(tmp_path: Path, manager: _Manager, source: DatedSource) -> InsiderTradesStrategy:
    data = FakeBacktestDataSource()
    data.set_sessions(SESSIONS)
    for symbol, price in {**PRICES, "SPY": 400.0}.items():
        data.set_bars(Asset(symbol), _bars([price] * (len(PRIOR) + len(SESSIONS))))
    strategy = InsiderTradesStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="insider_trades"),
        mode=TradingMode.BACKTESTING,
        project_root=tmp_path,
        settings=PARAMS,
        source=source,
    )
    strategy._agents = cast(AgentManager, manager)
    strategy.run_backtesting(start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=data, budget=Decimal(10000), benchmark="SPY")
    return strategy


def _runs(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("runs.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _state(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "data" / "insider_trades_state_backtesting.json").read_text(encoding="utf-8"))


def _day(session_index: int) -> str:
    return SESSIONS[session_index].open.date().isoformat()


def test_the_bot_checks_every_session_but_acts_only_on_changes(tmp_path: Path) -> None:
    manager = _Manager()

    _run(tmp_path, manager, DatedSource(_data(*BASE)))

    runs = _runs(tmp_path)
    assert [r["date"] for r in runs] == [s.open.date().isoformat() for s in SESSIONS]  # a check every session
    # the first run, F2 (filed the 16th, known the 17th), F1 leaving the window (the 21st), F3 (filed the 22nd, known the 23rd)
    changes = {r["date"]: r["changed"] for r in runs if r["changed"]}
    assert changes == {_day(0): ["AAA", "BBB", "CCC"], _day(3): ["BBB"], _day(5): ["AAA"], _day(7): ["CCC"]}
    # a day after a change only checks the orders sent (they fill on the next bar), then the book is quiet
    assert [r["outcome"] for r in runs] == [
        "completed_pending", "pending_retry", "pending_complete",  # 14-16 Sep
        "completed_pending", "pending_retry",  # 17-18 Sep
        "completed_pending", "pending_retry",  # 21-22 Sep
        "completed_pending", "pending_retry", "pending_complete",  # 23-25 Sep
        "nothing_new", "nothing_new", "nothing_new", "nothing_new",  # 28 Sep - 1 Oct
    ]  # fmt: skip
    # the researcher read AAA, BBB and CCC once each; the expiry (AAA left its window) got a note written by code, with no researcher
    assert [(day, ticker) for day, ticker, _ in manager.read] == [(_day(0), "AAA"), (_day(3), "BBB"), (_day(7), "CCC")]
    assert manager.runs() == {"researcher": 3, "trader": 4, "trader_retry": 0}  # the trader ran on the 4 change days only, nothing was re-run on the others
    assert {day for day, *_ in manager.sent} <= set(changes)
    state = _state(tmp_path)
    assert state["notes"]["AAA"]["summary"].startswith("no open-market insider trade") and state["notes"]["AAA"]["as_of"] == _day(5)
    assert [state["notes"][t]["class"] for t in ("AAA", "BBB", "CCC")] == ["none", "selling", "buying"]
    assert state["last_run"] == _day(9) and state["pending_trade"] is None and state["abandoned_streak"] == 0
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report


@pytest.mark.parametrize("leaky", [False, True], ids=["dated source", "source that leaks the future"])
def test_the_agents_never_see_a_filing_before_its_filing_date(tmp_path: Path, leaky: bool) -> None:
    manager = _Manager()

    _run(tmp_path, manager, DatedSource(_data(*BASE, F4), leaky=leaky))

    # F4 is filed on the 23rd, the day the researcher reads CCC: it is not in that answer, and is in the next day's
    seen = [(day, ticker, [t["accession"] for i in result["insiders"] for t in i["trades"]]) for day, ticker, result in manager.read]
    assert seen == [
        (_day(0), "AAA", ["AAA-F1"]),
        (_day(3), "BBB", ["BBB-F2"]),
        (_day(7), "CCC", ["CCC-F3"]),
        (_day(8), "CCC", ["CCC-F4", "CCC-F3"]),
    ]
    for day, _, result in manager.read:
        assert result["as_of"] == day
        assert all(t["filed"] < day for i in result["insiders"] for t in i["trades"])  # filed strictly before the session
        assert all(t["label"] == "opportunistic" for i in result["insiders"] for t in i["trades"])  # the code's labels, from the history before the window
    assert [r["bought"]["opportunistic"] for _, ticker, result in manager.read if ticker == "CCC" for r in result["insiders"]] == [5000, 7000]
    assert "CCC-F4" in _state(tmp_path)["processed"]


def test_orders_go_out_once_and_fill_on_a_later_bar(tmp_path: Path) -> None:
    manager = _Manager()

    strategy = _run(tmp_path, manager, DatedSource(_data(*BASE)))

    runs = _runs(tmp_path)
    assert manager.sent == [
        (_day(0), "AAA", "buy", 80), (_day(0), "BBB", "buy", 100), (_day(0), "CCC", "buy", 125),  # the first book: AAA buying 40%, the others 25%
        (_day(3), "BBB", "sell", 60),  # BBB turns selling: down to 10%
        (_day(5), "AAA", "sell", 30),  # AAA's signal expired: down to 25%
        (_day(7), "CCC", "buy", 75),  # CCC turns buying: up to 40%
    ]  # fmt: skip
    # a daily backtest fills a market order on the next bar: still working at the tick that sent it
    for index in (0, 3, 5, 7):
        assert {o["status"] for o in runs[index]["orders"]} == {"working"}
        assert [(o["symbol"], o["side"]) for o in runs[index]["orders"]] == [(s, k) for d, s, k, _ in manager.sent if d == _day(index)]
    # the next tick re-checks without re-sending: nothing off target counting the working orders, and no trading agent ran
    for index in (1, 4, 6, 8):
        assert runs[index]["outcome"] == "pending_retry" and runs[index]["shortfalls_before"] == [] and "orders" not in runs[index]
    assert manager.runs()["trader_retry"] == 0
    fills = pd.read_parquet(next(tmp_path.rglob("trades.parquet")))
    fills = fills[fills["status"] == "fill"]
    expected = [(pd.Timestamp(SESSIONS[[s.open.date().isoformat() for s in SESSIONS].index(day) + 1].close), symbol, side, float(quantity)) for day, symbol, side, quantity in manager.sent]
    assert list(zip(fills["time"], fills["symbol"], fills["side"], fills["filled_quantity"], strict=True)) == expected  # each order filled once, at the next session's close
    # the account ends at the last target, within the rebalance band
    target = _state(tmp_path)["target"]
    assert target == {"AAA": 0.25, "BBB": 0.1, "CCC": 0.4}
    value = strategy.broker.get_account().portfolio_value
    held = {p.asset.symbol: p.quantity for p in strategy.broker.pull_positions()}
    assert held == {"AAA": 50, "BBB": 40, "CCC": 200}
    for symbol, weight in target.items():
        assert abs(float(held[symbol] * Decimal(str(PRICES[symbol])) / value) - weight) <= PARAMS.rebalance_band
    assert _state(tmp_path)["pending_trade"] is None and _state(tmp_path)["traded"] == ["AAA", "BBB", "CCC"]


def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(tmp_path: Path) -> None:
    manager = _Manager()

    def dead(tools: Tools, context: Any) -> None:
        raise AgentError("llm down")

    manager.scripts["researcher"] = dead

    with pytest.raises(BacktestError, match="runs abandoned in a row"):
        _run(tmp_path, manager, DatedSource(_data(*BASE)))

    assert manager.handles["researcher"].runs == 6  # three abandoned runs, each with its one forced retry
    assert list(tmp_path.rglob("metrics.json")) == []
    assert [(r["date"], r["outcome"], r["stage"]) for r in _runs(tmp_path)] == [(_day(i), "abandoned", "researcher") for i in range(3)]
    assert manager.runs()["trader"] == 0 and manager.sent == []  # nothing was traded
    state = _state(tmp_path)
    assert state["abandoned_streak"] == 3 and state["processed"] == []  # the filings stay new
