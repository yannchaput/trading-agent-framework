"""End to end: a daily backtest with the real executor, the real backtest broker, fake agents (they call the real tools) and a fake filings source."""

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
from trading_agent_framework.congress.annual import AssetHolding, tier_of
from trading_agent_framework.congress.ptr import FilingRef
from trading_agent_framework.congress.source import CongressSource, KnownFilings
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.utils.errors import AgentError, BacktestError

PRIOR = weekday_sessions(date(2026, 9, 9), 3)  # history before the window: the first tick needs a last price
SESSIONS = weekday_sessions(date(2026, 9, 14), 7)
ANNUAL = FilingRef("A1", "Nancy Pelosi", "annual", date(2026, 5, 15), 2025)
PTR1 = FilingRef("P1", "Nancy Pelosi", "ptr", date(2026, 6, 23), 2026)
PTR2 = FilingRef("P2", "Nancy Pelosi", "ptr", date(2026, 9, 17), 2026)  # filed on the 4th session of the window: known from the 5th (filed strictly before today)


def _asset(ticker: str, low: int, high: int) -> AssetHolding:
    return AssetHolding("A1", "spouse", ticker, f"{ticker} Inc.", Decimal(low), Decimal(high), tier_of(Decimal(low)))


def _known(*ptrs: FilingRef) -> KnownFilings:
    return KnownFilings(
        annual_ref=ANNUAL,
        period_end=date(2025, 12, 31),
        assets=[_asset("AAA", 5_000_001, 25_000_000), _asset("BBB", 1_000_001, 5_000_000)],
        transactions=[],
        refs=sorted([ANNUAL, *ptrs], key=lambda r: (r.filed, r.doc_id), reverse=True),
        unparsed_filings=0,
        skipped_non_stock=0,
    )


class DatedSource:
    """What the Clerk would have published by each simulated day: PTR2 (filed the 17th) only from the 18th on."""

    def __init__(self) -> None:
        self.as_ofs: list[Any] = []

    def known(self, as_of: Any) -> KnownFilings:
        self.as_ofs.append(as_of)
        return _known(PTR1, PTR2) if as_of.date() > PTR2.filed else _known(PTR1)

    def new_since(self, known: KnownFilings, processed: Any) -> list[FilingRef]:
        return CongressSource.new_since(known, processed)


class _Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _research(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_holdings"]([{"ticker": b["ticker"], "value_low": b["value_low"], "value_high": b["value_high"]} for b in context["baseline"]])


def _portfolio(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_target"]([{"ticker": h["ticker"], "weight": h["baseline_weight"] or 0, "reason": "by tier"} for h in context["holdings"]])


def _trade(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    for shortfall in sorted(context["shortfalls"], key=lambda s: s["kind"] != "sell"):
        price = {"AAA": 50.0, "BBB": 25.0}[shortfall["symbol"]]
        quantity = math.floor(shortfall["dollars"] / price)
        if quantity > 0:
            tools["place_order"](shortfall["symbol"], shortfall["kind"], quantity)
    checked = tools["check_orders"]()
    tools["submit_trade_report"]([{"order_id": o["order_id"], **({} if o["status"] == "filled" else {"reason": "still working at the check"})} for o in checked["orders"]])


class _Manager:
    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": _research, "portfolio_manager": _portfolio, "trader": _trade}

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run


def _bars(closes: list[float]) -> pd.DataFrame:
    assert len(closes) == len(PRIOR) + len(SESSIONS)
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in [*PRIOR, *SESSIONS]], name="timestamp"),
    )


def _run(tmp_path: Path, manager: _Manager, source: DatedSource, settings: CongressParams | None = None) -> CongressTradesStrategy:
    data = FakeBacktestDataSource()
    data.set_sessions(SESSIONS)
    for symbol, price in {"AAA": 50.0, "BBB": 25.0, "SPY": 400.0}.items():
        data.set_bars(Asset(symbol), _bars([price] * (len(PRIOR) + len(SESSIONS))))
    strategy = CongressTradesStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="congress_trades"),
        mode=TradingMode.BACKTESTING,
        project_root=tmp_path,
        settings=settings,
        source=source,
    )
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=data, budget=Decimal(10000), benchmark="SPY")
    return strategy


def _runs(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("runs.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_bot_checks_every_session_but_acts_only_on_new_filings(tmp_path: Path) -> None:
    manager = _Manager()

    _run(tmp_path, manager, DatedSource())

    runs = _runs(tmp_path)
    assert [r["date"] for r in runs] == [s.open.date().isoformat() for s in SESSIONS]  # a check every session
    assert [r["outcome"] for r in runs] == [
        "completed_pending",  # day 1: the first run builds the portfolio; the orders fill on a later bar
        "pending_retry",  # day 2: they are still working: only the trading stage is re-checked
        "pending_complete",  # day 3: filled
        "nothing_new",  # day 4 (the 17th): PTR2 is being filed today, so it is not known yet
        "completed",  # day 5 (the 18th): PTR2 appears and the account is already at the target
        "nothing_new",
        "nothing_new",
    ]
    # research and portfolio ran twice (day 1 and the day PTR2 became known); the trading agent only on day 1
    assert {name: handle.runs for name, handle in manager.handles.items()} == {"researcher": 2, "portfolio_manager": 2, "trader": 1}
    assert runs[3]["new_filings"] == [] and [f["doc_id"] for f in runs[4]["new_filings"]] == ["P2"]
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report


def test_orders_go_out_once_and_fill_on_a_later_bar(tmp_path: Path) -> None:
    _run(tmp_path, _Manager(), DatedSource())

    first = _runs(tmp_path)[0]
    assert {(o["symbol"], o["side"]) for o in first["orders"]} == {("AAA", "buy"), ("BBB", "buy")}
    assert {o["status"] for o in first["orders"]} == {"working"}  # a daily backtest fills on the next bar: still working at the 10:00 tick
    assert all(r.get("orders") in (None, []) or r["date"] == first["date"] for r in _runs(tmp_path))  # nothing is sent a second time


def test_the_agents_see_only_filings_known_on_the_simulated_day(tmp_path: Path) -> None:
    source = DatedSource()

    _run(tmp_path, _Manager(), source)

    assert all(as_of.tzinfo is not None and as_of.utcoffset() == timedelta(hours=-4) for as_of in source.as_ofs)
    assert [as_of.date() for as_of in source.as_ofs] == [s.open.date() for s in SESSIONS]


def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(tmp_path: Path) -> None:
    manager = _Manager()

    def dead(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
        raise AgentError("llm down")

    manager.scripts["researcher"] = dead

    with pytest.raises(BacktestError, match="runs abandoned in a row"):
        _run(tmp_path, manager, DatedSource())

    assert manager.handles["researcher"].runs == 6  # three abandoned runs, each with its one forced retry
    assert list(tmp_path.rglob("metrics.json")) == []
    assert [r["outcome"] for r in _runs(tmp_path)] == ["abandoned"] * 3
