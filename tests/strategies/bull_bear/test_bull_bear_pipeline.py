from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bull_bear.market_data import DailySeries, DataUnavailable
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewOutcome, ReviewPipeline
from trading_agent_framework.strategies.bull_bear.prompts import UNAVAILABLE_NOTE
from trading_agent_framework.strategies.bull_bear.state import BullBearState, ReviewLog, StateStore
from trading_agent_framework.strategies.common.rebalancer import PlacedOrder, Rebalancer
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.errors import AgentError, BrokerError, ConfigurationError

UNIVERSE = [f"S{i:02d}" for i in range(20)]  # S00 has the strongest momentum, S19 the weakest

Script = Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]


def _series(growth: float, sessions: int = 300, swing: float = 0.01) -> DailySeries:
    """A trend with a +-`swing` zigzag, so the volatility is above zero; 50 x 1e6 shares clears the dollar-volume filter."""
    closes = [50.0 * (1 + growth) ** i * (1 + swing if i % 2 else 1 - swing) for i in range(sessions)]
    return DailySeries(closes=closes, volumes=[1e6] * sessions)


def _default_series() -> dict[str, DailySeries]:
    return {symbol: _series(0.004 - index * 0.0001) for index, symbol in enumerate(UNIVERSE)}


# --- fakes -------------------------------------------------------------------------------------------


class FakeBars:
    def __init__(self, series: dict[str, DailySeries]) -> None:
        self.series = series
        self.error: DataUnavailable | None = None
        self.calls: list[tuple[list[str], list[str]]] = []

    def load(self, universe, held):  # noqa: ANN001, ANN201
        self.calls.append((list(universe), sorted(held)))
        if self.error is not None:
            raise self.error
        return {symbol: series for symbol, series in self.series.items() if symbol in universe}


class FakeAgent:
    """Runs `script` on every call, or the next of `overrides` first; a script calls the real submit tools."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self.overrides: list[Callable[..., Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self.tool_calls: list[ToolCallRecord] = []
        self.attempt_tool_calls: list[list[ToolCallRecord]] = []  # the next runs' tool calls, one list per run, before `tool_calls`

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool, "tool_budget": tool_budget})
        step = self.overrides.pop(0) if self.overrides else self.script
        step(self.tools, context)
        return AgentRunResult(output="done", tool_calls=self.attempt_tool_calls.pop(0) if self.attempt_tool_calls else self.tool_calls)


def notes(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_note"](ctx["fact_sheet"]["symbol"], "2026-09-30: quarterly revenue up 12% y/y")


def bull_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bull_case"]([{"symbol": s["fact_sheet"]["symbol"], "conviction": "high", "argument": "strong trend"} for s in ctx["stocks"]])


def bear_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bear_case"]([{"symbol": s["fact_sheet"]["symbol"], "risk": "low", "concern": "none", "argument": "no red flag"} for s in ctx["stocks"]])


def verdicts(chosen: Collection[str] | None = None) -> Script:
    """The judge puts `chosen` in (default: the first 10 stocks it is shown, the best ranked) and the others out."""

    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        put_in = set(symbols[:10]) if chosen is None else set(chosen)
        tools["submit_verdicts"]([{"symbol": s, "verdict": "in" if s in put_in else "out", "reason": "bull wins" if s in put_in else "bear wins"} for s in symbols])

    return step


def does_nothing(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    return None


@dataclass
class Harness:
    pipeline: ReviewPipeline
    broker: FakeBroker
    bars: FakeBars
    agents: dict[str, FakeAgent]
    rebalancer: Rebalancer
    store: StateStore
    log_path: Path
    orders: list[tuple[str, str, float]] = field(default_factory=list)

    def run(self) -> ReviewOutcome:
        self.broker.submitted.clear()
        outcome = self.pipeline.run()
        self.orders = [(order.asset.symbol, order.side.value, float(order.quantity or 0)) for order in self.broker.submitted]
        return outcome

    def log_lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []


def _harness(
    tmp_path: Path,
    *,
    held: dict[str, float] | None = None,
    series: dict[str, DailySeries] | None = None,
    params: BullBearParams | None = None,
    state: BullBearState | None = None,
) -> Harness:
    """Tuesday 2026-10-06 12:00, portfolio value 10,000; every stock costs 50, SHV 100; `held` maps a symbol to shares."""
    params = params or BullBearParams()
    held = held or {}
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 12)), strategy_name="bull_bear")
    broker.last_prices = {symbol: Decimal(50) for symbol in [*UNIVERSE, "OLD"]} | {"SHV": Decimal(100)}
    broker.positions = [Position(strategy_name="bull_bear", asset=Asset(symbol), quantity=Decimal(str(shares)), side=PositionSide.LONG) for symbol, shares in held.items()]
    invested = sum(shares * 50 for shares in held.values())
    broker.account = AccountBalances(cash=Decimal(str(10_000 - invested)), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    store = StateStore(tmp_path / "data" / "state.json")
    if state is not None:
        store.save(state)
    recorder = HandoffRecorder(params)
    agents = {"researcher": FakeAgent(notes), "bull": FakeAgent(bull_cases), "bear": FakeAgent(bear_cases), "judge": FakeAgent(verdicts())}
    for agent in agents.values():
        agent.tools = submit_tools(recorder)
    bars = FakeBars(series if series is not None else _default_series())
    rebalancer = Rebalancer(strategy, params)
    log_path = tmp_path / "logs" / "reviews.jsonl"
    pipeline = ReviewPipeline(
        strategy=strategy,
        params=params,
        agents=agents,
        recorder=recorder,
        state=store,
        review_log=ReviewLog(log_path),
        rebalancer=rebalancer,
        universe=UNIVERSE,
        bars=bars,
        sector_of=lambda symbol: "Technology",
        momentum=CONFIG,
    )
    return Harness(pipeline, broker, bars, agents, rebalancer, store, log_path)


def assert_no_agent_is_told_what_is_held(h: Harness) -> None:
    """No context any of the four agents received, at any depth, carries a `held` or a `weight` key."""
    for name, agent in h.agents.items():
        assert agent.calls, f"the {name} agent never ran"
        seen = json.dumps([call["context"] for call in agent.calls], default=str)
        assert '"held' not in seen and '"weight' not in seen, f"the {name} agent was told what is held"


def test_no_agent_is_told_what_is_held_or_at_what_weight(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20})  # S03 is a holding and a candidate: it is debated like any other

    assert h.run().completed

    (line,) = h.log_lines()
    assert {"symbol": "S03", "rank": 4, "held": True} in line["debate_set"]  # the review knows; the agents do not
    assert_no_agent_is_told_what_is_held(h)


# --- the happy path ---------------------------------------------------------------------------------


def test_a_review_debates_the_top_15_and_buys_what_the_judge_puts_in(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=True, abandoned_streak=0)
    research = h.agents["researcher"].calls
    assert [call["context"]["fact_sheet"]["symbol"] for call in research] == UNIVERSE[:15]
    assert all(call["tool_budget"] == 3 for call in research)
    bull_context = h.agents["bull"].calls[0]["context"]
    assert [stock["fact_sheet"]["symbol"] for stock in bull_context["stocks"]] == UNIVERSE[:15]
    assert bull_context["stocks"][0]["note"] == "2026-09-30: quarterly revenue up 12% y/y"
    assert h.agents["bear"].calls[0]["context"] == bull_context  # the same evidence, and not the bull's case
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in UNIVERSE[:10]]  # in the book's order
    state = h.store.load()
    assert state.last_completed_review == "2026-10-06" and state.abandoned_streak == 0
    assert set(state.verdicts) == set(UNIVERSE[:15])
    assert state.verdicts["S00"] == {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}
    assert state.verdicts["S14"]["verdict"] == "out"


def test_the_debaters_are_told_the_character_caps_that_reject_a_submission(tmp_path: Path) -> None:
    h = _harness(tmp_path, params=BullBearParams(argument_max_chars=123, reason_max_chars=77))

    assert h.run().completed

    bull, bear, judge = (h.agents[name].calls[0] for name in ("bull", "bear", "judge"))
    assert bull["context"]["constraints"] == bear["context"]["constraints"] == {"argument_max_chars": 123}
    assert judge["context"]["constraints"] == {"max_in": 10, "reason_max_chars": 77}
    assert "123 characters" in bull["task"] and "123 characters" in bear["task"] and "77 characters" in judge["task"]


def test_the_review_log_records_the_whole_review_as_one_line(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    h.run()

    (line,) = h.log_lines()
    assert line["date"] == "2026-10-06" and line["abandoned"] is False
    assert [stock["symbol"] for stock in line["debate_set"]] == UNIVERSE[:15]
    assert line["debate_set"][0] == {"symbol": "S00", "rank": 1, "held": False}
    assert line["forced_exits"] == []
    assert line["notes"] == {"written": 15, "failed": []}
    assert line["bull_conviction"] == {"high": 15} and line["bear_risk"] == {"low": 15}
    assert "picks" not in line and "drops" not in line
    assert line["verdicts"][0] == {"symbol": "S00", "verdict": "in", "reason": "bull wins", "what_changed": None, "previous_verdict": None} and len(line["verdicts"]) == 15
    assert line["verdict_counts"] == {"in": 10, "out": 5} and "veto_cap" not in line and "judge_verdicts" not in line
    assert line["book"] == UNIVERSE[:10]
    stock_weights = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert list(stock_weights) == UNIVERSE[:10]
    assert sum(stock_weights.values()) == pytest.approx(0.98)
    assert all(0.04 <= weight <= 0.20 for weight in stock_weights.values())
    assert [order["symbol"] for order in line["orders"]] == UNIVERSE[:10]


def test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20, "S19": 20, "OLD": 20}, params=BullBearParams(retention_rank=18))
    h.agents["judge"].script = verdicts(UNIVERSE[:10])

    outcome = h.run()

    assert outcome.completed
    (line,) = h.log_lines()
    assert [stock["symbol"] for stock in line["debate_set"]] == [*UNIVERSE[:15], "S16"]
    assert line["debate_set"][-1] == {"symbol": "S16", "rank": 17, "held": True}
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}, {"symbol": "S19", "reason": "rank 20"}]
    assert_no_agent_is_told_what_is_held(h)
    assert {"symbol": "S16", "verdict": "out", "reason": "bear wins", "what_changed": None, "previous_verdict": None} in line["verdicts"]
    assert line["book"] == UNIVERSE[:10]
    assert {("OLD", "sell", 20.0), ("S19", "sell", 20.0), ("S16", "sell", 20.0)} <= set(h.orders)  # a holding put out is sold
    assert h.bars.calls == [(UNIVERSE, ["OLD", "S16", "S19"])]


def test_a_holding_put_in_stays_whatever_its_rank_and_gets_no_preferred_slot(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20}, params=BullBearParams(retention_rank=18))
    h.agents["judge"].script = verdicts([*UNIVERSE[:9], "S16"])

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == [*UNIVERSE[:9], "S16"]  # rank order: the holding comes last
    assert not any(order[0] == "S16" for order in h.orders)  # held at 10%, target ~9.8%: inside the band


def test_the_judges_choice_not_the_rank_decides_the_book(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts({"S03", "S14", "S07", "S11", "S12"})

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == ["S03", "S07", "S11", "S12", "S14"]
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in line["book"]]


def test_fewer_than_five_stocks_put_in_leave_the_rest_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts({"S03", "S14"})

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == ["S03", "S14"]
    assert line["targets"] == pytest.approx({"S03": 0.20, "S14": 0.20, "SHV": 0.58})
    assert any(order[0] == "SHV" and order[1] == "buy" for order in h.orders)


def test_a_judge_putting_nothing_in_parks_everything(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S00": 20})
    h.agents["judge"].script = verdicts(set())

    outcome = h.run()

    assert outcome.completed  # not an abandonment
    (line,) = h.log_lines()
    assert line["book"] == [] and line["verdict_counts"] == {"out": 15}
    assert line["targets"] == pytest.approx({"SHV": 0.98})
    assert ("S00", "sell", 20.0) in h.orders and any(order[0] == "SHV" and order[1] == "buy" for order in h.orders)


def test_a_judge_putting_more_than_max_book_size_in_is_refused_and_corrects_itself_on_the_retry(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [verdicts(UNIVERSE[:11])]

    outcome = h.run()

    assert outcome.completed
    retry = h.agents["judge"].calls[1]
    assert retry["force_tool"] == "submit_verdicts" and "11 stocks in: put at most 10 in" in retry["task"]
    assert h.log_lines()[0]["book"] == UNIVERSE[:10]


def test_a_debate_smaller_than_max_book_size_still_completes(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:9]})

    assert h.run().completed
    assert h.log_lines()[0]["book"] == UNIVERSE[:9]


def test_max_volatility_filters_new_entries_but_never_forces_a_holding_out(tmp_path: Path) -> None:
    series = _default_series()
    series["S03"] = _series(0.0037, swing=0.06)  # held: annualised volatility far above max_volatility (80%)
    series["S04"] = _series(0.0036, swing=0.06)  # not held: the same volatility
    h = _harness(tmp_path, held={"S03": 20}, series=series)

    assert h.run().completed

    (line,) = h.log_lines()
    debated = {stock["symbol"]: stock for stock in line["debate_set"]}
    assert debated["S03"]["held"] is True
    assert "S04" not in debated
    assert line["forced_exits"] == []
    assert "S03" in line["book"]  # the high-volatility holding the judge put in, kept in the book...
    assert line["targets"]["S03"] == pytest.approx(0.04)  # ...and sized at the minimum weight


def test_holdings_the_judge_puts_out_are_sold(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={symbol: 20 for symbol in UNIVERSE[:12]})  # 12 holdings of 20 shares at 50 = 1,000 each

    outcome = h.run()

    assert outcome.completed
    (line,) = h.log_lines()
    assert line["book"] == UNIVERSE[:10]
    assert ("S10", "sell", 20.0) in h.orders
    assert ("S11", "sell", 20.0) in h.orders


# --- research failures --------------------------------------------------------------------------------


def test_a_stock_whose_research_fails_gets_research_unavailable_and_the_review_goes_on(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    def flaky(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        if ctx["fact_sheet"]["symbol"] == "S03":
            raise AgentError("timeout")
        notes(tools, ctx)

    h.agents["researcher"].script = flaky

    outcome = h.run()

    assert outcome.completed
    assert len(h.agents["researcher"].calls) == 16  # S03 once more with a forced submit_note
    stock = next(s for s in h.agents["bull"].calls[0]["context"]["stocks"] if s["fact_sheet"]["symbol"] == "S03")
    assert stock["note"] == UNAVAILABLE_NOTE
    assert h.log_lines()[0]["notes"] == {"written": 14, "failed": ["S03"]}


def test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    def misconfigured(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        raise ConfigurationError("SEC_EDGAR_USER_AGENT is not set")

    h.agents["researcher"].script = misconfigured

    with pytest.raises(ConfigurationError):
        h.run()
    assert h.store.load() == BullBearState() and h.orders == []


# --- abandoned reviews --------------------------------------------------------------------------------


@pytest.mark.parametrize(("agent", "tool"), [("bull", "submit_bull_case"), ("bear", "submit_bear_case"), ("judge", "submit_verdicts")])
def test_a_debater_without_a_valid_submission_abandons_the_review_with_no_order(tmp_path: Path, agent: str, tool: str) -> None:
    h = _harness(tmp_path, held={"OLD": 20})
    h.agents[agent].script = does_nothing

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    assert h.orders == []  # not even the forced exit of OLD
    assert len(h.agents[agent].calls) == 2 and h.agents[agent].calls[1]["force_tool"] == tool
    state = h.store.load()
    assert (state.last_completed_review, state.abandoned_streak) == (None, 1)
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == agent and tool in line["error"]
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}]  # what was known before the stage failed


def test_a_corrected_submission_on_the_forced_retry_completes_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["bear"].overrides = [lambda tools, ctx: tools["submit_bear_case"]([{"symbol": "S00", "risk": "high", "concern": "valuation", "argument": "too expensive"}])]

    outcome = h.run()

    assert outcome.completed
    retry = h.agents["bear"].calls[1]
    assert retry["force_tool"] == "submit_bear_case" and "no case for S01" in retry["task"]


def test_unusable_data_abandons_the_review_before_any_agent_runs(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.bars.error = DataUnavailable("Yahoo covers only 3/20 symbols")

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == [] and h.orders == []
    (line,) = h.log_lines()
    assert (line["stage"], line["error"]) == ("data", "Yahoo covers only 3/20 symbols")


def test_no_stock_to_debate_abandons_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={})

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == []
    (line,) = h.log_lines()
    assert line["stage"] == "data" and "no stock to debate" in line["error"]


def test_a_broker_error_before_the_rebalance_abandons_at_the_broker_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)

    def broken() -> list[str]:
        raise BrokerError("positions unavailable")

    monkeypatch.setattr(h.rebalancer, "holdings", broken)

    outcome = h.run()

    assert not outcome.completed and h.log_lines()[0]["stage"] == "broker"
    state = h.store.load()
    assert (state.last_completed_review, state.abandoned_streak) == (None, 1)


def test_a_broker_error_during_the_rebalance_logs_the_orders_already_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)

    def half_way(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        h.rebalancer.placed = [PlacedOrder("S00", "buy", 10.0)]
        raise BrokerError("connection reset")

    monkeypatch.setattr(h.rebalancer, "rebalance", half_way)

    outcome = h.run()

    assert not outcome.completed
    (line,) = h.log_lines()
    assert line["stage"] == "execution" and line["orders"] == [{"symbol": "S00", "side": "buy", "quantity": 10.0}]
    # The decision that led to those orders is in the same line.
    assert line["book"] == UNIVERSE[:10] and line["verdict_counts"] == {"in": 10, "out": 5}
    assert len(line["verdicts"]) == 15
    stock_targets = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert sorted(stock_targets) == UNIVERSE[:10] and "SHV" in line["targets"]
    assert sum(stock_targets.values()) == pytest.approx(0.98)
    state = h.store.load()
    assert (state.last_completed_review, state.abandoned_streak) == (None, 1)


def _raises_after(h: Harness, monkeypatch: pytest.MonkeyPatch, placed: list[PlacedOrder]) -> None:
    """The rebalance records `placed` as already sent, then fails: a connection lost half way."""

    def half_way(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        h.rebalancer.placed = list(placed)
        raise BrokerError("connection reset")

    monkeypatch.setattr(h.rebalancer, "rebalance", half_way)


def test_an_execution_abandonment_after_a_sell_stores_the_verdicts_and_the_sold_stocks_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(verdicts={"S03": {"verdict": "in", "reason": "old", "date": "2026-09-29"}}))

    def sells_s03(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        put_in = set(symbols[:10]) - {"S03"}
        tools["submit_verdicts"](
            [
                {"symbol": s, "verdict": "in" if s in put_in else "out", "reason": "bull wins" if s in put_in else "bear wins", **({"what_changed": "guidance cut"} if s == "S03" else {})}
                for s in symbols
            ]
        )

    h.agents["judge"].script = sells_s03
    _raises_after(h, monkeypatch, [PlacedOrder("S03", "sell", 20.0), PlacedOrder("S00", "buy", 10.0)])

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    state = h.store.load()
    assert state.verdicts["S03"] == {"verdict": "out", "reason": "bear wins", "date": "2026-10-06"}  # the stored "in" is replaced
    assert state.verdicts["S00"] == {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}
    assert set(state.verdicts) == set(UNIVERSE[:15]) | {"S03"}
    assert state.cooldowns == {"S03": 4}
    assert state.history == [] and state.last_completed_review is None and state.abandoned_streak == 1
    (line,) = h.log_lines()
    assert line["stage"] == "execution" and line["cooldowns_started"] == ["S03"]


def test_an_execution_abandonment_before_any_order_stores_the_verdicts_but_no_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path, held={"S03": 20})
    h.agents["judge"].script = verdicts(set(UNIVERSE[:10]) - {"S03"})
    _raises_after(h, monkeypatch, [])  # no order was sent: S03 is still held

    assert not h.run().completed

    state = h.store.load()
    assert state.verdicts["S03"]["verdict"] == "out" and state.verdicts["S00"]["verdict"] == "in"
    assert state.cooldowns == {}
    assert state.history == [] and state.last_completed_review is None and state.abandoned_streak == 1
    assert h.log_lines()[0]["cooldowns_started"] == []


def test_an_execution_abandonment_starts_cooldowns_only_for_the_judges_sells_that_were_placed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # S03 was sold (placed), S04 is put out and held but its sell never went out, S05 was sold by a trim of a stock put in
    h = _harness(tmp_path, held={"S03": 20, "S04": 20, "S05": 20})
    h.agents["judge"].script = verdicts(set(UNIVERSE[:10]) - {"S03", "S04"})
    _raises_after(h, monkeypatch, [PlacedOrder("S03", "sell", 20.0), PlacedOrder("S04", "buy", 1.0), PlacedOrder("S05", "sell", 5.0)])

    assert not h.run().completed

    assert h.store.load().cooldowns == {"S03": 4}
    assert h.log_lines()[0]["cooldowns_started"] == ["S03"]


def test_an_execution_abandonment_does_not_count_the_running_cooldowns_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(cooldowns={"S05": 2, "S06": 1}))
    h.agents["judge"].script = verdicts(set(UNIVERSE[:10]) - {"S03"})
    _raises_after(h, monkeypatch, [PlacedOrder("S03", "sell", 20.0)])

    assert not h.run().completed

    assert h.store.load().cooldowns == {"S05": 2, "S06": 1, "S03": 4}


@pytest.mark.parametrize("stage", ["broker", "judge"])
def test_an_abandonment_before_the_execution_stage_stores_nothing_but_the_streak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    earlier = {"S03": {"verdict": "in", "reason": "old", "date": "2026-09-29"}}
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(verdicts=earlier, cooldowns={"S05": 2}))
    if stage == "judge":
        h.agents["judge"].script = does_nothing
    else:
        monkeypatch.setattr(h.rebalancer, "holdings", lambda: (_ for _ in ()).throw(BrokerError("positions unavailable")))

    assert not h.run().completed

    state = h.store.load()
    assert state == BullBearState(abandoned_streak=1, verdicts=earlier, cooldowns={"S05": 2})


# --- the streak and the same-day rule -------------------------------------------------------------------


def test_a_completed_review_resets_the_streak_and_marks_the_day_done(tmp_path: Path) -> None:
    h = _harness(tmp_path, state=BullBearState(abandoned_streak=2))

    assert not h.pipeline.completed_today()
    h.run()

    assert h.store.load().abandoned_streak == 0 and h.pipeline.completed_today()


def test_an_abandoned_review_leaves_the_day_open_so_a_restart_reviews_again(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [does_nothing, does_nothing]

    assert not h.run().completed
    assert not h.pipeline.completed_today()

    assert h.run() == ReviewOutcome(completed=True, abandoned_streak=0)
    assert len(h.agents["judge"].calls) == 3


def test_a_hand_edited_review_date_never_stops_a_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, state=BullBearState(last_completed_review="next tuesday-ish"))

    assert not h.pipeline.completed_today()
    outcome = h.run()

    assert outcome.completed and h.pipeline.completed_today()
    assert h.store.load().last_completed_review == "2026-10-06"


# --- holdings the debate does not cover -------------------------------------------------------------------


def test_the_parking_instrument_is_never_forced_out_and_a_position_outside_the_universe_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path, held={"SHV": 10, "OLD": 20})
    seen: list[list[str]] = []
    real_rebalance = h.rebalancer.rebalance

    def spy(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        seen.append(list(forced_exits))
        return real_rebalance(target, forced_exits)

    monkeypatch.setattr(h.rebalancer, "rebalance", spy)

    outcome = h.run()

    assert outcome.completed
    assert h.bars.calls == [(UNIVERSE, ["OLD"])]  # SHV is not a holding of the ranking
    (line,) = h.log_lines()
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}]
    assert "forced_exits" not in h.agents["judge"].calls[0]["context"]  # forced exits never reach the agents
    assert_no_agent_is_told_what_is_held(h)
    assert seen == [["OLD"]]
    assert ("OLD", "sell", 20.0) in h.orders


def _forced_exit_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING and "forced exit" in record.getMessage()]


def test_each_forced_exit_is_logged_as_a_warning_before_its_sell(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # OLD is outside the universe (unranked), so code sells it; SHV is the parking instrument and is never forced out
    h = _harness(tmp_path, held={"OLD": 20, "SHV": 10})
    with caplog.at_level(logging.INFO):
        outcome = h.run()

    assert outcome.completed
    (message,) = _forced_exit_warnings(caplog)
    assert "OLD" in message and "unranked" in message
    assert ("OLD", "sell", 20.0) in h.orders


def test_a_review_without_forced_exits_logs_no_forced_exit_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path, held={"SHV": 10})
    with caplog.at_level(logging.INFO):
        outcome = h.run()

    assert outcome.completed
    assert _forced_exit_warnings(caplog) == []


def test_an_abandoned_review_does_not_log_a_forced_exit_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # nothing is sold when a review is abandoned, so no forced exit "occurred"
    h = _harness(tmp_path, held={"OLD": 20})
    h.agents["judge"].script = does_nothing  # the judge never submits: the review is abandoned at the judge stage
    with caplog.at_level(logging.INFO):
        outcome = h.run()

    assert not outcome.completed
    assert h.orders == []
    assert _forced_exit_warnings(caplog) == []


# --- the log carries what each agent submitted -------------------------------------------------------------


def test_each_researcher_note_is_logged_with_its_text(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    with caplog.at_level(logging.INFO):
        h.run()

    assert "[researcher] S00: 2026-09-30: quarterly revenue up 12% y/y" in caplog.text
    assert "[researcher] S14: 2026-09-30: quarterly revenue up 12% y/y" in caplog.text


def test_each_bull_case_is_logged_with_its_conviction_and_argument(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    with caplog.at_level(logging.INFO):
        h.run()

    assert "[bull] S00: high conviction: strong trend" in caplog.text
    assert "[bull] S14: high conviction: strong trend" in caplog.text


def test_each_bear_case_is_logged_with_its_risk_concern_and_argument(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    with caplog.at_level(logging.INFO):
        h.run()

    assert "[bear] S00: low risk (none): no red flag" in caplog.text
    assert "[bear] S14: low risk (none): no red flag" in caplog.text


def test_every_judge_verdict_is_logged_ins_and_outs_alike(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts({"S00", "S14"})
    with caplog.at_level(logging.INFO):
        h.run()

    assert "[judge] S00: in: bull wins" in caplog.text
    assert "[judge] S01: out: bear wins" in caplog.text
    assert "[judge] S14: in: bull wins" in caplog.text


def test_a_rejected_submission_is_logged_as_a_warning_with_the_reason_and_not_as_recorded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [verdicts(UNIVERSE[:11])]  # 11 in against a cap of 10: refused, then the retry puts the first 10 in
    with caplog.at_level(logging.INFO):
        assert h.run().completed

    (rejection,) = [record for record in caplog.records if "rejected" in record.getMessage()]
    assert rejection.levelno == logging.WARNING
    assert "[judge]" in rejection.getMessage() and "11 stocks in: put at most 10 in" in rejection.getMessage()
    # only the accepted retry is logged as recorded: none of the refused submission's verdicts
    assert "[judge] S10: in: bull wins" not in caplog.text
    assert "[judge] S10: out: bear wins" in caplog.text


def test_an_agent_that_ends_without_calling_its_submit_tool_is_warned_about_on_each_attempt(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = does_nothing  # no verdict ever lands: the review is abandoned
    with caplog.at_level(logging.INFO):
        assert not h.run().completed

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING and "[judge]" in record.getMessage()]
    assert len(warnings) == 2  # the first attempt and the forced retry
    assert all("without calling submit_verdicts" in message for message in warnings)
    assert not any("rejected" in message for message in warnings)  # nothing was submitted, so nothing was rejected
    assert "[judge] S00:" not in caplog.text  # and no verdict is logged as recorded


def test_the_judge_sees_its_previous_verdict_on_a_stock_it_judged_before(tmp_path: Path) -> None:
    earlier = {"S00": {"verdict": "out", "reason": "lawsuit", "date": "2026-09-29"}, "ZZZ": {"verdict": "in", "reason": "x", "date": "2026-09-29"}}
    h = _harness(tmp_path, state=BullBearState(verdicts=earlier))
    h.agents["judge"].script = verdicts(UNIVERSE[1:11])  # S00 stays out: no what_changed needed

    assert h.run().completed

    stocks = {s["fact_sheet"]["symbol"]: s for s in h.agents["judge"].calls[0]["context"]["stocks"]}
    assert stocks["S00"]["previous_verdict"] == {"verdict": "out", "reason": "lawsuit", "date": "2026-09-29"}
    assert "previous_verdict" not in stocks["S01"]
    (line,) = h.log_lines()
    assert line["verdicts"][0]["previous_verdict"] == "out"
    assert h.store.load().verdicts["ZZZ"] == earlier["ZZZ"]  # a stock not judged this time keeps its last verdict


def test_a_judge_flipping_a_verdict_without_what_changed_is_refused_then_corrects_itself(tmp_path: Path) -> None:
    # a verdict from months ago still counts
    h = _harness(tmp_path, state=BullBearState(verdicts={"S00": {"verdict": "out", "reason": "lawsuit", "date": "2026-07-07"}}))
    h.agents["judge"].overrides = [verdicts()]  # puts S00 in with no what_changed

    def with_what_changed(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        tools["submit_verdicts"]([{"symbol": s, "verdict": "in" if s in symbols[:10] else "out", "reason": "r", **({"what_changed": "lawsuit dismissed"} if s == "S00" else {})} for s in symbols])

    h.agents["judge"].script = with_what_changed

    assert h.run().completed

    assert "your previous verdict on S00 was out" in h.agents["judge"].calls[1]["task"]
    (line,) = h.log_lines()
    assert line["verdicts"][0]["what_changed"] == "lawsuit dismissed"


# --- the re-entry cooldown ---------------------------------------------------------------------------------


def test_a_holding_the_judge_puts_out_starts_its_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20})
    h.agents["judge"].script = verdicts(set(UNIVERSE[:10]) - {"S03"})

    assert h.run().completed

    assert h.store.load().cooldowns == {"S03": 4}
    (line,) = h.log_lines()
    assert line["cooldowns_started"] == ["S03"] and line["cooldowns_active"] == {}
    assert ("S03", "sell", 20.0) in h.orders


def test_a_stock_put_out_that_was_not_held_starts_no_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path)  # the default judge puts S10-S14 out; none is held

    assert h.run().completed
    assert h.store.load().cooldowns == {}


def test_a_stock_in_cooldown_is_not_debated_and_its_count_goes_down(tmp_path: Path) -> None:
    h = _harness(tmp_path, state=BullBearState(cooldowns={"S03": 2, "S05": 1}))

    assert h.run().completed

    (line,) = h.log_lines()
    debated = [stock["symbol"] for stock in line["debate_set"]]
    assert "S03" not in debated and "S05" not in debated and len(debated) == 13
    assert line["cooldowns_active"] == {"S03": 2, "S05": 1}
    assert h.store.load().cooldowns == {"S03": 1}


def test_an_abandoned_review_leaves_the_cooldowns_as_they_were(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(cooldowns={"S05": 2}))
    h.agents["judge"].script = does_nothing

    assert not h.run().completed
    assert h.store.load().cooldowns == {"S05": 2}


def test_a_holding_still_in_cooldown_is_sold_as_a_forced_exit(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S03": 20}, state=BullBearState(cooldowns={"S03": 3}))

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["forced_exits"] == [{"symbol": "S03", "reason": "cooldown"}]
    assert ("S03", "sell", 20.0) in h.orders


# --- outcome feedback ----------------------------------------------------------------------------------------


def test_the_judge_sees_how_its_recent_out_verdicts_did_since(tmp_path: Path) -> None:
    series = _default_series()
    now_close = {symbol: s.closes[-1] for symbol, s in series.items()}
    history = [
        {
            "date": "2026-09-29",
            "verdicts": {
                "S00": {"verdict": "in", "reason": "r", "close": now_close["S00"] / 1.10},
                "S18": {"verdict": "out", "reason": "dilution", "close": now_close["S18"] / 0.95},  # S18 is no candidate now, but priced
            },
        }
    ]
    h = _harness(tmp_path, series=series, state=BullBearState(history=history))

    assert h.run().completed

    shown = h.agents["judge"].calls[0]["context"]["recent_verdicts"]
    assert shown == [{"symbol": "S18", "date": "2026-09-29", "reason": "dilution", "return_since_pct": -5.0, "in_names_return_pct": 10.0}]
    assert h.log_lines()[0]["recent_verdicts"] == shown


def test_the_feedback_prices_the_then_close_from_todays_series_not_from_the_stored_close(tmp_path: Path) -> None:
    # a split since the review: the stored close (1000) is on the old basis; today's adjusted series is the truth
    sessions = [day.date() for day in pd.bdate_range(end=date(2026, 10, 5), periods=300)]
    series = {symbol: replace(data, dates=sessions) for symbol, data in _default_series().items()}
    before = sessions.index(date(2026, 9, 28))  # the last session strictly before the review of 2026-09-29
    expected = round((series["S18"].closes[-1] / series["S18"].closes[before] - 1) * 100, 1)
    history = [{"date": "2026-09-29", "verdicts": {"S18": {"verdict": "out", "reason": "dilution", "close": 1000.0}}}]
    h = _harness(tmp_path, series=series, state=BullBearState(history=history))

    assert h.run().completed

    (shown,) = h.agents["judge"].calls[0]["context"]["recent_verdicts"]
    assert shown["symbol"] == "S18" and shown["return_since_pct"] == expected
    assert expected != round((series["S18"].closes[-1] / 1000.0 - 1) * 100, 1)


def test_a_completed_review_stores_each_verdict_with_the_close_it_was_made_at(tmp_path: Path) -> None:
    series = _default_series()
    earlier = [{"date": "2026-09-22", "verdicts": {}}, {"date": "2026-09-29", "verdicts": {}}]
    h = _harness(tmp_path, series=series, params=BullBearParams(feedback_reviews=2), state=BullBearState(history=earlier))

    assert h.run().completed

    history = h.store.load().history
    assert [review["date"] for review in history] == ["2026-09-29", "2026-10-06"]  # trimmed to feedback_reviews
    assert history[-1]["verdicts"]["S00"] == {"verdict": "in", "reason": "bull wins", "close": series["S00"].closes[-1]}
    assert history[-1]["verdicts"]["S14"]["verdict"] == "out"


def test_the_judge_runs_with_its_memory_budget_and_its_lessons_are_logged(tmp_path: Path) -> None:
    h = _harness(tmp_path, params=BullBearParams(judge_memory_budget=4))
    h.agents["judge"].tool_calls = [
        ToolCallRecord("search_memory", {"query": "S03"}, "{}"),
        ToolCallRecord("remember_lesson", {"text": "out on a shelf filing alone was wrong", "symbol": "CRDO"}, "{}"),
        ToolCallRecord("submit_verdicts", {}, "{}"),
    ]

    assert h.run().completed

    assert h.agents["judge"].calls[0]["tool_budget"] == 4
    assert h.log_lines()[0]["memory"] == {"calls": 2, "lessons": ["out on a shelf filing alone was wrong"]}


def test_lessons_from_a_rejected_first_attempt_and_from_the_retry_are_both_logged(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [verdicts(UNIVERSE[:11])]  # 11 in against a cap of 10: refused, then the retry submits
    h.agents["judge"].attempt_tool_calls = [
        [ToolCallRecord("remember_lesson", {"text": "first lesson"}, '{"id": "lesson_1", "kind": "lesson", "status": "active"}'), ToolCallRecord("submit_verdicts", {}, '{"error": "too many in"}')],
        [
            ToolCallRecord("search_memory", {"query": "S03"}, "{}"),
            ToolCallRecord("remember_lesson", {"text": "second lesson"}, '{"id": "lesson_2", "kind": "lesson", "status": "active"}'),
            ToolCallRecord("submit_verdicts", {}, "{}"),
        ],
    ]

    assert h.run().completed

    assert len(h.agents["judge"].calls) == 2
    assert h.log_lines()[0]["memory"] == {"calls": 3, "lessons": ["first lesson", "second lesson"]}


def test_a_refused_or_invalid_remember_lesson_counts_as_a_call_but_is_not_a_lesson(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].tool_calls = [
        ToolCallRecord("remember_lesson", {"text": "stored"}, '{"id": "lesson_1", "kind": "lesson", "status": "active"}'),
        ToolCallRecord("remember_lesson", {"text": "over the budget"}, '{"error": "tool budget of 6 calls spent; remember_lesson was not run. Finish now: call submit_verdicts."}'),
        ToolCallRecord("remember_lesson", {"text": ""}, '{"error": "text must be a non-empty string"}'),
        ToolCallRecord("remember_lesson", {"text": "python repr"}, "{'error': 'text must be a non-empty string'}"),
        ToolCallRecord("remember_lesson", {"symbol": "CRDO"}, '{"id": "lesson_2", "kind": "lesson", "status": "active"}'),  # no text: not a lesson
        ToolCallRecord("remember_lesson", {"text": "never answered"}, ""),
        ToolCallRecord("submit_verdicts", {}, "{}"),
    ]

    assert h.run().completed

    assert h.log_lines()[0]["memory"] == {"calls": 6, "lessons": ["stored"]}


def test_an_abandoned_judge_stage_still_logs_the_memory_calls_it_made(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = does_nothing
    h.agents["judge"].attempt_tool_calls = [
        [ToolCallRecord("remember_lesson", {"text": "kept before the failure"}, '{"id": "lesson_1", "kind": "lesson", "status": "active"}')],
        [ToolCallRecord("search_memory", {"query": "S00"}, "{}")],
    ]

    outcome = h.run()

    assert not outcome.completed
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == "judge"
    assert line["memory"] == {"calls": 2, "lessons": ["kept before the failure"]}


def test_a_judge_that_used_no_memory_tool_logs_an_empty_record(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    assert h.run().completed

    assert h.log_lines()[0]["memory"] == {"calls": 0, "lessons": []}
