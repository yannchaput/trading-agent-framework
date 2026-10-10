from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult
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


def _series(growth: float, sessions: int = 300) -> DailySeries:
    """A trend with a +-1% zigzag, so the volatility is above zero; 50 x 1e6 shares clears the dollar-volume filter."""
    closes = [50.0 * (1 + growth) ** i * (1.01 if i % 2 else 0.99) for i in range(sessions)]
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

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool, "tool_budget": tool_budget})
        step = self.overrides.pop(0) if self.overrides else self.script
        step(self.tools, context)
        return AgentRunResult(output="done", tool_calls=[])


def notes(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_note"](ctx["fact_sheet"]["symbol"], "2026-09-30: quarterly revenue up 12% y/y")


def bull_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bull_case"]([{"symbol": s["fact_sheet"]["symbol"], "conviction": "high", "argument": "strong trend"} for s in ctx["stocks"]])


def bear_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bear_case"]([{"symbol": s["fact_sheet"]["symbol"], "risk": "low", "concern": "none", "argument": "no red flag"} for s in ctx["stocks"]])


def picks_first(count: int) -> Script:
    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        picked = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]][:count]
        drops = [{"symbol": symbol, "reason": "lost the debate"} for symbol in ctx["held"] if symbol not in picked]
        tools["submit_picks"]([{"symbol": symbol, "reason": "won the debate"} for symbol in picked], drops)

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
    agents = {"researcher": FakeAgent(notes), "bull": FakeAgent(bull_cases), "bear": FakeAgent(bear_cases), "judge": FakeAgent(picks_first(5))}
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


# --- the happy path ---------------------------------------------------------------------------------


def test_a_review_debates_the_top_15_and_buys_the_judges_picks(tmp_path: Path) -> None:
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
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in UNIVERSE[:5]]  # in the judge's order
    state = h.store.load()
    assert state.last_completed_review == "2026-10-06" and state.abandoned_streak == 0
    assert [pick["symbol"] for pick in state.last_picks] == UNIVERSE[:5]


def test_the_debaters_are_told_the_character_caps_that_reject_a_submission(tmp_path: Path) -> None:
    h = _harness(tmp_path, params=BullBearParams(argument_max_chars=123, reason_max_chars=77))

    assert h.run().completed

    bull, bear, judge = (h.agents[name].calls[0] for name in ("bull", "bear", "judge"))
    assert bull["context"]["constraints"] == bear["context"]["constraints"] == {"argument_max_chars": 123}
    assert judge["context"]["constraints"] == {"min_picks": 5, "max_picks": 10, "reason_max_chars": 77}
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
    assert [pick["symbol"] for pick in line["picks"]] == UNIVERSE[:5] and line["drops"] == []
    stock_weights = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert sum(stock_weights.values()) == pytest.approx(0.98)
    assert all(0.04 <= weight <= 0.20 for weight in stock_weights.values())
    assert [order["symbol"] for order in line["orders"]] == UNIVERSE[:5]


def test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20, "S19": 20, "OLD": 20}, params=BullBearParams(retention_rank=18))

    outcome = h.run()

    assert outcome.completed
    (line,) = h.log_lines()
    assert [stock["symbol"] for stock in line["debate_set"]] == [*UNIVERSE[:15], "S16"]
    assert line["debate_set"][-1] == {"symbol": "S16", "rank": 17, "held": True}
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}, {"symbol": "S19", "reason": "rank 20"}]
    assert h.agents["judge"].calls[0]["context"]["held"] == ["S16"]
    assert line["drops"] == [{"symbol": "S16", "reason": "lost the debate"}]
    assert {("OLD", "sell", 20.0), ("S19", "sell", 20.0), ("S16", "sell", 20.0)} <= set(h.orders)
    assert h.bars.calls == [(UNIVERSE, ["OLD", "S16", "S19"])]


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


@pytest.mark.parametrize(("agent", "tool"), [("bull", "submit_bull_case"), ("bear", "submit_bear_case"), ("judge", "submit_picks")])
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


def test_a_debate_set_smaller_than_min_picks_abandons_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:4]})

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == []
    (line,) = h.log_lines()
    assert line["stage"] == "data" and "fewer than min_picks (5)" in line["error"]


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
    assert [pick["symbol"] for pick in line["picks"]] == UNIVERSE[:5] and line["drops"] == []
    stock_targets = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert sorted(stock_targets) == UNIVERSE[:5] and "SHV" in line["targets"]
    assert sum(stock_targets.values()) == pytest.approx(0.98)
    state = h.store.load()
    assert (state.last_completed_review, state.abandoned_streak) == (None, 1)


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
    assert h.agents["judge"].calls[0]["context"]["held"] == []
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
