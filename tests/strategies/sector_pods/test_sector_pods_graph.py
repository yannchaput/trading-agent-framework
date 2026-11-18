import json
import logging
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.graph import AgentNode
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.agents.submit import NodeRecorder
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.common.daily_bars import DailySeries, DataUnavailable
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.sector_pods.graph import (
    Execute,
    GraphParts,
    LoadPrevious,
    MarketFacts,
    PodsState,
    build_graph,
    pm_reads,
    pod_reads,
    risk_reads,
)
from trading_agent_framework.strategies.sector_pods.handoff import (
    PM_SUBMIT,
    POD_SUBMIT,
    RISK_SUBMIT,
    UNAVAILABLE,
    pm_submit_tool,
    pod_submit_tool,
    risk_submit_tool,
    validate_pm,
    validate_pod,
    validate_risk,
)
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, PODS
from trading_agent_framework.strategies.sector_pods.store import StateStore, WeekState
from trading_agent_framework.utils.errors import AgentError, BrokerError
from trading_agent_framework.utils.log import ColorLogger

LOG = ColorLogger(logging.getLogger("test.sector_pods.graph"), "Graph")
MAIN = threading.current_thread()


def _series(index: int) -> DailySeries:
    closes = [50.0 + index + 0.1 * i for i in range(260)]
    return DailySeries(closes=closes, volumes=[1e6] * 260)


class FakeBars:
    def __init__(self) -> None:
        self.error: DataUnavailable | None = None

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        if self.error is not None:
            raise self.error
        return {symbol: _series(index) for index, symbol in enumerate(universe)}


class RecordingBroker(FakeBroker):
    """Records the thread of every submission; the `fail_on`-th submission raises `BrokerError`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.threads: list[threading.Thread] = []
        self.fail_on: int | None = None
        self.refuse_all = False  # every submission raises `BrokerError`, as a broker that rejects each order
        self.account_reads = 0
        self.fail_account_on: int | None = None

    def get_account(self) -> AccountBalances:
        self.account_reads += 1
        if self.fail_account_on == self.account_reads:
            raise BrokerError("account unavailable")
        return super().get_account()

    def _submit_order(self, order: Order) -> Order:
        self.threads.append(threading.current_thread())
        if self.refuse_all or (self.fail_on is not None and len(self.threads) == self.fail_on):
            raise BrokerError("broker down")
        return super()._submit_order(order)


Script = Callable[[Callable[..., dict[str, Any]], Any], None]


class FakeAgent:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.submit: Callable[..., dict[str, Any]] = lambda **_: {}
        self.contexts: list[Any] = []
        self.threads: list[threading.Thread] = []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None, max_steps: int | None = None) -> AgentRunResult:
        self.contexts.append(context)
        self.threads.append(threading.current_thread())
        self.script(self.submit, context)
        return AgentRunResult(output="done", tool_calls=[])


def pod_scores(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
    tool(scores=[{"symbol": s, "score": 1, "reason": "trend"} for s in ctx["facts"]], sector_view="fine")


def no_objections(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
    tool(objections=[], summary="calm")


def weights(book: dict[str, float]) -> Script:
    def script(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
        tool(weights=[{"symbol": s, "weight": w, "reason": "why"} for s, w in book.items()], overrides="", summary="done")

    return script


def does_nothing(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
    return None


def raises(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
    raise AgentError("model down")


@dataclass
class Harness:
    graph: Any
    broker: RecordingBroker
    bars: FakeBars
    agents: dict[str, FakeAgent]
    store: StateStore
    log_path: Path
    rebalancer: Rebalancer
    params: SectorPodsParams
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        return self.graph.invoke({})

    def log_lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []

    def symbols(self) -> list[str]:
        return [order.asset.symbol for order in self.broker.submitted]


def _harness(
    tmp_path: Path,
    *,
    held: dict[str, float] | None = None,
    state: WeekState | None = None,
    barrier: threading.Barrier | None = None,
    pm: Script | None = None,
    scripts: dict[str, Script] | None = None,
) -> Harness:
    """Value 10,000; every ETF costs 50, SHV 100; `held` maps a symbol to shares."""
    params = SectorPodsParams()
    broker = RecordingBroker(FakeClock(et(2026, 11, 10, 12)), strategy_name="sector_pods")
    broker.last_prices = {symbol: Decimal(50) for symbol in ALL_SYMBOLS} | {"SHV": Decimal(100)}
    broker.positions = [Position(strategy_name="sector_pods", asset=Asset(symbol), quantity=Decimal(str(shares)), side=PositionSide.LONG) for symbol, shares in (held or {}).items()]
    invested = sum(shares * 50 for shares in (held or {}).values())
    broker.account = AccountBalances(cash=Decimal(str(10_000 - invested)), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    rebalancer = Rebalancer(strategy, params)
    store = StateStore(tmp_path / "state.json", log=LOG)
    if state is not None:
        store.save(state)
    log_path = tmp_path / "run.jsonl"
    bars = FakeBars()
    agents: dict[str, FakeAgent] = {}

    def make(name: str, script: Script, validate: Any, tool_factory: Any, submit_tool: str, reads: Any, write: Callable[[Any], dict[str, Any]]) -> AgentNode:
        recorder = NodeRecorder(submit_tool, validate)
        agent = FakeAgent(script)
        agent.submit = tool_factory(recorder, params)
        agents[name] = agent
        return AgentNode(name=name, agent=agent, recorder=recorder, submit_tool=submit_tool, task=f"task {name}", reads=reads, write=write)

    def waiting(script: Script) -> Script:
        def run(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
            if barrier is not None:
                barrier.wait()
            script(tool, ctx)

        return run

    scripts = scripts or {}
    pods = {
        pod.key: make(
            pod.key,
            waiting(scripts.get(pod.key, pod_scores)),
            validate_pod(pod, params),
            pod_submit_tool,
            POD_SUBMIT,
            pod_reads(pod.key),
            lambda v, key=pod.key: {"pods": {key: UNAVAILABLE if v is None else v}},
        )
        for pod in PODS
    }
    warnings: list[str] = []
    errors: list[str] = []
    infos: list[str] = []
    parts = GraphParts(
        load_previous=LoadPrevious(store),
        market_facts=MarketFacts(bars, regime=lambda: 1, holdings=rebalancer.current_weights, today=lambda: date(2026, 11, 10), on_warning=warnings.append),
        pods=pods,
        risk_manager=make("risk_manager", no_objections, validate_risk(params), risk_submit_tool, RISK_SUBMIT, risk_reads, lambda v: {"risk": UNAVAILABLE if v is None else v}),
        portfolio_manager=make(
            "portfolio_manager", pm or weights({"XLK": 0.2, "XLV": 0.2}), validate_pm(params), pm_submit_tool, PM_SUBMIT, pm_reads, lambda v: {"pm": UNAVAILABLE if v is None else v}
        ),
        execute=Execute(store, rebalancer, RunLog(log_path), params, on_info=infos.append, on_warning=warnings.append, on_error=errors.append),
    )
    return Harness(build_graph(parts), broker, bars, agents, store, log_path, rebalancer, params, warnings, errors, infos)


def test_graph_runs_pods_in_parallel(tmp_path: Path) -> None:
    h = _harness(tmp_path, barrier=threading.Barrier(5, timeout=5))

    final = h.run()

    assert final["abandoned"] is None
    assert set(final["pods"]) == {pod.key for pod in PODS} and all(view != UNAVAILABLE for view in final["pods"].values())


def test_single_nodes_run_on_the_calling_thread(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    h.run()

    assert h.agents["risk_manager"].threads == [MAIN] and h.agents["portfolio_manager"].threads == [MAIN]
    assert h.broker.threads and all(thread is MAIN for thread in h.broker.threads)
    assert all(h.agents[pod.key].threads[0] is not MAIN for pod in PODS)


def test_agents_see_only_their_slots(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    h.run()

    (tech,) = h.agents["tech_pod"].contexts
    assert set(tech) == {"date", "regime", "facts", "previous"}
    assert set(tech["facts"]) == set(PODS[0].sleeve) and tech["regime"] == "bullish" and tech["previous"] is None
    (risk,) = h.agents["risk_manager"].contexts
    assert len(risk["pods"]) == 5 and set(risk["facts"]) == {pod.key for pod in PODS}
    (pm,) = h.agents["portfolio_manager"].contexts
    assert {"pods", "risk", "outcome", "holdings", "unresearched"} <= set(pm)


def test_pm_sees_unresearched_holdings(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"XLF": 20}, scripts={"financials_pod": raises})

    h.run()

    (pm,) = h.agents["portfolio_manager"].contexts
    assert pm["unresearched"] == {"XLF": 0.10}


def test_completed_week_trades_and_saves_previous(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    final = h.run()

    assert set(h.symbols()) == {"XLK", "XLV", "SHV"}
    state = h.store.load()
    assert state.last_completed == "2026-11-10" and state.abandoned_streak == 0
    assert {w["symbol"]: w["weight"] for w in state.previous["pm"]["weights"]} == {"XLK": 0.2, "XLV": 0.2}
    assert set(state.previous) == {"date", "pods", "risk", "pm"}
    (line,) = h.log_lines()
    assert line["abandoned"] is None and line["date"] == "2026-11-10" and {o["symbol"] for o in line["orders"]} == {"XLK", "XLV", "SHV"}
    assert final["orders"] == line["orders"]
    assert line["holdings"] == {} and "outcome" in line


def test_risk_manager_unavailable_still_lets_the_pm_complete_the_week(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["risk_manager"].script = does_nothing

    final = h.run()

    assert final["risk"] == UNAVAILABLE and final["abandoned"] is None
    assert [e["node"] for e in final["errors"]] == ["risk_manager"]
    assert set(h.symbols()) == {"XLK", "XLV", "SHV"}


def test_failed_pod_keeps_or_trims_its_sleeve(tmp_path: Path) -> None:
    results: list[dict[str, Any]] = []

    def greedy_then_keep(tool: Callable[..., dict[str, Any]], ctx: Any) -> None:
        results.append(tool(weights=[{"symbol": "XLF", "weight": 0.15, "reason": "up"}], overrides="", summary="s"))
        results.append(tool(weights=[{"symbol": "XLF", "weight": 0.10, "reason": "keep"}], overrides="", summary="s"))

    h = _harness(tmp_path, held={"XLF": 20}, scripts={"financials_pod": raises}, pm=greedy_then_keep)

    final = h.run()

    assert final["pods"]["financials_pod"] == UNAVAILABLE
    assert [e["node"] for e in final["errors"]] == ["financials_pod"]
    assert "error" in results[0] and results[1] == {"status": "recorded"}
    assert "XLF" not in h.symbols()


def test_pm_unavailable_abandons_without_orders(tmp_path: Path) -> None:
    previous = {"date": "2026-11-03", "pods": {}, "risk": {}, "pm": {"weights": []}}
    h = _harness(tmp_path, state=WeekState(last_completed="2026-11-03", previous=previous), pm=does_nothing)

    final = h.run()

    assert h.broker.submitted == [] and final["abandoned"]
    state = h.store.load()
    assert state.abandoned_streak == 1 and state.previous == previous and state.last_completed == "2026-11-03"
    (line,) = h.log_lines()
    assert line["abandoned"] and line["abandoned_streak"] == 1 and line["orders"] == []


def test_data_unavailable_abandons_before_agents(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.bars.error = DataUnavailable("yahoo down")

    final = h.run()

    assert all(not agent.contexts for agent in h.agents.values())
    assert h.broker.submitted == [] and final["abandoned"] == "data: yahoo down"
    assert h.store.load().abandoned_streak == 1


def test_broker_error_during_rebalance_is_logged_and_abandoned(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.broker.fail_account_on = 2  # the 1st read is the holdings, the 2nd the rebalancer's own

    final = h.run()

    (line,) = h.log_lines()
    assert line["orders"] == [] and line["abandoned"].startswith("execution") and final["abandoned"] == line["abandoned"]
    assert h.errors and h.store.load().abandoned_streak == 1 and h.broker.submitted == []


def test_a_refused_order_does_not_abandon_the_week(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.broker.fail_on = 2  # the rebalancer logs a refused order and goes on; only the first reaches the broker log

    final = h.run()

    (line,) = h.log_lines()
    assert final["abandoned"] is None and [o["symbol"] for o in line["orders"]] == ["XLK", "SHV"]


def test_a_book_whose_every_order_is_refused_abandons_the_week_and_keeps_the_previous_views(tmp_path: Path) -> None:
    previous = {"date": "2026-11-03", "pods": {}, "risk": {}, "pm": {"weights": []}}
    h = _harness(tmp_path, state=WeekState(last_completed="2026-11-03", previous=previous))
    h.broker.refuse_all = True  # the rebalancer logs each refusal and returns normally

    final = h.run()

    (line,) = h.log_lines()
    assert final["abandoned"] == "execution: no order was accepted" and line["orders"] == [] and h.broker.submitted == []
    saved = h.store.load()
    assert saved.abandoned_streak == 1 and saved.last_completed == "2026-11-03" and saved.previous == previous
    assert h.errors == ["execute: no order was accepted although the book needs trades"]


def test_a_rebalance_that_sends_nothing_and_was_refused_nothing_completes_the_week_with_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)  # the book needs trades, but the rebalancer sends none and the broker refuses none: nothing could be funded
    monkeypatch.setattr(h.rebalancer, "rebalance", lambda target, forced_exits=(): [])

    final = h.run()

    (line,) = h.log_lines()
    assert final["abandoned"] is None and line["orders"] == [] and h.errors == []
    assert h.warnings == ["execute: the rebalancer sent no order although the book needs trades: nothing could be funded"]
    assert h.store.load().last_completed == "2026-11-10" and h.store.load().abandoned_streak == 0


def test_a_book_already_held_within_the_band_completes_the_week_with_no_orders(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"XLK": 50, "XLV": 50, "SHV": 48})  # the filled book already held: 25% + 25% in ETFs, 48% parked: nothing is due

    final = h.run()

    (line,) = h.log_lines()
    assert final["abandoned"] is None and line["orders"] == [] and h.store.load().last_completed == "2026-11-10"


def test_execute_refuses_weights_that_break_the_limits(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    execute = Execute(h.store, h.rebalancer, RunLog(h.log_path), h.params, on_info=h.infos.append, on_warning=h.warnings.append, on_error=h.errors.append)

    def week(pm: dict[str, Any]) -> PodsState:
        return PodsState(abandoned=None, date="2026-11-10", regime="bullish", holdings={}, facts={}, pods={}, pm=pm)

    too_big = execute(week({"weights": [{"symbol": "XLK", "weight": 0.9}]}))
    malformed = execute(week({"weights": "nope"}))

    assert too_big["abandoned"].startswith("execution") and too_big["orders"] == []
    assert malformed["abandoned"].startswith("execution") and h.broker.submitted == []
    assert len(h.errors) == 2 and h.store.load().abandoned_streak == 2


def test_execute_fills_the_book_and_records_it(tmp_path: Path) -> None:
    book = {"XLK": 0.12, "XLF": 0.08, "XLV": 0.10, "XLE": 0.10, "XLY": 0.10}
    h = _harness(tmp_path, pm=weights(book))

    h.run()

    (line,) = h.log_lines()
    assert line["book"] == pytest.approx({"XLK": 0.2352, "XLF": 0.1568, "XLV": 0.196, "XLE": 0.196, "XLY": 0.196})
    assert [w["weight"] for w in line["pm"]["weights"]] == list(book.values())  # what the PM submitted is kept
    assert h.infos == ["book filled from 50.0% to 98.0% invested"]


def test_a_capped_book_trades_below_98_and_says_so(tmp_path: Path) -> None:
    h = _harness(tmp_path)  # the default PM book, XLK 0.2 and XLV 0.2: two ETFs, each at the 25% cap

    final = h.run()

    (line,) = h.log_lines()
    assert final["abandoned"] is None and line["book"] == pytest.approx({"XLK": 0.25, "XLV": 0.25})
    assert h.infos == ["book filled from 40.0% to 50.0% invested (the caps hold it below 98.0%)"]


def test_a_refused_book_records_no_book(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    execute = Execute(h.store, h.rebalancer, RunLog(h.log_path), h.params, on_info=h.infos.append, on_warning=h.warnings.append, on_error=h.errors.append)
    week = PodsState(abandoned=None, date="2026-11-10", regime="bullish", holdings={}, facts={}, pods={}, pm={"weights": [{"symbol": "XLK", "weight": 0.9}]})

    assert execute(week)["abandoned"].startswith("execution")

    assert h.log_lines()[-1]["book"] is None and h.infos == []


def test_previous_week_reaches_agents_and_outcome(tmp_path: Path) -> None:
    tech_view = {"scores": [{"symbol": "XLK", "score": 2, "reason": "x"}, {"symbol": "XLC", "score": -1, "reason": "y"}], "sector_view": "old"}
    previous = {
        "date": "2026-11-03",
        "pods": {"tech_pod": tech_view},
        "risk": {"objections": [], "summary": "old"},
        "pm": {"weights": [{"symbol": "XLK", "weight": 0.2, "reason": "r"}], "overrides": "", "summary": "old"},
    }
    h = _harness(tmp_path, state=WeekState(last_completed="2026-11-03", previous=previous))

    h.run()

    (pm,) = h.agents["portfolio_manager"].contexts
    assert isinstance(pm["outcome"]["portfolio_1w_pct"], float) and pm["previous"] == previous["pm"]
    (tech,) = h.agents["tech_pod"].contexts
    assert tech["previous"] == tech_view
    (risk,) = h.agents["risk_manager"].contexts
    assert risk["previous"] == previous["risk"]


def test_refused_book_reports_no_orders_even_after_a_week_with_orders(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.run()
    assert h.rebalancer.placed  # last week's orders are still on the long-lived rebalancer
    execute = Execute(h.store, h.rebalancer, RunLog(h.log_path), h.params, on_info=h.infos.append, on_warning=h.warnings.append, on_error=h.errors.append)
    refused = PodsState(abandoned=None, date="2026-11-17", regime="bullish", holdings={}, facts={}, pods={}, pm={"weights": [{"symbol": "XLK", "weight": 0.9}]})

    result = execute(refused)

    assert result["orders"] == [] and result["abandoned"].startswith("execution")
    assert h.log_lines()[-1]["orders"] == [] and h.log_lines()[-1]["date"] == "2026-11-17"


def test_broker_error_after_the_sells_keeps_the_orders_already_placed(tmp_path: Path) -> None:
    previous = {"date": "2026-11-03", "pods": {}, "risk": {}, "pm": {"weights": []}}
    h = _harness(tmp_path, held={"XLF": 20}, state=WeekState(last_completed="2026-11-03", previous=previous))
    h.broker.fail_account_on = 3  # holdings, the rebalancer's first read, then the re-read after the sells

    final = h.run()

    (line,) = h.log_lines()
    assert [(o["symbol"], o["side"]) for o in line["orders"]] == [("XLF", "sell")]
    assert line["abandoned"].startswith("execution") and final["orders"] == line["orders"]
    state = h.store.load()
    assert state.abandoned_streak == 1 and state.previous == previous


def test_unavailable_views_are_not_fed_back_and_a_bad_saved_book_warns(tmp_path: Path) -> None:
    previous = {"date": "2026-11-03", "pods": {"tech_pod": UNAVAILABLE}, "risk": UNAVAILABLE, "pm": {"weights": "nope"}}
    h = _harness(tmp_path, state=WeekState(last_completed="2026-11-03", previous=previous))

    h.run()

    assert h.agents["tech_pod"].contexts[0]["previous"] is None
    assert h.agents["risk_manager"].contexts[0]["previous"] is None
    assert h.agents["portfolio_manager"].contexts[0]["outcome"] == {}
    assert any("unreadable" in w for w in h.warnings)
