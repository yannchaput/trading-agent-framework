from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bill_ackman.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewOutcome, ReviewPipeline
from trading_agent_framework.strategies.bill_ackman.rebalancer import Rebalancer
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, ConfigurationError, FundamentalsError

Step = Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]


def _candidate(symbol: str, rank: int = 1) -> Candidate:
    return Candidate(
        symbol=symbol,
        rank=rank,
        score=0.8,
        sic=5812,
        market_cap=Decimal("100000000000"),
        fcf_yield=0.05,
        fcf_margin=0.2,
        operating_margin=0.25,
        operating_margin_stdev=0.02,
        revenue_growth=0.08,
        net_debt_to_operating_income=1.0,
        debt_reported=True,
        fiscal_year_end=date(2025, 12, 31),
        filed=date(2026, 2, 15),
    )


# --- fakes -------------------------------------------------------------------------------------------


class FakeScreen:
    """Returns the universe result for a universe call and the holdings result for a call with `top_n`."""

    def __init__(self, candidates: list[Candidate], *, holdings: list[Candidate] | None = None, holding_rejections: dict[str, str] | None = None) -> None:
        self.universe = ScreenResult(candidates=candidates, rejections={})
        self.holdings = ScreenResult(candidates=holdings or [], rejections=holding_rejections or {})
        self.calls: list[tuple[list[str], int | None]] = []
        self.error: FundamentalsError | None = None

    def run(self, symbols, *, as_of, price_of, top_n=None) -> ScreenResult:  # noqa: ANN001
        self.calls.append((list(symbols), top_n))
        if self.error is not None:
            raise self.error
        return self.universe if top_n is None else self.holdings


class FakeAgent:
    """Runs one scripted step per call; a step calls the real submit tools (or raises `AgentError`)."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.calls: list[dict[str, Any]] = []
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self.tool_calls: list[ToolCallRecord] = []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None) -> AgentRunResult:
        self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool})
        if self.steps:
            self.steps.pop(0)(self.tools, context)
        return AgentRunResult(output="done", tool_calls=list(self.tool_calls))


def ranks(*symbols: str) -> Step:
    return lambda tools, ctx: tools["submit_ranking"]([{"symbol": symbol, "reason": f"{symbol} is simple"} for symbol in symbols])


def judges(**verdicts: str) -> Step:
    return lambda tools, ctx: tools["submit_verdicts"](
        [
            {"symbol": symbol, "verdict": verdict, "reason": f"{verdict} reason", "what_changed": "new facts", **({"concern": "debt"} if verdict == "fail" else {})}
            for symbol, verdict in verdicts.items()
        ]
    )


def holds(**weights: float) -> Step:
    return lambda tools, ctx: tools["submit_portfolio"]([{"symbol": symbol, "weight": weight, "reason": "best idea"} for symbol, weight in weights.items()])


def does_nothing(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    return None


def crashes(message: str) -> Step:
    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        raise AgentError(message)

    return step


@dataclass
class Harness:
    pipeline: ReviewPipeline
    broker: FakeBroker
    screen: FakeScreen
    researcher: FakeAgent
    short_seller: FakeAgent
    trader: FakeAgent
    store: StateStore
    log_path: Path
    orders: list[tuple[str, str, float]] = field(default_factory=list)

    def run(self) -> ReviewOutcome:
        self.broker.submitted.clear()
        outcome = self.pipeline.run()
        self.orders = [(order.asset.symbol, order.side.value, float(order.quantity)) for order in self.broker.submitted]
        return outcome

    def log_lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []


def _harness(
    tmp_path: Path,
    screen: FakeScreen,
    *,
    held: dict[str, float] | None = None,
    params: AckmanParams | None = None,
    state: ReviewState | None = None,
) -> Harness:
    """Portfolio value 10,000. `held` maps a symbol to a number of shares; prices: AAA 50, BBB 25, HHH 50, SHV 100."""
    params = params or AckmanParams()
    prices = {"AAA": 50, "BBB": 25, "CCC": 20, "HHH": 50, "SHV": 100}
    held = held or {}
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.last_prices = {symbol: Decimal(price) for symbol, price in prices.items()}
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(shares), side=PositionSide.LONG) for symbol, shares in held.items()]
    invested = sum(shares * prices[symbol] for symbol, shares in held.items())
    broker.account = AccountBalances(cash=Decimal(10_000 - invested), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    store = StateStore(tmp_path / "data" / "state.json")
    if state is not None:
        store.save(state)
    recorder = HandoffRecorder(params)
    agents = {"researcher": FakeAgent(), "short_seller": FakeAgent(), "trader": FakeAgent()}
    for agent in agents.values():
        agent.tools = submit_tools(recorder)
    log_path = tmp_path / "logs" / "reviews.jsonl"
    pipeline = ReviewPipeline(
        strategy=strategy,
        params=params,
        screen=screen,
        agents=agents,
        recorder=recorder,
        state=store,
        review_log=ReviewLog(log_path),
        rebalancer=Rebalancer(strategy, params),
        universe=["AAA", "BBB", "CCC"],
    )
    return Harness(pipeline, broker, screen, agents["researcher"], agents["short_seller"], agents["trader"], store, log_path)


# --- the happy path ---------------------------------------------------------------------------------


def test_a_review_runs_the_three_agents_in_order_and_places_the_orders(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2), _candidate("CCC", 3)]))
    h.researcher.steps = [ranks("AAA", "BBB")]
    h.short_seller.steps = [judges(AAA="survive", BBB="fail")]
    h.trader.steps = [holds(AAA=0.3)]

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=True, abandoned_streak=0)
    # 30% in AAA, the rest but the 2% buffer parked (a floating-point weight of 0.68 floors to 67.999999 shares)
    assert h.orders[0] == ("AAA", "buy", 60.0)
    assert h.orders[1][:2] == ("SHV", "buy") and h.orders[1][2] == pytest.approx(68.0, abs=1e-5)
    assert len(h.orders) == 2
    state = h.store.load()
    assert (state.last_review, state.last_ranking, state.abandoned_streak) == ("2026-09-14", ["AAA", "BBB"], 0)
    assert state.last_verdicts == {
        "AAA": {"verdict": "survive", "reason": "survive reason", "concern": None, "date": "2026-09-14"},
        "BBB": {"verdict": "fail", "reason": "fail reason", "concern": "debt", "date": "2026-09-14"},
    }


def test_the_review_log_records_the_whole_review_as_one_line(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2)]))
    h.researcher.steps = [ranks("AAA", "BBB")]
    h.short_seller.steps = [judges(AAA="survive", BBB="fail")]
    h.trader.steps = [holds(AAA=0.3)]

    h.run()

    (line,) = h.log_lines()
    assert line["date"] == "2026-09-14" and line["abandoned"] is False
    assert [c["symbol"] for c in line["candidates"]] == ["AAA", "BBB"]
    assert [idea["symbol"] for idea in line["ranking"]] == ["AAA", "BBB"]
    assert line["review_set"] == ["AAA", "BBB"]
    assert {(v["symbol"], v["verdict"], v["source"]) for v in line["verdicts"]} == {("AAA", "survive", "llm"), ("BBB", "fail", "llm")}
    assert line["allowed"] == ["AAA"]
    assert line["portfolio"] == [{"symbol": "AAA", "weight": 0.3, "reason": "best idea"}]
    assert line["targets"]["AAA"] == 0.3 and line["targets"]["SHV"] == pytest.approx(0.68)
    assert line["orders"][0] == {"symbol": "AAA", "side": "buy", "quantity": 60.0}
    assert line["orders"][1]["symbol"] == "SHV" and line["orders"][1]["quantity"] == pytest.approx(68.0, abs=1e-5)
    assert line["forced_exits"] == [] and line["fail_counts_after"] == {}


def test_the_researcher_is_asked_for_the_number_of_ideas_the_candidates_allow(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA"), _candidate("BBB"), _candidate("CCC")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    h.run()

    assert "Rank your best 3 ideas" in h.researcher.calls[0]["task"]


def test_the_agents_get_the_context_the_design_promises(tmp_path: Path) -> None:
    state = ReviewState(last_ranking=["OLD"], fail_counts={"HHH": 1})
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, state=state)
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="survive")]
    h.trader.steps = [holds(AAA=0.3, HHH=0.2)]

    h.run()

    researcher = h.researcher.calls[0]["context"]
    assert researcher["current_datetime"].startswith("2026-09-14T10:00")
    assert [sheet["symbol"] for sheet in researcher["candidates"]] == ["AAA"]
    assert researcher["holdings"] == [{"symbol": "HHH", "weight": 0.5}]
    assert researcher["previous_ranking"] == ["OLD"]
    seller = h.short_seller.calls[0]["context"]["to_judge"]
    assert [entry["fact_sheet"]["symbol"] for entry in seller] == ["AAA", "HHH"]  # the ranked first, then the holdings
    assert seller[0]["researcher_reason"] == "AAA is simple" and seller[0]["held"] is False
    assert seller[1] == {**seller[1], "held": True, "current_weight": 0.5, "fail_count": 1, "researcher_reason": None}
    trader = h.trader.calls[0]["context"]
    by_symbol = {entry["symbol"]: entry for entry in trader["allowed"]}
    assert by_symbol["AAA"]["research_rank"] == 1 and by_symbol["AAA"]["verdict"] == "survive" and by_symbol["AAA"]["pending_fail_count"] == 0
    assert by_symbol["HHH"]["research_rank"] is None and by_symbol["HHH"]["verdict"] == "survive" and by_symbol["HHH"]["pending_fail_count"] == 0
    assert by_symbol["HHH"]["current_weight"] == 0.5
    assert trader["forced_exits"] == []
    assert trader["constraints"]["max_positions"] == 5 and trader["constraints"]["max_total_weight"] == pytest.approx(0.98)


# --- holdings, hysteresis and forced exits ------------------------------------------------------------


def test_a_holding_that_fails_twice_in_a_row_is_sold_even_if_the_trader_would_keep_it(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})

    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]
    first = h.run()
    assert first.completed
    assert h.store.load().fail_counts == {"HHH": 1}
    (pending,) = h.trader.calls[0]["context"]["allowed"]
    assert (pending["symbol"], pending["verdict"], pending["pending_fail_count"]) == ("HHH", "fail", 1)
    assert h.orders[0] == ("HHH", "sell", 30.0)  # trimmed to 35%: it is a pending fail, not yet a forced exit

    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]  # would keep it, but the trader is not even asked: nothing is allowed
    second = h.run()

    assert second.completed
    assert h.store.load().fail_counts == {"HHH": 2}
    assert len(h.trader.calls) == 1  # only the first day
    # The first day's trim (30 shares) is still open in the fake broker, which never fills: the forced exit sells the rest.
    assert h.orders[0] == ("HHH", "sell", 70.0)
    assert h.log_lines()[-1]["forced_exits"] == ["HHH"]


def test_a_holding_that_recovers_resets_its_counter(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.35)]
    h.run()
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert h.store.load().fail_counts == {}


def test_the_forced_exit_threshold_comes_from_the_parameters(tmp_path: Path) -> None:
    screen = FakeScreen([], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, params=AckmanParams(forced_exit_fails=1))
    h.short_seller.steps = [judges(HHH="fail")]

    h.run()

    assert h.orders[0] == ("HHH", "sell", 100.0)
    assert h.trader.calls == []


def test_a_holding_the_screen_rejected_on_quality_fails_by_code_and_never_reaches_the_short_seller(tmp_path: Path) -> None:
    screen = FakeScreen([], holding_rejections={"HHH": "negative_fcf"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert h.short_seller.calls == []
    (line,) = h.log_lines()
    assert line["verdicts"] == [{"symbol": "HHH", "verdict": "fail", "reason": "screen: negative_fcf", "concern": None, "what_changed": None, "source": "screen"}]
    assert h.store.load().fail_counts == {"HHH": 1}
    assert h.trader.calls[0]["context"]["allowed"][0]["verdict_reason"] == "screen: negative_fcf"


def test_a_holding_the_screen_could_not_describe_goes_to_the_short_seller_with_a_reduced_sheet(tmp_path: Path) -> None:
    screen = FakeScreen([], holding_rejections={"HHH": "no_data"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    (entry,) = h.short_seller.calls[0]["context"]["to_judge"]
    assert entry["fact_sheet"]["screen_unavailable"] == "no_data"
    assert entry["fact_sheet"]["price"] == 50.0
    assert h.log_lines()[0]["verdicts"][0]["source"] == "llm"


def test_the_holdings_are_screened_without_truncation(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail", HHH="survive")]
    h.trader.steps = [holds(HHH=0.35)]

    h.run()

    assert screen.calls == [(["AAA", "BBB", "CCC"], None), (["HHH"], 1)]


def test_a_pending_fail_the_trader_drops_is_refused_and_the_holding_is_kept(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="fail")]
    h.trader.steps = [holds(AAA=0.3), holds(AAA=0.3, HHH=0.2)]

    h.run()

    assert h.trader.calls[0]["context"]["required"] == ["HHH"]
    assert h.trader.calls[1]["force_tool"] == "submit_portfolio"
    assert "HHH failed once and is kept until a second consecutive fail" in h.trader.calls[1]["task"]
    assert ("HHH", "sell", 60.0) in h.orders  # shrunk from 50% to 20%, not sold out
    (line,) = h.log_lines()
    assert line["required"] == ["HHH"]


def test_a_forced_exit_starts_its_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=ReviewState(fail_counts={"HHH": 1}))
    h.short_seller.steps = [judges(HHH="fail")]

    h.run()

    assert ("HHH", "sell", 100.0) in h.orders
    assert h.store.load().cooldowns == {"HHH": 4}
    assert h.log_lines()[0]["cooldowns"] == {"HHH": 4}


def test_a_symbol_cooling_down_is_hidden_from_the_researcher_and_its_cooldown_counts_down(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2)]), state=ReviewState(cooldowns={"BBB": 2}))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [holds(AAA=0.3)]

    h.run()

    assert [sheet["symbol"] for sheet in h.researcher.calls[0]["context"]["candidates"]] == ["AAA"]
    assert h.store.load().cooldowns == {"BBB": 1}
    (line,) = h.log_lines()
    assert [c["symbol"] for c in line["candidates"]] == ["AAA"]


def test_an_abandoned_review_leaves_the_cooldowns_alone(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]), state=ReviewState(cooldowns={"BBB": 2}))
    h.researcher.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert outcome.completed is False
    assert h.store.load().cooldowns == {"BBB": 2}


# --- failures ----------------------------------------------------------------------------------------


def test_an_invalid_submission_is_corrected_inside_the_same_run(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))

    def bad_then_good(tools, ctx) -> None:  # noqa: ANN001
        assert "error" in tools["submit_ranking"]([{"symbol": "ZZZ", "reason": "x"}])
        ranks("AAA")(tools, ctx)

    h.researcher.steps = [bad_then_good]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed
    assert len(h.researcher.calls) == 1


def test_a_missing_submission_is_fixed_by_one_forced_retry_in_the_same_run(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [does_nothing, ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed

    first, retry = h.researcher.calls
    assert (first["force_tool"], retry["force_tool"]) == (None, "submit_ranking")
    assert first["run_id"] == retry["run_id"]
    assert "ended without calling submit_ranking" in retry["task"]


def test_the_retry_prompt_quotes_the_last_tool_error(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [lambda tools, ctx: tools["submit_ranking"]([{"symbol": "ZZZ", "reason": "x"}]), ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    h.run()

    assert "not one of the candidates" in h.researcher.calls[1]["task"]


def test_a_stage_with_no_valid_submission_after_the_retry_abandons_the_review_and_trades_nothing(tmp_path: Path) -> None:
    before = ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, last_ranking=["OLD"], abandoned_streak=1)
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=before)
    h.researcher.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=2)
    assert h.orders == []
    assert h.short_seller.calls == [] and h.trader.calls == []
    assert h.store.load() == ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, last_ranking=["OLD"], abandoned_streak=2)
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == "researcher" and line["abandoned_streak"] == 2


def test_an_agent_error_on_both_attempts_abandons_the_review_with_its_message(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [crashes("llm down"), crashes("llm down")]

    outcome = h.run()

    assert not outcome.completed
    assert "llm down" in h.log_lines()[0]["error"]


def test_a_later_stage_can_abandon_the_review_too(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert not outcome.completed
    assert h.log_lines()[0]["stage"] == "short_seller"
    assert h.orders == []


def test_a_screen_outage_abandons_the_review_before_any_agent_runs(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA")])
    screen.error = FundamentalsError("sec is down")
    h = _harness(tmp_path, screen)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    assert h.researcher.calls == []
    assert h.log_lines()[0]["stage"] == "screen"


def test_a_completed_review_resets_the_abandoned_streak(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]), state=ReviewState(abandoned_streak=2))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().abandoned_streak == 0
    assert h.store.load().abandoned_streak == 0


# --- degenerate days ---------------------------------------------------------------------------------


def test_with_no_candidates_and_no_holdings_no_agent_runs_and_everything_is_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([]))

    outcome = h.run()

    assert outcome.completed
    assert (h.researcher.calls, h.short_seller.calls, h.trader.calls) == ([], [], [])
    assert h.orders == [("SHV", "buy", 98.0)]


def test_when_nothing_survives_the_trader_is_not_asked_and_everything_is_parked(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="fail")]

    assert h.run().completed

    assert h.trader.calls == []
    assert h.orders == [("SHV", "buy", 98.0)]


def test_an_empty_portfolio_from_the_trader_parks_everything(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [lambda tools, ctx: tools["submit_portfolio"]([])]

    assert h.run().completed
    assert h.orders == [("SHV", "buy", 98.0)]


# --- inputs the design implies but does not spell out ---------------------------------------------------------


def test_a_failed_price_lookup_never_stops_a_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.broker.market_data_error = BrokerError("market data is down")
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [holds(AAA=0.3)]

    outcome = h.run()

    assert outcome.completed
    (sheet,) = h.researcher.calls[0]["context"]["candidates"]
    assert sheet["price"] is None and sheet["price_return_12m"] is None  # the fact sheet says what it could not compute
    assert h.orders == []  # no price, nothing to size an order with


def test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state_alone(tmp_path: Path) -> None:
    before = ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, abandoned_streak=1)
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=before)

    def no_model(tools, ctx) -> None:  # noqa: ANN001
        raise ConfigurationError("No model id given and LLM_MODEL is not set")

    h.researcher.steps = [no_model]

    with pytest.raises(ConfigurationError, match="LLM_MODEL"):
        h.run()

    assert h.orders == []
    assert h.store.load() == before  # an unusable LLM setup is a configuration problem, not an abandoned review
    assert h.log_lines() == []


def test_a_position_the_screen_does_not_know_can_be_dropped_by_the_trader_and_is_then_sold(tmp_path: Path) -> None:
    # A dedicated account is assumed, so any position is reviewed; an ETF has no SEC data and reaches the short seller with a reduced sheet.
    screen = FakeScreen([], holding_rejections={"HHH": "no_data"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds()]  # the trader chooses to hold nothing

    assert h.run().completed

    assert h.orders[0] == ("HHH", "sell", 100.0)


def test_the_counters_survive_a_restart(tmp_path: Path) -> None:
    first_day = _harness(tmp_path / "one", FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100})
    first_day.short_seller.steps = [judges(HHH="fail")]
    first_day.trader.steps = [holds(HHH=0.35)]
    first_day.run()
    saved = first_day.store.load()
    assert saved.fail_counts == {"HHH": 1}

    # A new process: a new pipeline, a new state file, seeded from what the old one saved on disk.
    next_day = _harness(tmp_path / "two", FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=saved)
    next_day.short_seller.steps = [judges(HHH="fail")]

    assert next_day.run().completed

    assert next_day.log_lines()[0]["forced_exits"] == ["HHH"]
    assert next_day.trader.calls == []


# --- a broker or backtest failure inside a review ----------------------------------------------------------------


def _account_fails_on_call(h: Harness, monkeypatch: pytest.MonkeyPatch, number: int, error: Exception) -> None:
    """Make the `number`-th `get_account` call raise `error` (call 1 is current_weights, 2 the top of rebalance, 3 the post-sells read)."""
    original = h.broker.get_account
    calls = {"count": 0}

    def get_account() -> AccountBalances:
        calls["count"] += 1
        if calls["count"] == number:
            raise error
        return original()

    monkeypatch.setattr(h.broker, "get_account", get_account)


@pytest.mark.parametrize("error", [BrokerError("down"), BacktestError("data gone")])
def test_a_broker_failure_before_the_rebalance_abandons_the_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    _account_fails_on_call(h, monkeypatch, 1, error)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    assert h.orders == []
    assert h.store.load().abandoned_streak == 1
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == "broker" and line["abandoned_streak"] == 1
    assert str(error) in line["error"] and "orders" not in line


def test_a_failure_after_a_sell_went_out_abandons_the_review_and_logs_the_orders_already_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, abandoned_streak=1)
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=before)
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="survive")]
    h.trader.steps = [holds(AAA=0.3)]  # HHH is dropped: the rebalancer sells it, then sizes the buys
    _account_fails_on_call(h, monkeypatch, 3, BacktestError("data gone"))

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=2)
    assert h.orders == [("HHH", "sell", 100.0)]
    assert h.store.load() == ReviewState(last_review="2026-09-11", fail_counts={"HHH": 1}, abandoned_streak=2)  # the counters are not saved
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == "execution" and line["abandoned_streak"] == 2
    assert line["orders"] == [{"symbol": "HHH", "side": "sell", "quantity": 100.0}]
    assert "data gone" in line["error"]


def test_an_agent_error_on_the_retry_is_not_reported_with_the_first_attempts_tool_error(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]))
    h.researcher.steps = [lambda tools, ctx: tools["submit_ranking"]([{"symbol": "ZZZ", "reason": "x"}]), crashes("llm down")]

    outcome = h.run()

    assert not outcome.completed
    error = h.log_lines()[0]["error"]
    assert "llm down" in error and "not one of the candidates" not in error
    assert "not one of the candidates" in h.researcher.calls[1]["task"]  # the retry prompt still quotes it


def test_every_agent_tool_call_is_logged_at_debug_level(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2)]))
    h.researcher.steps = [ranks("AAA", "BBB")]
    h.short_seller.steps = [judges(AAA="survive", BBB="fail")]
    h.short_seller.tool_calls = [ToolCallRecord(name="search_news", args={"symbols": ["AAA"]}, result='{"articles": []}')]
    h.trader.steps = [holds(AAA=0.3)]

    with caplog.at_level(logging.DEBUG):
        h.run()

    lines = [record for record in caplog.records if "search_news" in record.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.DEBUG
    assert "short_seller" in lines[0].getMessage() and "'AAA'" in lines[0].getMessage()
