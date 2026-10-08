from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.congress.annual import AssetHolding, tier_of
from trading_agent_framework.congress.ptr import FilingRef, Transaction
from trading_agent_framework.congress.source import CongressSource, KnownFilings
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.congress_trades.desk import TradeDesk
from trading_agent_framework.strategies.congress_trades.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.strategies.congress_trades.pipeline import CongressPipeline
from trading_agent_framework.strategies.congress_trades.state import CongressState, PendingTrade, RunLog, StateStore
from trading_agent_framework.utils.errors import AgentError, BrokerError, CongressDataError

NOW = et(2026, 9, 14, 10)
ANNUAL = FilingRef("A1", "Nancy Pelosi", "annual", date(2026, 5, 15), 2025)
PTR1 = FilingRef("P1", "Nancy Pelosi", "ptr", date(2026, 6, 23), 2026)
PTR2 = FilingRef("P2", "Nancy Pelosi", "ptr", date(2026, 9, 10), 2026)
PRICES = {"AAA": 50.0, "BBB": 25.0, "CCC": 10.0, "OLD": 10.0, "NEW": 20.0}
# With CongressParams() the baseline is AAA 0.15, BBB 0.15 (both capped) and CCC 0.1433: $1,500, $1,500 and $1,433 of a $10,000 account.


def _asset(ticker: str, low: int, high: int) -> AssetHolding:
    return AssetHolding("A1", "spouse", ticker, f"{ticker} Inc.", Decimal(low), Decimal(high), tier_of(Decimal(low)))


def known_filings(*, ptrs: tuple[FilingRef, ...] = (PTR1,), transactions: list[Transaction] | None = None) -> KnownFilings:
    return KnownFilings(
        annual_ref=ANNUAL,
        period_end=date(2025, 12, 31),
        assets=[_asset("AAA", 5_000_001, 25_000_000), _asset("BBB", 1_000_001, 5_000_000), _asset("CCC", 250_001, 500_000)],
        transactions=transactions or [],
        refs=sorted([ANNUAL, *ptrs], key=lambda r: (r.filed, r.doc_id), reverse=True),
        unparsed_filings=0,
        skipped_non_stock=79,
    )


class FakeSource:
    def __init__(self, known: KnownFilings) -> None:
        self.filings = known
        self.error: Exception | None = None
        self.calls = 0

    def known(self, as_of: Any) -> KnownFilings:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.filings

    def new_since(self, known: KnownFilings, processed: Any) -> list[FilingRef]:
        return CongressSource.new_since(known, processed)


Script = Callable[[dict[str, Callable[..., dict[str, Any]]], dict[str, Any]], None]


class Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Script) -> None:
        self.tools, self.script = tools, script
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.calls.append((task_prompt, context, force_tool))
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def research_baseline(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
    tools["submit_holdings"]([{"ticker": b["ticker"], "value_low": b["value_low"], "value_high": b["value_high"]} for b in context["baseline"]])


def portfolio_by_baseline(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
    tools["submit_target"]([{"ticker": h["ticker"], "weight": h["baseline_weight"] or 0, "reason": "by tier"} for h in context["holdings"]])


def no_submission(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
    return None


class Rig:
    def __init__(self, tmp_path: Path, *, params: CongressParams | None = None, known: KnownFilings | None = None, positions: dict[str, float] | None = None) -> None:
        self.params = params or CongressParams()
        self.broker = FakeBroker(FakeClock(NOW), strategy_name="congress_trades")
        self.broker.last_prices = {s: Decimal(str(p)) for s, p in PRICES.items()}
        self.broker.account = AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(10_000))
        self.broker.positions = []
        for symbol, quantity in (positions or {}).items():
            self._set_position(symbol, quantity)
        self.strategy = Strategy(self.broker, mode=TradingMode.PAPER, project_root=tmp_path)
        self.recorder = HandoffRecorder(self.params)
        self.desk = TradeDesk(self.strategy, self.params)
        self.source = FakeSource(known or known_filings())
        self.state = StateStore(tmp_path / "data" / "state.json")
        self.log_path = tmp_path / "logs" / "runs.jsonl"
        submit = submit_tools(self.recorder)
        desk_tools = {t.__name__: t for t in self.desk.tools()}
        self.handles = {
            "researcher": Handle({"submit_holdings": submit["submit_holdings"]}, research_baseline),
            "portfolio_manager": Handle({"submit_target": submit["submit_target"]}, portfolio_by_baseline),
            "trader": Handle({**desk_tools, "submit_trade_report": submit["submit_trade_report"]}, self._trade),
        }
        self.fill = True  # the trader's orders fill at once
        self.pipeline = CongressPipeline(
            strategy=self.strategy,
            params=self.params,
            source=self.source,
            agents=self.handles,
            recorder=self.recorder,
            state=self.state,
            run_log=RunLog(self.log_path),
            desk=self.desk,
        )

    # --- the account ---------------------------------------------------------------------------------------

    def _set_position(self, symbol: str, quantity: float) -> None:
        self.broker.positions = [p for p in self.broker.positions if p.asset.symbol != symbol]
        if quantity > 0:
            self.broker.positions.append(Position(strategy_name="congress_trades", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG))

    def held(self, symbol: str) -> float:
        return next((float(p.quantity) for p in self.broker.positions if p.asset.symbol == symbol), 0.0)

    def fill_order(self, order: Order) -> None:
        price = self.broker.last_prices[order.asset.symbol]
        self.broker.tracker.process_trade_event(order, OrderEvent.FILLED, price=price, filled_quantity=order.quantity)
        sign = 1 if order.side is OrderSide.BUY else -1
        self._set_position(order.asset.symbol, self.held(order.asset.symbol) + sign * float(order.quantity or 0))

    def fill_all_open(self) -> None:
        for order in self.broker.tracker.get_active_orders():
            self.fill_order(order)

    # --- the trading agent -----------------------------------------------------------------------------------

    def _trade(self, tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        for shortfall in sorted(context["shortfalls"], key=lambda s: s["kind"] != "sell"):
            price = float(self.broker.last_prices[shortfall["symbol"]])
            quantity = math.floor(shortfall["dollars"] / price)
            if quantity > 0:
                tools["place_order"](shortfall["symbol"], shortfall["kind"], quantity)
        if self.fill:
            for order in self.desk.orders:
                self.fill_order(order)
        tools["check_orders"]()
        report = [{"order_id": oid, **({} if view.filled else {"reason": "still working at the check"})} for oid, view in self.desk.orders_view().items()]
        tools["submit_trade_report"](report)

    # --- reading ----------------------------------------------------------------------------------------------

    def runs(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []

    def submitted(self) -> list[tuple[str, str, float]]:
        return [(o.asset.symbol, o.side.value, float(o.quantity or 0)) for o in self.broker.submitted]

    def calls(self) -> dict[str, int]:
        return {name: len(handle.calls) for name, handle in self.handles.items()}


# --- the daily rule ----------------------------------------------------------------------------------------


def test_the_first_run_treats_every_known_filing_as_new_and_builds_the_portfolio(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    outcome = rig.pipeline.run()

    assert outcome.completed and outcome.abandoned_streak == 0
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 1}
    assert rig.submitted() == [("AAA", "buy", 30.0), ("BBB", "buy", 60.0), ("CCC", "buy", 143.0)]
    state = rig.state.load()
    assert state.processed == ["P1", "A1"]
    assert state.target == {"AAA": 0.15, "BBB": 0.15, "CCC": 0.1433}
    assert state.traded == ["AAA", "BBB", "CCC"]
    assert state.pending_trade is None and state.last_run == "2026-09-14" and state.abandoned_streak == 0
    assert state.holdings["AAA"] == {"tier": 8, "value_low": 5_000_001, "value_high": 25_000_000}


def test_nothing_new_and_nothing_pending_runs_no_agent_sends_no_order_and_leaves_the_state_alone(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    before = rig.state.load()
    orders = len(rig.broker.submitted)

    outcome = rig.pipeline.run()  # the next day: the same filings

    assert outcome.completed and outcome.abandoned_streak == 0
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 1}  # no new agent call
    assert len(rig.broker.submitted) == orders
    assert rig.state.load() == before
    assert [r["outcome"] for r in rig.runs()] == ["completed", "nothing_new"]
    assert rig.runs()[1]["new_filings"] == []


def test_a_new_trade_report_runs_all_three_agents_again_and_updates_processed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.source.filings = known_filings(ptrs=(PTR1, PTR2))

    rig.pipeline.run()

    assert rig.calls() == {"researcher": 2, "portfolio_manager": 2, "trader": 1}  # the account was already at the target: no trading agent
    assert rig.handles["researcher"].calls[1][1]["new_filings"] == [{"doc_id": "P2", "kind": "ptr", "filed": "2026-09-10"}]
    assert rig.state.load().processed == ["P2", "P1", "A1"]
    assert rig.runs()[1]["outcome"] == "completed"


def test_a_quiet_day_trades_nothing_even_when_the_account_drifted(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig._set_position("AAA", 5)  # the account drifted far below the target

    rig.pipeline.run()

    assert rig.calls()["trader"] == 1
    assert len(rig.broker.submitted) == 3


def test_the_research_context_has_the_baseline_the_new_filings_and_only_known_filings(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.pipeline.run()
    _, context, force = rig.handles["researcher"].calls[0]

    assert force is None
    assert context["base_report"] == {"doc_id": "A1", "filed": "2026-05-15", "period_end": "2025-12-31"}
    assert [f["doc_id"] for f in context["new_filings"]] == ["P1", "A1"]
    assert [(b["ticker"], b["tier"]) for b in context["baseline"]] == [("AAA", 8), ("BBB", 7), ("CCC", 5)]
    assert context["unparsed_filings"] == 0 and context["skipped_non_stock"] == 79
    assert context["constraints"] == {"max_holdings": 20}


def test_the_portfolio_context_carries_tiers_baseline_weights_and_the_previous_target(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.source.filings = known_filings(ptrs=(PTR1, PTR2))

    rig.pipeline.run()
    _, context, _ = rig.handles["portfolio_manager"].calls[1]

    assert [(h["ticker"], h["tier"], h["baseline_weight"]) for h in context["holdings"]] == [("AAA", 8, 0.15), ("BBB", 7, 0.15), ("CCC", 5, 0.1433)]
    assert context["previous_target"] == {"AAA": 0.15, "BBB": 0.15, "CCC": 0.1433}
    assert context["constraints"]["max_positions"] == 15 and context["constraints"]["unallocated_money"] == "stays in cash"


def test_a_holding_beyond_the_position_limit_has_no_baseline_weight(tmp_path: Path) -> None:
    rig = Rig(tmp_path, params=CongressParams(max_positions=2))

    rig.pipeline.run()
    context = rig.handles["portfolio_manager"].calls[0][1]

    assert {h["ticker"]: h["baseline_weight"] for h in context["holdings"]} == {"AAA": 0.15, "BBB": 0.15, "CCC": None}


# --- the research agent's changes -----------------------------------------------------------------------------


def test_the_tier_is_recomputed_from_the_value_the_research_agent_submits(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def research(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        tools["submit_holdings"](
            [
                {"ticker": "AAA", "value_low": 5_000_001, "value_high": 25_000_000},
                {"ticker": "BBB", "value_low": 5_000_001, "value_high": 25_000_000, "reason": "a purchase the baseline could not read"},
                {"ticker": "CCC", "drop": True, "reason": "sold in full in the August report"},
            ]
        )

    rig.handles["researcher"].script = research

    rig.pipeline.run()

    holdings = rig.state.load().holdings
    assert set(holdings) == {"AAA", "BBB"}  # CCC dropped
    assert holdings["BBB"]["tier"] == 8
    assert [h["ticker"] for h in rig.handles["portfolio_manager"].calls[0][1]["holdings"]] == ["AAA", "BBB"]


def test_no_holdings_skips_the_portfolio_agent_and_sells_what_was_bought_before(tmp_path: Path) -> None:
    rig = Rig(tmp_path, positions={"OLD": 100})
    rig.state.save(CongressState(traded=["OLD"]))

    def research(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        tools["submit_holdings"]([{"ticker": b["ticker"], "drop": True, "reason": "sold"} for b in context["baseline"]])

    rig.handles["researcher"].script = research

    rig.pipeline.run()

    assert rig.calls()["portfolio_manager"] == 0
    assert rig.submitted() == [("OLD", "sell", 100.0)]
    assert rig.state.load().target == {}


# --- abandonment -------------------------------------------------------------------------------------------


def test_a_stage_with_no_valid_submission_is_forced_once_then_abandons_the_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["researcher"].script = no_submission

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    calls = rig.handles["researcher"].calls
    assert [force for _, _, force in calls] == [None, "submit_holdings"]
    assert "Your previous run ended without a valid submit_holdings call" in calls[1][0]
    assert rig.calls()["portfolio_manager"] == 0 and rig.broker.submitted == []
    state = rig.state.load()
    assert state.processed == [] and state.abandoned_streak == 1  # the filings stay new: tomorrow tries again
    assert rig.runs()[0]["outcome"] == "abandoned" and rig.runs()[0]["stage"] == "researcher"


def test_the_last_error_is_quoted_in_the_forced_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["researcher"].script = lambda tools, context: tools["submit_holdings"]([{"ticker": "ZZZZ", "value_low": 1, "value_high": 2, "reason": "x"}])

    rig.pipeline.run()

    assert "ZZZZ does not appear in any known filing" in rig.handles["researcher"].calls[1][0]


def test_a_successful_run_resets_the_abandoned_streak(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["researcher"].script = no_submission
    rig.pipeline.run()
    assert rig.pipeline.run().abandoned_streak == 2
    rig.handles["researcher"].script = research_baseline

    outcome = rig.pipeline.run()

    assert outcome.completed and rig.state.load().abandoned_streak == 0


def test_an_agent_error_counts_as_a_failed_attempt(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def failing(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        raise AgentError("the model server is down")

    rig.handles["portfolio_manager"].script = failing

    outcome = rig.pipeline.run()

    assert not outcome.completed
    assert rig.runs()[0]["stage"] == "portfolio_manager" and "model server is down" in rig.runs()[0]["error"]
    assert rig.broker.submitted == []  # nothing is traded when the portfolio stage fails


def test_a_filings_failure_abandons_the_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.source.error = CongressDataError("No readable yearly report")

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    assert rig.runs()[0]["stage"] == "filings"
    assert rig.calls() == {"researcher": 0, "portfolio_manager": 0, "trader": 0}


def test_the_trading_agent_without_a_valid_report_and_without_orders_abandons_the_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["trader"].script = no_submission

    outcome = rig.pipeline.run()

    assert not outcome.completed
    assert rig.state.load().processed == []
    assert rig.runs()[0]["stage"] == "trader"


def test_orders_sent_but_never_reported_do_not_discard_the_research_and_are_checked_later(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def trade_without_report(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        tools["place_order"]("AAA", "buy", 30)

    rig.handles["trader"].script = trade_without_report

    outcome = rig.pipeline.run()

    assert outcome.completed
    state = rig.state.load()
    assert state.processed == ["P1", "A1"] and state.traded == ["AAA"]
    assert state.pending_trade is not None and state.pending_trade.days == 0
    assert rig.runs()[0]["outcome"] == "completed_pending"


def test_a_broker_error_in_the_trading_stage_abandons_with_the_orders_already_sent_logged(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def trade_then_fail(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        tools["place_order"]("AAA", "buy", 30)
        raise BrokerError("the connection dropped")

    rig.handles["trader"].script = trade_then_fail

    outcome = rig.pipeline.run()

    assert not outcome.completed
    record = rig.runs()[0]
    assert record["stage"] == "execution" and "connection dropped" in record["error"]
    assert [o["symbol"] for o in record["orders"]] == ["AAA"]
    assert rig.state.load().traded == ["AAA"]  # a stock we may now hold is remembered as ours


# --- the pending trade --------------------------------------------------------------------------------------


def _unfilled_first_run(rig: Rig) -> None:
    rig.fill = False
    rig.pipeline.run()
    rig.fill = True


def test_orders_still_working_leave_a_pending_trade_and_the_run_completes(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    _unfilled_first_run(rig)

    state = rig.state.load()
    assert state.pending_trade == PendingTrade(target={"AAA": 0.15, "BBB": 0.15, "CCC": 0.1433}, days=0)
    assert state.processed == ["P1", "A1"]
    assert rig.runs()[0]["outcome"] == "completed_pending"
    assert {o["status"] for o in rig.runs()[0]["orders"]} == {"working"}


def test_the_next_day_only_the_trading_stage_is_checked_and_the_pending_trade_is_cleared_once_filled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.fill_all_open()  # the orders filled overnight
    orders = len(rig.broker.submitted)

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 1}  # no agent at all: the account is at the target
    assert len(rig.broker.submitted) == orders
    assert rig.state.load().pending_trade is None
    assert rig.runs()[1]["outcome"] == "pending_complete"


def test_a_pending_trade_with_a_shortfall_re_runs_only_the_trading_agent_against_the_stored_target(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    for order in list(rig.broker.tracker.get_active_orders()):
        rig.broker.tracker.process_trade_event(order, OrderEvent.CANCELED)  # the day orders expired unfilled

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 2}
    context = rig.handles["trader"].calls[1][1]
    assert context["target"] == [{"ticker": "AAA", "weight": 0.15}, {"ticker": "BBB", "weight": 0.15}, {"ticker": "CCC", "weight": 0.1433}]
    assert {s["symbol"] for s in context["shortfalls"]} == {"AAA", "BBB", "CCC"}
    assert rig.state.load().pending_trade is None  # the second attempt filled
    assert rig.runs()[1]["outcome"] == "pending_complete"


def test_orders_that_stay_working_are_never_sent_twice_and_the_pending_trade_is_given_up_after_the_limit(tmp_path: Path) -> None:
    rig = Rig(tmp_path, params=CongressParams(max_trade_retries=2))
    _unfilled_first_run(rig)
    orders = len(rig.broker.submitted)

    outcomes = [rig.pipeline.run() for _ in range(4)]

    assert all(o.completed for o in outcomes)
    assert len(rig.broker.submitted) == orders  # nothing was re-sent
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 1}
    assert [r["outcome"] for r in rig.runs()] == ["completed_pending", "pending_retry", "pending_retry", "pending_gave_up", "nothing_new"]
    assert rig.state.load().pending_trade is None


def test_a_new_filing_replaces_the_pending_trade(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.fill_all_open()
    rig.source.filings = known_filings(ptrs=(PTR1, PTR2))

    rig.pipeline.run()

    assert rig.calls()["researcher"] == 2
    assert rig.state.load().pending_trade is None
    assert rig.runs()[1]["outcome"] == "completed" and rig.runs()[1]["new_filings"][0]["doc_id"] == "P2"


def test_a_failed_attempt_on_a_pending_trade_still_uses_up_a_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    for order in list(rig.broker.tracker.get_active_orders()):
        rig.broker.tracker.process_trade_event(order, OrderEvent.CANCELED)
    rig.handles["trader"].script = no_submission

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    state = rig.state.load()
    assert state.pending_trade == PendingTrade(target={"AAA": 0.15, "BBB": 0.15, "CCC": 0.1433}, days=1)


def test_an_account_already_at_the_target_needs_no_trading_agent(tmp_path: Path) -> None:
    rig = Rig(tmp_path, positions={"AAA": 30, "BBB": 60, "CCC": 143})

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.calls() == {"researcher": 1, "portfolio_manager": 1, "trader": 0}
    assert rig.state.load().pending_trade is None
    assert rig.runs()[0]["shortfalls_before"] == []


def test_a_dropped_holding_this_strategy_bought_before_is_sold(tmp_path: Path) -> None:
    rig = Rig(tmp_path, positions={"AAA": 30, "BBB": 60, "CCC": 143, "OLD": 100})
    rig.state.save(CongressState(processed=["stale"], traded=["AAA", "BBB", "CCC", "OLD"]))

    rig.pipeline.run()

    assert rig.submitted() == [("OLD", "sell", 100.0)]
    assert rig.held("OLD") == 0


# --- the run log -----------------------------------------------------------------------------------------------


def test_the_run_log_records_the_whole_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.pipeline.run()
    [record] = rig.runs()

    assert record["outcome"] == "completed" and record["date"] == "2026-09-14"
    assert [f["doc_id"] for f in record["new_filings"]] == ["P1", "A1"]
    assert record["baseline"][0]["ticker"] == "AAA" and record["baseline"][0]["baseline_weight"] == 0.15
    assert [h["ticker"] for h in record["holdings"]] == ["AAA", "BBB", "CCC"]
    assert [p["ticker"] for p in record["target"]] == ["AAA", "BBB", "CCC"]
    assert len(record["shortfalls_before"]) == 3 and record["shortfalls_after"] == []
    assert [o["status"] for o in record["orders"]] == ["filled"] * 3


def test_the_pipeline_writes_nothing_without_a_run_directory(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline._log = RunLog(None)

    assert rig.pipeline.run().completed
    assert not rig.log_path.exists()


def test_a_corrupt_state_file_starts_from_empty_and_rebuilds_the_portfolio(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "state.json").write_text("{broken", encoding="utf-8")

    assert rig.pipeline.run().completed
    assert rig.calls()["researcher"] == 1


def test_state_is_not_saved_when_nothing_changed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    path = tmp_path / "data" / "state.json"
    before = path.stat().st_mtime_ns

    rig.pipeline.run()

    assert path.stat().st_mtime_ns == before
