from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.trade_desk import TradeDesk
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing, Form4Trade
from trading_agent_framework.strategies.insider_trades.form4.source import KnownInsiderData, LoadReport
from trading_agent_framework.strategies.insider_trades.handoff import InsiderRecorder, submit_tools
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.strategies.insider_trades.pipeline import InsiderPipeline
from trading_agent_framework.strategies.insider_trades.state import PendingTrade, StateStore
from trading_agent_framework.strategies.insider_trades.tools import insider_research_tools
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError

NOW = et(2025, 10, 14, 10)
WATCHLIST = ("AAA", "BBB", "CCC", "DDD", "EEE")
PRICES = {"AAA": 50.0, "BBB": 25.0, "CCC": 10.0, "DDD": 20.0, "EEE": 40.0}
RESEARCHER_TASK, TRADER_TASK, TRADER_RETRY_TASK = "research this stock", "build the book", "finish the trade"


def retry_prompt(tool: str, error: str) -> str:
    return f"RETRY {tool}: {error}"


def trade(accession: str, ticker: str, cik: str, *, filed: date, traded: date | None = None, code: str = "S", shares: int = 100, owned_after: int = 900) -> Form4Trade:
    return Form4Trade(accession, filed, ticker, cik, f"OWNER {cik}", "CEO", code, traded or filed, Decimal(shares), Decimal(10), Decimal(owned_after), False, ())


def filing(accession: str, ticker: str, trades: list[Form4Trade], *, filed: date) -> Form4Filing:
    return Form4Filing(accession, filed, ticker, "0000000001", ticker, (), None, (), tuple(trades), False, 0, 0, {})


def history(ticker: str, cik: str, months: tuple[int, int, int]) -> list[Form4Filing]:
    """The insider traded once in each of 2022, 2023 and 2024, in `months`: the same month every year is routine, three different ones are not."""
    return [
        filing(f"{ticker}-H{year}", ticker, [trade(f"{ticker}-H{year}", ticker, cik, filed=date(year, month, 5))], filed=date(year, month, 5))
        for year, month in zip((2022, 2023, 2024), months, strict=True)
    ]


def base_data() -> dict[str, list[Form4Filing]]:
    """AAA: an opportunistic purchase (a buying stock); BBB: an opportunistic sale (selling); CCC: a routine sale (no signal); DDD and EEE: no trade."""
    return {
        "AAA": [*history("AAA", "I1", (3, 6, 9)), filing("AAA-P1", "AAA", [trade("AAA-P1", "AAA", "I1", filed=date(2025, 10, 2), code="P", shares=1000, owned_after=9000)], filed=date(2025, 10, 2))],
        "BBB": [*history("BBB", "I1", (3, 6, 9)), filing("BBB-S1", "BBB", [trade("BBB-S1", "BBB", "I1", filed=date(2025, 10, 3), shares=1000, owned_after=4000)], filed=date(2025, 10, 3))],
        "CCC": [*history("CCC", "I2", (10, 10, 10)), filing("CCC-S1", "CCC", [trade("CCC-S1", "CCC", "I2", filed=date(2025, 10, 2), traded=date(2025, 10, 1))], filed=date(2025, 10, 2))],
        "DDD": [],
        "EEE": [],
    }


def routine_purchase_data() -> dict[str, list[Form4Filing]]:
    """Only DDD has a trade: a purchase by an insider who buys every October, so it is routine and the stock has no signal."""
    purchase = filing("DDD-P1", "DDD", [trade("DDD-P1", "DDD", "I4", filed=date(2025, 10, 8), code="P", shares=1000, owned_after=9000)], filed=date(2025, 10, 8))
    return {**{t: [] for t in WATCHLIST}, "DDD": [*history("DDD", "I4", (10, 10, 10)), purchase]}


class FakeSource:
    """What `Form4Source` does that the pipeline sees: `load` snapshots the data (and which tickers fail), `known` reads the snapshot."""

    def __init__(self, data: dict[str, list[Form4Filing]]) -> None:
        self.data = data
        self.failing: dict[str, str] = {}
        self.loads: list[tuple[datetime, date]] = []
        self.loaded_on: date | None = None
        self._rows: dict[str, list[Form4Filing]] = {}
        self._failed: dict[str, str] = {}

    def load(self, *, as_of: datetime, since: date) -> LoadReport:
        self.loads.append((as_of, since))
        self._failed = dict(self.failing)
        self._rows = {t: list(rows) for t, rows in self.data.items() if t not in self._failed}
        self.loaded_on = as_of.astimezone(MARKET_TZ).date()
        return LoadReport(loaded=tuple(self._rows), failed=dict(self._failed))

    def known(self, as_of: datetime) -> KnownInsiderData:
        today = as_of.astimezone(MARKET_TZ).date()
        return KnownInsiderData({t: [f for f in rows if f.filed < today] for t, rows in self._rows.items()}, dict(self._failed))


Script = Callable[[dict[str, Callable[..., dict[str, Any]]], dict[str, Any]], None]


class Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Script) -> None:
        self.tools, self.script = tools, script
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.calls.append((task_prompt, context, force_tool))
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def no_submission(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
    return None


class Rig:
    def __init__(self, tmp_path: Path, *, mode: TradingMode = TradingMode.PAPER, params: InsiderParams | None = None, data: dict[str, list[Form4Filing]] | None = None) -> None:
        self.params = params or InsiderParams(watchlist=WATCHLIST)
        self.clock = FakeClock(NOW)
        self.broker = FakeBroker(self.clock, strategy_name="insider_trades")
        self.broker.last_prices = {s: Decimal(str(p)) for s, p in PRICES.items()}
        self.broker.account = AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(10_000))
        self.broker.positions = []
        self.strategy = Strategy(self.broker, mode=mode, project_root=tmp_path)
        self.recorder = InsiderRecorder(self.params)
        self.desk = TradeDesk(self.strategy, self.params, log_tag="insider_trades")
        self.source = FakeSource(data if data is not None else base_data())
        self.facts = InsiderFacts(self.strategy, self.source, self.params)
        self.state = StateStore(tmp_path / "data" / "state.json")
        self.log_path = tmp_path / "logs" / "runs.jsonl"
        self.overrides: dict[str, list[dict[str, Any]]] = {}
        self.fill = True  # the trader's orders fill at once
        submit = submit_tools(self.recorder)
        desk_tools = {t.__name__: t for t in self.desk.tools()}
        research_tools = {t.__name__: t for t in insider_research_tools(self.strategy, self.facts, self.recorder.facts_fetched)}
        self.handles = {
            "researcher": Handle({**research_tools, "submit_note": submit["submit_note"]}, self._research),
            "trader": Handle({**desk_tools, "submit_target": submit["submit_target"], "submit_trade_report": submit["submit_trade_report"]}, self._build_book),
            "trader_retry": Handle({**desk_tools, "submit_trade_report": submit["submit_trade_report"]}, self._finish_trade),
        }
        self.pipeline = InsiderPipeline(
            strategy=self.strategy,
            params=self.params,
            source=self.source,
            facts=self.facts,
            agents=self.handles,
            recorder=self.recorder,
            state=self.state,
            run_log=RunLog(self.log_path),
            desk=self.desk,
            load_window=lambda: (self.clock.now(), self.clock.now().date() - timedelta(days=365 * self.params.history_years + 31)),
            researcher_task=RESEARCHER_TASK,
            trader_task=TRADER_TASK,
            trader_retry_task=TRADER_RETRY_TASK,
            retry_prompt=retry_prompt,
        )

    def day(self, days: int = 1) -> None:
        self.clock.advance(86_400 * days)

    # --- the account ---------------------------------------------------------------------------------------

    def _set_position(self, symbol: str, quantity: float) -> None:
        self.broker.positions = [p for p in self.broker.positions if p.asset.symbol != symbol]
        if quantity > 0:
            self.broker.positions.append(Position(strategy_name="insider_trades", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG))

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

    def cancel_all_open(self) -> None:
        for order in list(self.broker.tracker.get_active_orders()):
            self.broker.tracker.process_trade_event(order, OrderEvent.CANCELED)

    # --- the agents -------------------------------------------------------------------------------------------

    def _research(self, tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        ticker = context["ticker"]
        tools["get_insider_trades"](ticker)
        tools["submit_note"](f"{ticker}: insider activity read", self.overrides.get(ticker, []))

    @staticmethod
    def weights_for(classes: dict[str, str]) -> dict[str, float]:
        """Even when no stock has a signal, else a tilt: buying 20%, none 15%, selling 5%."""
        if all(c == "none" for c in classes.values()):
            return dict.fromkeys(classes, 0.19)
        return {t: {"buying": 0.20, "none": 0.15, "selling": 0.05}[c] for t, c in classes.items()}

    def _build_book(self, tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        classes = {s["ticker"]: s["class"] for s in context["stocks"]}
        weights = self.weights_for(classes)
        even = all(c == "none" for c in classes.values())
        answer = tools["submit_target"]([{"ticker": t, "weight": w, **({} if even else {"reason": "tilt to the signal"})} for t, w in weights.items()])
        assert "error" not in answer, answer
        self._place(tools, answer["shortfalls"])

    def _finish_trade(self, tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        self._place(tools, context["shortfalls"])

    def _place(self, tools: dict[str, Callable[..., dict[str, Any]]], shortfalls: list[dict[str, Any]]) -> None:
        for shortfall in sorted(shortfalls, key=lambda s: s["kind"] != "sell"):
            quantity = math.floor(shortfall["dollars"] / float(self.broker.last_prices[shortfall["symbol"]]))
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

    def researched(self) -> list[str]:
        return [context["ticker"] for _, context, force in self.handles["researcher"].calls if force is None]

    def calls(self) -> dict[str, int]:
        return {name: len(handle.calls) for name, handle in self.handles.items()}


FIRST_BOOK = [("AAA", "buy", 40.0), ("BBB", "buy", 20.0), ("CCC", "buy", 150.0), ("DDD", "buy", 75.0), ("EEE", "buy", 37.0)]


# --- the first run --------------------------------------------------------------------------------------------


def test_first_run_researches_every_ticker_with_trades_and_builds_the_book(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    outcome = rig.pipeline.run()

    assert outcome.completed and outcome.abandoned_streak == 0
    assert rig.researched() == ["AAA", "BBB", "CCC"]  # DDD and EEE have no trade in their window: a code note, no agent
    assert rig.calls() == {"researcher": 3, "trader": 1, "trader_retry": 0}
    assert rig.submitted() == FIRST_BOOK
    state = rig.state.load()
    assert len(state.processed) == 12 and {"AAA-P1", "BBB-S1", "CCC-S1"} <= set(state.processed)
    assert state.in_window == {"AAA": ["AAA-P1"], "BBB": ["BBB-S1"], "CCC": ["CCC-S1"], "DDD": [], "EEE": []}
    assert {t: n["class"] for t, n in state.notes.items()} == {"AAA": "buying", "BBB": "selling", "CCC": "none", "DDD": "none", "EEE": "none"}
    assert state.notes["AAA"]["summary"] == "AAA: insider activity read"
    assert state.notes["DDD"]["summary"] == "no open-market insider trade in the last 30 days"
    assert state.notes["DDD"]["overrides"] == [] and state.notes["DDD"]["as_of"] == "2025-10-14"
    assert state.target == {"AAA": 0.2, "BBB": 0.05, "CCC": 0.15, "DDD": 0.15, "EEE": 0.15}
    assert state.traded == list(WATCHLIST)
    assert state.pending_trade is None and state.abandoned_streak == 0 and state.last_run == "2025-10-14"
    assert [r["outcome"] for r in rig.runs()] == ["completed"]


def test_the_researcher_context_and_the_trader_context(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.day()
    rig.source.data["BBB"].append(filing("BBB-S2", "BBB", [trade("BBB-S2", "BBB", "I3", filed=date(2025, 10, 14))], filed=date(2025, 10, 14)))

    rig.pipeline.run()

    task, context, force = rig.handles["researcher"].calls[-1]
    assert (task, force) == (RESEARCHER_TASK, None)
    assert context["ticker"] == "BBB" and context["window_days"] == 30 and context["current_datetime"].startswith("2025-10-15T10:00")
    assert context["previous_note"]["class"] == "selling" and context["previous_note"]["summary"] == "BBB: insider activity read"
    first = rig.handles["researcher"].calls[0][1]
    assert first["previous_note"] is None
    task, context, _ = rig.handles["trader"].calls[-1]
    assert task == TRADER_TASK
    assert context["constraints"] == {"min_weight": 0.03, "max_weight": 0.2, "max_total_weight": 0.95, "even_weight": 0.19}
    assert [(s["ticker"], s["class"]) for s in context["stocks"]] == [("AAA", "buying"), ("BBB", "selling"), ("CCC", "none"), ("DDD", "none"), ("EEE", "none")]
    aaa, bbb, ccc, ddd = context["stocks"][:4]
    assert aaa["insiders"] == [{"name": "OWNER I1", "role": "CEO", "opportunistic_bought": 10_000, "opportunistic_sold": 0, "max_buy_fraction": 0.1111, "sell_fraction": 0.0}]
    assert bbb["insiders"][0]["opportunistic_sold"] == 10_000 and bbb["insiders"][0]["sell_fraction"] == 0.2
    assert {i["name"] for i in bbb["insiders"]} == {"OWNER I1", "OWNER I3"}
    assert ccc["insiders"][0]["opportunistic_bought"] == 0 and ddd["insiders"] == []
    # AAA is held at 40 x $50 of $10,000, DDD at 75 x $20 and BBB 20 x $25: the weights are read from the book, the previous ones from the state
    assert (aaa["current_weight"], aaa["previous_weight"]) == (0.2, 0.2)
    assert (bbb["current_weight"], bbb["previous_weight"]) == (0.05, 0.05)
    assert (ddd["current_weight"], ddd["previous_weight"]) == (0.15, 0.15)


def test_a_first_run_with_no_filing_at_all_builds_an_even_book_with_no_researcher(tmp_path: Path) -> None:
    rig = Rig(tmp_path, data={t: [] for t in WATCHLIST})

    rig.pipeline.run()

    assert rig.calls() == {"researcher": 0, "trader": 1, "trader_retry": 0}
    context = rig.handles["trader"].calls[0][1]
    assert {s["class"] for s in context["stocks"]} == {"none"}
    assert {s["current_weight"] for s in context["stocks"]} == {0.0} and {s["previous_weight"] for s in context["stocks"]} == {None}
    assert rig.state.load().target == dict.fromkeys(WATCHLIST, 0.19)


# --- act only on change ----------------------------------------------------------------------------------------


def test_a_quiet_day_runs_no_agent_and_sends_no_order(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    before = rig.state.load()
    orders = len(rig.broker.submitted)
    rig.day()

    outcome = rig.pipeline.run()

    assert outcome.completed and outcome.abandoned_streak == 0
    assert rig.calls() == {"researcher": 3, "trader": 1, "trader_retry": 0}
    assert len(rig.broker.submitted) == orders
    assert rig.state.load() == before
    assert [r["outcome"] for r in rig.runs()] == ["completed", "nothing_new"]
    assert rig.runs()[1]["changed"] == []


def test_a_new_routine_filing_still_wakes_the_researcher_for_that_ticker_only(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.day()
    # a routine sale (I2 sells every October): it never moves the class, but the researcher must get the chance to verify the label
    rig.source.data["BBB"] += [*history("BBB", "I2", (10, 10, 10)), filing("BBB-S2", "BBB", [trade("BBB-S2", "BBB", "I2", filed=date(2025, 10, 14))], filed=date(2025, 10, 14))]

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.researched() == ["AAA", "BBB", "CCC", "BBB"]
    assert rig.calls()["trader"] == 2
    context = rig.handles["trader"].calls[1][1]
    assert [s["class"] for s in context["stocks"]] == ["buying", "selling", "none", "none", "none"]
    state = rig.state.load()
    assert "BBB-S2" in state.processed and state.in_window["BBB"] == ["BBB-S1", "BBB-S2"]
    assert len(rig.broker.submitted) == len(FIRST_BOOK)  # the account is already at the target: the trader sent nothing
    assert [r["outcome"] for r in rig.runs()] == ["completed", "completed"] and rig.runs()[1]["changed"] == ["BBB"]


def test_a_filing_with_no_open_market_trade_never_wakes_anyone(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.day()
    rig.source.data["DDD"].append(filing("DDD-X1", "DDD", [], filed=date(2025, 10, 14)))  # an award or a gift: no P/S row

    rig.pipeline.run()

    assert [r["outcome"] for r in rig.runs()] == ["completed", "nothing_new"]
    assert rig.calls()["trader"] == 1


def test_a_signal_expiring_reruns_the_trader_with_a_code_note_and_no_researcher(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.day(31)  # the 30-day window no longer reaches the purchase, the sale and the routine sale

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.calls() == {"researcher": 3, "trader": 2, "trader_retry": 0}
    context = rig.handles["trader"].calls[1][1]
    assert [s["class"] for s in context["stocks"]] == ["none"] * 5
    assert {s["summary"] for s in context["stocks"]} == {"no open-market insider trade in the last 30 days"}
    assert [s["insiders"] for s in context["stocks"]] == [[]] * 5
    state = rig.state.load()
    assert state.in_window == {t: [] for t in WATCHLIST}
    assert {n["class"] for n in state.notes.values()} == {"none"}
    assert state.target == dict.fromkeys(WATCHLIST, 0.19)
    assert rig.runs()[1]["changed"] == ["AAA", "BBB", "CCC"]


def test_a_stock_that_did_not_change_keeps_its_stored_note_and_class(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    rig.day()
    rig.source.data["CCC"].append(filing("CCC-S2", "CCC", [trade("CCC-S2", "CCC", "I2", filed=date(2025, 10, 14))], filed=date(2025, 10, 14)))
    rig.handles["researcher"].script = lambda tools, context: (tools["get_insider_trades"](context["ticker"]), tools["submit_note"]("CCC again", []))[1]

    rig.pipeline.run()

    context = rig.handles["trader"].calls[1][1]
    by_ticker = {s["ticker"]: s for s in context["stocks"]}
    assert by_ticker["AAA"]["summary"] == "AAA: insider activity read" and by_ticker["AAA"]["class"] == "buying"
    assert by_ticker["CCC"]["summary"] == "CCC again"
    notes = rig.state.load().notes
    assert notes["AAA"]["as_of"] == "2025-10-14" and notes["CCC"]["summary"] == "CCC again"


def test_an_override_changes_the_class_the_trader_must_respect(tmp_path: Path) -> None:
    # DDD's purchase is routine, so no signal, until the researcher relabels it
    data = routine_purchase_data()
    rig = Rig(tmp_path, data=data)
    rig.overrides["DDD"] = [{"insider_id": "I4", "from_label": "routine", "to_label": "opportunistic", "reason": "the footnote says it was an open-market purchase after earnings"}]

    rig.pipeline.run()

    context = rig.handles["trader"].calls[0][1]
    ddd = next(s for s in context["stocks"] if s["ticker"] == "DDD")
    assert ddd["class"] == "buying"
    assert ddd["insiders"][0]["opportunistic_bought"] == 10_000 and ddd["insiders"][0]["max_buy_fraction"] == 0.1111
    note = rig.state.load().notes["DDD"]
    assert note["class"] == "buying"
    assert note["overrides"] == [{"insider_id": "I4", "from_label": "routine", "to_label": "opportunistic", "reason": "the footnote says it was an open-market purchase after earnings"}]
    assert rig.state.load().target["DDD"] == 0.2
    # the stored override still applies on a later day the stock did not change
    rig.day()
    rig.pipeline.run()
    assert rig.calls()["trader"] == 1 and rig.runs()[1]["outcome"] == "nothing_new"


def test_a_stored_override_still_shapes_the_next_trader_context(tmp_path: Path) -> None:
    data = routine_purchase_data()
    rig = Rig(tmp_path, data=data)
    rig.overrides["DDD"] = [{"insider_id": "I4", "from_label": "routine", "to_label": "opportunistic", "reason": "discretionary purchase"}]
    rig.pipeline.run()
    rig.day()
    rig.source.data["EEE"].append(filing("EEE-S1", "EEE", [trade("EEE-S1", "EEE", "I5", filed=date(2025, 10, 14))], filed=date(2025, 10, 14)))

    rig.pipeline.run()

    ddd = next(s for s in rig.handles["trader"].calls[1][1]["stocks"] if s["ticker"] == "DDD")
    assert ddd["class"] == "buying" and ddd["insiders"][0]["opportunistic_bought"] == 10_000
    assert rig.researched() == ["DDD", "EEE"]


# --- the trader --------------------------------------------------------------------------------------------------


def test_the_trader_cannot_order_before_its_target_is_recorded(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    answers: list[dict[str, Any]] = []
    build_book = rig.handles["trader"].script

    def eager(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        answers.append(tools["place_order"]("AAA", "buy", 10))
        build_book(tools, context)

    rig.handles["trader"].script = eager

    rig.pipeline.run()

    assert "no target yet" in answers[0]["error"]
    assert rig.submitted() == FIRST_BOOK  # only what the target allowed


def test_a_target_without_a_report_completes_and_leaves_a_pending_trade(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def target_only(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        classes = {s["ticker"]: s["class"] for s in context["stocks"]}
        tools["submit_target"]([{"ticker": t, "weight": w, "reason": "tilt"} for t, w in rig.weights_for(classes).items()])

    rig.handles["trader"].script = target_only

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert [force for _, _, force in rig.handles["trader"].calls] == [None, "submit_trade_report"]
    assert rig.handles["trader"].calls[1][0].startswith("RETRY submit_trade_report: ")
    state = rig.state.load()
    assert state.target["AAA"] == 0.2 and len(state.processed) == 12
    assert state.pending_trade == PendingTrade(target=state.target, days=0)
    assert rig.runs()[0]["outcome"] == "completed_pending"


def test_a_trader_that_never_records_a_target_abandons_the_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["trader"].script = no_submission

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    assert [force for _, _, force in rig.handles["trader"].calls] == [None, "submit_target"]
    assert rig.broker.submitted == []
    state = rig.state.load()
    assert state.processed == [] and state.notes == {} and state.abandoned_streak == 1
    assert rig.runs()[0]["outcome"] == "abandoned" and rig.runs()[0]["stage"] == "trader"


def test_a_broker_error_while_trading_abandons_with_the_orders_already_sent_logged(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    def trade_then_fail(tools: dict[str, Callable[..., dict[str, Any]]], context: dict[str, Any]) -> None:
        classes = {s["ticker"]: s["class"] for s in context["stocks"]}
        tools["submit_target"]([{"ticker": t, "weight": w, "reason": "tilt"} for t, w in rig.weights_for(classes).items()])
        tools["place_order"]("AAA", "buy", 40)
        raise BrokerError("the connection dropped")

    rig.handles["trader"].script = trade_then_fail

    outcome = rig.pipeline.run()

    assert not outcome.completed
    record = rig.runs()[0]
    assert record["stage"] == "execution" and "connection dropped" in record["error"]
    assert [o["symbol"] for o in record["orders"]] == ["AAA"]
    assert rig.state.load().traded == ["AAA"] and rig.state.load().processed == []


# --- the pending trade --------------------------------------------------------------------------------------------


def _unfilled_first_run(rig: Rig) -> None:
    rig.fill = False
    rig.pipeline.run()
    rig.fill = True


def test_unfilled_orders_leave_a_pending_trade_retried_by_trader_retry_only(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    state = rig.state.load()
    assert state.pending_trade == PendingTrade(target={"AAA": 0.2, "BBB": 0.05, "CCC": 0.15, "DDD": 0.15, "EEE": 0.15}, days=0)
    assert rig.runs()[0]["outcome"] == "completed_pending"
    rig.cancel_all_open()  # the day orders expired unfilled
    rig.day()

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.calls() == {"researcher": 3, "trader": 1, "trader_retry": 1}  # no researcher, and the trader proper is not run again
    task, context, force = rig.handles["trader_retry"].calls[0]
    assert (task, force) == (TRADER_RETRY_TASK, None)
    assert context["target"][0] == {"ticker": "AAA", "weight": 0.2}
    assert {s["symbol"] for s in context["shortfalls"]} == set(WATCHLIST)
    assert rig.state.load().pending_trade is None
    assert rig.runs()[1]["outcome"] == "pending_complete"
    assert rig.submitted()[len(FIRST_BOOK) :] == FIRST_BOOK  # sent again once, after the first ones were canceled


def test_working_orders_are_never_sent_twice_and_the_pending_trade_is_given_up_after_the_limit(tmp_path: Path) -> None:
    rig = Rig(tmp_path, params=InsiderParams(watchlist=WATCHLIST, max_trade_retries=2))
    _unfilled_first_run(rig)
    orders = len(rig.broker.submitted)

    outcomes = []
    for _ in range(4):
        rig.day()
        outcomes.append(rig.pipeline.run())

    assert all(o.completed for o in outcomes)
    assert len(rig.broker.submitted) == orders
    assert rig.calls() == {"researcher": 3, "trader": 1, "trader_retry": 0}
    assert [r["outcome"] for r in rig.runs()] == ["completed_pending", "pending_retry", "pending_retry", "pending_gave_up", "nothing_new"]
    assert rig.state.load().pending_trade is None


def test_the_next_day_the_pending_trade_clears_once_the_orders_filled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.fill_all_open()
    rig.day()

    rig.pipeline.run()

    assert rig.calls()["trader_retry"] == 0
    assert rig.state.load().pending_trade is None and rig.runs()[1]["outcome"] == "pending_complete"


def test_a_change_replaces_the_pending_trade(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.fill_all_open()
    rig.day()
    rig.source.data["EEE"].append(filing("EEE-S1", "EEE", [trade("EEE-S1", "EEE", "I5", filed=date(2025, 10, 14))], filed=date(2025, 10, 14)))

    rig.pipeline.run()

    assert rig.researched()[-1] == "EEE"
    assert rig.state.load().pending_trade is None and rig.runs()[1]["outcome"] == "completed"


def test_a_failed_attempt_on_a_pending_trade_still_uses_up_a_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.cancel_all_open()
    rig.handles["trader_retry"].script = no_submission
    rig.day()

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    state = rig.state.load()
    assert state.pending_trade is not None and state.pending_trade.days == 1
    assert [force for _, _, force in rig.handles["trader_retry"].calls] == [None, "submit_trade_report"]
    assert rig.runs()[1]["stage"] == "trader"


def test_orders_sent_by_the_retry_but_never_reported_complete_with_a_pending_trade(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _unfilled_first_run(rig)
    rig.cancel_all_open()
    rig.handles["trader_retry"].script = lambda tools, context: tools["place_order"]("AAA", "buy", 40)
    rig.day()

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.state.load().pending_trade is not None and rig.runs()[1]["outcome"] == "pending_retry"


# --- abandonment ---------------------------------------------------------------------------------------------------


def test_a_researcher_that_never_submits_abandons_the_run_and_keeps_the_filings_new(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["researcher"].script = no_submission

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    calls = rig.handles["researcher"].calls
    assert [force for _, _, force in calls] == [None, "submit_note"]
    assert calls[1][0].startswith("RETRY submit_note: the agent ended without calling submit_note")
    assert rig.calls()["trader"] == 0 and rig.broker.submitted == []
    state = rig.state.load()
    assert state.processed == [] and state.notes == {} and state.abandoned_streak == 1
    assert rig.runs()[0]["outcome"] == "abandoned" and rig.runs()[0]["stage"] == "researcher"
    # tomorrow tries again with everything still new, and a completed run resets the streak
    rig.handles["researcher"].script = rig._research
    rig.day()
    assert rig.pipeline.run().completed
    assert rig.researched()[-3:] == ["AAA", "BBB", "CCC"] and rig.state.load().abandoned_streak == 0


def test_the_last_error_is_quoted_in_the_forced_retry(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.handles["researcher"].script = lambda tools, context: tools["submit_note"]("read nothing", [])  # never fetched the facts

    rig.pipeline.run()

    assert "call get_insider_trades for AAA first" in rig.handles["researcher"].calls[1][0]


# --- loading the Form 4s ---------------------------------------------------------------------------------------------


def test_in_a_backtest_a_failed_ticker_abandons_the_run_and_the_next_tick_loads_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path, mode=TradingMode.BACKTESTING)
    rig.source.failing = {"BBB": "SEC throttled the request"}

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    assert rig.runs()[0]["stage"] == "filings" and "BBB" in rig.runs()[0]["error"]
    assert rig.calls() == {"researcher": 0, "trader": 0, "trader_retry": 0}
    assert len(rig.source.loads) == 1
    rig.source.failing = {}
    rig.day()

    outcome = rig.pipeline.run()

    assert outcome.completed and len(rig.source.loads) == 2
    assert rig.researched() == ["AAA", "BBB", "CCC"]
    rig.day()
    rig.pipeline.run()
    assert len(rig.source.loads) == 2  # a backtest loads once nothing is missing


def test_in_paper_a_failed_ticker_keeps_its_note_and_the_others_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    before = rig.state.load()
    rig.day()
    rig.source.failing = {"BBB": "SEC throttled the request"}
    rig.source.data["CCC"].append(filing("CCC-S2", "CCC", [trade("CCC-S2", "CCC", "I2", filed=date(2025, 10, 14))], filed=date(2025, 10, 14)))

    outcome = rig.pipeline.run()

    assert outcome.completed
    assert rig.researched() == ["AAA", "BBB", "CCC", "CCC"]  # not BBB again
    context = rig.handles["trader"].calls[1][1]
    bbb = next(s for s in context["stocks"] if s["ticker"] == "BBB")
    assert bbb["class"] == "selling" and bbb["summary"] == "BBB: insider activity read" and bbb["insiders"] == []
    state = rig.state.load()
    assert state.notes["BBB"] == before.notes["BBB"]
    assert state.in_window["BBB"] == ["BBB-S1"]
    assert set(before.processed) <= set(state.processed) and "CCC-S2" in state.processed  # BBB's filings are not forgotten
    assert rig.runs()[1]["failed"] == {"BBB": "SEC throttled the request"}


def test_a_ticker_that_failed_on_the_first_run_has_a_placeholder_and_no_stored_note(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.source.failing = {"EEE": "no such company"}

    outcome = rig.pipeline.run()

    assert outcome.completed
    eee = next(s for s in rig.handles["trader"].calls[0][1]["stocks"] if s["ticker"] == "EEE")
    assert eee["class"] == "none" and "could not be loaded" in eee["summary"] and eee["insiders"] == []
    state = rig.state.load()
    assert "EEE" not in state.notes and "EEE" not in state.in_window
    assert state.target["EEE"] == 0.15  # the target still names every watchlist stock
    # tomorrow it loads: it has no note, so it is researched (here: nothing in its window, a code note)
    rig.source.failing = {}
    rig.day()
    rig.pipeline.run()
    assert rig.state.load().notes["EEE"]["summary"] == "no open-market insider trade in the last 30 days"


def test_in_paper_every_ticker_failing_abandons_the_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.source.failing = {t: "offline" for t in WATCHLIST}

    outcome = rig.pipeline.run()

    assert not outcome.completed and outcome.abandoned_streak == 1
    assert rig.runs()[0]["stage"] == "filings"
    assert rig.calls() == {"researcher": 0, "trader": 0, "trader_retry": 0}


def test_the_source_is_loaded_once_per_market_date(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.pipeline.run()
    rig.clock.advance(3_600)  # a second tick the same market date
    rig.pipeline.run()

    assert len(rig.source.loads) == 1
    rig.day()
    rig.pipeline.run()
    assert len(rig.source.loads) == 2
    as_of, since = rig.source.loads[1]
    assert as_of == rig.clock.now() and since == date(2025, 10, 15) - timedelta(days=365 * 3 + 31)


# --- state and log -------------------------------------------------------------------------------------------------


def test_a_corrupt_state_file_starts_from_empty_and_builds_the_book_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "state.json").write_text("{broken", encoding="utf-8")

    assert rig.pipeline.run().completed
    assert rig.researched() == ["AAA", "BBB", "CCC"]


def test_a_stored_note_that_cannot_be_read_back_is_researched_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    state = rig.state.load()
    bad = {**state.notes["AAA"], "overrides": [{"insider_id": "I1", "from_label": "nonsense", "to_label": "routine", "reason": "x"}]}
    rig.state.save(replace(state, notes={**state.notes, "AAA": bad}))
    rig.day()

    rig.pipeline.run()

    assert rig.researched() == ["AAA", "BBB", "CCC", "AAA"]


def test_the_run_log_records_the_whole_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    rig.pipeline.run()
    [record] = rig.runs()

    assert record["outcome"] == "completed" and record["date"] == "2025-10-14" and record["failed"] == {}
    assert record["changed"] == list(WATCHLIST)
    assert [(s["ticker"], s["class"], s["changed"]) for s in record["stocks"]][:2] == [("AAA", "buying", True), ("BBB", "selling", True)]
    assert [t["ticker"] for t in record["target"]] == list(WATCHLIST)
    assert record["shortfalls_after"] == [] and [o["status"] for o in record["orders"]] == ["filled"] * 5


def test_the_pipeline_writes_nothing_without_a_run_directory(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline._log = RunLog(None)

    assert rig.pipeline.run().completed
    assert not rig.log_path.exists()


def test_state_is_not_saved_when_nothing_changed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.pipeline.run()
    path = tmp_path / "data" / "state.json"
    before = path.stat().st_mtime_ns
    rig.day()

    rig.pipeline.run()

    assert path.stat().st_mtime_ns == before
