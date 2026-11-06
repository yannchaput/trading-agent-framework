from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from tests.fakes import et

from trading_agent_framework.strategies.common.handoff import OrderView
from trading_agent_framework.strategies.common.pipeline import PendingTrade, ReviewPipeline, RunOutcome, StageAbandoned, decode_pending, valid_pending
from trading_agent_framework.strategies.common.trade_desk import Audit, Shortfall
from trading_agent_framework.utils.errors import BacktestError, BrokerError


@dataclass(frozen=True)
class ToyState:
    abandoned_streak: int = 0
    saved_by: str = ""


class FakeStore:
    def __init__(self, state: ToyState) -> None:
        self.state = state
        self.saved: list[ToyState] = []

    def load(self) -> ToyState:
        return self.state

    def save(self, state: ToyState) -> None:
        self.saved.append(state)


class FakeLog:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class ToyDataError(Exception):
    pass


class Toy(ReviewPipeline[ToyState]):
    LOG_PREFIX: ClassVar[str] = "[toy] "
    WHAT: ClassVar[str] = "round"
    DATA_ERRORS: ClassVar[tuple[type[Exception], ...]] = (ToyDataError,)

    def __init__(self, store: FakeStore, log: FakeLog) -> None:
        self.errors: list[str] = []
        strategy = SimpleNamespace(clock=SimpleNamespace(now=lambda: et(2026, 9, 14, 10)), run_id="r1", log_error=self.errors.append)
        super().__init__(strategy=cast(Any, strategy), agents={}, recorder=None, state=cast(Any, store), log=cast(Any, log))
        self.body: Exception | RunOutcome = RunOutcome(True, 0)
        self.executing = False
        self.begins = 0
        self.calls: list[str] = []
        self.seen: tuple[datetime, ToyState, dict[str, Any]] | None = None

    def _begin(self) -> None:
        self.begins += 1
        self.calls.append("begin")

    def _run(self, now: datetime, state: ToyState, record: dict[str, Any]) -> RunOutcome:
        self.calls.append("run")
        self.seen = (now, state, record)
        record["late"] = 1
        if isinstance(self.body, Exception):
            raise self.body
        return self.body

    def _executing(self) -> bool:
        return self.executing

    def _on_abandoned(self, state: ToyState, streak: int, stage: str) -> tuple[ToyState, dict[str, Any]]:
        return replace(state, abandoned_streak=streak, saved_by=stage), {"extra": stage}


def _toy(streak: int = 1) -> tuple[Toy, FakeStore, FakeLog]:
    store, log = FakeStore(ToyState(abandoned_streak=streak)), FakeLog()
    return Toy(store, log), store, log


def test_template_success_returns_body_outcome() -> None:
    toy, store, log = _toy()
    toy.body = RunOutcome(True, 7)

    assert toy.run() == RunOutcome(True, 7)

    assert store.saved == [] and log.records == [] and toy.errors == []
    now, state, record = cast(Any, toy.seen)
    assert now.isoformat() == "2026-09-14T10:00:00-04:00" and state == ToyState(abandoned_streak=1)
    assert record == {"date": "2026-09-14", "run_id": "r1", "abandoned": False, "late": 1}


def test_stage_abandoned() -> None:
    toy, store, log = _toy(streak=1)
    toy.body = StageAbandoned("judge", "no verdict")

    assert toy.run() == RunOutcome(completed=False, abandoned_streak=2)

    assert toy.errors == ["[toy] round abandoned at the judge stage (streak 2): no verdict"]
    assert store.saved == [ToyState(abandoned_streak=2, saved_by="judge")]
    assert log.records == [{"date": "2026-09-14", "run_id": "r1", "abandoned": True, "late": 1, "stage": "judge", "error": "no verdict", "abandoned_streak": 2, "extra": "judge"}]


def test_data_error_is_filings_stage() -> None:
    toy, store, log = _toy(streak=0)
    toy.body = ToyDataError("bad xml")

    assert toy.run() == RunOutcome(False, 1)

    assert log.records[0]["stage"] == "filings" and log.records[0]["error"] == "bad xml"
    assert toy.errors == ["[toy] round abandoned at the filings stage (streak 1): bad xml"]
    assert store.saved == [ToyState(abandoned_streak=1, saved_by="filings")]


@pytest.mark.parametrize("error", [BrokerError("down"), BacktestError("gone")])
@pytest.mark.parametrize(("executing", "stage"), [(False, "broker"), (True, "execution")])
def test_broker_error_execution_vs_broker(error: Exception, executing: bool, stage: str) -> None:
    toy, _, log = _toy()
    toy.body, toy.executing = error, executing

    assert toy.run() == RunOutcome(False, 2)

    assert log.records[0]["stage"] == stage and log.records[0]["error"] == str(error)


def test_an_unrelated_error_propagates_without_a_record() -> None:
    toy, store, log = _toy()
    toy.body = ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        toy.run()

    assert store.saved == [] and log.records == []


def test_record_flag_key_order() -> None:
    class Flagged(Toy):
        RECORD_FLAG: ClassVar[tuple[str, Any, Any]] = ("outcome", None, "abandoned")

    store, log = FakeStore(ToyState()), FakeLog()
    toy = Flagged(store, log)
    toy.body = StageAbandoned("trader", "boom")

    toy.run()

    assert list(log.records[0].items()) == [
        ("date", "2026-09-14"),
        ("run_id", "r1"),
        ("outcome", "abandoned"),  # the flag keeps the slot it had in the record
        ("late", 1),
        ("stage", "trader"),
        ("error", "boom"),
        ("abandoned_streak", 1),
        ("extra", "trader"),
    ]


def test_begin_called_once_per_run() -> None:
    toy, _, _ = _toy()

    toy.run()
    assert toy.begins == 1
    assert toy.calls == ["begin", "run"]
    toy.body = StageAbandoned("x", "y")
    toy.run()
    assert toy.begins == 2
    assert toy.calls == ["begin", "run", "begin", "run"]


def test_consecutive_runs_each_get_begin() -> None:
    class Phased(Toy):
        def _begin(self) -> None:
            super()._begin()
            self.executing = False

        def _run(self, now: datetime, state: ToyState, record: dict[str, Any]) -> RunOutcome:
            if self.begins % 2 == 1:
                self.executing = True
            return super()._run(now, state, record)

    store, log = FakeStore(ToyState()), FakeLog()
    toy = Phased(store, log)
    toy.body = BrokerError("down")

    toy.run()
    toy.run()

    assert [r["stage"] for r in log.records] == ["execution", "broker"]


# --- the pending-trade steps (congress_trades and insider_trades only) -------------------------------------------------


@dataclass(frozen=True)
class PendState:
    abandoned_streak: int = 0
    traded: list[str] = field(default_factory=list)
    pending_trade: PendingTrade | None = None
    last_run: str | None = None


class FakeDesk:
    def __init__(self) -> None:
        self.traded: list[str] = ["ZZZ"]
        self.views = {"o1": OrderView("o1", "AAA", "buy", 10.0, "working", 0.0, True)}
        self.audits: list[Audit] = []

    def orders_view(self) -> dict[str, OrderView]:
        return self.views

    def audit(self) -> Audit:
        return self.audits.pop(0)


class PendToy(ReviewPipeline[PendState]):
    LOG_PREFIX: ClassVar[str] = "[toy] "

    def __init__(self, store: FakeStore, log: FakeLog) -> None:
        self.lines: list[tuple[str, str]] = []
        strategy = SimpleNamespace(
            log_info=lambda m: self.lines.append(("info", m)),
            log_warning=lambda m: self.lines.append(("warning", m)),
            log_error=lambda m: self.lines.append(("error", m)),
        )
        super().__init__(strategy=cast(Any, strategy), agents={}, recorder=None, state=cast(Any, store), log=cast(Any, log))
        self.executing = False

    def _run(self, now: datetime, state: PendState, record: dict[str, Any]) -> RunOutcome:
        raise NotImplementedError

    def _executing(self) -> bool:
        return self.executing

    def _on_abandoned(self, state: PendState, streak: int, stage: str) -> tuple[PendState, dict[str, Any]]:
        raise NotImplementedError


def _pend() -> tuple[PendToy, FakeStore, FakeLog, FakeDesk]:
    store, log = FakeStore(cast(Any, PendState())), FakeLog()
    return PendToy(store, log), store, log, FakeDesk()


_SHORT = Shortfall("AAA", "buy", 1234.5, "AAA is 2.0% under its 5.0% target")
_VIEW = OrderView("o1", "AAA", "buy", 10.0, "working", 0.0, True)


def test_pending_trade_codec() -> None:
    assert valid_pending(None) and valid_pending({"target": {"A": 0.1}, "days": 0})
    assert not valid_pending({"target": {"A": 0.1}}) and not valid_pending({"target": {"A": 0.1}, "days": -1}) and not valid_pending({"target": {"A": 2}, "days": True})
    assert decode_pending(None) is None
    assert decode_pending({"target": {"A": 0.1}, "days": 2}) == PendingTrade(target={"A": 0.1}, days=2)


def test_orders_record() -> None:
    toy, _, _, desk = _pend()
    assert toy._orders_record(cast(Any, desk)) == [{"order_id": "o1", "symbol": "AAA", "side": "buy", "quantity": 10.0, "status": "working", "filled_quantity": 0.0, "checked": True}]


def test_at_target_false_with_shortfalls_and_silent() -> None:
    toy, *_ = _pend()
    assert toy._at_target(Audit(unfilled=[], shortfalls=[_SHORT])) is False
    assert toy.lines == []


def test_at_target_in_flight_line() -> None:
    toy, *_ = _pend()
    assert toy._at_target(Audit(unfilled=[], shortfalls=[], in_flight=["AAA", "BBB"])) is True
    assert toy.lines == [("info", "[trader] the account is at the target counting orders still working (AAA, BBB): no new order, checked again later")]


def test_at_target_already_line() -> None:
    toy, *_ = _pend()
    assert toy._at_target(Audit(unfilled=[], shortfalls=[])) is True
    assert toy.lines == [("info", "[trader] the account is already at the target: no order needed")]


def test_audited_records_and_warns() -> None:
    toy, _, _, desk = _pend()
    desk.audits = [Audit(unfilled=[_VIEW], shortfalls=[_SHORT])]
    record: dict[str, Any] = {}

    after = toy._audited(cast(Any, desk), record)

    assert after is not None and after.shortfalls == [_SHORT]
    assert record == {
        "orders": [{"order_id": "o1", "symbol": "AAA", "side": "buy", "quantity": 10.0, "status": "working", "filled_quantity": 0.0, "checked": True}],
        "shortfalls_after": [{"symbol": "AAA", "kind": "buy", "value": 1234.5, "detail": "AAA is 2.0% under its 5.0% target"}],
    }
    assert toy.lines == [("warning", "[trader] buy 10 AAA is working: it will be checked again"), ("warning", "[trader] off target: AAA is 2.0% under its 5.0% target")]


def test_give_up_pending_continues_below_the_limit() -> None:
    toy, store, log, _ = _pend()
    assert toy._give_up_pending(PendState(), PendingTrade({"A": 0.1}, days=2), {"d": 1}, max_retries=3) is None
    assert store.saved == [] and log.records == [] and toy.lines == []


def test_give_up_pending_at_the_limit() -> None:
    toy, store, log, _ = _pend()
    state = PendState(abandoned_streak=2, pending_trade=PendingTrade({"A": 0.1}, days=3))

    outcome = toy._give_up_pending(state, cast(PendingTrade, state.pending_trade), {"d": 1}, max_retries=3)

    assert outcome == RunOutcome(completed=True, abandoned_streak=2)
    assert toy.lines == [("warning", "[toy] giving up on the unfinished trade after 3 later checks: the target was not reached")]
    assert store.saved == [replace(state, pending_trade=None)]
    assert log.records == [{"d": 1, "outcome": "pending_gave_up", "target": {"A": 0.1}}]


@pytest.mark.parametrize(("complete", "remaining", "outcome"), [(True, None, "pending_complete"), (False, PendingTrade({"A": 0.1}, days=2), "pending_retry")])
def test_finish_pending(complete: bool, remaining: PendingTrade | None, outcome: str) -> None:
    toy, store, log, desk = _pend()
    state = PendState(abandoned_streak=4, traded=["Q"], pending_trade=PendingTrade({"A": 0.1}, days=1))
    audit = Audit(unfilled=[] if complete else [_VIEW], shortfalls=[])

    result = toy._finish_pending(cast(Any, desk), state, cast(PendingTrade, state.pending_trade), audit, {"d": 1}, et(2026, 9, 14, 10))

    assert result == RunOutcome(completed=True, abandoned_streak=0)
    assert store.saved == [PendState(abandoned_streak=0, traded=["ZZZ"], pending_trade=remaining, last_run="2026-09-14")]
    assert log.records == [{"d": 1, "outcome": outcome, "target": {"A": 0.1}}]


@pytest.mark.parametrize(("retry", "days"), [(True, 2), (False, 1)])
def test_abandoned_with_pending(retry: bool, days: int) -> None:
    toy, _, _, desk = _pend()
    state = PendState(traded=["B"], pending_trade=PendingTrade({"A": 0.1}, days=1))

    saved, extra = toy._abandoned_with_pending(cast(Any, desk), state, 3, pending_retry=retry, trading=True)

    assert saved == PendState(abandoned_streak=3, traded=["B", "ZZZ"], pending_trade=PendingTrade({"A": 0.1}, days=days))
    assert extra == {"orders": toy._orders_record(cast(Any, desk))}


def test_abandoned_with_pending_not_trading_and_no_pending() -> None:
    toy, _, _, desk = _pend()
    saved, extra = toy._abandoned_with_pending(cast(Any, desk), PendState(), 1, pending_retry=True, trading=False)
    assert saved == PendState(abandoned_streak=1, traded=["ZZZ"]) and extra == {"orders": []}
