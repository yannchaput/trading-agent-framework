"""The skeleton the four agent pipelines share: one run, and what happens when it is abandoned.

A pipeline (`AckmanPipeline`, `BullBearPipeline`, `CongressPipeline`, `InsiderPipeline`) runs its stages through agents; a stage that
never yields a valid submission, a data failure or a broker failure abandons the run. `ReviewPipeline.run` is the template: it loads
the state, builds the record, calls the body (`_run`) and, on an abandonment, bumps the streak, logs one error line, saves the state
the subclass chose and appends the abandoned record. Each pipeline keeps its own `_run_stage`, `retry_prompt` and body.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from trading_agent_framework.strategies.common.json_state import is_int, valid_weights
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.agents.results import AgentRunResult
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.common.json_state import JsonStateStore
    from trading_agent_framework.strategies.common.run_log import RunLog
    from trading_agent_framework.strategies.common.trade_desk import Audit, TradeDesk


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...


class AgentLookup(Protocol):
    """What a pipeline needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike: ...


class StageAbandoned(Exception):
    """A stage could not produce what the run needs; the run ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class RunOutcome:
    completed: bool
    abandoned_streak: int  # abandoned runs in a row after this one (0 after a completed run)


@dataclass(frozen=True, slots=True)
class PendingTrade:
    """A target the trading stage could not reach: later ticks re-run only that stage against it, `days` ticks used so far."""

    target: dict[str, float]  # ticker -> weight, as submitted by the portfolio agent
    days: int = 0


def valid_pending(value: Any) -> bool:
    return value is None or (isinstance(value, dict) and set(value) == {"target", "days"} and valid_weights(value["target"]) and is_int(value["days"]) and value["days"] >= 0)


def decode_pending(raw: dict[str, Any] | None) -> PendingTrade | None:
    """The `PendingTrade` of a state that passed `valid_pending`."""
    return None if raw is None else PendingTrade(target=dict(raw["target"]), days=raw["days"])


class Streaked(Protocol):
    """A pipeline's state: it counts the abandoned runs in a row."""

    @property
    def abandoned_streak(self) -> int: ...


class ReviewPipeline[S: Streaked](ABC):
    """The template `run`. A subclass sets the class constants, implements `_run`, `_executing` and `_on_abandoned`, and may override `_begin`."""

    LOG_PREFIX: ClassVar[str] = ""  # what the abandoned error line starts with, after the logger's own tag
    WHAT: ClassVar[str] = "review"  # "review" or "run": what that line calls one pass
    RECORD_FLAG: ClassVar[tuple[str, Any, Any]] = ("abandoned", False, True)  # the record key that tells an abandoned line, with its value before and after
    DATA_ERRORS: ClassVar[tuple[type[Exception], ...]] = ()  # data failures that abandon the run at the "filings" stage

    def __init__(self, *, strategy: Strategy, agents: AgentLookup, recorder: Any, state: JsonStateStore[S], log: RunLog) -> None:
        self._strategy = strategy
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = log

    def run(self) -> RunOutcome:
        state = self._state.load()
        now = self._strategy.clock.now().astimezone(MARKET_TZ)
        key, initial, abandoned = self.RECORD_FLAG
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, key: initial}
        self._begin()
        try:
            return self._run(now, state, record)
        except StageAbandoned as exc:
            stage, error = exc.stage, exc
        except self.DATA_ERRORS as exc:
            stage, error = "filings", exc
        except (BrokerError, BacktestError) as exc:
            # A broker or data failure anywhere in the run (ConfigurationError is not one: it propagates).
            stage, error = ("execution" if self._executing() else "broker"), exc
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"{self.LOG_PREFIX}{self.WHAT} abandoned at the {stage} stage (streak {streak}): {error}")
        saved, extra = self._on_abandoned(state, streak, stage)
        self._state.save(saved)
        self._log.append({**record, key: abandoned, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
        return RunOutcome(completed=False, abandoned_streak=streak)

    def _begin(self) -> None:  # noqa: B027 -- an optional hook, not abstract
        """Reset what a failure is attributed to, at the start of each run. Nothing by default."""

    @abstractmethod
    def _run(self, now: datetime, state: S, record: dict[str, Any]) -> RunOutcome:
        """The run itself; `record` is the line it fills in."""

    @abstractmethod
    def _executing(self) -> bool:
        """Whether a broker failure now happened while orders were going out (the `execution` stage) rather than before (`broker`)."""

    @abstractmethod
    def _on_abandoned(self, state: S, streak: int, stage: str) -> tuple[S, dict[str, Any]]:
        """The state to save for an abandoned run, and the fields to add to its record."""

    # --- pending-trade steps: concrete, called only by congress_trades and insider_trades ---------------------------------

    def _orders_record(self, desk: TradeDesk) -> list[dict[str, Any]]:
        return [asdict(view) for view in desk.orders_view().values()]

    def _at_target(self, before: Audit) -> bool:
        """True when the audit before the trading stage shows nothing off target; logs why no order is needed."""
        if before.shortfalls:
            return False
        if before.in_flight:
            self._strategy.log_info(f"[trader] the account is at the target counting orders still working ({', '.join(before.in_flight)}): no new order, checked again later")
        else:
            self._strategy.log_info("[trader] the account is already at the target: no order needed")
        return True

    def _audited(self, desk: TradeDesk, record: dict[str, Any]) -> Audit:
        """The desk's audit after the trading stage, with the orders and what is still off target recorded and logged."""
        strategy = self._strategy
        after = desk.audit()
        record["orders"] = self._orders_record(desk)
        record["shortfalls_after"] = [asdict(s) for s in after.shortfalls]
        for view in after.unfilled:
            strategy.log_warning(f"[trader] {view.side} {view.quantity:g} {view.symbol} is {view.status}: it will be checked again")
        for shortfall in after.shortfalls:
            strategy.log_warning(f"[trader] off target: {shortfall.detail}")
        return after

    def _give_up_pending(self, state: S, pending: PendingTrade, record: dict[str, Any], *, max_retries: int) -> RunOutcome | None:
        """Drop a pending trade that used up its retries (None when it still has some)."""
        if pending.days < max_retries:
            return None
        self._strategy.log_warning(f"{self.LOG_PREFIX}giving up on the unfinished trade after {pending.days} later checks: the target was not reached")
        self._state.save(replace(state, pending_trade=None))  # type: ignore[type-var]  # S is a dataclass with a pending_trade field
        self._log.append({**record, "outcome": "pending_gave_up", "target": pending.target})
        return RunOutcome(completed=True, abandoned_streak=state.abandoned_streak)

    def _finish_pending(self, desk: TradeDesk, state: S, pending: PendingTrade, audit: Audit, record: dict[str, Any], now: datetime) -> RunOutcome:
        """Save the state after a re-check of the pending trade: settled, or kept one more day."""
        remaining = None if audit.complete else PendingTrade(target=pending.target, days=pending.days + 1)
        self._state.save(replace(state, traded=desk.traded, pending_trade=remaining, abandoned_streak=0, last_run=now.date().isoformat()))  # type: ignore[type-var]
        self._log.append({**record, "outcome": "pending_complete" if audit.complete else "pending_retry", "target": pending.target})
        return RunOutcome(completed=True, abandoned_streak=0)

    def _abandoned_with_pending(self, desk: TradeDesk, state: S, streak: int, *, pending_retry: bool, trading: bool) -> tuple[S, dict[str, Any]]:
        """`_on_abandoned` for a pipeline with a pending trade: the streak, the retry the failed attempt used up, the tickers ordered."""
        pending = state.pending_trade  # type: ignore[attr-defined]
        if pending_retry and pending is not None:
            pending = replace(pending, days=pending.days + 1)  # an attempt that fails still uses up one of the retries
        traded = sorted({*state.traded, *desk.traded})  # type: ignore[attr-defined]
        orders = self._orders_record(desk) if trading else []
        return replace(state, abandoned_streak=streak, pending_trade=pending, traded=traded), {"orders": orders}  # type: ignore[type-var]
