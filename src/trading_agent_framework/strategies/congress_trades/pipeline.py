"""One daily check: is there a new filing? If so research, portfolio, trading; if a trade is left over, re-check only the trading.

The agents decide; this module validates (through the `HandoffRecorder`), remembers (`StateStore`) and calls the `TradeDesk`.
The daily rule (see the design spec): the strategy ticks every day but acts only when the member has filed something since the
last COMPLETED run. Code decides that, not an agent: with no new filing and no pending trade no agent runs and no order is
sent. A trade the trading stage could not finish (orders still working, or a position off its target) is kept in the state and
re-checked on the following days by running only the trading stage against the stored target, at most `max_trade_retries`
times; a new filing replaces it. A stage that never yields a valid submission abandons the run: nothing is traded, the
filings stay "new" and the streak goes up.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.congress_trades.congress import holdings as holdings_module
from trading_agent_framework.strategies.congress_trades.congress.annual import tier_of
from trading_agent_framework.strategies.congress_trades.congress.holdings import Holding
from trading_agent_framework.strategies.congress_trades.congress.ptr import FilingRef
from trading_agent_framework.strategies.congress_trades.congress.source import KnownFilings
from trading_agent_framework.strategies.congress_trades.desk import Audit, TradeDesk
from trading_agent_framework.strategies.congress_trades.handoff import HandoffRecorder, HoldingSubmission, TargetPosition
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.strategies.congress_trades.prompts import PORTFOLIO_TASK, RESEARCHER_TASK, TRADER_TASK, retry_prompt
from trading_agent_framework.strategies.congress_trades.state import CongressState, PendingTrade, RunLog, StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, CongressDataError

if TYPE_CHECKING:
    from collections.abc import Collection

    from trading_agent_framework.core.strategy import Strategy

AGENT_RESEARCHER = "researcher"
AGENT_PORTFOLIO = "portfolio_manager"
AGENT_TRADER = "trader"


class SourceLike(Protocol):
    def known(self, as_of: datetime) -> KnownFilings: ...

    def new_since(self, known: KnownFilings, processed: Collection[str]) -> list[FilingRef]: ...


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...


class AgentLookup(Protocol):
    """What the pipeline needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike: ...


class RunAbandoned(Exception):
    """A stage could not produce what the run needs; the run ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class RunOutcome:
    completed: bool
    abandoned_streak: int  # abandoned runs in a row after this one (unchanged by a quiet day, 0 after a completed run)


class CongressPipeline:
    def __init__(
        self,
        *,
        strategy: Strategy,
        params: CongressParams,
        source: SourceLike,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        run_log: RunLog,
        desk: TradeDesk,
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._source = source
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = run_log
        self._desk = desk
        self._phase = "filings"  # where a failure is attributed: filings, researcher, portfolio, trading
        self._pending_retry = False

    # --- one daily check ---------------------------------------------------------------------------

    def run(self) -> RunOutcome:
        state = self._state.load()
        now = self._strategy.clock.now().astimezone(MARKET_TZ)
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "outcome": None}
        self._phase, self._pending_retry = "filings", False
        try:
            return self._run(now, state, record)
        except RunAbandoned as exc:
            stage, error = exc.stage, exc
        except CongressDataError as exc:
            stage, error = "filings", exc
        except (BrokerError, BacktestError) as exc:
            stage, error = ("execution" if self._phase == "trading" else "broker"), exc
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"[congress_trades] run abandoned at the {stage} stage (streak {streak}): {error}")
        pending = state.pending_trade
        if self._pending_retry and pending is not None:
            pending = replace(pending, days=pending.days + 1)  # an attempt that fails still uses up one of the retries
        traded = sorted({*state.traded, *self._desk.traded})
        self._state.save(replace(state, abandoned_streak=streak, pending_trade=pending, traded=traded))
        orders = self._orders_record() if self._phase == "trading" else []
        self._log.append({**record, "outcome": "abandoned", "stage": stage, "error": str(error), "abandoned_streak": streak, "orders": orders})
        return RunOutcome(completed=False, abandoned_streak=streak)

    def _run(self, now: datetime, state: CongressState, record: dict[str, Any]) -> RunOutcome:
        strategy = self._strategy
        known = self._source.known(strategy.clock.now())
        new = self._source.new_since(known, state.processed)
        record["new_filings"] = [{"doc_id": ref.doc_id, "kind": ref.kind, "filed": ref.filed.isoformat()} for ref in new]
        if new:
            return self._full_run(now, state, known, new, record)
        if state.pending_trade is None:
            strategy.log_info("[researcher] nothing new: no filing since the last run, so nothing to do")
            self._log.append({**record, "outcome": "nothing_new"})
            return RunOutcome(completed=True, abandoned_streak=state.abandoned_streak)
        return self._retry_pending(now, state, state.pending_trade, record)

    # --- a pending trade: only the trading stage ---------------------------------------------------------

    def _retry_pending(self, now: datetime, state: CongressState, pending: PendingTrade, record: dict[str, Any]) -> RunOutcome:
        strategy, params = self._strategy, self._params
        if pending.days >= params.max_trade_retries:
            strategy.log_warning(f"[congress_trades] giving up on the unfinished trade after {pending.days} later checks: the target was not reached")
            self._state.save(replace(state, pending_trade=None))
            self._log.append({**record, "outcome": "pending_gave_up", "target": pending.target})
            return RunOutcome(completed=True, abandoned_streak=state.abandoned_streak)
        strategy.log_info(f"[trader] checking the unfinished trade again (check {pending.days + 1} of {params.max_trade_retries})")
        self._phase, self._pending_retry = "trading", True
        audit = self._trade(pending.target, state, record)
        remaining = None if audit.complete else PendingTrade(target=pending.target, days=pending.days + 1)
        self._state.save(replace(state, traded=self._desk.traded, pending_trade=remaining, abandoned_streak=0, last_run=now.date().isoformat()))
        self._log.append({**record, "outcome": "pending_complete" if audit.complete else "pending_retry", "target": pending.target})
        return RunOutcome(completed=True, abandoned_streak=0)

    # --- a new filing: research, portfolio, trading ------------------------------------------------------

    def _full_run(self, now: datetime, state: CongressState, known: KnownFilings, new: list[FilingRef], record: dict[str, Any]) -> RunOutcome:
        strategy, params = self._strategy, self._params
        current = holdings_module.reconstruct(known.assets, known.transactions, period_end=known.period_end)
        baseline_weights = self._weights(current)
        strategy.log_info(
            f"[researcher] {len(new)} new filing(s): {', '.join(f'{r.doc_id} ({r.kind}, {r.filed})' for r in new)}; "
            f"baseline from report {known.annual_ref.doc_id} (period end {known.period_end}): {len(current)} holdings"
        )
        record["baseline"] = [_holding_record(h, baseline_weights.get(h.ticker)) for h in current]

        # 1. Research agent: what she owns today.
        self._phase = "researcher"
        tickers = {a.ticker for a in known.assets} | {t.ticker for t in known.transactions}
        self._recorder.expect_holdings(tickers, {h.ticker: (h.value_low, h.value_high) for h in current})
        context = {
            "current_datetime": now.isoformat(),
            "new_filings": [{"doc_id": r.doc_id, "kind": r.kind, "filed": r.filed.isoformat()} for r in new],
            "base_report": {"doc_id": known.annual_ref.doc_id, "filed": known.annual_ref.filed.isoformat(), "period_end": known.period_end.isoformat()},
            "baseline": [_holding_context(h) for h in sorted(current, key=lambda h: (h.tier, h.midpoint), reverse=True)],
            "unparsed_filings": known.unparsed_filings,
            "skipped_non_stock": known.skipped_non_stock,
            "constraints": {"max_holdings": params.max_holdings},
        }
        self._run_stage(AGENT_RESEARCHER, "submit_holdings", RESEARCHER_TASK, context)
        submitted: list[HoldingSubmission] = self._recorder.submission
        final = self._final_holdings(submitted, current)
        record["holdings"] = [asdict(s) for s in submitted]
        for entry in submitted:
            note = f" ({entry.reason})" if entry.reason else ""
            strategy.log_info(f"[researcher]   {entry.ticker} {'dropped' if entry.dropped else f'${entry.value_low:,.0f} - ${entry.value_high:,.0f}'}{note}")

        # 2. Portfolio agent: the target mix.
        self._phase = "portfolio"
        weights = self._weights(final)
        positions: list[TargetPosition] = []
        if final:
            self._recorder.expect_target({h.ticker: h.tier for h in final})
            context = {
                "current_datetime": now.isoformat(),
                "holdings": [
                    {**_holding_context(h), "baseline_weight": float(weights[h.ticker]) if h.ticker in weights else None} for h in sorted(final, key=lambda h: (h.tier, h.midpoint), reverse=True)
                ],
                "previous_target": state.target,
                "constraints": {
                    "min_weight": params.min_weight,
                    "max_weight": params.max_position_weight,
                    "max_total_weight": params.max_total_weight,
                    "max_positions": params.max_positions,
                    "unallocated_money": "stays in cash",
                },
            }
            self._run_stage(AGENT_PORTFOLIO, "submit_target", PORTFOLIO_TASK, context)
            positions = self._recorder.submission
        else:
            strategy.log_info("[portfolio_manager] no holdings: the target is empty")
        target = {p.ticker: p.weight for p in positions if p.weight > 0}
        record["target"] = [asdict(p) for p in positions]
        strategy.log_info(f"[portfolio_manager] target: {len(target)} positions, {sum(target.values()):.1%} invested")
        for position in positions:
            strategy.log_info(f"[portfolio_manager]   {position.ticker} {position.weight:.1%}: {position.reason}")

        # 3. Trading agent: the orders, and the check that they filled.
        self._phase, self._pending_retry = "trading", False
        audit = self._trade(target, state, record)

        pending = None if audit.complete else PendingTrade(target=target, days=0)
        self._state.save(
            CongressState(
                processed=[ref.doc_id for ref in known.refs],
                holdings={h.ticker: {"tier": h.tier, "value_low": int(h.value_low), "value_high": int(h.value_high)} for h in final},
                target=target,
                traded=self._desk.traded,
                pending_trade=pending,
                abandoned_streak=0,
                last_run=now.date().isoformat(),
            )
        )
        self._log.append({**record, "outcome": "completed" if audit.complete else "completed_pending"})
        return RunOutcome(completed=True, abandoned_streak=0)

    # --- the trading stage ---------------------------------------------------------------------------------

    def _trade(self, target: Mapping[str, float], state: CongressState, record: dict[str, Any]) -> Audit:
        """Run the trading agent against `target`; returns the audit afterwards (complete = nothing unfilled, nothing off target)."""
        strategy, desk = self._strategy, self._desk
        desk.begin_run(target, state.traded)
        before = desk.audit()
        record["shortfalls_before"] = [asdict(s) for s in before.shortfalls]
        if not before.shortfalls:
            if before.in_flight:
                strategy.log_info(f"[trader] the account is at the target counting orders still working ({', '.join(before.in_flight)}): no new order, checked again later")
            else:
                strategy.log_info("[trader] the account is already at the target: no order needed")
            return before
        self._recorder.expect_report(desk.orders_view)
        context = {
            "current_datetime": strategy.clock.now().astimezone(MARKET_TZ).isoformat(),
            "target": [{"ticker": ticker, "weight": weight} for ticker, weight in sorted(target.items())],
            "shortfalls": [{"symbol": s.symbol, "kind": s.kind, "dollars": round(s.value), "detail": s.detail} for s in before.shortfalls],
        }
        try:
            self._run_stage(AGENT_TRADER, "submit_trade_report", TRADER_TASK, context)
        except RunAbandoned as exc:
            if not desk.orders:
                raise  # nothing was sent: abandon the whole run
            # Orders went out but were never reported: do not discard the research; they are checked again on later days.
            strategy.log_error(f"[trader] no valid trade report, but {len(desk.orders)} order(s) were sent and will be checked again later: {exc}")
        after = desk.audit()
        record["orders"] = self._orders_record()
        record["shortfalls_after"] = [asdict(s) for s in after.shortfalls]
        for view in after.unfilled:
            strategy.log_warning(f"[trader] {view.side} {view.quantity:g} {view.symbol} is {view.status}: it will be checked again")
        for shortfall in after.shortfalls:
            strategy.log_warning(f"[trader] off target: {shortfall.detail}")
        return after

    # --- helpers -------------------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any]) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon."""
        agent = self._agents[agent_name]
        run_id = uuid.uuid4().hex
        prompt, force_tool, error = task, None, ""
        for _ in range(2):
            self._recorder.last_error, error = None, ""  # an error from the first attempt must not be reported for the second
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool)
                self._strategy.log_info(f"[{agent_name}] {result.output}")
                for i, tool_call in enumerate(result.tool_calls):
                    self._strategy.log_debug(f"[{agent_name}] tool_call_{i}: {tool_call}")
            except AgentError as exc:
                error = str(exc)
                self._strategy.log_error(f"[{agent_name}] run failed: {exc}")
            if self._recorder.submitted:
                return
            error = self._recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = retry_prompt(tool, error), tool
        raise RunAbandoned(agent_name, error)

    def _weights(self, holdings: list[Holding]) -> dict[str, Decimal]:
        params = self._params
        return holdings_module.baseline_weights(
            holdings,
            max_total=Decimal(str(params.max_total_weight)),
            max_position=Decimal(str(params.max_position_weight)),
            min_weight=Decimal(str(params.min_weight)),
            max_positions=params.max_positions,
            tier_base=Decimal(str(params.tier_weight_base)),
        )

    @staticmethod
    def _final_holdings(submitted: list[HoldingSubmission], baseline: list[Holding]) -> list[Holding]:
        """The kept holdings of the research agent's submission, with the tier recomputed from the value it gave."""
        known = {h.ticker: h for h in baseline}
        final = []
        for entry in submitted:
            if entry.dropped:
                continue
            old = known.get(entry.ticker)
            mid = (entry.value_low + entry.value_high) / 2
            final.append(
                Holding(
                    ticker=entry.ticker,
                    asset_name=old.asset_name if old else entry.ticker,
                    value_low=entry.value_low,
                    value_high=entry.value_high,
                    tier=tier_of(mid),
                    sources=old.sources if old else (),
                )
            )
        return sorted(final, key=lambda h: h.ticker)

    def _orders_record(self) -> list[dict[str, Any]]:
        return [asdict(view) for view in self._desk.orders_view().values()]


def _holding_context(holding: Holding) -> dict[str, Any]:
    return {"ticker": holding.ticker, "tier": holding.tier, "value_low": int(holding.value_low), "value_high": int(holding.value_high), "filings": list(holding.sources)}


def _holding_record(holding: Holding, weight: Decimal | None) -> dict[str, Any]:
    return {**_holding_context(holding), "baseline_weight": float(weight) if weight is not None else None}
