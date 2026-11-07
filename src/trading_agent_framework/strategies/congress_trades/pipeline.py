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
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from trading_agent_framework.strategies.common.pipeline import AgentLookup, PendingTrade, ReviewPipeline, RunOutcome, StageAbandoned
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.trade_desk import Audit, TradeDesk
from trading_agent_framework.strategies.congress_trades.congress import holdings as holdings_module
from trading_agent_framework.strategies.congress_trades.congress.annual import tier_of
from trading_agent_framework.strategies.congress_trades.congress.holdings import Holding
from trading_agent_framework.strategies.congress_trades.congress.ptr import FilingRef
from trading_agent_framework.strategies.congress_trades.congress.source import KnownFilings
from trading_agent_framework.strategies.congress_trades.handoff import HandoffRecorder, HoldingSubmission, TargetPosition
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.strategies.congress_trades.prompts import PORTFOLIO_TASK, RESEARCHER_TASK, TRADER_TASK, retry_prompt
from trading_agent_framework.strategies.congress_trades.state import CongressState, StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, CongressDataError

if TYPE_CHECKING:
    from collections.abc import Collection

    from trading_agent_framework.core.strategy import Strategy

AGENT_RESEARCHER = "researcher"
AGENT_PORTFOLIO = "portfolio_manager"
AGENT_TRADER = "trader"


class SourceLike(Protocol):
    """The filings source the pipeline reads, as a protocol so tests can fake it."""

    def known(self, as_of: datetime) -> KnownFilings:
        """The congressional filings known at `as_of`."""
        ...

    def new_since(self, known: KnownFilings, processed: Collection[str]) -> list[FilingRef]:
        """Filings in `known` that are not among the `processed` ids."""
        ...


class CongressPipeline(ReviewPipeline[CongressState]):
    """Congress-trades review pipeline: find new disclosures, let the agent review them and trade the resulting portfolio."""

    LOG_PREFIX: ClassVar[str] = "[congress_trades] "
    WHAT: ClassVar[str] = "run"
    RECORD_FLAG: ClassVar[tuple[str, Any, Any]] = ("outcome", None, "abandoned")
    DATA_ERRORS: ClassVar[tuple[type[Exception], ...]] = (CongressDataError,)

    def __init__(
        self,
        *,
        strategy: Strategy,
        params: CongressParams,
        source: SourceLike,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        log: RunLog,
        desk: TradeDesk,
    ) -> None:
        """Wire the pipeline to the strategy, its parameters, the filings source and the agents."""
        super().__init__(strategy=strategy, agents=agents, recorder=recorder, state=state, log=log)
        self._params = params
        self._source = source
        self._desk = desk
        self._phase = "filings"  # where a failure is attributed: filings, researcher, portfolio, trading
        self._pending_retry = False

    # --- one daily check ---------------------------------------------------------------------------

    def _begin(self) -> None:
        """Reset the per-run phase to filings and clear any pending retry."""
        self._phase, self._pending_retry = "filings", False

    def _executing(self) -> bool:
        """Whether the run has reached the trading phase."""
        return self._phase == "trading"

    def _on_abandoned(self, state: CongressState, streak: int, stage: str) -> tuple[CongressState, dict[str, Any]]:
        """Build the state and record for a run abandoned after repeated failures, keeping any pending trade for retry."""
        return self._abandoned_with_pending(self._desk, state, streak, pending_retry=self._pending_retry, trading=self._executing())

    def _run(self, now: datetime, state: CongressState, record: dict[str, Any]) -> RunOutcome:
        """Run one pipeline pass: look for new filings and review them."""
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
        """Retry the trade left pending by an earlier abandoned run."""
        strategy, params = self._strategy, self._params
        if (gave_up := self._give_up_pending(state, pending, record, max_retries=params.max_trade_retries)) is not None:
            return gave_up
        strategy.log_info(f"[trader] checking the unfinished trade again (check {pending.days + 1} of {params.max_trade_retries})")
        self._phase, self._pending_retry = "trading", True
        audit = self._trade(pending.target, state, record)
        return self._finish_pending(self._desk, state, pending, audit, record, now)

    # --- a new filing: research, portfolio, trading ------------------------------------------------------

    def _full_run(self, now: datetime, state: CongressState, known: KnownFilings, new: list[FilingRef], record: dict[str, Any]) -> RunOutcome:
        """Review the new filings with the agent and execute the resulting target portfolio."""
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
        if self._at_target(before):
            return before
        self._recorder.expect_report(desk.orders_view)
        context = {
            "current_datetime": strategy.clock.now().astimezone(MARKET_TZ).isoformat(),
            "target": [{"ticker": ticker, "weight": weight} for ticker, weight in sorted(target.items())],
            "shortfalls": [{"symbol": s.symbol, "kind": s.kind, "dollars": round(s.value), "detail": s.detail} for s in before.shortfalls],
        }
        try:
            self._run_stage(AGENT_TRADER, "submit_trade_report", TRADER_TASK, context)
        except StageAbandoned as exc:
            if not desk.orders:
                raise  # nothing was sent: abandon the whole run
            # Orders went out but were never reported: do not discard the research; they are checked again on later days.
            strategy.log_error(f"[trader] no valid trade report, but {len(desk.orders)} order(s) were sent and will be checked again later: {exc}")
        return self._audited(desk, record)

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
        raise StageAbandoned(agent_name, error)

    def _weights(self, holdings: list[Holding]) -> dict[str, Decimal]:
        """Baseline portfolio weights for the holdings, from their tiers."""
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


def _holding_context(holding: Holding) -> dict[str, Any]:
    """A holding as the plain context handed to the agent."""
    return {"ticker": holding.ticker, "tier": holding.tier, "value_low": int(holding.value_low), "value_high": int(holding.value_high), "filings": list(holding.sources)}


def _holding_record(holding: Holding, weight: Decimal | None) -> dict[str, Any]:
    """A holding as a run-log record, with its baseline weight."""
    return {**_holding_context(holding), "baseline_weight": float(weight) if weight is not None else None}
