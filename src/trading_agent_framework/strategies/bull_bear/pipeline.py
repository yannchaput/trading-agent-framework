"""One review: data → debate set → researcher (one run per stock) → bull → bear → judge → sizing → rebalance.

The agents decide which stocks win; this module validates (through the `HandoffRecorder`), sizes, remembers
(`StateStore`) and calls the `Rebalancer`. A debater stage that never yields a valid submission abandons the review:
no order at all, forced exits included. One stock's failed research only gives it the note "research unavailable".
Free text from an agent is logged and otherwise ignored. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bull_bear.debate_set import build_debate_set
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Picks
from trading_agent_framework.strategies.bull_bear.market_data import DailyBars, DataUnavailable
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.prompts import BEAR_TASK, BULL_TASK, JUDGE_TASK, UNAVAILABLE_NOTE, researcher_task, retry_prompt
from trading_agent_framework.strategies.bull_bear.sizing import capped_inverse_volatility
from trading_agent_framework.strategies.bull_bear.state import BullBearState, ReviewLog, StateStore
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.scoring import MomentumRow, rank, score_stock
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...


class AgentLookup(Protocol):
    """What the pipeline needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike: ...


class ReviewAbandoned(Exception):
    """A stage could not produce what the review needs; the review ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    completed: bool
    abandoned_streak: int  # abandoned reviews in a row after this one (0 after a completed review)


class ReviewPipeline:
    def __init__(
        self,
        *,
        strategy: Strategy,
        params: BullBearParams,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        review_log: ReviewLog,
        rebalancer: Rebalancer,
        universe: Sequence[str],
        bars: DailyBars,
        sector_of: Callable[[str], str],
        momentum: Mapping[str, Any],
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = review_log
        self._rebalancer = rebalancer
        self._universe = list(universe)
        self._bars = bars
        self._sector_of = sector_of
        self._momentum = momentum  # cross_momentum's CONFIG: the score weights, the skip and the filters
        self._stage = "setup"  # "setup" until the rebalancer starts, then "execution": what a broker failure is attributed to

    def _now(self) -> datetime:
        return self._strategy.clock.now().astimezone(MARKET_TZ)

    def completed_today(self) -> bool:
        """Whether today's review (market date) already completed: a restart then does not run it again."""
        return self._state.load().last_completed_review == self._now().date().isoformat()

    # --- one review --------------------------------------------------------------------------------

    def run(self) -> ReviewOutcome:
        state = self._state.load()
        now = self._now()
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
        self._stage = "setup"
        extra: dict[str, Any] = {}
        try:
            self._review(now, record)
        except ReviewAbandoned as exc:
            stage, error = exc.stage, exc
        except (BrokerError, BacktestError) as exc:
            # A broker or data failure anywhere in the review (ConfigurationError is not one: it propagates).
            stage, error = ("execution" if self._stage == "execution" else "broker"), exc
            if stage == "execution":
                extra["orders"] = [asdict(order) for order in self._rebalancer.placed]  # sent before the failure
        else:
            return ReviewOutcome(completed=True, abandoned_streak=0)
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"[bull_bear] review abandoned at the {stage} stage (streak {streak}): {error}")
        self._state.save(replace(state, abandoned_streak=streak))
        self._log.append({**record, "abandoned": True, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
        return ReviewOutcome(completed=False, abandoned_streak=streak)

    def _review(self, now: datetime, record: dict[str, Any]) -> None:
        params, strategy = self._params, self._strategy
        today = now.date().isoformat()
        moment = now.isoformat()

        # 1. Data and ranking.
        holdings = self._rebalancer.holdings()
        try:
            series = self._bars.load(self._universe, holdings)
        except DataUnavailable as exc:
            raise ReviewAbandoned("data", str(exc)) from exc
        rows: list[MomentumRow] = []
        for symbol in self._universe:
            if symbol in series:
                row = score_stock(symbol, series[symbol].closes, series[symbol].volumes, self._momentum)
                if row is not None:
                    rows.append(row)
        ranked = rank(rows)

        # 2. The debate set, and the holdings code sells without asking the agents.
        debate = build_debate_set(ranked, holdings, shortlist_size=params.shortlist_size, retention_rank=params.retention_rank)
        symbols = debate.symbols
        record.update(
            ranked=len(ranked),
            debate_set=[{"symbol": stock.symbol, "rank": stock.rank, "held": stock.held} for stock in debate.stocks],
            forced_exits=[asdict(exit) for exit in debate.forced_exits],
        )
        strategy.log_info(
            f"[bull_bear] {len(ranked)} stocks ranked; debate: {', '.join(f'{s.symbol} (#{s.rank}{", held" if s.held else ""})' for s in debate.stocks)}; "
            f"forced exits: {', '.join(f'{e.symbol} ({e.reason})' for e in debate.forced_exits) or 'none'}"
        )
        if len(symbols) < params.min_picks:
            raise ReviewAbandoned("data", f"only {len(symbols)} stocks in the debate set, fewer than min_picks ({params.min_picks})")

        # 3. Fact sheet.
        weights = self._rebalancer.current_weights()
        sheets = {stock.symbol: fact_sheet_row(stock, sector=self._sector_of(stock.symbol), weight=weights.get(stock.symbol, 0.0)) for stock in debate.stocks}

        # 4. Researcher: one short run per stock, so one failure costs one note, not the review.
        notes: dict[str, str] = {}
        for symbol in symbols:
            self._recorder.expect_note(symbol)
            try:
                self._run_stage(
                    "researcher",
                    "submit_note",
                    researcher_task(symbol, params.note_max_chars),
                    {"current_datetime": moment, "fact_sheet": sheets[symbol]},
                    tool_budget=params.research_tool_budget,
                )
            except ReviewAbandoned as exc:
                strategy.log_warning(f"[researcher] no note for {symbol}: {exc}")
                notes[symbol] = UNAVAILABLE_NOTE
                continue
            note: Note = self._recorder.submission
            notes[symbol] = note.note
        failed = [symbol for symbol in symbols if notes[symbol] == UNAVAILABLE_NOTE]
        record["notes"] = {"written": len(symbols) - len(failed), "failed": failed}
        evidence = [{"fact_sheet": sheets[symbol], "note": notes[symbol]} for symbol in symbols]

        # 5. Bull, then bear: the same evidence; neither sees the other's case.
        self._recorder.expect_bull(symbols)
        self._run_stage("bull", "submit_bull_case", BULL_TASK, {"current_datetime": moment, "stocks": evidence})
        bull: dict[str, BullCase] = {case.symbol: case for case in self._recorder.submission}
        self._recorder.expect_bear(symbols)
        self._run_stage("bear", "submit_bear_case", BEAR_TASK, {"current_datetime": moment, "stocks": evidence})
        bear: dict[str, BearCase] = {case.symbol: case for case in self._recorder.submission}
        record.update(bull_conviction=dict(Counter(case.conviction for case in bull.values())), bear_risk=dict(Counter(case.risk for case in bear.values())))
        for symbol in symbols:
            strategy.log_info(f"[debate]   {symbol}: bull {bull[symbol].conviction} ({bull[symbol].argument}); bear {bear[symbol].risk}/{bear[symbol].concern} ({bear[symbol].argument})")

        # 6. Judge.
        held = debate.held
        self._recorder.expect_picks(symbols, held=held)
        context = {
            "current_datetime": moment,
            "stocks": [
                {
                    "fact_sheet": sheets[symbol],
                    "note": notes[symbol],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                }
                for symbol in symbols
            ],
            "held": held,
            "forced_exits": [asdict(exit) for exit in debate.forced_exits],
            "constraints": {"min_picks": params.min_picks, "max_picks": params.max_picks},
        }
        self._run_stage("judge", "submit_picks", JUDGE_TASK, context)
        picks: Picks = self._recorder.submission
        strategy.log_info(f"[judge] picks: {', '.join(f'{p.symbol} ({p.reason})' for p in picks.picks)}; drops: {', '.join(f'{d.symbol} ({d.reason})' for d in picks.drops) or 'none'}")

        # 7. Sizing and execution.
        volatilities = {pick.symbol: debate.stock(pick.symbol).row.volatility for pick in picks.picks}
        target_weights = capped_inverse_volatility(volatilities, total=params.investable, min_weight=params.min_weight, max_weight=params.max_weight)
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        self._stage = "execution"
        orders = self._rebalancer.rebalance(target, [exit.symbol for exit in debate.forced_exits])

        # 8. Remember and log.
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, last_picks=[{"symbol": pick.symbol, "reason": pick.reason, "date": today} for pick in picks.picks]))
        self._log.append(
            {
                **record,
                "bull": [asdict(case) for case in bull.values()],
                "bear": [asdict(case) for case in bear.values()],
                "picks": [asdict(pick) for pick in picks.picks],
                "drops": [asdict(drop) for drop in picks.drops],
                "targets": {**target.weights, params.parking_symbol: target.parking_weight},
                "orders": [asdict(order) for order in orders],
            }
        )

    # --- helpers -----------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any], tool_budget: int | None = None) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon."""
        agent = self._agents[agent_name]
        run_id = uuid.uuid4().hex
        prompt, force_tool, error = task, None, ""
        for _ in range(2):
            self._recorder.last_error, error = None, ""  # an error from the first attempt must not be reported for the second
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool, tool_budget=tool_budget)
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
        raise ReviewAbandoned(agent_name, error)
