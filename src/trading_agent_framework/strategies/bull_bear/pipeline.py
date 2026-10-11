"""One review: data → candidates → researcher (one run per stock) → bull → bear → judge → book → sizing → rebalance.

The judge puts every candidate in or out; code validates (through the `HandoffRecorder`), builds the book from the
stocks put in (rank order), sizes it, remembers (`StateStore`) and calls the `Rebalancer`. A debater stage that
never yields a valid submission abandons the review: no order at all, forced exits included. One stock's failed
research only gives it the note "research unavailable". Free text from an agent is logged and otherwise ignored.
See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md and
docs/superpowers/specs/2026-10-11-bull-bear-debate-realignment-design.md.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bull_bear.book import build_book
from trading_agent_framework.strategies.bull_bear.cooldown import advance_cooldowns
from trading_agent_framework.strategies.bull_bear.debate_set import build_debate_set
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.bull_bear.feedback import recent_outcomes
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdicts
from trading_agent_framework.strategies.bull_bear.market_data import DailyBars, DataUnavailable
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.prompts import UNAVAILABLE_NOTE, bear_task, bull_task, judge_task, researcher_task, retry_prompt
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
        # max_volatility gates new entries only: holdings are scored with a copy that lifts it (CONFIG itself is never changed)
        self._holding_momentum = {**momentum, "max_volatility": math.inf}
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
            self._review(now, record, state)
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

    def _review(self, now: datetime, record: dict[str, Any], state: BullBearState) -> None:
        params, strategy = self._params, self._strategy
        today = now.date().isoformat()
        moment = now.isoformat()

        # 1. Data and ranking.
        holdings = self._rebalancer.holdings()
        try:
            series = self._bars.load(self._universe, holdings)
        except DataUnavailable as exc:
            raise ReviewAbandoned("data", str(exc)) from exc
        held_now = set(holdings)
        max_volatility = self._momentum["max_volatility"]
        rows: list[MomentumRow] = []
        for symbol in self._universe:
            if symbol not in series:
                continue
            closes, volumes = series[symbol].closes, series[symbol].volumes
            if symbol in held_now:
                # A holding: every filter but max_volatility. A volatility spike after a crash must not force it out
                # near its low; at or below the cap this is the very row the full filters would give.
                row = score_stock(symbol, closes, volumes, self._holding_momentum)
                if row is not None and row.volatility > max_volatility:
                    strategy.log_info(f"[bull_bear] {symbol} is held at a volatility of {row.volatility:.0%}, above max_volatility ({max_volatility:.0%}): kept in the ranking")
            else:
                # A possible new entry: cross_momentum's filters in full, max_volatility included.
                row = score_stock(symbol, closes, volumes, self._momentum)
            if row is not None:
                rows.append(row)
        ranked = rank(rows)
        closes_now = {symbol: data.closes[-1] for symbol, data in series.items() if data.closes}

        # 2. The debate set, and the holdings code sells without asking the agents.
        debate = build_debate_set(ranked, holdings, candidate_count=params.candidate_count, retention_rank=params.retention_rank, excluded=state.cooldowns)
        symbols = debate.symbols
        record.update(
            ranked=len(ranked),
            debate_set=[{"symbol": stock.symbol, "rank": stock.rank, "held": stock.held} for stock in debate.stocks],
            forced_exits=[asdict(exit) for exit in debate.forced_exits],
            cooldowns_active=dict(state.cooldowns),
        )
        strategy.log_info(
            f"[bull_bear] {len(ranked)} stocks ranked; debate: {', '.join(f'{s.symbol} (#{s.rank}{", held" if s.held else ""})' for s in debate.stocks)}; "
            f"forced exits: {', '.join(f'{e.symbol} ({e.reason})' for e in debate.forced_exits) or 'none'}"
        )
        if not symbols:
            raise ReviewAbandoned("data", "no stock to debate")

        # 3. Fact sheet.
        sheets = {stock.symbol: fact_sheet_row(stock, sector=self._sector_of(stock.symbol)) for stock in debate.stocks}

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
        limits = {"argument_max_chars": params.argument_max_chars}
        self._recorder.expect_bull(symbols)
        self._run_stage("bull", "submit_bull_case", bull_task(params.argument_max_chars), {"current_datetime": moment, "stocks": evidence, "constraints": limits})
        bull: dict[str, BullCase] = {case.symbol: case for case in self._recorder.submission}
        self._recorder.expect_bear(symbols)
        self._run_stage("bear", "submit_bear_case", bear_task(params.argument_max_chars), {"current_datetime": moment, "stocks": evidence, "constraints": limits})
        bear: dict[str, BearCase] = {case.symbol: case for case in self._recorder.submission}
        record.update(bull_conviction=dict(Counter(case.conviction for case in bull.values())), bear_risk=dict(Counter(case.risk for case in bear.values())))
        for symbol in symbols:
            strategy.log_info(f"[debate]   {symbol}: bull {bull[symbol].conviction} ({bull[symbol].argument}); bear {bear[symbol].risk}/{bear[symbol].concern} ({bear[symbol].argument})")

        # 6. Judge: in or out on every candidate, on merit (no agent is told which stocks are held), starting from its own previous verdict on a stock when it gave one.
        previous = {symbol: state.verdicts[symbol] for symbol in symbols if symbol in state.verdicts}
        self._recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})
        recent = recent_outcomes(state.history, closes_now)
        context = {
            "current_datetime": moment,
            "stocks": [
                {
                    "fact_sheet": sheets[symbol],
                    "note": notes[symbol],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                    **({"previous_verdict": previous[symbol]} if symbol in previous else {}),
                }
                for symbol in symbols
            ],
            "recent_verdicts": recent,
            "constraints": {"max_in": params.max_book_size, "reason_max_chars": params.reason_max_chars},
        }
        self._run_stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context)
        verdicts: Verdicts = self._recorder.submission
        outs = ", ".join(f"{verdict.symbol} ({verdict.reason})" for verdict in verdicts.verdicts if verdict.verdict == "out")
        strategy.log_info(f"[judge] in: {', '.join(verdicts.chosen) or 'none'}; out: {outs or 'none'}")

        # 7. The book: every stock the judge put in, in rank order.
        book = build_book(debate, verdicts.chosen)
        strategy.log_info(f"[bull_bear] book: {', '.join(book) or 'empty, everything parked'}")

        # 8. Sizing and execution: inverse volatility within the weight bounds; below 5 stocks the rest is parked.
        volatilities = {symbol: debate.stock(symbol).row.volatility for symbol in book}
        total = min(params.investable, len(book) * params.max_weight)
        target_weights = capped_inverse_volatility(volatilities, total=total, min_weight=params.min_weight, max_weight=params.max_weight) if book else {}
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        # A held stock the judge put out starts a cooldown: it stays out of the candidates for the next reviews.
        sold = [stock.symbol for stock in debate.stocks if stock.held and stock.symbol not in set(verdicts.chosen)]
        # The decision goes into the record before the first order: an abandonment during execution still says why.
        record.update(
            verdicts=[{**asdict(verdict), "previous_verdict": previous.get(verdict.symbol, {}).get("verdict")} for verdict in verdicts.verdicts],
            verdict_counts=dict(Counter(verdict.verdict for verdict in verdicts.verdicts)),
            book=list(book),
            cooldowns_started=sold,
            recent_verdicts=recent,
            targets={**target.weights, params.parking_symbol: target.parking_weight},
        )
        # A forced exit is sold by code, never debated: say so at warning level, before its sell order goes out.
        for forced in debate.forced_exits:
            strategy.log_warning(f"[bull_bear] forced exit: {forced.symbol} ({forced.reason}) is sold by code, outside the debate")
        self._stage = "execution"
        orders = self._rebalancer.rebalance(target, [forced.symbol for forced in debate.forced_exits])

        # 9. Remember and log.
        decided = {verdict.symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "date": today} for verdict in verdicts.verdicts}
        cooldowns = advance_cooldowns(state.cooldowns, started=sold, length=params.reentry_cooldown_reviews)
        review = {"date": today, "verdicts": {verdict.symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "close": closes_now[verdict.symbol]} for verdict in verdicts.verdicts}}
        history = [*state.history, review][-params.feedback_reviews :]
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, verdicts={**state.verdicts, **decided}, cooldowns=cooldowns, history=history))
        self._log.append(
            {
                **record,
                "bull": [asdict(case) for case in bull.values()],
                "bear": [asdict(case) for case in bear.values()],
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
                for line in _submission_lines(agent_name, self._recorder.submission):
                    self._strategy.log_info(line)
                return
            if self._recorder.last_error:
                self._strategy.log_warning(f"[{agent_name}] submission rejected: {self._recorder.last_error}")
            elif not error:  # a failed run was already logged as an error above
                self._strategy.log_warning(f"[{agent_name}] the agent ended without calling {tool}")
            error = self._recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = retry_prompt(tool, error), tool
        raise ReviewAbandoned(agent_name, error)


def _submission_lines(agent_name: str, submission: Any) -> list[str]:
    """One log line per item of a recorded submission: what the agent said, not its "submitted successfully"."""
    if isinstance(submission, Note):
        return [f"[{agent_name}] {submission.symbol}: {submission.note}"]
    if agent_name == "bull":
        return [f"[bull] {case.symbol}: {case.conviction} conviction: {case.argument}" for case in submission]
    if agent_name == "bear":
        return [f"[bear] {case.symbol}: {case.risk} risk ({case.concern}): {case.argument}" for case in submission]
    return [f"[{agent_name}] {verdict.symbol}: {verdict.verdict}: {verdict.reason}" + (f" (changed: {verdict.what_changed})" if verdict.what_changed else "") for verdict in submission.verdicts]
