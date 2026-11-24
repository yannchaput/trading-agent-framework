"""One weekly review as a LangGraph graph: rank → fact_sheets → research → bull_case → bear_case → judge → execute.

Code ranks the universe and builds the candidates (`rank`), reads every candidate's fact sheet (`fact_sheets`), runs
the researcher once per stock (`research`), the bull and then the bear on the same evidence (`bull_case`,
`bear_case`) and the judge (`judge`), who puts every candidate in or out. Code validates every submission (through the
`HandoffRecorder`), builds the book from the stocks put in (rank order), sizes it, calls the `Rebalancer`, remembers
(`StateStore`) and appends one line to `reviews.jsonl` (`execute`). No agent has an order tool.

A stage that never yields a valid submission, unusable data, no candidate or a broker failure abandons the review: the
node raises `StageAbandoned` (or a `BrokerError`/`BacktestError` comes up from the broker), it passes straight through
`graph.invoke`, and `run` stores the streak and appends the abandoned line. No order goes out before `execute`, forced
exits included. One stock's failed research only gives it the note "research unavailable". Free text from an agent is
logged and otherwise ignored.

The graph state carries what flows from one step to the next. What an abandonment needs after a node raised is kept on
the review instead (`_record`, `_stage`, `_pending`): a failed `invoke` returns no state. The graph runs with a plain
`invoke` (no streaming, no `step_timeout`, no subgraphs), so every node, `execute` and its orders included, runs on the
strategy thread.

See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md,
docs/superpowers/specs/2026-10-11-bull-bear-debate-realignment-design.md and
docs/superpowers/specs/2026-11-24-bull-bear-langgraph-design.md.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bull_bear.book import build_book
from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet, build_debate_set
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.bull_bear.fundamentals import unavailable_fundamentals
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdicts
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.prompts import UNAVAILABLE_NOTE, bear_task, bull_task, judge_task, researcher_task, retry_prompt
from trading_agent_framework.strategies.bull_bear.sizing import equal_weights
from trading_agent_framework.strategies.bull_bear.state import BullBearState, StateStore
from trading_agent_framework.strategies.common.cooldowns import advance_cooldowns
from trading_agent_framework.strategies.common.daily_bars import DailyBars, DataUnavailable
from trading_agent_framework.strategies.common.pipeline import AgentLike as CommonAgentLike
from trading_agent_framework.strategies.common.pipeline import RunOutcome, StageAbandoned
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, rank, score_stock
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, AgentLoopError, BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class AgentLike(CommonAgentLike, Protocol):
    """The slice of an agent handle bull_bear uses."""

    def run(
        self,
        task_prompt: str,
        *,
        context: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        force_tool: str | None = None,
        tool_budget: int | None = None,
        max_steps: int | None = None,
    ) -> AgentRunResult:
        """Run the agent on `task_prompt` and return its output."""
        ...


class AgentLookup(Protocol):
    """What the review needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike:
        """The agent registered under `name`."""
        ...


class AgentLooped(StageAbandoned):
    """An agent run used up its step limit: raised at once, with no forced retry. The review is abandoned, except for the researcher, whose stock only loses its note."""


@dataclass(frozen=True, slots=True)
class _Pending:
    """What a review decided, kept for an abandonment during execution (the judge's verdicts are out in the market)."""

    decided: dict[str, dict[str, str]]  # symbol -> the judge's {verdict, reason, date, strength}
    sold: list[str]  # the held stocks the judge put out


class ReviewState(TypedDict):
    """What one review hands from node to node; `now` and `saved` are the input, each other slot has one writer.

    Slot names differ from node names (LangGraph refuses a node named like a state key): `bull_case` writes `bull`.
    """

    now: datetime  # the review time, market timezone
    saved: BullBearState  # the state loaded at the start of the review
    debate: DebateSet  # rank: the candidates and the forced exits
    sheets: dict[str, dict[str, Any]]  # fact_sheets: symbol -> fact sheet row
    notes: dict[str, str]  # research: symbol -> note (UNAVAILABLE_NOTE where its run failed)
    bull: dict[str, BullCase]  # bull_case
    bear: dict[str, BearCase]  # bear_case
    verdicts: Verdicts  # judge
    previous: dict[str, dict[str, str]]  # judge: symbol -> the judge's previous verdict entry, for the stocks it judged before


class BullBearReview:
    """The weekly review: a straight-line LangGraph graph, compiled once, run by `run`."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        params: BullBearParams,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        log: RunLog,
        rebalancer: Rebalancer,
        universe_on: Callable[[date], Sequence[str]],
        bars: DailyBars,
        sector_of: Callable[[str], str],
        fundamentals_of: Callable[[str, datetime], Mapping[str, Any]],
        momentum: Mapping[str, Any],
    ) -> None:
        """Wire the review to the strategy, parameters, agents, handoff recorder, state, log and data sources, and compile the graph."""
        self._strategy = strategy
        self._params = params
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = log
        self._rebalancer = rebalancer
        self._universe_on = universe_on  # the stocks to rank on a market date: a constant list in paper/live, the month's list in a backtest
        self._bars = bars
        self._sector_of = sector_of
        self._fundamentals_of = fundamentals_of  # (symbol, the strategy clock's now) -> the SEC fields of the fact sheet
        self._momentum = momentum  # cross_momentum's CONFIG: the score weights, the skip and the filters
        # max_volatility gates new entries only: holdings are scored with a copy that lifts it (CONFIG itself is never changed)
        self._holding_momentum = {**momentum, "max_volatility": math.inf}
        # What an abandonment needs once a node has raised; reset by `run` at the start of every review.
        self._record: dict[str, Any] = {}  # the reviews.jsonl line being filled in
        self._stage = "setup"  # "setup" until the rebalancer starts, then "execution": what a broker failure is attributed to
        self._pending: _Pending | None = None  # set once the judge has decided, just before the stage becomes "execution"
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Compile START -> rank -> fact_sheets -> research -> bull_case -> bear_case -> judge -> execute -> END."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(ReviewState)
        steps = [
            ("rank", self._rank_node),
            ("fact_sheets", self._fact_sheets_node),
            ("research", self._research_node),
            ("bull_case", self._bull_node),
            ("bear_case", self._bear_node),
            ("judge", self._judge_node),
            ("execute", self._execute_node),
        ]
        previous = START
        for name, node in steps:
            graph.add_node(name, node)
            graph.add_edge(previous, name)
            previous = name
        graph.add_edge(previous, END)
        return graph.compile()

    def _now(self) -> datetime:
        """Current time in the market timezone."""
        return self._strategy.clock.now().astimezone(MARKET_TZ)

    def _fundamentals(self, symbol: str, now: datetime) -> Mapping[str, Any]:
        """One stock's SEC fields as of the strategy clock; whatever goes wrong costs that stock its figures, never the review."""
        try:
            return self._fundamentals_of(symbol, now)
        except Exception as exc:  # noqa: BLE001 -- a source that raises must not abandon the review
            self._strategy.log_warning(f"[bull_bear] fundamentals for {symbol} unavailable: {exc!r}")
            return unavailable_fundamentals("error")

    def completed_today(self) -> bool:
        """Whether today's review (market date) already completed: a restart then does not run it again."""
        return self._state.load().last_completed_review == self._now().date().isoformat()

    # --- one review --------------------------------------------------------------------------------

    def run(self) -> RunOutcome:
        """Run one review through the graph; on an abandonment, store the streak and append the abandoned line.

        An exception that is not an abandonment (a `ConfigurationError` from an agent) propagates with the state untouched.
        """
        state = self._state.load()
        now = self._now()
        self._record = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
        self._stage, self._pending = "setup", None
        try:
            self._graph.invoke({"now": now, "saved": state})
            return RunOutcome(completed=True, abandoned_streak=0)
        except StageAbandoned as exc:
            stage, error = exc.stage, exc
        except (BrokerError, BacktestError) as exc:
            # A broker or data failure anywhere in the review: orders may be out once the rebalancer started.
            stage, error = ("execution" if self._stage == "execution" else "broker"), exc
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"[bull_bear] review abandoned at the {stage} stage (streak {streak}): {error}")
        saved, extra = self._on_abandoned(state, streak, stage)
        self._state.save(saved)
        self._log.append({**self._record, "abandoned": True, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
        return RunOutcome(completed=False, abandoned_streak=streak)

    def _on_abandoned(self, state: BullBearState, streak: int, stage: str) -> tuple[BullBearState, dict[str, Any]]:
        """Build the state and record for a run abandoned after repeated failures."""
        extra: dict[str, Any] = {}
        if stage == "execution":
            extra["orders"] = [asdict(order) for order in self._rebalancer.placed]  # sent before the failure
        if stage == "execution" and self._pending is not None:
            # The judge decided and orders may be out: remember its verdicts and the cooldown of each sell actually placed,
            # so a stock sold today is not bought back next week. Running cooldowns are not counted down and the day stays open.
            length = self._params.reentry_cooldown_reviews
            sent = {order.symbol for order in self._rebalancer.placed if order.side == "sell"}
            started = [symbol for symbol in self._pending.sold if symbol in sent] if length > 0 else []
            extra["cooldowns_started"] = started
            saved = replace(state, abandoned_streak=streak, verdicts={**state.verdicts, **self._pending.decided}, cooldowns={**state.cooldowns, **{symbol: length for symbol in started}})
        else:
            saved = replace(state, abandoned_streak=streak)
        return saved, extra

    # --- the nodes ---------------------------------------------------------------------------------

    def _rank_node(self, state: ReviewState) -> dict[str, Any]:
        """Rank the universe and build the debate set: the candidates, and the holdings code sells without asking the agents."""
        params, strategy = self._params, self._strategy
        holdings = self._rebalancer.holdings()
        ranked = self._rank(state["now"], holdings)
        debate = build_debate_set(ranked, holdings, candidate_count=params.candidate_count, retention_rank=params.retention_rank, excluded=state["saved"].cooldowns)
        self._record.update(
            ranked=len(ranked),
            debate_set=[{"symbol": stock.symbol, "rank": stock.rank, "held": stock.held} for stock in debate.stocks],
            forced_exits=[asdict(exit) for exit in debate.forced_exits],
            cooldowns_active=dict(state["saved"].cooldowns),
        )
        strategy.log_info(
            f"[bull_bear] {len(ranked)} stocks ranked; debate: {', '.join(f'{s.symbol} (#{s.rank}{", held" if s.held else ""})' for s in debate.stocks)}; "
            f"forced exits: {', '.join(f'{e.symbol} ({e.reason})' for e in debate.forced_exits) or 'none'}"
        )
        if not debate.symbols:
            raise StageAbandoned("data", "no stock to debate")
        return {"debate": debate}

    def _fact_sheets_node(self, state: ReviewState) -> dict[str, Any]:
        """One fact sheet per candidate, its SEC fields read as of the strategy clock."""
        as_of = self._strategy.clock.now()
        sheets = {stock.symbol: fact_sheet_row(stock, sector=self._sector_of(stock.symbol), fundamentals=self._fundamentals(stock.symbol, as_of)) for stock in state["debate"].stocks}
        return {"sheets": sheets}

    def _research_node(self, state: ReviewState) -> dict[str, Any]:
        """The researcher: one short run per stock, so one failure costs one note (`UNAVAILABLE_NOTE`), not the review."""
        params, strategy = self._params, self._strategy
        symbols, sheets, moment = state["debate"].symbols, state["sheets"], state["now"].isoformat()
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
            except StageAbandoned as exc:  # a looping run too: it costs this stock its note, like any other researcher failure
                strategy.log_warning(f"[researcher] no note for {symbol}: {exc}")
                notes[symbol] = UNAVAILABLE_NOTE
                continue
            note: Note = self._recorder.submission
            notes[symbol] = note.note
        failed = [symbol for symbol in symbols if notes[symbol] == UNAVAILABLE_NOTE]
        self._record["notes"] = {"written": len(symbols) - len(failed), "failed": failed}
        return {"notes": notes}

    def _evidence(self, state: ReviewState) -> list[dict[str, Any]]:
        """What the bull and the bear both read: each candidate's fact sheet and note, in rank order."""
        return [{"fact_sheet": state["sheets"][symbol], "note": state["notes"][symbol]} for symbol in state["debate"].symbols]

    def _bull_node(self, state: ReviewState) -> dict[str, Any]:
        """The bull's case for every candidate; it never sees the bear's."""
        symbols, limit = state["debate"].symbols, self._params.argument_max_chars
        self._recorder.expect_bull(symbols)
        self._run_stage("bull", "submit_bull_case", bull_task(limit), {"current_datetime": state["now"].isoformat(), "stocks": self._evidence(state), "constraints": {"argument_max_chars": limit}})
        return {"bull": {case.symbol: case for case in self._recorder.submission}}

    def _bear_node(self, state: ReviewState) -> dict[str, Any]:
        """The bear's case against every candidate, from the same evidence; then both sides are logged stock by stock."""
        symbols, limit, bull = state["debate"].symbols, self._params.argument_max_chars, state["bull"]
        self._recorder.expect_bear(symbols)
        self._run_stage("bear", "submit_bear_case", bear_task(limit), {"current_datetime": state["now"].isoformat(), "stocks": self._evidence(state), "constraints": {"argument_max_chars": limit}})
        bear: dict[str, BearCase] = {case.symbol: case for case in self._recorder.submission}
        self._record.update(bull_conviction=dict(Counter(case.conviction for case in bull.values())), bear_risk=dict(Counter(case.risk for case in bear.values())))
        for symbol in symbols:
            self._strategy.log_info(f"[debate]   {symbol}: bull {bull[symbol].conviction} ({bull[symbol].argument}); bear {bear[symbol].risk}/{bear[symbol].concern} ({bear[symbol].argument})")
        return {"bear": bear}

    def _judge_node(self, state: ReviewState) -> dict[str, Any]:
        """The judge: in or out on every candidate, on merit (no agent is told which stocks are held), from its own previous verdict on a stock when it gave one."""
        params, symbols, saved = self._params, state["debate"].symbols, state["saved"]
        sheets, notes, bull, bear = state["sheets"], state["notes"], state["bull"], state["bear"]
        previous = {symbol: saved.verdicts[symbol] for symbol in symbols if symbol in saved.verdicts}
        self._recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})
        context = {
            "current_datetime": state["now"].isoformat(),
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
            "constraints": {"max_in": params.max_book_size, "reason_max_chars": params.reason_max_chars},
        }
        self._run_stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context)
        verdicts: Verdicts = self._recorder.submission
        outs = ", ".join(f"{verdict.symbol} ({verdict.reason})" for verdict in verdicts.verdicts if verdict.verdict == "out")
        ins = ", ".join(f"{verdict.symbol} ({verdict.strength})" for verdict in verdicts.verdicts if verdict.verdict == "in")
        self._strategy.log_info(f"[judge] in: {ins or 'none'}; out: {outs or 'none'}")
        return {"verdicts": verdicts, "previous": previous}

    def _execute_node(self, state: ReviewState) -> dict[str, Any]:
        """Build the book, size it, rebalance, then remember the review and append its line to `reviews.jsonl`."""
        params, strategy, debate, verdicts, previous, saved = self._params, self._strategy, state["debate"], state["verdicts"], state["previous"], state["saved"]
        today = state["now"].date().isoformat()
        # The book: every stock the judge put in, in rank order.
        book = build_book(debate, verdicts.chosen)
        strategy.log_info(f"[bull_bear] book: {', '.join(book) or 'empty, everything parked'}")

        # Sizing: every stock in the book gets the same weight (no volatility, no strength); the rest is parked.
        target_weights = equal_weights(book, weight=params.stock_weight)
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        # A held stock the judge put out starts a cooldown: it stays out of the candidates for the next reviews.
        chosen = set(verdicts.chosen)
        sold = [stock.symbol for stock in debate.stocks if stock.held and stock.symbol not in chosen]
        # What a completed review remembers, built before the first order.
        decided = {verdict.symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "date": today, "strength": verdict.strength} for verdict in verdicts.verdicts}
        self._pending = _Pending(decided=decided, sold=sold)
        # The decision goes into the record before the first order: an abandonment during execution still says why.
        self._record.update(
            verdicts=[{**asdict(verdict), "previous_verdict": previous.get(verdict.symbol, {}).get("verdict")} for verdict in verdicts.verdicts],
            verdict_counts=dict(Counter(verdict.verdict for verdict in verdicts.verdicts)),
            strength_counts=dict(Counter(verdict.strength for verdict in verdicts.verdicts if verdict.strength)),
            book=list(book),
            cooldowns_started=sold,
            targets={**target.weights, params.parking_symbol: target.parking_weight},
        )
        # A forced exit is sold by code, never debated: say so at warning level, before its sell order goes out.
        for forced in debate.forced_exits:
            strategy.log_warning(f"[bull_bear] forced exit: {forced.symbol} ({forced.reason}) is sold by code, outside the debate")
        self._stage = "execution"
        orders = self._rebalancer.rebalance(target, [forced.symbol for forced in debate.forced_exits])

        # Remember and log.
        cooldowns = advance_cooldowns(saved.cooldowns, started=sold, length=params.reentry_cooldown_reviews)
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, verdicts={**saved.verdicts, **decided}, cooldowns=cooldowns))
        self._log.append(
            {
                **self._record,
                "bull": [asdict(case) for case in state["bull"].values()],
                "bear": [asdict(case) for case in state["bear"].values()],
                "orders": [asdict(order) for order in orders],
            }
        )
        return {}

    # --- helpers -----------------------------------------------------------------------------------

    def _rank(self, now: datetime, holdings: Sequence[str]) -> list[RankedRow]:
        """Score the universe (the bars come from `self._bars`) and rank it; a holding keeps every filter but max_volatility."""
        universe = list(self._universe_on(now.date()))
        try:
            series = self._bars.load(universe, holdings)
        except DataUnavailable as exc:
            raise StageAbandoned("data", str(exc)) from exc
        held_now = set(holdings)
        max_volatility = self._momentum["max_volatility"]
        rows: list[MomentumRow] = []
        for symbol in universe:
            if symbol not in series:
                continue
            daily = series[symbol]
            closes, volumes = daily.closes, daily.volumes
            held = symbol in held_now
            # A holding: every filter but max_volatility. A volatility spike after a crash must not force it out
            # near its low; at or below the cap this is the very row the full filters would give. A possible new
            # entry: cross_momentum's filters in full, max_volatility included.
            row = score_stock(symbol, closes, volumes, self._holding_momentum if held else self._momentum, traded_closes=daily.traded_closes, traded_volumes=daily.traded_volumes)
            if held and row is not None and row.volatility > max_volatility:
                self._strategy.log_info(f"[bull_bear] {symbol} is held at a volatility of {row.volatility:.0%}, above max_volatility ({max_volatility:.0%}): kept in the ranking")
            if row is not None:
                rows.append(row)
        return rank(rows)

    def _run_stage(
        self,
        agent_name: str,
        tool: str,
        task: str,
        context: dict[str, Any],
        tool_budget: int | None = None,
    ) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon.

        Attempt 1 runs `task` as is. If it records no valid submission, attempt 2 runs a fresh agent run (same `run_id`)
        with `retry_prompt` (quoting the rejection, or saying the agent ended without the tool) and `tool` forced, so the
        model must call it. On success the submit tool is a final tool, so the run ends without a closing model turn.
        `AgentLoopError` is not retried: it raises `AgentLooped` at once.

        Args:
            agent_name: key into `self._agents` (`"researcher"`, `"bull"`, `"bear"` or `"judge"`).
            tool: the submit tool the agent must call, e.g. `"submit_verdicts"`.
            task: the first-attempt prompt; the retry prompt is built from it.
            context: the template variables passed to the agent run.
            tool_budget: cap on the agent's non-submit tool calls per run (the researcher's), or `None`.

        Returns:
            None. The accepted submission is in `self._recorder.submission`.

        Raises:
            AgentLooped: the run hit the step limit (`AgentLoopError`); no retry.
            StageAbandoned: neither attempt recorded a valid `tool` call.
        """
        agent = self._agents[agent_name]
        # Both attempts share one run_id, so the first run and its forced retry are recorded under the same id.
        run_id = uuid.uuid4().hex
        # `prompt` and `force_tool` change only for the retry: `force_tool=None` lets the model choose.
        prompt, force_tool, error = task, None, ""
        for attempt in range(2):
            self._recorder.last_error, error, stop = None, "", ""  # an error from the first attempt must not be reported for the second
            if attempt:
                self._strategy.log_info(f"[{agent_name}] retrying with {tool} forced")
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool, tool_budget=tool_budget, max_steps=self._params.max_agent_steps)
                stop = _stop_detail(result)  # how the run's last model call ended, for the warnings below
                if result.output:  # empty when a final tool ended the run with no closing turn (the normal case), or the model spent its window thinking (warned about below, with `stop`)
                    self._strategy.log_info(f"[{agent_name}] {result.output}")
                for i, tool_call in enumerate(result.tool_calls):
                    self._strategy.log_debug(f"[{agent_name}] tool_call_{i}: {tool_call}")
            except AgentLoopError as exc:
                # The run reached the maximum number of steps: the model looped. The review is abandoned, no retry: a retry would only loop again.
                self._strategy.log_error(f"[{agent_name}] run stopped as a loop (too many steps): {exc}")
                raise AgentLooped(agent_name, str(exc)) from exc
            except AgentError as exc:
                # Any other failure than AgentLoopError: the run will trigger a retry with a force submission.
                error = str(exc)
                self._strategy.log_error(f"[{agent_name}] run failed: {exc}")
            # `submitted` is set by the recorder when the submit tool was accepted: the only success exit.
            if self._recorder.submitted:
                if attempt:
                    self._strategy.log_info(f"[{agent_name}] forced retry recorded {tool}")
                for line in _submission_lines(agent_name, self._recorder.submission):
                    self._strategy.log_info(line)
                return
            if self._recorder.last_error:  # the model called the tool and it was refused: the refusal is the retry's reason
                self._strategy.log_warning(f"[{agent_name}] submission rejected: {self._recorder.last_error}{stop}")
            elif not error:  # a failed run was already logged as an error above
                self._strategy.log_warning(f"[{agent_name}] the agent ended without calling {tool}{stop}")
            # The retry prompt names the most specific reason: the rejection, else the run's failure, else "no call".
            error = self._recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = retry_prompt(agent_name, tool, error, task), tool
            self._strategy.log_debug(f"[{agent_name}] retry with prompt: {prompt}")
        raise StageAbandoned(agent_name, error)


def _stop_detail(result: AgentRunResult) -> str:
    """` (finish_reason=length, 10069 input, 22699 output, 22699 reasoning, 32768 total tokens)`: how the run's last model call ended, for a run that reports it; else empty.

    A `length` stop with every output token spent on reasoning is what a reasoning model that hit its output cap (or the window) looks like: no text, no tool call.
    """
    parts = [
        f"finish_reason={result.finish_reason}" if result.finish_reason else "",
        f"{result.input_tokens} input" if result.input_tokens is not None else "",
        f"{result.output_tokens} output" if result.output_tokens is not None else "",
        f"{result.reasoning_tokens} reasoning" if result.reasoning_tokens is not None else "",
        f"{result.total_tokens} total tokens" if result.total_tokens is not None else "",
    ]
    known = [part for part in parts if part]
    return f" ({', '.join(known)})" if known else ""


def _submission_lines(agent_name: str, submission: Any) -> list[str]:
    """One log line per item of a recorded submission: what the agent said, not its "submitted successfully"."""
    if isinstance(submission, Note):
        return [f"[{agent_name}] {submission.symbol}: {submission.note}"]
    if agent_name == "bull":
        return [f"[bull] {case.symbol}: {case.conviction} conviction: {case.argument}" for case in submission]
    if agent_name == "bear":
        return [f"[bear] {case.symbol}: {case.risk} risk ({case.concern}): {case.argument}" for case in submission]
    return [f"[{agent_name}] {verdict.symbol}: {verdict.verdict}: {verdict.reason}" + (f" (changed: {verdict.what_changed})" if verdict.what_changed else "") for verdict in submission.verdicts]
