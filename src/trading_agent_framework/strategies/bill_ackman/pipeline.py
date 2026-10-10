"""One review: screen, researcher, short seller, trader, rebalancer (see the design spec, section 2).

The agents decide; this module validates (through the `HandoffRecorder`), remembers (`StateStore`) and calls the
`Rebalancer`. A stage that never yields a valid submission abandons the review: nothing is traded and no counter
moves. Free text from an agent is logged and otherwise ignored.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bill_ackman.fact_sheet import fact_sheet, price_return, unavailable_fact_sheet
from trading_agent_framework.strategies.bill_ackman.handoff import FAIL, SURVIVE, HandoffRecorder, Idea, PortfolioPosition, Verdict
from trading_agent_framework.strategies.bill_ackman.hysteresis import advance_cooldowns, apply_verdicts
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.prompts import SHORT_SELLER_TASK, TRADER_TASK, researcher_task, retry_prompt
from trading_agent_framework.strategies.bill_ackman.screen import Candidate, ScreenResult
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, FundamentalsError, TradingFrameworkError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

# A holding the screen rejects for one of these reasons failed a quality gate: code gives it the verdict `fail`.
# The other reasons (no_data, no_price, no_split_data, duplicate_listing) are data problems, not judgements.
QUALITY_REJECTIONS = frozenset({"insufficient_history", "stale_filing", "operating_loss", "negative_fcf", "shrinking_revenue", "debt_unknown", "too_much_debt", "excluded_sector"})


class ScreenLike(Protocol):
    def run(self, symbols: Sequence[str], *, as_of: datetime, price_of: Callable[[str], Decimal | None], top_n: int | None = None, label: str = "screen") -> ScreenResult: ...


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
        params: AckmanParams,
        screen: ScreenLike,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        review_log: ReviewLog,
        rebalancer: Rebalancer,
        universe: Sequence[str],
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._screen = screen
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = review_log
        self._rebalancer = rebalancer
        self._universe = list(universe)
        self._stage = "setup"  # "setup" until the rebalancer starts, then "execution": what a broker failure is attributed to

    # --- one review --------------------------------------------------------------------------------

    def run(self) -> ReviewOutcome:
        state = self._state.load()
        now = self._strategy.clock.now().astimezone(MARKET_TZ)
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
        self._stage = "setup"
        extra: dict[str, Any] = {}
        try:
            self._review(now, state, record)
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
        self._strategy.log_error(f"review abandoned at the {stage} stage (streak {streak}): {error}")
        self._state.save(replace(state, abandoned_streak=streak))
        self._log.append({**record, "abandoned": True, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
        return ReviewOutcome(completed=False, abandoned_streak=streak)

    def _review(self, now: datetime, state: ReviewState, record: dict[str, Any]) -> None:
        params = self._params
        strategy = self._strategy
        price_of = strategy.get_last_price

        # 1-2. The screen over the universe, and over the holdings (so each holding has metrics or a reason).
        try:
            universe_result = self._screen.run(self._universe, as_of=now, price_of=price_of, label="universe")
            holdings = self._rebalancer.holdings()
            holdings_result = self._screen.run(holdings, as_of=now, price_of=price_of, top_n=len(holdings), label="holdings") if holdings else ScreenResult(candidates=[], rejections={})
        except FundamentalsError as exc:
            raise ReviewAbandoned("screen", str(exc)) from exc
        weights = self._rebalancer.current_weights()
        cooling = set(state.cooldowns)  # recent forced exits: kept out of the candidates and the allowed set
        candidates = [candidate for candidate in universe_result.candidates if candidate.symbol not in cooling]
        sheets = self._fact_sheets(candidates, holdings, holdings_result)
        record.update(candidates=[{"symbol": c.symbol, "rank": c.rank, "fcf_yield": round(c.fcf_yield, 4)} for c in candidates], holdings=holdings)

        # 3. Researcher.
        ranking: list[Idea] = []
        if candidates:
            symbols = [c.symbol for c in candidates]
            self._recorder.expect_ranking(symbols)
            strategy.log_info(f"[researcher] {len(candidates)} candidates, by FCF yield: {', '.join(f'{c.symbol} (#{c.rank}, {c.fcf_yield:.1%})' for c in candidates)}")
            context = {
                "current_datetime": now.isoformat(),
                "candidates": [sheets[symbol] for symbol in symbols],
                "holdings": [{"symbol": symbol, "weight": weight} for symbol, weight in weights.items()],
            }
            self._run_stage("researcher", "submit_ranking", researcher_task(min(params.research_top_n, len(symbols))), context)
            ranking = self._recorder.submission
        ranked = [idea.symbol for idea in ranking]
        researcher_reasons = {idea.symbol: idea.reason for idea in ranking}

        # 4-5. The review set, and the verdicts code gives to holdings the screen rejected on quality.
        review_set = list(dict.fromkeys([*ranked, *holdings]))
        verdicts: dict[str, Verdict] = {
            symbol: Verdict(symbol, FAIL, f"screen: {holdings_result.rejections[symbol]}") for symbol in holdings if holdings_result.rejections.get(symbol) in QUALITY_REJECTIONS
        }
        sources = {symbol: "screen" for symbol in verdicts}

        # 6. Short seller.
        to_judge = [symbol for symbol in review_set if symbol not in verdicts]
        if to_judge:
            previous = {symbol: state.last_verdicts[symbol] for symbol in to_judge if symbol in state.last_verdicts}
            self._recorder.expect_verdicts(to_judge, previous={symbol: record["verdict"] for symbol, record in previous.items()})
            strategy.log_info(f"[short_seller] {len(to_judge)} to judge: {', '.join(f'{symbol} (held)' if symbol in holdings else symbol for symbol in to_judge)}")
            context = {
                "current_datetime": now.isoformat(),
                "to_judge": [
                    {
                        "fact_sheet": sheets[symbol],
                        "researcher_reason": researcher_reasons.get(symbol),
                        "held": symbol in holdings,
                        "current_weight": weights.get(symbol, 0.0),
                        "fail_count": state.fail_counts.get(symbol, 0),
                        "previous_verdict": previous.get(symbol),
                    }
                    for symbol in to_judge
                ],
            }
            # One or two checks per name; past the budget only submit_verdicts runs (it is exempt), so a long
            # review set cannot overflow the model's context.
            self._run_stage("short_seller", "submit_verdicts", SHORT_SELLER_TASK, context, tool_budget=2 * len(to_judge))
            for verdict in self._recorder.submission:
                verdicts[verdict.symbol] = verdict
                sources[verdict.symbol] = "llm"

        survivors = [symbol for symbol in review_set if verdicts[symbol].verdict == SURVIVE]
        failures = [f"{symbol} ({verdicts[symbol].concern or verdicts[symbol].reason})" for symbol in review_set if verdicts[symbol].verdict == FAIL]
        strategy.log_info(f"[short_seller] reviewed {len(review_set)}: {len(survivors)} survive ({', '.join(survivors) or 'none'}); {len(failures)} fail ({', '.join(failures) or 'none'})")
        for symbol in review_set:
            verdict = verdicts[symbol]
            concern = f" ({verdict.concern})" if verdict.concern else ""
            changed = f" [changed: {verdict.what_changed}]" if verdict.what_changed else ""
            strategy.log_info(f"[short_seller]   {symbol} {verdict.verdict}{concern}: {verdict.reason}{changed}")

        # 7. Hysteresis.
        outcome = apply_verdicts(
            holdings=holdings,
            ranking=ranked,
            verdicts={symbol: verdict.verdict for symbol, verdict in verdicts.items()},
            fail_counts=state.fail_counts,
            forced_exit_fails=params.forced_exit_fails,
        )

        allowed = [symbol for symbol in outcome.allowed if symbol not in cooling]
        required = [symbol for symbol in holdings if symbol in allowed]  # held and not failed twice: may be shrunk, not dropped
        if len(required) > params.max_positions:  # the trader could not hold them all: require the first max_positions
            left_out = required[params.max_positions :]
            strategy.log_warning(
                f"[bill_ackman] {len(required)} holdings to keep but max_positions is {params.max_positions}: "
                f"requiring only {', '.join(required[: params.max_positions])}; left to the trader's choice: {', '.join(left_out)}"
            )
            required = required[: params.max_positions]
        elif len(required) == params.max_positions:
            blocked = [symbol for symbol in allowed if symbol not in required]
            strategy.log_info(
                f"[book] full: {len(required)} holdings to keep fill max_positions ({params.max_positions}); blocked newcomers: {', '.join(blocked) or 'none'}"
            )

        # 8. Trader.
        positions: list[PortfolioPosition] = []
        if allowed:
            self._recorder.expect_portfolio(allowed, required=required)
            strategy.log_info(
                f"[trader] {len(allowed)} allowed: {', '.join(allowed)}; required: {', '.join(required) or 'none'}; forced exits: {', '.join(outcome.forced_exits) or 'none'}"
            )
            context = {
                "current_datetime": now.isoformat(),
                "allowed": [
                    {
                        "symbol": symbol,
                        "research_rank": ranked.index(symbol) + 1 if symbol in ranked else None,
                        "verdict": verdicts[symbol].verdict,
                        "verdict_reason": verdicts[symbol].reason,
                        "researcher_reason": researcher_reasons.get(symbol),
                        "pending_fail_count": outcome.fail_counts.get(symbol, 0),
                        "current_weight": weights.get(symbol, 0.0),
                        "fact_sheet": sheets[symbol],
                    }
                    for symbol in allowed
                ],
                "required": required,
                "forced_exits": outcome.forced_exits,
                "constraints": {
                    "max_positions": params.max_positions,
                    "min_weight": params.min_weight,
                    "max_weight": params.max_weight,
                    "max_total_weight": params.max_total_weight,
                    "unallocated_money": f"parked in {params.parking_symbol} by code",
                },
            }
            self._run_stage("trader", "submit_portfolio", TRADER_TASK, context)
            positions = self._recorder.submission
        else:
            strategy.log_info("[trader] nothing allowed: everything goes to the parking instrument")
        if positions:
            strategy.log_info(f"[trader] decision: {len(positions)} positions, {sum(p.weight for p in positions):.1%} invested")
            for position in positions:
                strategy.log_info(f"[trader]   {position.symbol} {position.weight:.1%}: {position.reason}")
        elif allowed:
            strategy.log_info(f"[trader] decision: hold nothing, everything goes to {params.parking_symbol}")

        chosen = {position.symbol for position in positions}
        in_use = f"slots in use {len(chosen)}/{params.max_positions}"
        for symbol in holdings:
            if symbol not in chosen:
                strategy.log_info(f"[book] slot freed: {symbol} ({'forced exit' if symbol in outcome.forced_exits else 'cooling down'}); {in_use}")
        for position in positions:
            if position.symbol not in holdings:
                strategy.log_info(f"[book] slot filled: {position.symbol} at {position.weight:.1%}; {in_use}")

        # 9. Targets and execution.
        target = target_portfolio({position.symbol: position.weight for position in positions}, cash_buffer=params.cash_buffer)
        self._stage = "execution"
        orders = self._rebalancer.rebalance(target, outcome.forced_exits)

        # 10. Remember and log.
        cooldowns = advance_cooldowns(state.cooldowns, forced_exits=outcome.forced_exits, reviews=params.reentry_cooldown_reviews)
        self._state.save(
            ReviewState(
                last_review=now.date().isoformat(),
                fail_counts=outcome.fail_counts,
                last_ranking=ranked,
                last_verdicts={
                    symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "concern": verdict.concern, "date": now.date().isoformat()}
                    for symbol, verdict in verdicts.items()
                    if sources[symbol] == "llm"  # a verdict code gave a rejected holding is not the short seller's own
                },
                abandoned_streak=0,
                cooldowns=cooldowns,
            )
        )
        self._log.append(
            {
                **record,
                "ranking": [asdict(idea) for idea in ranking],
                "review_set": review_set,
                "verdicts": [{**asdict(verdict), "source": sources[verdict.symbol]} for verdict in verdicts.values()],
                "fail_counts_before": state.fail_counts,
                "fail_counts_after": outcome.fail_counts,
                "forced_exits": outcome.forced_exits,
                "allowed": allowed,
                "required": required,
                "cooldowns": cooldowns,
                "portfolio": [asdict(position) for position in positions],
                "targets": {**target.weights, params.parking_symbol: target.parking_weight},
                "orders": [asdict(order) for order in orders],
            }
        )

    # --- helpers -----------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any], tool_budget: int | None = None) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon.

        `tool_budget` caps each attempt's research tool calls (see `AgentHandle.run`).
        """
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

    def _fact_sheets(self, candidates: Sequence[Candidate], holdings: Sequence[str], holdings_result: ScreenResult) -> dict[str, dict[str, Any]]:
        """A fact sheet per candidate and per holding; a holding the screen could not describe gets the price facts and the reason."""
        sheets: dict[str, dict[str, Any]] = {}
        for candidate in [*candidates, *holdings_result.candidates]:
            if candidate.symbol not in sheets:
                price, change = self._price_facts(candidate.symbol)
                sheets[candidate.symbol] = fact_sheet(candidate, price=price, price_return_12m=change)
        for symbol in holdings:
            if symbol not in sheets:
                price, change = self._price_facts(symbol)
                sheets[symbol] = unavailable_fact_sheet(symbol, reason=holdings_result.rejections.get(symbol, "no_data"), price=price, price_return_12m=change)
        return sheets

    def _price_facts(self, symbol: str) -> tuple[float | None, float | None]:
        """(last price, 12-month return); each None when it cannot be computed (a failed lookup never stops a review)."""
        strategy = self._strategy
        try:
            last = strategy.get_last_price(symbol)
        except TradingFrameworkError:
            last = None
        try:
            bars = strategy.get_historical_prices(symbol, 253)
        except TradingFrameworkError:
            bars = None
        closes = [float(close) for close in bars.df["close"]] if bars is not None else []
        return (float(last) if last is not None else None, price_return(closes))
