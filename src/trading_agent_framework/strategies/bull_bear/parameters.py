"""`BullBearParams`: every threshold of the bull_bear strategy in one frozen dataclass.

The momentum parameters (score weights, skip, filters) are NOT here: they are cross_momentum's `CONFIG`, so the
debated ranking is cross_momentum's by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.utils.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class BullBearParams:
    retention_rank: int = 35  # a holding ranked worse (or unranked) is a forced exit; above candidate_count
    candidate_count: int = 15  # the top ranks the judge rules on every review (plus the holdings ranked up to retention_rank)
    max_book_size: int = 10  # most stocks the judge may put in
    strong_weight: float = 0.14  # per-stock weight by the judge's strength, fractions of portfolio value; there is no weak weight (a weak stock is out)
    standard_weight: float = 0.09  # ten standard stocks make 90%; above 98% every weight is scaled by one factor
    cash_buffer: float = 0.02  # never invested (fees, fill drift)
    rebalance_band: float = 0.03  # drift that triggers a trade
    min_trade_pct: float = 0.005  # smallest order
    parking_symbol: str = "SHV"  # where unspent money goes
    research_tool_budget: int = 5  # tool calls per researcher run; submit_note is exempt
    research_news: bool = True  # the researcher's news tool; False for the news-off comparison run
    # LangGraph steps one agent run may take before it counts as a loop and abandons the review: a model resubmitting a rejected
    # call for ever. LangGraph supersteps: a tool call is a round trip of 3, a run takes 3 x calls + 5 at worst, so a normal run
    # needs a limit of at most 21 (the researcher using its 5 tool calls; the judge with 3 memory calls needs 15); a rejected resubmission adds 3.
    max_agent_steps: int = 60
    # Memory tool calls per judge run; submit_verdicts is exempt. The 10-23 run (Qwen 3.8) made 107 memory calls, each kept in its context.
    judge_memory_budget: int = 3
    note_max_chars: int = 500  # the length the researcher is asked for in a submitted note
    note_hard_max_chars: int = 600  # the length the recorder refuses above: a model cannot count characters (10-19: 25 notes 1 to 20% over 500)
    argument_max_chars: int = 300  # the bull's and the bear's submitted argument
    reason_max_chars: int = 300  # the judge's submitted reason per verdict
    reentry_cooldown_reviews: int = 4  # completed reviews a stock the judge sold stays out of the candidates
    feedback_reviews: int = 4  # completed reviews whose out verdicts the judge sees with their outcome
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    agent_temperature: float | None = 0.3  # all four agents; None leaves the LLM server's default
    # Reasoning tokens per model call for the researcher, bull and bear (None: no budget), sent to the server, which cuts the
    # reasoning there and lets the model write its answer. A safety net set high on purpose: a healthy turn never reaches it
    # (the 10-25 run, Qwen 3.8 at effort medium, peaked at 11.6k output tokens), and a budget below normal use would cut good
    # reasoning. A turn that reaches it is logged as a warning.
    thinking_token_budget: int | None = 28672
    # The judge's own budget (it reasons over every candidate at once); None: NO budget for the judge, not thinking_token_budget.
    judge_thinking_token_budget: int | None = 36864
    # Output tokens per model call, reasoning included (the researcher, bull and bear; None: no cap). Each cap is its budget plus
    # 8,192: the room left after a cut for the submission. One run per review covers every stock (up to about 25): the judge's
    # reason (up to 300 characters per stock) plus what_changed (up to 300 on each flip) come to about 15,000 characters, 4-5k
    # tokens, at most; the bull's and the bear's argument is up to 300 per stock; the rest is margin for text written after the cut.
    # The cap is the backstop for an answer that runs on and for a server that ignores the budget: it counts every output token,
    # and a call that spends it all on reasoning stops with finish_reason=length.
    max_output_tokens: int | None = 36864
    # The judge's own cap (researcher, bull and bear keep max_output_tokens); None: NO cap for the judge, not max_output_tokens.
    # Window check: the judge's prompt peaked at 17.4k tokens in the 10-25 run and 20.5k in the 10-23 run, so 20k + 45,056 is
    # about 65k of the 81,920 window (a prompt plus the cap above the window makes output_cap's fallback send the call uncapped).
    judge_max_output_tokens: int | None = 45056
    rebalance_time: str = "12:00"  # market time (ET) of the weekly review
    rebalance_weekday: int = 1  # 0 = Monday ... 4 = Friday; 1 = Tuesday, as cross_momentum
    yahoo_retry_delays: tuple[float, ...] = (60.0, 180.0)  # seconds before each new attempt at an unusable Yahoo batch
    min_yahoo_coverage: float = 0.5  # share of the universe a Yahoo batch must cover

    def __post_init__(self) -> None:
        floats = (self.strong_weight, self.standard_weight, self.cash_buffer, self.rebalance_band, self.min_trade_pct, self.min_yahoo_coverage)
        investable = 1 - self.cash_buffer
        # a standard stock in the fullest book of strong ones, once scaled down to what is investable (checked only for a book that makes sense)
        fullest_book = (self.max_book_size - 1) * self.strong_weight + self.standard_weight
        smallest_target = min(self.standard_weight, self.standard_weight * investable / fullest_book) if fullest_book > 0 else math.inf
        try:
            parse_rebalance_time(self.rebalance_time)
            time_problem = ""
        except ConfigurationError as exc:
            time_problem = str(exc)
        problems = {
            "strong_weight, standard_weight, cash_buffer, rebalance_band, min_trade_pct and min_yahoo_coverage must be finite": not all(math.isfinite(value) for value in floats),
            "candidate_count and max_book_size must be at least 1": self.candidate_count < 1 or self.max_book_size < 1,
            "retention_rank must be above candidate_count": self.retention_rank <= self.candidate_count,
            "0 < standard_weight < strong_weight <= 1": not 0 < self.standard_weight < self.strong_weight <= 1,
            "a standard stock in a full book of strong ones must stay above rebalance_band (it would never be bought)": smallest_target <= self.rebalance_band,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "research_tool_budget must be at least 1": self.research_tool_budget < 1,
            "max_agent_steps must be at least 1": self.max_agent_steps < 1,
            "judge_memory_budget must be at least 0": self.judge_memory_budget < 0,
            "note_max_chars, argument_max_chars and reason_max_chars must be at least 1": min(self.note_max_chars, self.argument_max_chars, self.reason_max_chars) < 1,
            "note_hard_max_chars must be at least note_max_chars": self.note_hard_max_chars < self.note_max_chars,
            "reentry_cooldown_reviews must be at least 0": self.reentry_cooldown_reviews < 0,
            "feedback_reviews must be at least 1": self.feedback_reviews < 1,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
            "max_output_tokens must be None or at least 1": self.max_output_tokens is not None and self.max_output_tokens < 1,
            "judge_max_output_tokens must be None or at least 1": self.judge_max_output_tokens is not None and self.judge_max_output_tokens < 1,
            "thinking_token_budget must be None or at least 1": self.thinking_token_budget is not None and self.thinking_token_budget < 1,
            "judge_thinking_token_budget must be None or at least 1": self.judge_thinking_token_budget is not None and self.judge_thinking_token_budget < 1,
            f"rebalance_time: {time_problem}": bool(time_problem),
            "rebalance_weekday must be a weekday (0 = Monday ... 4 = Friday)": not 0 <= self.rebalance_weekday <= 4,
            "yahoo_retry_delays must be finite and at least 0": not all(math.isfinite(delay) and delay >= 0 for delay in self.yahoo_retry_delays),
            "min_yahoo_coverage must be in (0, 1]": not 0 < self.min_yahoo_coverage <= 1,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("BullBearParams: " + "; ".join(failed))

    @property
    def investable(self) -> float:
        """The share of portfolio value the stock weights sum to: everything but the cash buffer."""
        return 1 - self.cash_buffer

    @property
    def weights_by_strength(self) -> dict[str, float]:
        """The weight of a stock the judge rates strong or standard (a weak stock is never in the book)."""
        return {"strong": self.strong_weight, "standard": self.standard_weight}
