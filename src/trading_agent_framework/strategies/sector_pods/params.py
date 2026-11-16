"""`SectorPodsParams`: every threshold of the sector_pods strategy in one frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from trading_agent_framework.agents.tools.text import TextLimit
from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.utils.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class SectorPodsParams:
    """Parameters of the sector_pods strategy; satisfies `RebalanceParams`.

    Attributes:
        rebalance_weekday: 0 is Monday ... 4 is Friday.
        rebalance_time: Market time (ET) of the weekly review.
        cash_buffer: Share of the portfolio never invested.
        rebalance_band: Drift that triggers a trade.
        min_trade_pct: Smallest order.
        parking_symbol: Where unspent money goes.
        max_etf_weight: Largest weight of one ETF.
        max_pod_weight: Largest total weight of one pod's sleeve.
        pod_tool_budget: Tool calls per pod run; the submit tool is exempt.
        pm_memory_budget: `search_memory` calls per portfolio manager run (`remember_lesson` is not counted).
        max_agent_steps: LangGraph steps one agent run may take before it counts as a loop.
        agent_temperature: Sampling temperature of every agent; None leaves the server's default.
        thinking_token_budget: Reasoning tokens per model call; None for no budget.
        max_output_tokens: Output tokens per model call, reasoning included; None for no cap.
        reason_max_chars: Length of a reason the prompts ask for; a longer one is refused while attempts remain.
        summary_max_chars: Length of a summary, sector view or overrides text the prompts ask for; same rule.
        reason_hard_max_chars: Where the last attempt cuts a longer reason (never quoted to the models).
        summary_hard_max_chars: Where the last attempt cuts a longer summary, sector view or overrides text.
        max_submit_attempts: Submissions per agent run before over-long texts are cut instead of refused; only a refusal
            for length counts, 1 never refuses.
        max_consecutive_abandoned: Abandoned reviews in a row that abort a backtest.
        yahoo_retry_delays: Seconds before each new attempt at an unusable Yahoo batch.
        min_yahoo_coverage: Share of the universe a Yahoo batch must cover.
    """

    rebalance_weekday: int = 1  # 0 = Monday ... 4 = Friday; 1 = Tuesday
    rebalance_time: str = "12:00"  # market time (ET) of the weekly review
    cash_buffer: float = 0.02  # never invested
    rebalance_band: float = 0.03  # drift that triggers a trade
    min_trade_pct: float = 0.005  # smallest order
    parking_symbol: str = "SHV"  # where unspent money goes
    max_etf_weight: float = 0.25  # largest weight of one ETF
    max_pod_weight: float = 0.40  # largest total weight of one pod
    pod_tool_budget: int = 4  # tool calls per pod run
    pm_memory_budget: int = 3  # search_memory calls per portfolio manager run
    max_agent_steps: int = 60  # LangGraph steps before a run counts as a loop
    agent_temperature: float | None = 0.3  # None leaves the LLM server's default
    thinking_token_budget: int | None = 16384  # reasoning tokens per model call
    max_output_tokens: int | None = 24576  # output tokens per model call, reasoning included
    reason_max_chars: int = 200  # length of a reason the prompts ask for
    summary_max_chars: int = 400  # length of a summary the prompts ask for
    reason_hard_max_chars: int = 300  # where the last attempt cuts a reason
    summary_hard_max_chars: int = 700  # where the last attempt cuts a summary
    max_submit_attempts: int = 3  # submissions per run before over-long texts are cut, not refused
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    yahoo_retry_delays: tuple[float, ...] = (60.0, 180.0)  # seconds before each new attempt at an unusable batch
    min_yahoo_coverage: float = 0.9  # share of the universe a Yahoo batch must cover

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ConfigurationError: Naming every parameter that is out of range.
        """
        parse_rebalance_time(self.rebalance_time)
        problems = {
            "max_etf_weight, max_pod_weight and min_yahoo_coverage must be in (0, 1]": not all(0 < value <= 1 for value in (self.max_etf_weight, self.max_pod_weight, self.min_yahoo_coverage)),
            "max_etf_weight must not exceed max_pod_weight": self.max_etf_weight > self.max_pod_weight,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "rebalance_weekday must be a weekday (0 = Monday ... 4 = Friday)": not 0 <= self.rebalance_weekday <= 4,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "reason_max_chars and summary_max_chars must be at least 1": min(self.reason_max_chars, self.summary_max_chars) < 1,
            "reason_hard_max_chars must be at least reason_max_chars": self.reason_hard_max_chars < self.reason_max_chars,
            "summary_hard_max_chars must be at least summary_max_chars": self.summary_hard_max_chars < self.summary_max_chars,
            "max_submit_attempts must be at least 1": self.max_submit_attempts < 1,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ConfigurationError("SectorPodsParams: " + "; ".join(failed))

    @property
    def reason_limit(self) -> TextLimit:
        """The stated and hard lengths of a reason."""
        return TextLimit(self.reason_max_chars, self.reason_hard_max_chars)

    @property
    def summary_limit(self) -> TextLimit:
        """The stated and hard lengths of a summary, sector view or overrides text."""
        return TextLimit(self.summary_max_chars, self.summary_hard_max_chars)
