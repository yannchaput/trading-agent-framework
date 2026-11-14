"""`SectorPodsParams`: every threshold of the sector_pods strategy in one frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass

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
        reason_max_chars: Longest submitted reason; a longer one is trimmed.
        summary_max_chars: Longest submitted summary, sector view or overrides text; a longer one is trimmed.
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
    max_agent_steps: int = 40  # LangGraph steps before a run counts as a loop
    agent_temperature: float | None = 0.3  # None leaves the LLM server's default
    thinking_token_budget: int | None = 16384  # reasoning tokens per model call
    max_output_tokens: int | None = 24576  # output tokens per model call, reasoning included
    reason_max_chars: int = 200  # longest submitted reason
    summary_max_chars: int = 400  # longest submitted summary
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
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ConfigurationError("SectorPodsParams: " + "; ".join(failed))
