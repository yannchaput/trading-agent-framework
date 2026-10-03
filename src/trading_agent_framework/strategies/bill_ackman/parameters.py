"""`AckmanParams`: every threshold of the Bill Ackman strategy in one frozen dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from trading_agent_framework.strategies.bill_ackman.screen import ScreenParams


@dataclass(frozen=True, slots=True)
class AckmanParams:
    screen: ScreenParams = field(default_factory=ScreenParams)
    research_top_n: int = 8  # ideas the researcher submits
    max_positions: int = 5  # stocks in the target portfolio
    max_weight: float = 0.35  # largest weight of one stock, as a fraction of portfolio value
    min_weight: float = 0.05  # smallest weight of one stock; must be at least `rebalance_band`
    cash_buffer: float = 0.02  # share of portfolio value never invested (fees, fill drift)
    forced_exit_fails: int = 2  # consecutive `fail` verdicts that force a holding out
    reentry_cooldown_reviews: int = 4  # completed reviews a forced exit stays out of the candidates and the allowed set (0 disables)
    rebalance_band: float = 0.05  # drift, as a fraction of portfolio value, that triggers a trade
    min_trade_pct: float = 0.005  # smallest order, as a fraction of portfolio value
    parking_symbol: str = "SHV"  # where unallocated money goes
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    reason_max_chars: int = 300  # longest `reason` a submit tool accepts
    agent_temperature: float | None = 0.3  # sampling temperature of the three agents; None leaves the LLM server's default

    def __post_init__(self) -> None:
        floats = (self.max_weight, self.min_weight, self.cash_buffer, self.rebalance_band, self.min_trade_pct)
        problems = {
            "research_top_n must be at least 1": self.research_top_n < 1,
            "max_positions must be at least 1": self.max_positions < 1,
            "weights, cash_buffer, rebalance_band and min_trade_pct must be finite": not all(math.isfinite(value) for value in floats),
            "min_weight must be above 0 and at most max_weight, which is at most 1": not 0 < self.min_weight <= self.max_weight <= 1,
            "min_weight must be at least rebalance_band (a chosen stock below the band would never be bought)": self.min_weight < self.rebalance_band,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "forced_exit_fails must be at least 1": self.forced_exit_fails < 1,
            "reentry_cooldown_reviews must be at least 0": self.reentry_cooldown_reviews < 0,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "max_positions x min_weight must fit in 1 - cash_buffer": self.max_positions * self.min_weight > 1 - self.cash_buffer + 1e-9,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("AckmanParams: " + "; ".join(failed))

    @property
    def max_total_weight(self) -> float:
        """The largest sum of stock weights: everything but the cash buffer."""
        return 1 - self.cash_buffer
