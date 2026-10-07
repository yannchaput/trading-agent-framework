"""`CongressParams`: every threshold of the congress_trades strategy in one frozen dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CongressParams:
    politician: str = "Nancy Pelosi"  # whose disclosures are followed: the exact first and last name as the Clerk lists them
    max_holdings: int = 20  # holdings the research agent may submit
    max_positions: int = 15  # stocks in the target portfolio
    max_total_weight: float = 0.95  # largest sum of stock weights, as a fraction of portfolio value; the rest stays cash
    max_position_weight: float = 0.15  # largest weight of one stock
    min_weight: float = 0.01  # smallest weight of one stock; must be at least `rebalance_band`
    rebalance_band: float = 0.01  # drift, as a fraction of portfolio value, below which a position is left alone
    min_trade_pct: float = 0.005  # smallest order, as a fraction of portfolio value
    max_trade_retries: int = 3  # later ticks that re-run the trading stage after orders were left unfilled (0 disables)
    max_consecutive_abandoned: int = 3  # abandoned runs in a row that abort a backtest
    reason_max_chars: int = 300  # longest `reason` a submit tool accepts
    agent_temperature: float | None = 0.3  # sampling temperature of the three agents; None leaves the LLM server's default
    order_wait_seconds: float = 60.0  # how long `check_orders` waits for the orders to reach a final state

    def __post_init__(self) -> None:
        floats = (self.max_total_weight, self.max_position_weight, self.min_weight, self.rebalance_band, self.min_trade_pct, self.order_wait_seconds)
        finite = all(math.isfinite(value) for value in floats)
        problems = {
            "politician must not be blank": not self.politician.strip(),
            "max_holdings must be at least 1": self.max_holdings < 1,
            "max_positions must be at least 1": self.max_positions < 1,
            "weights, rebalance_band, min_trade_pct and order_wait_seconds must be finite": not finite,
            "max_total_weight must be in (0, 1]": finite and not 0 < self.max_total_weight <= 1,
            "min_weight must be above 0 and at most max_position_weight, which is at most max_total_weight": finite and not 0 < self.min_weight <= self.max_position_weight <= self.max_total_weight,
            "min_weight must be at least rebalance_band (a chosen stock below the band would never be bought)": finite and self.min_weight < self.rebalance_band,
            "rebalance_band must be in (0, 1)": finite and not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": finite and not 0 <= self.min_trade_pct < 1,
            "order_wait_seconds must be above 0": finite and self.order_wait_seconds <= 0,
            "max_trade_retries must be at least 0": self.max_trade_retries < 0,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
            "max_positions x min_weight must fit in max_total_weight": finite and self.max_positions * self.min_weight > self.max_total_weight + 1e-9,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("CongressParams: " + "; ".join(failed))
