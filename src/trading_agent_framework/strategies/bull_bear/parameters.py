"""`BullBearParams`: every threshold of the bull_bear strategy in one frozen dataclass.

The momentum parameters (score weights, skip, filters) are NOT here: they are cross_momentum's `CONFIG`, so the
debated ranking is cross_momentum's by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.utils.errors import ConfigurationError

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class BullBearParams:
    retention_rank: int = 35  # a holding ranked worse (or unranked) is a forced exit; above debate_size
    book_size: int = 10  # stocks held after every completed review
    max_vetoes: int = 5  # most stocks the judge may veto in one review
    min_weight: float = 0.04  # per-stock weight bounds, fractions of portfolio value
    max_weight: float = 0.20
    cash_buffer: float = 0.02  # never invested (fees, fill drift)
    rebalance_band: float = 0.03  # drift that triggers a trade
    min_trade_pct: float = 0.005  # smallest order
    parking_symbol: str = "SHV"  # where unspent money goes
    research_tool_budget: int = 3  # tool calls per researcher run; submit_note is exempt
    note_max_chars: int = 500
    argument_max_chars: int = 300
    reason_max_chars: int = 300
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    agent_temperature: float | None = 0.3  # all four agents; None leaves the LLM server's default
    rebalance_time: str = "12:00"  # market time (ET) of the weekly review
    rebalance_weekday: int = 1  # 0 = Monday ... 4 = Friday; 1 = Tuesday, as cross_momentum
    yahoo_retry_delays: tuple[float, ...] = (60.0, 180.0)  # seconds before each new attempt at an unusable Yahoo batch
    min_yahoo_coverage: float = 0.5  # share of the universe a Yahoo batch must cover

    def __post_init__(self) -> None:
        floats = (self.min_weight, self.max_weight, self.cash_buffer, self.rebalance_band, self.min_trade_pct, self.min_yahoo_coverage)
        investable = 1 - self.cash_buffer
        try:
            parse_rebalance_time(self.rebalance_time)
            time_problem = ""
        except ConfigurationError as exc:
            time_problem = str(exc)
        problems = {
            "weights, cash_buffer, rebalance_band, min_trade_pct and min_yahoo_coverage must be finite": not all(math.isfinite(value) for value in floats),
            "book_size and max_vetoes must be at least 1": self.book_size < 1 or self.max_vetoes < 1,
            "retention_rank must be above debate_size (book_size + max_vetoes)": self.retention_rank <= self.book_size + self.max_vetoes,
            "book_size x min_weight must fit in 1 - cash_buffer": self.book_size * self.min_weight > investable + _EPS,
            "book_size x max_weight must reach 1 - cash_buffer (the book is always fully invested)": self.book_size * self.max_weight < investable - _EPS,
            "min_weight must be above 0 and at most max_weight, which is at most 1": not 0 < self.min_weight <= self.max_weight <= 1,
            "min_weight must be above rebalance_band (a new stock below the band would never be bought)": self.min_weight <= self.rebalance_band,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "research_tool_budget must be at least 1": self.research_tool_budget < 1,
            "note_max_chars, argument_max_chars and reason_max_chars must be at least 1": min(self.note_max_chars, self.argument_max_chars, self.reason_max_chars) < 1,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
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
    def debate_size(self) -> int:
        """The top ranks debated: enough for `book_size` stocks to survive `max_vetoes` vetoes."""
        return self.book_size + self.max_vetoes
