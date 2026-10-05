"""`DriftParams`: every threshold of the earnings_drift strategy in one frozen dataclass (spec §9)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DriftParams:
    agent_enabled: bool = True  # False: baseline mode, every gated candidate is bought at default_trail_percent
    max_positions: int = 8
    max_holding_sessions: int = 10  # sessions from the entry fill through the review, both included
    min_trail_percent: float = 3.0  # trailing stop bounds, in percent (8 = 8%), as Alpaca and IBKR take them
    default_trail_percent: float = 8.0
    max_trail_percent: float = 15.0
    min_abnormal_pct: float = 0.03  # reaction-day return minus the benchmark's, as a fraction
    min_hold_ratio: float = 0.5
    min_close_location: float = 0.5
    min_rel_volume: float = 2.0
    min_price: float = 10.0
    min_dollar_volume: float = 20_000_000.0
    volume_baseline_sessions: int = 20
    bars_lookback_sessions: int = 75  # daily bars read per name: the 60-session run-up plus margin
    surprise_lookback_hours: float = 2.0  # news is read from this long before the 8-K's acceptance
    news_symbols_per_call: int = 5
    news_limit: int = 50
    event_lookback_days: int = 10  # older SEC submission pages are read back to the cycle's date minus this
    live_bar_delay_seconds: float = 300.0  # paper/live wait after the close, so the daily bar is final
    cancel_wait_seconds: float = 30.0
    tool_budget_per_item: int = 4  # tool calls per candidate or holding in one agent run (order tools and skip exempt)
    agent_temperature: float | None = 0.3
    max_consecutive_agent_failures: int = 3
    max_consecutive_hollow_scans: int = 3
    sec_hollow_fraction: float = 0.5
    sec_hollow_min_failures: int = 20
    reason_max_chars: int = 300  # a longer reason is cut before it is logged or kept as a thesis

    def __post_init__(self) -> None:
        floats = (
            self.min_trail_percent,
            self.default_trail_percent,
            self.max_trail_percent,
            self.min_abnormal_pct,
            self.min_hold_ratio,
            self.min_close_location,
            self.min_rel_volume,
            self.min_price,
            self.min_dollar_volume,
            self.surprise_lookback_hours,
            self.live_bar_delay_seconds,
            self.cancel_wait_seconds,
            self.sec_hollow_fraction,
        )
        problems = {
            "thresholds must be finite": not all(math.isfinite(value) for value in floats),
            "max_positions must be at least 1": self.max_positions < 1,
            "max_holding_sessions must be at least 1": self.max_holding_sessions < 1,
            "trail percents must satisfy 0 < min <= default <= max <= 100": not (0 < self.min_trail_percent <= self.default_trail_percent <= self.max_trail_percent <= 100),
            "min_hold_ratio must be in [0, 1]": not 0 <= self.min_hold_ratio <= 1,
            "min_close_location must be in [0, 1]": not 0 <= self.min_close_location <= 1,
            "min_rel_volume must be above 0": not self.min_rel_volume > 0,
            "min_price and min_dollar_volume must be at least 0": self.min_price < 0 or self.min_dollar_volume < 0,
            "volume_baseline_sessions must be at least 2": self.volume_baseline_sessions < 2,
            "bars_lookback_sessions must exceed volume_baseline_sessions + 1": self.bars_lookback_sessions <= self.volume_baseline_sessions + 1,
            "surprise_lookback_hours must be at least 0": self.surprise_lookback_hours < 0,
            "news_symbols_per_call and news_limit must be at least 1": self.news_symbols_per_call < 1 or self.news_limit < 1,
            "event_lookback_days must be at least 0": self.event_lookback_days < 0,
            "live_bar_delay_seconds and cancel_wait_seconds must be at least 0": self.live_bar_delay_seconds < 0 or self.cancel_wait_seconds < 0,
            "tool_budget_per_item must be at least 1": self.tool_budget_per_item < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
            "max_consecutive_agent_failures and max_consecutive_hollow_scans must be at least 1": self.max_consecutive_agent_failures < 1 or self.max_consecutive_hollow_scans < 1,
            "sec_hollow_fraction must be in (0, 1]": not 0 < self.sec_hollow_fraction <= 1,
            "sec_hollow_min_failures must be at least 1": self.sec_hollow_min_failures < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("DriftParams: " + "; ".join(failed))
