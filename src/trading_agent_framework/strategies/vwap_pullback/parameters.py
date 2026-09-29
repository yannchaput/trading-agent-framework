"""Every threshold, window and risk knob of the vwap_pullback strategy (spec §8), in one frozen dataclass.

Ratios are floats (they multiply bar maths); `risk.py` turns them into `Decimal` before touching money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True, slots=True)
class VwapPullbackParameters:
    # stage 1 (daily)
    stage1_lookback_sessions: int = 70
    min_price: float = 5.0
    atr_pct_band: tuple[float, float] = (0.015, 0.08)
    dollar_volume_percentile: float = 0.60
    stage1_size: int = 150
    beta_lookback_sessions: int = 60
    atr_length: int = 14
    # stage 2 (intraday)
    rvol_baseline_sessions: int = 10
    rvol_min: float = 1.5
    tracked_size: int = 30
    bar_minutes: int = 5
    ema_length: int = 9
    # setups
    impulse_move_atr: float = 0.8
    pullback_min_retrace: float = 0.25
    pullback_max_retrace: float = 0.618
    bearish_body_atr: float = 0.25
    selling_volume_ratio: float = 1.5
    # risk
    stop_buffer_atr: float = 0.1
    r_band_atr: tuple[float, float] = (0.15, 1.0)
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.25
    cash_buffer: float = 0.95
    max_positions: int = 4
    max_daily_loss_pct: float = 0.015
    chase_guard_r: float = 0.3
    entry_limit_atr: float = 0.05
    no_entry_before: time = time(9, 45)
    no_entry_after: time = time(15, 0)
    # agents
    exit_review_minutes: int = 15
    headlines_per_symbol: int = 3
    news_calls_per_run: int = 4
    none_catalyst_min_z: float = 2.0
    tp1_fraction_band: tuple[float, float] = (0.25, 0.5)
    trail_atr_band: tuple[float, float] = (0.5, 2.0)
    # order handling
    cancel_wait_seconds: float = 10.0
    flatten_wait_seconds: float = 60.0
    live_bar_delay_seconds: float = 20.0
