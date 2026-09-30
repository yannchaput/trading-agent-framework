"""Every threshold, window and risk knob of the vwap_pullback strategy (spec §8), in one frozen dataclass.

Ratios are floats (they multiply bar maths); `risk.py` turns them into `Decimal` before touching money.
Nothing else in the package hard-codes a threshold: the prompts, the screens, the state machine and the
desk all read them from here, so tuning happens in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True, slots=True)
class VwapPullbackParameters:
    """Tunable settings, grouped by the pipeline stage that reads them.

    "ATR" below always means the symbol's daily ATR(`atr_length`) unless a field says "5-minute ATR".
    "R" is a trade's initial risk per share: trigger close minus protective stop.
    """

    # --- stage 1 (daily bars, once per session before the open; `screening.select_stage1`) ---
    stage1_lookback_sessions: int = 70  # daily bars fetched per symbol (ATR, 20-day momentum, 60-day beta)
    min_price: float = 5.0  # last close floor, in dollars (no penny stocks)
    atr_pct_band: tuple[float, float] = (0.015, 0.08)  # ATR / close must fall in [low, high]: moves enough, not a lottery ticket
    dollar_volume_percentile: float = 0.60  # keep the top 60% by 20-day dollar volume (a percentile: IEX volume is a slice of the tape)
    stage1_size: int = 150  # survivors kept for the intraday scan
    beta_lookback_sessions: int = 60  # daily returns used for beta to SPY (feeds the beta-adjusted RS)
    atr_length: int = 14  # ATR window, for daily ATR and the 5-minute bar ATR alike

    # --- stage 2 (intraday, every tick; `screening.rank_stage2`) ---
    rvol_baseline_sessions: int = 10  # prior sessions averaged into the per-minute RVOL baseline
    rvol_min: float = 1.5  # RVOL floor to be ranked at all (cumulative volume vs the baseline at the same minute)
    tracked_size: int = 30  # candidates tracked by the setup state machine each tick
    bar_minutes: int = 5  # bar size the state machine and the agents reason on
    ema_length: int = 9  # EMA over those bars, a trailing reference for the exit agent

    # --- setup state machine (`setups.py`) ---
    impulse_move_atr: float = 0.8  # WATCH -> IMPULSE once session high - open >= this many ATRs
    pullback_min_retrace: float = 0.25  # IMPULSE -> PULLBACK once 25% of the impulse leg is given back
    pullback_max_retrace: float = 0.618  # BROKEN beyond this retracement (the move has failed)
    bearish_body_atr: float = 0.25  # BROKEN on a red bar whose body exceeds this many ATRs (plan deviation 2)
    selling_volume_ratio: float = 1.5  # BROKEN on a red bar with volume above this multiple of the impulse average

    # --- risk and sizing (`risk.py`, enforced by `desk.Desk`) ---
    stop_buffer_atr: float = 0.1  # protective stop = pullback low - this many ATRs
    r_band_atr: tuple[float, float] = (0.15, 1.0)  # R must lie in [low, high] ATRs: not noise-tight, not meaningless
    risk_per_trade: float = 0.005  # equity fraction lost if the stop is hit (0.5%)
    max_position_pct: float = 0.50  # cap on one position's value, as a fraction of equity (0.25 left most of the risk budget unused)
    cash_buffer: float = 0.95  # spend at most 95% of available cash (fees, price drift)
    max_positions: int = 4  # concurrent open trades + pending entries
    max_daily_loss_pct: float = 0.015  # circuit breaker: no new entries once session P&L <= -1.5% of opening equity
    chase_guard_r: float = 0.3  # refuse an entry when price already ran more than 0.3R past the trigger close
    entry_limit_atr: float = 0.05  # marketable limit = last price + this many ATRs
    no_entry_before: time = time(9, 45)  # market time; the open's price discovery is skipped
    no_entry_after: time = time(15, 0)  # market time; late entries have no room before the 15:50 flatten

    # --- agents (`prompts.py`, `tools.py`, `desk.Desk.exit_review_due`) ---
    exit_review_minutes: int = 15  # an open trade is reviewed at least this often, event or not
    headlines_per_symbol: int = 3  # headlines attached to a setup or trade row
    news_calls_per_run: int = 4  # search_news budget per agent run (local models loop on search tools otherwise)
    bars_calls_per_run: int = 4  # get_intraday_bars budget per agent run (14 parallel fetches once overflowed the 32k context)
    max_passes_per_symbol: int = 2  # after this many passes a symbol's triggers no longer reach the entry agent (same setup, same verdict)
    tp1_fraction_band: tuple[float, float] = (0.25, 0.5)  # share of the position take_partial_profit may sell
    trail_atr_band: tuple[float, float] = (0.5, 2.0)  # trailing distance in 5-minute ATRs allowed to the exit agent

    # --- order handling (`desk.Desk`, strategy) ---
    cancel_wait_seconds: float = 10.0  # how long a stop hand-off waits for a cancel to be confirmed
    flatten_wait_seconds: float = 60.0  # live: how long the 15:50 flatten waits for its sells before re-checking
    live_bar_delay_seconds: float = 20.0  # live: wait at each tick so the last minute bar is published (plan deviation 7)
