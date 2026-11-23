"""Small shared helpers: thread capacity, log directories, fractional quantities, buying-power error parsing, the market-fall signal."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_agent_framework.config import TradingMode

logger = logging.getLogger(__name__)

# Share of the portfolio value `Strategy.spendable_cash` keeps back from every buy (fees, fill drift). Not a parameter.
BUY_BUFFER = Decimal("0.02")

# `in_market_fall` rules, fixed by the 2026-11 "VIX alone" research (docs/superpowers/specs/2026-11-22-*). Not parameters.
FALL_RATIO_ENTER = 0.95  # VIX / VIX3M at or above this is market stress
FALL_VIXY_ENTER = 0.10  # VIXY up this much over FALL_VIXY_SESSIONS sessions is market stress
FALL_VIXY_SESSIONS = 5
FALL_RATIO_CALM = 0.90  # VIX / VIX3M below this is a calm session
FALL_CALM_SESSIONS = 3  # consecutive calm sessions that end a fall


def get_thread_capacity() -> int:
    """Return the number of threads to use for parallel processing.

    Returns half the CPU count, with a minimum of 1.
    """
    cpu_count = os.cpu_count() or 1
    dedicated_threads = max(1, cpu_count // 2)
    logger.debug("CPU available: %d, using %d dedicated threads for parallel processing", cpu_count, dedicated_threads)
    return dedicated_threads


def build_logs(strategy_name: str, trading_mode: TradingMode = TradingMode.BACKTESTING) -> Path:
    """Helper function to create a log directory for the current run.

    Args:
        trading_mode (TradingMode): The trading mode (BACKTESTING, PAPER_TRADING, or LIVE_TRADING).
        strategy_name (str): The name of the strategy being run, used to create a subdirectory for the logs.

    Returns:
        Path: The path to the created log directory.
    """
    run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = Path("logs") / strategy_name / trading_mode.value / f"{run_ts}_{trading_mode.value}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def fractional_qty(value: float, decimals: int = 6) -> float:
    """Floor a quantity to the specified number of decimal places.

    Using floor (not round) guarantees the computed quantity never exceeds
    the dollar budget — a quantity that rounds up could produce a cost
    greater than available cash.

    Args:
        value: Raw quantity computed as dollar_amount / share_price.
        decimals: Number of decimal places to keep. Default 6 (Alpaca
            supports 9; 6 provides a safe margin).

    Returns:
        Floored quantity as a float.
    """
    factor = 10**decimals
    return math.floor(value * factor) / factor


def parse_insufficient_buying_power(error: Exception) -> float | None:
    """Extract the broker's real buying_power from a rejected Alpaca order error.

    Alpaca's APIError.__str__ returns the raw JSON error body, e.g.
    '{"buying_power":"132.45","code":40310000,"message":"insufficient buying power"}'.
    Returns None for any error that isn't this specific rejection shape, so a
    caller can safely try this against any broker exception.

    Args:
        error: The exception raised by strategy.submit_order().

    Returns:
        The broker-reported real buying power as a float, or None.
    """
    try:
        payload = json.loads(str(error))
    except json.JSONDecodeError, TypeError:
        return None
    if not isinstance(payload, dict) or payload.get("message") != "insufficient buying power":
        return None
    try:
        return float(payload["buying_power"])
    except KeyError, TypeError, ValueError:
        return None


def in_market_fall(vix: Sequence[float], vix3m: Sequence[float], vixy: Sequence[float]) -> bool:
    """Whether the market is in a fall (defensive posture) at the next open, by the "VIX alone" rule.

    This indicator does not provide the direction of trading within the day: only the trend on an extended time window.
    It can not be used for intraday trading.

    The rule is the one tested in the 2026-11 research. A fall **starts** on a session whose close shows market
    stress: VIX / VIX3M at or above 0.95, or VIXY up 10% or more over 5 sessions (its close against the close 5
    sessions earlier). It **ends** after 3 consecutive calm sessions (VIX / VIX3M below 0.90); the session that started
    it is not one of the 3, and a session between 0.90 and 0.95 is neither stressed nor calm (it resets the count).
    The signal switched at a median of 1.6% below the SPY high on 2016-2024 and caught all 3 falls of 2025-2026 at 2.0%
    to 2.6%, with about 4 false alarms a year; it returns to normal late (a median of 32 sessions after the trough).

    The function replays the whole history it is given, because the verdict depends on whether a fall began earlier
    and has not yet ended. It assumes the first session is not inside a fall, so pass a long history: an episode lasted
    up to about 90 sessions in the research, 250 is safe. Nothing in the framework trades on this signal.

    Args:
        vix: VIX closes, oldest first; the last value is the most recent completed session.
        vix3m: VIX3M closes, same sessions.
        vixy: VIXY closes, same sessions. NaN (VIXY did not exist before 2011) or fewer than 6 sessions disables the
            VIXY leg for that session; the ratio rule still applies.

    Returns:
        True when the market is in a fall, to be acted on at the next open. The caller must pass only closes that are
        known when it decides: a close from the session being traded would be look-ahead.

    Raises:
        ValueError: If a series is empty, the three series differ in length (a silent misalignment would read the
            wrong sessions), or a close is zero or negative (NaN is read as missing, not rejected).
    """
    n = len(vix)
    if n == 0:
        raise ValueError("in_market_fall needs at least one session of closes")
    if len(vix3m) != n or len(vixy) != n:
        raise ValueError("vix, vix3m and vixy must have the same length")
    if any(close <= 0 for series in (vix, vix3m, vixy) for close in series):
        raise ValueError("closes must be positive (NaN is read as missing)")
    in_fall, calm_sessions = False, 0
    for i in range(n):
        ratio = vix[i] / vix3m[i]
        if not in_fall:
            vixy_rise = vixy[i] / vixy[i - FALL_VIXY_SESSIONS] - 1 if i >= FALL_VIXY_SESSIONS else math.nan
            in_fall, calm_sessions = ratio >= FALL_RATIO_ENTER or vixy_rise >= FALL_VIXY_ENTER, 0
        else:
            calm_sessions = calm_sessions + 1 if ratio < FALL_RATIO_CALM else 0
            in_fall = calm_sessions < FALL_CALM_SESSIONS
    return in_fall
