from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

from trading_agent_framework.config import TradingMode

logger = logging.getLogger(__name__)


def get_thread_capacity() -> int:
    """Return the number of threads to use for parallel processing.

    Returns half the CPU count, with a minimum of 1.
    """
    cpu_count = os.cpu_count() or 1
    dedicated_threads = max(1, cpu_count // 2)
    logger.debug("CPU available: %d, using %d dedicated threads for parallel processing", cpu_count, dedicated_threads)
    return dedicated_threads


def build_logs(strategy_name: str, trading_mode: TradingMode = TradingMode.BACKTESTING) -> Path:
    """
    Helper function to create a log directory for the current run.

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
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("message") != "insufficient buying power":
        return None
    try:
        return float(payload["buying_power"])
    except (KeyError, TypeError, ValueError):
        return None
