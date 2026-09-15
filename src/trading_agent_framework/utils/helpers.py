from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

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


def compute_atr_from_df(df: pd.DataFrame, period: int = 14) -> float | None:
    """Compute Average True Range over `period` bars from a DataFrame with OHLC columns."""
    if len(df) < period + 1:
        return None
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    tr_values = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
    return sum(tr_values[-period:]) / period
