"""Coloured strategy logging, written to lumibot-style run log files.

`setup_strategy_logging` mirrors lumibot's `Trader(logfile=...)` combined with
the user's `build_logs` helper: one directory per run,
`logs/<strategy>/<mode>/<YYYY-mm-dd_HHMMSS>_<mode>/<mode>.log`.

`ColorLogger` reproduces `WrappingStrategy.log_*`: each message is wrapped in
an ANSI colour code inside the message itself, so the colour is kept in the
log file (read with VS Code's "ANSI Colors" plugin). The escape codes are
hand-written on purpose: `termcolor` disables colour whenever stdout is not a
TTY (nohup, systemd, cron), which would strip it from background runs' files.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from trading_agent_framework.config.env import TradingMode, find_project_root

PACKAGE_LOGGER_NAME = "trading_agent_framework"

ANSI_RESET = "\x1b[0m"
ANSI_GREY = "\x1b[90m"
ANSI_BLUE = "\x1b[34m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED = "\x1b[31m"

LOG_FILE_NAMES: dict[TradingMode, str] = {
    TradingMode.LIVE: "live.log",
    TradingMode.PAPER: "paper.log",
    TradingMode.BACKTESTING: "backtest.log",
}

_NOISY_LOGGERS = ("urllib3", "websockets", "httpx2")
_installed_handlers: list[logging.Handler] = []


class LumibotStyleFormatter(logging.Formatter):
    """lumibot's line format: the source location is shown for WARNING and above only."""

    _short = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s | %(message)s")
    _long = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._long if record.levelno >= logging.WARNING else self._short
        return formatter.format(record)


def setup_strategy_logging(
    strategy_name: str,
    mode: TradingMode,
    *,
    project_root: Path | None = None,
    level: int = logging.INFO,
    started_at: datetime | None = None,
) -> Path:
    """Send the package's logs to the console and a fresh run log file; return its path.

    Idempotent: a second call replaces the handlers installed by the first.
    """
    root = project_root if project_root is not None else find_project_root()
    stamp = (started_at or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    log_dir = root / "logs" / strategy_name / mode.value / f"{stamp}_{mode.value}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAMES[mode]

    reset_strategy_logging()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    formatter = LumibotStyleFormatter()
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        package_logger.addHandler(handler)
        _installed_handlers.append(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


def reset_strategy_logging() -> None:
    """Remove and close the handlers installed by `setup_strategy_logging`."""
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in _installed_handlers:
        package_logger.removeHandler(handler)
        handler.close()
    _installed_handlers.clear()
    package_logger.setLevel(logging.NOTSET)
    package_logger.propagate = True


class ColorLogger:
    """`WrappingStrategy`-style coloured log methods bound to one logger and prefix.

    Each method logs `[<prefix>] <colour>message<reset>` and returns the plain
    message. `stacklevel` works like `logging.Logger.log`'s: 1 attributes the
    record to whoever called the method.
    """

    def __init__(self, logger: logging.Logger, prefix: str) -> None:
        self._logger = logger
        self._prefix = prefix

    def log_debug(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.DEBUG, ANSI_GREY, message, stacklevel)

    def log_info(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.INFO, ANSI_BLUE, message, stacklevel)

    def log_warning(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.WARNING, ANSI_YELLOW, message, stacklevel)

    def log_error(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.ERROR, ANSI_RED, message, stacklevel)

    def log_critical(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.CRITICAL, ANSI_RED, message, stacklevel)

    def _log(self, level: int, color: str, message: object, stacklevel: int) -> str:
        text = str(message)
        # +2 skips this helper and the public log_* method.
        self._logger.log(
            level,
            "[%s] %s%s%s",
            self._prefix,
            color,
            text,
            ANSI_RESET,
            stacklevel=stacklevel + 2,
        )
        return text
