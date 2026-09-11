from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import (
    ANSI_BLUE,
    ANSI_GREY,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    PACKAGE_LOGGER_NAME,
    ColorLogger,
    reset_strategy_logging,
    setup_strategy_logging,
)

_STARTED = datetime(2026, 9, 10, 14, 30, 5)


def _setup(tmp_path: Path, **kwargs: object) -> Path:
    return setup_strategy_logging(
        "momentum",
        TradingMode.PAPER,
        project_root=tmp_path,
        started_at=_STARTED,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def _color_logger() -> ColorLogger:
    return ColorLogger(logging.getLogger(f"{PACKAGE_LOGGER_NAME}.tests"), "momentum")


def _last_line(log_file: Path) -> str:
    for handler in logging.getLogger(PACKAGE_LOGGER_NAME).handlers:
        handler.flush()
    return log_file.read_text(encoding="utf-8").splitlines()[-1]


def test_setup_creates_lumibot_style_run_directory(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    expected = tmp_path / "logs" / "momentum" / "paper" / "2026-09-10_143005_paper" / "paper.log"
    assert log_file == expected
    assert log_file.is_file()


@pytest.mark.parametrize(
    ("mode", "file_name"),
    [
        (TradingMode.LIVE, "live.log"),
        (TradingMode.PAPER, "paper.log"),
        (TradingMode.BACKTESTING, "backtest.log"),
    ],
)
def test_log_file_name_per_mode(tmp_path: Path, mode: TradingMode, file_name: str) -> None:
    log_file = setup_strategy_logging("momentum", mode, project_root=tmp_path, started_at=_STARTED)
    assert log_file.name == file_name
    assert log_file.parent.name == f"2026-09-10_143005_{mode.value}"


def test_info_line_keeps_ansi_colour_in_file(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    returned = _color_logger().log_info("hello")
    line = _last_line(log_file)
    assert returned == "hello"
    pattern = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \| INFO \| \[momentum\] "
    assert re.fullmatch(pattern + re.escape(f"{ANSI_BLUE}hello{ANSI_RESET}"), line)


def test_warning_line_names_the_calling_site(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    _color_logger().log_warning("careful")
    line = _last_line(log_file)
    assert "| WARNING | test_log.py:test_warning_line_names_the_calling_site:" in line
    assert line.endswith(f"[momentum] {ANSI_YELLOW}careful{ANSI_RESET}")


@pytest.mark.parametrize(
    ("method", "level", "color"),
    [
        ("log_debug", "DEBUG", ANSI_GREY),
        ("log_info", "INFO", ANSI_BLUE),
        ("log_warning", "WARNING", ANSI_YELLOW),
        ("log_error", "ERROR", ANSI_RED),
        ("log_critical", "CRITICAL", ANSI_RED),
    ],
)
def test_each_level_uses_its_colour(tmp_path: Path, method: str, level: str, color: str) -> None:
    log_file = _setup(tmp_path, level=logging.DEBUG)
    getattr(_color_logger(), method)("msg")
    line = _last_line(log_file)
    assert f"| {level} |" in line
    assert line.endswith(f"{color}msg{ANSI_RESET}")


def test_debug_is_dropped_at_info_level(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    _color_logger().log_debug("hidden")
    _color_logger().log_info("shown")
    _last_line(log_file)
    content = log_file.read_text(encoding="utf-8")
    assert "hidden" not in content
    assert "shown" in content


def test_setup_is_idempotent_and_closes_previous_file(tmp_path: Path) -> None:
    first = _setup(tmp_path)
    second = setup_strategy_logging(
        "momentum", TradingMode.PAPER, project_root=tmp_path, started_at=datetime(2026, 9, 10, 15)
    )
    assert len(logging.getLogger(PACKAGE_LOGGER_NAME).handlers) == 2
    _color_logger().log_info("only in second")
    _last_line(second)
    assert "only in second" not in first.read_text(encoding="utf-8")
    assert "only in second" in second.read_text(encoding="utf-8")


def test_setup_isolates_package_logger_and_quiets_noisy_libraries(tmp_path: Path) -> None:
    _setup(tmp_path)
    assert logging.getLogger(PACKAGE_LOGGER_NAME).propagate is False
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("websockets").level == logging.WARNING


def test_reset_restores_default_logger_state(tmp_path: Path) -> None:
    _setup(tmp_path)
    reset_strategy_logging()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    assert package_logger.propagate is True
    assert package_logger.handlers == []
    assert package_logger.level == logging.NOTSET
