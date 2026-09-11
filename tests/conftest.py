from __future__ import annotations

from collections.abc import Iterator

import pytest

from trading_agent_framework.utils.log import reset_strategy_logging


@pytest.fixture(autouse=True)
def _reset_strategy_logging() -> Iterator[None]:
    """setup_strategy_logging turns off propagation on the package logger, which
    would starve other tests' `caplog`; always undo it after each test."""
    yield
    reset_strategy_logging()
