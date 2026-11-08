from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.fakes import FakeRegimeSource

from trading_agent_framework.utils.log import reset_strategy_logging


@pytest.fixture(autouse=True)
def _reset_strategy_logging() -> Iterator[None]:
    """setup_strategy_logging turns off propagation on the package logger, which
    would starve other tests' `caplog`; always undo it after each test.
    """
    yield
    reset_strategy_logging()


@pytest.fixture(autouse=True)
def _no_regime_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `Strategy` gets an empty fake regime source: no test reaches Yahoo for SPY/VIX/VIX3M."""
    monkeypatch.setattr("trading_agent_framework.core.strategy.YahooRegimeData", lambda: FakeRegimeSource())
