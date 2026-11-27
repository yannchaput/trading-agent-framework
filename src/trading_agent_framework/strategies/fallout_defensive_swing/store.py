"""What fallout_defensive_swing remembers between ticks and restarts: its `SwingState`."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore
from trading_agent_framework.strategies.fallout_defensive_swing.rule import Phase, SwingState
from trading_agent_framework.utils.log import ColorLogger

STATE_VERSION = 1


def _amount(value: Any) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:  # an ArithmeticError, which the store does not catch
        raise InvalidState from exc
    if not amount.is_finite() or amount <= 0:
        raise InvalidState
    return amount


def _decode(raw: dict[str, Any]) -> SwingState:
    phase, basis, last_sale = Phase(raw["phase"]), raw["basis"], raw["last_sale_session"]
    state = SwingState(
        phase=phase,
        basis=None if basis is None else _amount(basis),
        sale_prices={symbol: _amount(price) for symbol, price in raw["sale_prices"].items()},
        last_sale_session=None if last_sale is None else date.fromisoformat(last_sale),
        selling=raw["selling"],
    )
    # the two things `decide` relies on
    if not isinstance(state.selling, bool) or (phase is Phase.HOLDING and state.basis is None) or (phase is Phase.CASH and not state.sale_prices):
        raise InvalidState
    return state


class StateStore(JsonStateStore[SwingState]):
    """JSON store of the `SwingState`; amounts and dates are saved as strings."""

    def __init__(self, path: Path, *, log: ColorLogger, fresh: bool = False) -> None:
        """Open the store.

        Args:
            path: State file.
            log: Logger for warnings about an unreadable or invalid file.
            fresh: Delete any existing file first, so a backtest never inherits the last run's state.
        """
        # `atomic_write_json` writes Decimal and date through `str`, which `_decode` reads back
        super().__init__(path, version=STATE_VERSION, empty=SwingState, decode=_decode, encode=asdict, log=log, fresh=fresh)
