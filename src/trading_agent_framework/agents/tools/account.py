"""Plain typed account tools for a LangChain agent: balances and positions."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def _lean_position(position: Position) -> dict[str, Any]:
    lean: dict[str, Any] = {
        "symbol": position.asset.symbol,
        "quantity": float(position.quantity),
        "side": position.side.value,
    }
    if position.avg_fill_price is not None:
        lean["avg_fill_price"] = float(position.avg_fill_price)
    if position.current_price is not None:
        lean["current_price"] = float(position.current_price)
    if position.market_value is not None:
        lean["market_value"] = float(position.market_value)
    if position.unrealized_pnl is not None:
        lean["unrealized_pnl"] = float(position.unrealized_pnl)
    return lean


def account_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Account tools bound to `strategy`."""

    def get_account_balance() -> dict[str, Any]:
        """Get cash, portfolio value and buying power."""
        try:
            account = strategy.broker.get_account()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
        }

    def get_positions() -> dict[str, Any]:
        """List every open position."""
        try:
            positions = strategy.get_positions()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"positions": [_lean_position(p) for p in positions]}

    def get_position(symbol: str) -> dict[str, Any]:
        """Look up the open position in one symbol."""
        try:
            position = strategy.get_position(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        return _lean_position(position) if position is not None else {"position": None}

    return [get_account_balance, get_positions, get_position]
