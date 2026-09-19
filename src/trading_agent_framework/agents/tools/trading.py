"""Plain typed trading tools for a LangChain agent: submit, cancel, and close orders.

Each docstring is a single line on purpose: it becomes the tool description sent to the model
on every call. Broker/order-validation failures come back as `{"error": ...}` so the model can
correct itself, mirroring `memory/tools.py`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def _lean_order(order: Order) -> dict[str, Any]:
    lean: dict[str, Any] = {
        "identifier": order.identifier,
        "symbol": order.asset.symbol,
        "side": order.side.value,
        "order_type": order.order_type.value,
        "status": order.status.value,
    }
    if order.quantity is not None:
        lean["quantity"] = float(order.quantity)
    if order.limit_price is not None:
        lean["limit_price"] = float(order.limit_price)
    if order.stop_price is not None:
        lean["stop_price"] = float(order.stop_price)
    if order.filled_quantity:
        lean["filled_quantity"] = float(order.filled_quantity)
    if order.avg_fill_price is not None:
        lean["avg_fill_price"] = float(order.avg_fill_price)
    return lean


def trading_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Order tools bound to `strategy`."""

    def submit_order(
        symbol: str,
        quantity: float,
        side: str,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit a market, limit, stop or stop-limit order; the type follows from the prices given."""
        try:
            order = strategy.create_order(
                symbol, quantity, side,
                limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force,
            )
            submitted = strategy.submit_order(order)
        except (OrderValidationError, BrokerError) as exc:
            return {"error": str(exc)}
        return _lean_order(submitted)

    def cancel_order(order_id: str) -> dict[str, Any]:
        """Cancel a tracked order by its identifier."""
        try:
            order = strategy.get_order(order_id)
        except Exception as exc:  # strategy.get_order can fall through to a raw SDK lookup
            return {"error": f"failed to look up order_id {order_id!r}: {exc}"}
        if order is None:
            return {"error": f"unknown order_id {order_id!r}"}
        try:
            strategy.cancel_order(order)
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"identifier": order.identifier, "status": "cancel_requested"}

    def cancel_open_orders() -> dict[str, Any]:
        """Cancel every open order."""
        try:
            strategy.cancel_open_orders()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"status": "ok"}

    def close_position(symbol: str, fraction: float = 1.0) -> dict[str, Any]:
        """Close (or partially close) the position in a symbol."""
        try:
            order = strategy.close_position(symbol, fraction)
        except BrokerError as exc:
            return {"error": str(exc)}
        return _lean_order(order) if order is not None else {"status": "no position"}

    def sell_all() -> dict[str, Any]:
        """Close every open position."""
        try:
            orders = strategy.sell_all()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"orders": [_lean_order(order) for order in orders]}

    def get_orders() -> dict[str, Any]:
        """List every tracked order."""
        return {"orders": [_lean_order(order) for order in strategy.get_orders()]}

    def get_order(order_id: str) -> dict[str, Any]:
        """Look up one order by its identifier."""
        try:
            order = strategy.get_order(order_id)
        except Exception as exc:  # strategy.get_order can fall through to a raw SDK lookup
            return {"error": f"failed to look up order_id {order_id!r}: {exc}"}
        return _lean_order(order) if order is not None else {"error": f"unknown order_id {order_id!r}"}

    for tool in (submit_order, cancel_order, cancel_open_orders, close_position, sell_all):
        vars(tool)["mutates_trading"] = True

    return [submit_order, cancel_order, cancel_open_orders, close_position, sell_all, get_orders, get_order]
