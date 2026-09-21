"""Plain typed trading tools for a LangChain agent: submit, cancel, and close orders.

Each docstring is a single line on purpose: it becomes the tool description sent to the model
on every call. Broker/order-validation failures come back as `{"error": ...}` so the model can
correct itself, mirroring `memory/tools.py`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.order import Order
from trading_agent_framework.memory.tools import RunMemo, current_run_id
from trading_agent_framework.utils.errors import BrokerError

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


class _OrdersThisRun:
    """Identifiers of the orders the current agent run placed; only the latest run is kept.

    An order placed in a run is final for that run: a local LLM that submits before it has finished deciding
    used to cancel its own order (dropping the trade, or resubmitting it). Live, a market order may already
    have filled by then, so the cancel fails and the resubmission doubles the position.
    """

    def __init__(self) -> None:
        self._run_id: str | None = None
        self._identifiers: set[str] = set()

    def add(self, order: Order) -> None:
        run_id = current_run_id()
        if run_id is None:
            return
        if run_id != self._run_id:
            self._run_id, self._identifiers = run_id, set()
        self._identifiers.add(order.identifier)

    def __contains__(self, identifier: str) -> bool:
        run_id = current_run_id()
        return run_id is not None and run_id == self._run_id and identifier in self._identifiers


def trading_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Order tools bound to `strategy`."""
    placed = _OrdersThisRun()
    submissions = RunMemo()  # an identical order within one agent run is refused, not placed twice

    def submit_order(
        symbol: str,
        quantity: float,
        side: str,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit a market, limit, stop or stop-limit order; the type follows from the prices given."""
        key = (symbol, side, float(quantity), limit_price, stop_price, time_in_force)
        return submissions.once(
            current_run_id(),
            key,
            lambda: _submit(symbol, quantity, side, limit_price, stop_price, time_in_force),
            on_repeat=lambda first: {
                "error": f"an identical order was already placed in this run ({first['identifier']}); not placed again"
            },
        )

    def _submit(
        symbol: str, quantity: float, side: str, limit_price: float | None, stop_price: float | None, time_in_force: str
    ) -> dict[str, Any]:
        try:
            order = strategy.create_order(
                symbol, quantity, side,
                limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force,
            )
            submitted = strategy.submit_order(order)
        except Exception as exc:  # AlpacaBroker._submit_order re-raises a raw SDK exception on rejection
            return {"error": str(exc)}
        placed.add(submitted)
        return _lean_order(submitted)

    def cancel_order(order_id: str) -> dict[str, Any]:
        """Cancel a tracked order from an earlier run, by its identifier."""
        try:
            order = strategy.get_order(order_id)
        except Exception as exc:  # strategy.get_order can fall through to a raw SDK lookup
            return {"error": f"failed to look up order_id {order_id!r}: {exc}"}
        if order is None:
            return {"error": f"unknown order_id {order_id!r}"}
        if order.identifier in placed:
            return {"error": f"order {order_id!r} was placed in this run and is final for this run; do not cancel it"}
        try:
            strategy.cancel_order(order)
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"identifier": order.identifier, "status": "cancel_requested"}

    def cancel_open_orders() -> dict[str, Any]:
        """Cancel every open order from earlier runs."""
        try:
            strategy.cancel_orders([o for o in strategy.broker.tracker.get_active_orders() if o.identifier not in placed])
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"status": "ok"}

    def close_position(symbol: str, fraction: float = 1.0) -> dict[str, Any]:
        """Close (or partially close) the position in a symbol."""
        try:
            order = strategy.close_position(symbol, fraction)
        except BrokerError as exc:
            return {"error": str(exc)}
        if order is None:
            return {"status": "no position"}
        placed.add(order)
        return _lean_order(order)

    def sell_all() -> dict[str, Any]:
        """Close every open position."""
        try:
            orders = strategy.sell_all()
        except BrokerError as exc:
            return {"error": str(exc)}
        for order in orders:
            placed.add(order)
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
