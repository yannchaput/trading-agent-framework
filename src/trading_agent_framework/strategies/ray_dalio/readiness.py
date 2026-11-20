"""Lumibot's `ORDER_READINESS_REQUIRED` rule: no order before the account, positions and price were read.

Within one agent run (`current_run_id()`), `submit_order(symbol, ...)` runs only if `get_account_balance`
and `get_positions` succeeded since the last accepted order (or the start of the run) and
`get_last_price` succeeded for that symbol in this run. Outside an agent run nothing is gated, as in
`news_builtin/grounding.py`.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from trading_agent_framework.memory.tools import current_run_id

_NAMES = ("get_account_balance", "get_positions", "get_last_price", "submit_order")


@dataclass
class _Checks:
    """What the current run has verified so far."""

    account: bool = False
    positions: bool = False
    priced: set[str] = field(default_factory=set)


def require_readiness_before_orders(
    tools: list[Callable[..., dict[str, Any]]],
) -> list[Callable[..., dict[str, Any]]]:
    """`tools` with `submit_order` refused until the account, positions and the symbol's price were read.

    An accepted order (no `"error"` key) clears the account and positions checks but keeps the prices;
    a refused one clears nothing. The list keeps its order and every tool its name, signature and
    docstring. Raises `ValueError` if any of the four tools is missing.
    """
    by_name = {tool.__name__: tool for tool in tools}
    missing = [name for name in _NAMES if name not in by_name]
    if missing:
        raise ValueError(f"require_readiness_before_orders needs these tools: {', '.join(missing)}")

    state: dict[str, Any] = {"run_id": None, "checks": _Checks()}

    def _checks() -> _Checks | None:
        run_id = current_run_id()
        if run_id is None:
            return None
        if run_id != state["run_id"]:
            state["run_id"], state["checks"] = run_id, _Checks()
        return state["checks"]

    def _symbol(tool: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        try:
            bound = inspect.signature(tool).bind_partial(*args, **kwargs)
        except TypeError:
            return ""
        return str(bound.arguments.get("symbol", "")).upper()

    def _recorder(name: str) -> Callable[..., dict[str, Any]]:
        inner = by_name[name]

        @functools.wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = inner(*args, **kwargs)
            checks = _checks()
            if checks is not None and "error" not in result:
                if name == "get_account_balance":
                    checks.account = True
                elif name == "get_positions":
                    checks.positions = True
                else:
                    checks.priced.add(_symbol(inner, args, kwargs))
            return result

        return wrapped

    submit_order = by_name["submit_order"]

    @functools.wraps(submit_order)
    def guarded_submit_order(*args: Any, **kwargs: Any) -> dict[str, Any]:
        checks = _checks()
        if checks is None:
            return submit_order(*args, **kwargs)
        symbol = _symbol(submit_order, args, kwargs)
        todo = []
        if not checks.account:
            todo.append("get_account_balance")
        if not checks.positions:
            todo.append("get_positions")
        if symbol not in checks.priced:
            todo.append(f"get_last_price(symbol='{symbol}')")
        if todo:
            return {"error": (f"ORDER_READINESS_REQUIRED: before submitting an order, call {', '.join(todo)} in this same agent run.")}
        result = submit_order(*args, **kwargs)
        if "error" not in result:
            checks.account = checks.positions = False
        return result

    replacements: dict[str, Callable[..., dict[str, Any]]] = {
        "get_account_balance": _recorder("get_account_balance"),
        "get_positions": _recorder("get_positions"),
        "get_last_price": _recorder("get_last_price"),
        "submit_order": guarded_submit_order,
    }
    return [replacements.get(tool.__name__, tool) for tool in tools]
