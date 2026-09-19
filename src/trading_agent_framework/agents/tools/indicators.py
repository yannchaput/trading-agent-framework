"""Plain typed indicator tool for a LangChain agent: any pandas-ta-classic indicator by name."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.core.indicators import IndicatorRow

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def indicator_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Indicator tool bound to `strategy`."""

    def get_indicator(
        name: str, symbol: str, timestep: str = "day", params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute a pandas-ta-classic indicator (e.g. sma, rsi, bbands) for a symbol."""
        try:
            indicator = getattr(strategy.indicators, name)
            value = indicator(symbol, timestep, **(params or {}))
        except AttributeError as exc:
            return {"error": str(exc)}
        result = value.as_dict() if isinstance(value, IndicatorRow) else value
        return {"indicator": name, "symbol": symbol.upper(), "value": result}

    return [get_indicator]
