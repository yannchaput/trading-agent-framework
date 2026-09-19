"""Plain typed macro tool for a LangChain agent: FRED time series, gated on the strategy clock."""

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.config.env import FredCredentials
from trading_agent_framework.utils.errors import MacroDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 250


class FredSeriesClient(Protocol):
    def get_series(self, series_id: str, **kwargs: object) -> Any: ...


def _default_fred_client() -> FredSeriesClient:
    import fredapi  # deferred: a strategy that never wires in the macro tool doesn't pay for it

    return fredapi.Fred(api_key=FredCredentials.from_env().api_key)


def _fetch_series(client: FredSeriesClient, series_id: str, start_date: str | None, cutoff: date) -> Any:
    try:
        return client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=cutoff,
            realtime_end=cutoff,
        )
    except Exception as exc:
        raise MacroDataError(f"Failed to fetch FRED series {series_id!r}: {exc}") from exc


def macro_tools(
    strategy: "Strategy", *, fred_client_factory: Callable[[], FredSeriesClient] = _default_fred_client  # noqa: UP037
) -> list[Callable[..., dict[str, Any]]]:
    """FRED macro tool bound to `strategy`."""

    def get_fred_series(series_id: str, start_date: str | None = None, limit: int = 60) -> dict[str, Any]:
        """Fetch a FRED macro time series (e.g. M2SL, FEDFUNDS, CPIAUCSL, UNRATE) up to the current date."""
        clamped_limit = min(max(int(limit), MIN_LIMIT), MAX_LIMIT)
        cutoff = strategy.clock.now().date()
        try:
            series = _fetch_series(fred_client_factory(), series_id, start_date, cutoff)
        except MacroDataError as exc:
            return {"error": str(exc)}
        observations = [
            {"date": index.date().isoformat() if hasattr(index, "date") else str(index), "value": float(value)}
            for index, value in series.items()
        ]
        return {
            "series_id": series_id,
            "as_of": cutoff.isoformat(),
            "observations": observations[-clamped_limit:],
        }

    return [get_fred_series]
