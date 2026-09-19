"""Plain typed macro tool for a LangChain agent: FRED time series, gated on the strategy clock."""

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, cast

from trading_agent_framework.config.env import FredCredentials
from trading_agent_framework.utils.errors import MacroDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 250


class FredSeriesClient(Protocol):
    """Documented interface for what this module needs from a FRED client.

    Not a literal match to `fredapi.Fred.get_series`'s real signature (`get_series(self,
    series_id, observation_start=None, observation_end=None, **kwargs)`) -- the real method
    accepts `realtime_start`/`realtime_end` only via `**kwargs`, with no `None` filtering, so
    passing `None` for either gets URL-encoded literally and FRED rejects it. See
    `_fetch_series` below, which always pins both to a concrete date.
    """

    def get_series(
        self,
        series_id: str,
        observation_start: object = None,
        observation_end: object = None,
        realtime_start: object = None,
        realtime_end: object = None,
    ) -> Any: ...


def _default_fred_client() -> FredSeriesClient:
    import fredapi  # deferred: a strategy that never wires in the macro tool doesn't pay for it

    return cast(FredSeriesClient, fredapi.Fred(api_key=FredCredentials.from_env().api_key))


def _fetch_series(client: FredSeriesClient, series_id: str, start_date: str | None, cutoff: date) -> Any:
    try:
        # Pin the vintage to a closed [cutoff, cutoff] interval. `fredapi.Fred.get_series`
        # takes `realtime_start`/`realtime_end` only via **kwargs with no None-filtering, so
        # passing `realtime_start=None` gets URL-encoded literally as "None" and FRED 400s.
        # Omitting it entirely isn't safe either: FRED then defaults realtime_start to today,
        # and rejects realtime_start > realtime_end -- which a backtest's past `cutoff` would
        # trigger. Pinning both to `cutoff` avoids both failure modes and returns the series
        # as it was known on that date (no future revisions leaking into a backtest).
        return client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=cutoff,
            realtime_start=cutoff,
            realtime_end=cutoff,
        )
    except Exception as exc:
        raise MacroDataError(f"Failed to fetch FRED series {series_id!r}: {exc}") from exc


def macro_tools(
    strategy: "Strategy", *, fred_client_factory: Callable[[], FredSeriesClient] | None = None  # noqa: UP037
) -> list[Callable[..., dict[str, Any]]]:
    """FRED macro tool bound to `strategy`."""
    if fred_client_factory is None:
        # Fail fast at wiring time -- consistent with `fundamentals_tools`, which validates
        # its own credentials (SEC_EDGAR_USER_AGENT) at construction time rather than
        # deferring to the first tool call. A caller who supplies their own factory (e.g. a
        # test double) doesn't need FRED_API_KEY, so this check only applies to the default.
        FredCredentials.from_env()
        fred_client_factory = _default_fred_client

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
