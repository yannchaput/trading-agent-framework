from __future__ import annotations

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import ConfigurationError


class _FakeFred:
    def __init__(self, series: pd.Series, calls: list[dict[str, object]]) -> None:
        self._series = series
        self._calls = calls

    def get_series(
        self,
        series_id: str,
        observation_start: object = None,
        observation_end: object = None,
        realtime_start: object = None,
        realtime_end: object = None,
    ) -> object:
        self._calls.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
            }
        )
        return self._series


def _strategy() -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))


def _tool(series: pd.Series, calls: list[dict[str, object]]):
    strategy = _strategy()
    [tool] = macro_tools(strategy, fred_client_factory=lambda: _FakeFred(series, calls))
    return tool, strategy


def test_get_fred_series_returns_observations() -> None:
    series = pd.Series(
        [21000.0, 21050.0],
        index=pd.to_datetime(["2026-08-01", "2026-09-01"]),
    )
    calls: list[dict[str, object]] = []
    tool, _ = _tool(series, calls)

    result = tool("M2SL")

    assert result["series_id"] == "M2SL"
    assert result["observations"] == [
        {"date": "2026-08-01", "value": 21000.0},
        {"date": "2026-09-01", "value": 21050.0},
    ]


def test_get_fred_series_pins_realtime_and_observation_end_to_the_clock() -> None:
    calls: list[dict[str, object]] = []
    tool, strategy = _tool(pd.Series(dtype=float), calls)

    tool("FEDFUNDS", start_date="2020-01-01")

    [call] = calls
    cutoff = strategy.clock.now().date()
    assert call["observation_start"] == "2020-01-01"
    assert call["observation_end"] == cutoff
    assert call["realtime_end"] == cutoff


def test_get_fred_series_pins_realtime_start_to_the_cutoff_not_none() -> None:
    """Regression test for the FRED 400 bug: `fredapi.Fred.get_series` only declares
    `observation_start`/`observation_end`; `realtime_start`/`realtime_end` fall into its
    `**kwargs` and get URL-encoded with no `None` filtering, so `realtime_start=None`
    becomes a literal "&realtime_start=None" query param that FRED rejects. The tool must
    always pass a concrete date for `realtime_start`, never `None`."""
    calls: list[dict[str, object]] = []
    tool, strategy = _tool(pd.Series(dtype=float), calls)

    tool("FEDFUNDS")

    [call] = calls
    cutoff = strategy.clock.now().date()
    assert call["realtime_start"] == cutoff
    assert call["realtime_start"] is not None


def test_get_fred_series_clamps_limit_to_the_last_n_observations() -> None:
    series = pd.Series([float(i) for i in range(10)], index=pd.date_range("2026-01-01", periods=10, freq="D"))
    calls: list[dict[str, object]] = []
    tool, _ = _tool(series, calls)

    result = tool("CPIAUCSL", limit=3)

    assert len(result["observations"]) == 3
    assert result["observations"][-1]["value"] == 9.0


def test_macro_tools_raises_configuration_error_eagerly_without_fred_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: `macro_tools` must fail fast at wiring time (like `fundamentals_tools`
    does for SEC_EDGAR_USER_AGENT) rather than deferring a missing FRED_API_KEY to the first
    `get_fred_series(...)` call, where a raised `ConfigurationError` would otherwise crash
    `on_trading_iteration()` mid-run instead of surfacing at strategy setup."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    strategy = _strategy()

    with pytest.raises(ConfigurationError, match="FRED_API_KEY"):
        macro_tools(strategy)


def test_get_fred_series_returns_an_error_dict_on_failure() -> None:
    class _Boom:
        def get_series(
            self,
            series_id: str,
            observation_start: object = None,
            observation_end: object = None,
            realtime_start: object = None,
            realtime_end: object = None,
        ) -> object:
            raise RuntimeError("network down")

    strategy = _strategy()
    [tool] = macro_tools(strategy, fred_client_factory=lambda: _Boom())

    result = tool("M2SL")

    assert "error" in result
    assert "M2SL" in result["error"]
