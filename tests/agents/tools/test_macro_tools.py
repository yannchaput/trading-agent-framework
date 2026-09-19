from __future__ import annotations

import pandas as pd
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.core.strategy import Strategy


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


def test_get_fred_series_clamps_limit_to_the_last_n_observations() -> None:
    series = pd.Series([float(i) for i in range(10)], index=pd.date_range("2026-01-01", periods=10, freq="D"))
    calls: list[dict[str, object]] = []
    tool, _ = _tool(series, calls)

    result = tool("CPIAUCSL", limit=3)

    assert len(result["observations"]) == 3
    assert result["observations"][-1]["value"] == 9.0


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
