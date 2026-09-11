"""Technical indicators computed with pandas-ta-classic over the strategy's own bars.

`strategy.indicators.sma(asset, length=200)` fetches recent bars through
`strategy.get_historical_prices`, runs the pandas-ta-classic function of the same name, and
returns its latest value. There is no cache and no look-ahead guard: every call re-fetches,
and the latest bar is the current one (live and paper trading only, for now).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.asset import Asset

if TYPE_CHECKING:
    import pandas as pd

    from trading_agent_framework.core.strategy import Strategy

_OHLCV = ("open", "high", "low", "close", "volume")
_MIN_BARS = 50
_BARS_PER_LENGTH = 3


def default_bars(kwargs: Mapping[str, Any]) -> int:
    """Bars fetched when the caller passes no `bars`: 3x the indicator's own `length`
    (warm-up for smoothed indicators), never fewer than 50."""
    return max(_MIN_BARS, int(kwargs.get("length", _MIN_BARS)) * _BARS_PER_LENGTH)


class IndicatorRow:
    """Latest row of a multi-column indicator: `row.BBL_20_2_0` or `row["BBL_20_2.0"]`."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, float | None]) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> float | None:
        if name.startswith("_"):
            raise AttributeError(name)
        for key, value in self._values.items():
            if name in (key, key.replace(".", "_").replace("-", "_")):
                return value
        raise AttributeError(f"no column {name!r}; columns are {list(self._values)}")

    def __getitem__(self, key: str) -> float | None:
        return self._values[key]

    def as_dict(self) -> dict[str, float | None]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"IndicatorRow({self._values!r})"


type IndicatorValue = float | IndicatorRow | None


class Indicators:
    """`strategy.indicators`: any pandas-ta-classic indicator by name, plus `custom`."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def __getattr__(self, name: str) -> Callable[..., IndicatorValue]:
        if name.startswith("_"):
            raise AttributeError(name)
        function = getattr(_pandas_ta(), name, None)
        if not callable(function):
            raise AttributeError(
                f"pandas_ta_classic has no indicator named {name!r}; "
                "use indicators.custom(name, fn, asset, ...) for your own"
            )

        def compute(df: pd.DataFrame, **kwargs: Any) -> object:
            return function(**{column: df[column] for column in _OHLCV}, **kwargs)

        def indicator(asset: Asset | str, timestep: str = "day", **kwargs: Any) -> IndicatorValue:
            return self._evaluate(name, compute, asset, timestep, kwargs)

        indicator.__name__ = name
        return indicator

    def custom(
        self,
        name: str,
        fn: Callable[..., object],
        asset: Asset | str,
        timestep: str = "day",
        **kwargs: Any,
    ) -> IndicatorValue:
        """Run `fn(df, **kwargs) -> Series | DataFrame` over the fetched bars."""
        if not callable(fn):
            raise TypeError(f"custom indicator {name!r}: fn must be callable, got {type(fn).__name__}")
        return self._evaluate(name, fn, asset, timestep, kwargs)

    def _evaluate(
        self,
        name: str,
        compute: Callable[..., object],
        asset: Asset | str,
        timestep: str,
        kwargs: Mapping[str, Any],
    ) -> IndicatorValue:
        options = dict(kwargs)
        count = options.pop("bars", None)
        include_after_hours = options.pop("include_after_hours", True)
        if count is None:
            count = default_bars(options)
        bars = self._strategy.get_historical_prices(
            asset, count, timestep, include_after_hours=include_after_hours
        )
        if bars is None or bars.df.empty:
            return None
        return _latest(name, compute(bars.df, **options))


def _pandas_ta() -> Any:
    import pandas_ta_classic  # deferred: keeps `import trading_agent_framework.core` light

    return pandas_ta_classic


def _latest(name: str, result: object) -> IndicatorValue:
    import pandas as pd

    if result is None:  # pandas-ta returns None when there are too few bars
        return None
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        return IndicatorRow({str(column): _number(v) for column, v in result.iloc[-1].items()})
    if isinstance(result, pd.Series):
        return None if result.empty else _number(result.iloc[-1])
    raise TypeError(
        f"indicator {name!r} returned {type(result).__name__}; expected a pandas Series or DataFrame"
    )


def _number(value: object) -> float | None:
    import pandas as pd

    return None if pd.isna(value) else float(value)  # ty: ignore[invalid-argument-type]
