from __future__ import annotations

import subprocess
import sys

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.core.indicators import IndicatorRow, Indicators, default_bars
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset

_CLOSES = [float(c) for c in range(1, 31)]  # 1.0 .. 30.0, steadily rising


def _strategy(closes: list[float] = _CLOSES) -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    broker.bar_frames["SPY"] = make_bars_frame(closes)
    return Strategy(broker), broker


def test_strategy_builds_one_indicators_accessor() -> None:
    strategy, _ = _strategy()

    assert isinstance(strategy.indicators, Indicators)
    assert strategy.indicators is strategy.indicators


def test_single_column_indicators_return_the_latest_value() -> None:
    strategy, _ = _strategy()

    assert strategy.indicators.sma("SPY", length=5) == pytest.approx(28.0)
    assert strategy.indicators.rsi(Asset("SPY"), length=14) == pytest.approx(100.0)


def test_indicators_receive_every_ohlcv_column() -> None:
    strategy, _ = _strategy()

    # bop needs open/high/low/close (pandas-ta-classic's `open_` param -- regression
    # coverage for the open/open_ kwarg mismatch); mfi needs high/low/close/volume.
    # Together every one of the five OHLCV columns is genuinely exercised.
    assert strategy.indicators.bop("SPY") == pytest.approx(0.0, abs=1e-9)
    assert strategy.indicators.mfi("SPY", length=14) == pytest.approx(100.0)


def test_multi_column_indicators_return_an_indicator_row() -> None:
    strategy, _ = _strategy()

    row = strategy.indicators.bbands("SPY", length=20, std=2)

    assert isinstance(row, IndicatorRow)
    assert row.BBM_20_2_0 == pytest.approx(20.5)  # mean of closes 11..30
    assert row["BBM_20_2.0"] == pytest.approx(20.5)
    assert set(row.as_dict()) == {"BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"}


@pytest.mark.parametrize(("kwargs", "expected"), [({"length": 40}, 120), ({"length": 5}, 50), ({}, 150)])
def test_default_bars_is_three_times_the_length_with_a_floor_of_50(
    kwargs: dict[str, int], expected: int
) -> None:
    assert default_bars(kwargs) == expected


def test_the_default_lookback_decides_how_many_bars_are_fetched() -> None:
    strategy, broker = _strategy()

    strategy.indicators.sma("SPY", length=40)
    strategy.indicators.macd("SPY")

    assert [call[1] for call in broker.bars_calls] == [120, 150]


def test_bars_and_include_after_hours_control_the_fetch() -> None:
    strategy, broker = _strategy()

    value = strategy.indicators.sma(
        "SPY", timestep="minute", length=5, bars=10, include_after_hours=False
    )

    assert value == pytest.approx(28.0)  # the last 10 closes still end at 26..30
    assert broker.bars_calls == [(("SPY",), 10, "minute", False)]


def test_too_few_bars_is_none() -> None:
    strategy, _ = _strategy(_CLOSES[:3])

    assert strategy.indicators.sma("SPY", length=5) is None


def test_a_nan_latest_value_is_none() -> None:
    strategy, _ = _strategy()

    slow = strategy.indicators.custom("slow", lambda df: df["close"].rolling(100).mean(), "SPY")

    assert slow is None


def test_no_bars_is_none() -> None:
    strategy, _ = _strategy()

    assert strategy.indicators.sma("AAPL", length=5) is None


def test_custom_indicators_get_the_bars_frame_and_their_kwargs() -> None:
    strategy, _ = _strategy()

    value = strategy.indicators.custom(
        "shifted", lambda df, offset: df["close"] - offset, "SPY", offset=5
    )

    assert value == pytest.approx(25.0)


def test_custom_rejects_a_non_callable() -> None:
    strategy, _ = _strategy()

    with pytest.raises(TypeError, match="callable"):
        strategy.indicators.custom("bad", 42, "SPY")  # ty: ignore[invalid-argument-type]


def test_custom_indicators_must_return_pandas() -> None:
    strategy, _ = _strategy()

    with pytest.raises(TypeError, match="Series or DataFrame"):
        strategy.indicators.custom("scalar", lambda df: 1.0, "SPY")


def test_an_unknown_indicator_name_raises_attribute_error() -> None:
    strategy, _ = _strategy()

    with pytest.raises(AttributeError, match="no indicator named 'not_an_indicator'"):
        strategy.indicators.not_an_indicator  # noqa: B018


def test_indicator_row_normalises_names_and_rejects_unknown_ones() -> None:
    row = IndicatorRow({"BBL_20_2.0": 1.5, "MACD-x": None})

    assert row.BBL_20_2_0 == 1.5
    assert row.MACD_x is None
    with pytest.raises(AttributeError):
        row.missing  # noqa: B018


def test_importing_core_imports_neither_pandas_nor_alpaca() -> None:
    """Subprocess: other tests in this session have already imported both."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.core\n"
            "import sys\n"
            "assert 'pandas' not in sys.modules, 'pandas'\n"
            "assert 'alpaca' not in sys.modules, 'alpaca'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
