from __future__ import annotations

import pytest

from trading_agent_framework.core.regime import REGIME_LABELS, REGIME_LINE, RegimeParameters, classify_regime

SMALL = RegimeParameters(sma_fast=3, sma_slow=5, vol_window=2, vol_lookback=4)


def test_defaults_need_273_daily_closes() -> None:
    params = RegimeParameters()
    assert (params.sma_fast, params.sma_slow, params.vol_window, params.vol_lookback, params.vol_percentile) == (50, 200, 20, 252, 0.80)
    assert params.min_bars == 273
    assert SMALL.min_bars == 7


def test_the_line_name_and_labels() -> None:
    assert REGIME_LINE == "Regime"
    assert REGIME_LABELS == {1: "bullish", 0: "neutral", -1: "bearish"}


def test_an_uptrend_with_calm_volatility_is_bullish() -> None:
    reading = classify_regime([1, 2, 3, 4, 5, 6, 7], SMALL)
    assert reading is not None
    assert reading.regime == 1 and not reading.stressed
    assert (reading.close, reading.sma_fast, reading.sma_slow) == (7, 6, 5)


def test_a_downtrend_is_bearish() -> None:
    reading = classify_regime([7, 6, 5, 4, 3, 2, 1], SMALL)
    assert reading is not None and reading.regime == -1


def test_a_close_above_the_slow_average_with_the_fast_average_below_is_neutral() -> None:
    reading = classify_regime([5, 5, 10, 10, 1, 1, 8], SMALL)
    assert reading is not None and reading.regime == 0


def test_a_volatility_spike_caps_a_bullish_trend_at_neutral() -> None:
    reading = classify_regime([10, 10.1, 10.2, 10.3, 10.4, 10.5, 13], SMALL)
    assert reading is not None
    assert reading.stressed and reading.vol > reading.vol_threshold
    assert reading.regime == 0


def test_a_volatility_spike_leaves_a_bearish_trend_bearish() -> None:
    reading = classify_regime([10, 9.9, 9.8, 9.7, 9.6, 9.5, 7], SMALL)
    assert reading is not None and reading.stressed
    assert reading.regime == -1


def test_a_flat_volatility_series_is_not_stressed() -> None:
    # Constant daily return: every rolling volatility is equal up to float noise, which must not read as a spike.
    reading = classify_regime([100 * 1.01**i for i in range(7)], SMALL)
    assert reading is not None
    assert not reading.stressed and reading.regime == 1


def test_only_the_last_min_bars_matter_for_the_volatility_window() -> None:
    # A huge move long before the lookback window must not change the reading.
    calm = [100 * 1.01**i for i in range(7)]
    assert classify_regime([1.0, 500.0, *calm], SMALL) == classify_regime(calm, SMALL)


def test_too_few_closes_give_no_reading() -> None:
    assert classify_regime([1, 2, 3, 4, 5, 6], SMALL) is None
    assert classify_regime([], SMALL) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_close_that_is_not_finite_and_positive_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        classify_regime([1, 2, 3, bad, 5, 6, 7], SMALL)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sma_fast": 0},
        {"sma_fast": 200, "sma_slow": 200},
        {"vol_window": 1},
        {"vol_lookback": 0},
        {"vol_percentile": 0.0},
        {"vol_percentile": 1.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RegimeParameters(**kwargs)
