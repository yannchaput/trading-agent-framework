"""The parking sleeve: GLD and IEF each hold their share while they trend, SHV holds the rest."""

import math

import pytest

from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.utils import sleeve_symbols, sleeve_weights, trend_reading

ASSETS = ("GLD", "IEF")
RISING, FALLING = [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]  # last close above / below the 3-day SMA (2.0)


def _weights(closes_by_asset, assets=ASSETS):
    return sleeve_weights(closes_by_asset, assets, 3, "SHV")


def test_the_config_names_the_sleeve():
    parking = CONFIG["parking"]
    assert parking["symbol"] == "SHV"
    assert parking["trend_assets"] == ("GLD", "IEF")
    assert parking["trend_sma_window"] == 200
    assert parking["min_trade_pct"] == 0.01


def test_sleeve_symbols_are_the_fallback_then_the_trend_assets():
    assert sleeve_symbols({"symbol": "SHV", "trend_assets": ("GLD", "IEF")}) == ("SHV", "GLD", "IEF")


def test_trend_reading_is_the_last_close_and_the_sma_of_the_last_window_closes():
    assert trend_reading(RISING, 3) == (3.0, 2.0)
    assert trend_reading([100.0, *RISING], 3) == (3.0, 2.0)


@pytest.mark.parametrize("closes", [[1.0, 2.0], [], [1.0, 2.0, math.nan], [math.inf, 2.0, 3.0]])
def test_trend_reading_is_none_when_unusable(closes):
    assert trend_reading(closes, 3) is None


def test_both_trending_split_the_sleeve():
    assert _weights({"GLD": RISING, "IEF": RISING}) == {"SHV": 0.0, "GLD": 0.5, "IEF": 0.5}


def test_one_trending_takes_its_half_and_shv_the_other():
    assert _weights({"GLD": RISING, "IEF": FALLING}) == {"SHV": 0.5, "GLD": 0.5, "IEF": 0.0}


def test_none_trending_is_all_shv():
    assert _weights({"GLD": FALLING, "IEF": FALLING}) == {"SHV": 1.0, "GLD": 0.0, "IEF": 0.0}


def test_a_close_equal_to_its_sma_is_off():
    assert _weights({"GLD": [2.0, 2.0, 2.0], "IEF": RISING}) == {"SHV": 0.5, "GLD": 0.0, "IEF": 0.5}


@pytest.mark.parametrize("gld", [None, [2.0, 3.0], [1.0, 2.0, math.nan]])
def test_missing_short_or_nan_history_is_off(gld):
    closes = {"IEF": RISING} if gld is None else {"GLD": gld, "IEF": RISING}

    assert _weights(closes) == {"SHV": 0.5, "GLD": 0.0, "IEF": 0.5}


def test_no_trend_assets_is_all_shv():
    assert _weights({}, assets=()) == {"SHV": 1.0}


@pytest.mark.parametrize("gld", [RISING, FALLING])
@pytest.mark.parametrize("ief", [RISING, FALLING])
@pytest.mark.parametrize("tlt", [RISING, FALLING])
def test_weights_always_sum_to_one(gld, ief, tlt):
    weights = sleeve_weights({"GLD": gld, "IEF": ief, "TLT": tlt}, ("GLD", "IEF", "TLT"), 3, "SHV")

    assert sum(weights.values()) == pytest.approx(1.0)
    assert list(weights) == ["SHV", "GLD", "IEF", "TLT"]
