from __future__ import annotations

import math
import statistics

import pytest

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, momentum_inputs, rank, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

CLOSES = [100.0 + i for i in range(300)]
VOLUMES = [1e6] * 300


def test_momentum_inputs_are_cross_momentums_returns_volatility_and_dollar_volume() -> None:
    inputs = momentum_inputs(CLOSES, VOLUMES, CONFIG)

    assert inputs is not None
    assert (inputs.price, inputs.trading_days) == (399.0, 300)
    assert inputs.ret_12_1m == pytest.approx(2.0)
    assert inputs.ret_6_1m == pytest.approx(0.5)
    assert inputs.ret_3m == pytest.approx(0.1875)
    assert inputs.avg_dollar_volume == pytest.approx(399e6)
    logs = [math.log(CLOSES[i] / CLOSES[i - 1]) for i in range(-20, 0)]
    assert inputs.volatility == pytest.approx(statistics.stdev(logs) * math.sqrt(252))


def test_a_history_shorter_than_min_trading_days_has_no_inputs() -> None:
    assert momentum_inputs(CLOSES[:249], VOLUMES[:249], CONFIG) is None


def test_a_history_too_short_for_the_12_month_return_has_no_inputs() -> None:
    assert momentum_inputs(CLOSES[:273], VOLUMES[:273], CONFIG) is None  # needs 252 + 21 + 1 = 274 closes


def test_score_stock_weights_the_returns_like_cross_momentum() -> None:
    row = score_stock("AAA", CLOSES, VOLUMES, CONFIG)

    assert isinstance(row, MomentumRow)
    assert row.symbol == "AAA"
    assert row.score == pytest.approx(0.5 * 2.0 + 0.3 * 0.5 + 0.2 * 0.1875)
    assert row.volatility == row.inputs.volatility
    assert row.closes == tuple(CLOSES)


def test_score_stock_applies_cross_momentums_filters() -> None:
    cheap = [c / 100 for c in CLOSES]  # last close 3.99, under min_price 10
    thin = [1_000.0] * 300  # 399 x 1,000 = 0.4M a day, under min_dollar_volume 20M

    assert score_stock("CHEAP", cheap, VOLUMES, CONFIG) is None
    assert score_stock("THIN", CLOSES, thin, CONFIG) is None


def test_score_stock_needs_a_volatility_above_zero() -> None:
    flat = [100.0] * 300  # every log return is exactly 0: zero volatility (a geometric series leaves float noise)

    assert score_stock("FLAT", flat, VOLUMES, CONFIG) is None


def test_rank_orders_by_score_best_first_from_1() -> None:
    slow = score_stock("SLOW", [100.0 + 0.5 * i for i in range(300)], VOLUMES, CONFIG)
    fast = score_stock("FAST", CLOSES, VOLUMES, CONFIG)
    assert slow is not None and fast is not None

    ranked = rank([slow, fast])

    assert ranked == [RankedRow(1, fast), RankedRow(2, slow)]


def test_rank_of_nothing_is_empty() -> None:
    assert rank([]) == []
