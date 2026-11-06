from __future__ import annotations

import math
import statistics

import pytest

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, compute_return_from_prices, momentum_inputs, rank, score_stock
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


TRADED = [c * 10 for c in CLOSES]
TRADED_VOLUMES = [v / 10 for v in VOLUMES]


def test_the_filter_inputs_come_from_the_traded_series() -> None:
    plain = momentum_inputs(CLOSES, VOLUMES, CONFIG)
    traded = momentum_inputs(CLOSES, VOLUMES, CONFIG, traded_closes=TRADED, traded_volumes=TRADED_VOLUMES)

    assert plain is not None and traded is not None
    assert traded.price == 3990.0
    # The mean of each of the last 20 sessions' traded close x 1e5 shares (closes 380..399 x 10), not the last close x the mean volume.
    assert traded.avg_dollar_volume == pytest.approx(sum(t * v for t, v in zip(TRADED[-20:], TRADED_VOLUMES[-20:], strict=True)) / 20)
    assert traded.avg_dollar_volume == pytest.approx(389.5e6)
    assert (traded.ret_12_1m, traded.ret_6_1m, traded.ret_3m) == (plain.ret_12_1m, plain.ret_6_1m, plain.ret_3m)
    assert traded.volatility == plain.volatility


def test_a_stock_under_min_price_on_adjusted_prices_passes_on_its_traded_price() -> None:
    adjusted = [c / 100 for c in CLOSES]  # last adjusted close 3.99, under min_price 10

    assert score_stock("SPLIT", adjusted, VOLUMES, CONFIG) is None
    assert score_stock("SPLIT", adjusted, VOLUMES, CONFIG, traded_closes=CLOSES, traded_volumes=VOLUMES) is not None  # traded at 399, $399M a day


def test_a_dividend_payer_passes_the_dollar_volume_filter_on_traded_prices() -> None:
    # Adjusted closes rise linearly from 5 to 15 with 1e6 shares a day: 15 x 1e6 = $15M, under min_dollar_volume ($20M).
    # The dividends adjusted away left it trading 40% higher: the last 20 sessions average about 14.7 x 1.4 x 1e6 = $20.6M, over. The price (15, 21), the
    # volatility (a gentle linear ramp) and the 300 trading days all pass either way, so only the dollar volume decides.
    adjusted = [5.0 + 10.0 * i / 299 for i in range(300)]
    traded = [c * 1.4 for c in adjusted]

    assert adjusted[-1] * 1e6 < CONFIG["min_dollar_volume"] <= traded[-1] * 1e6
    assert score_stock("DIV", adjusted, VOLUMES, CONFIG) is None
    assert score_stock("DIV", adjusted, VOLUMES, CONFIG, traded_closes=traded, traded_volumes=VOLUMES) is not None


def test_one_traded_series_alone_is_ignored() -> None:
    plain = momentum_inputs(CLOSES, VOLUMES, CONFIG)

    assert momentum_inputs(CLOSES, VOLUMES, CONFIG, traded_closes=TRADED) == plain
    assert momentum_inputs(CLOSES, VOLUMES, CONFIG, traded_volumes=TRADED_VOLUMES) == plain


def test_the_dollar_volume_averages_each_sessions_traded_dollars_across_a_split() -> None:
    # Every session at $30M a day, a 2:1 split 10 sessions before the end: 100 x 300,000 shares before, 50 x 600,000 after.
    # Last close x average shares would read 50 x 450,000 = $22.5M; the sessions' own dollars average to $30M.
    traded = [100.0] * 290 + [50.0] * 10
    traded_volumes = [300_000.0] * 290 + [600_000.0] * 10
    inputs = momentum_inputs(CLOSES, VOLUMES, CONFIG, traded_closes=traded, traded_volumes=traded_volumes)

    assert inputs is not None
    assert inputs.price == 50.0
    assert inputs.avg_dollar_volume == pytest.approx(30e6)


def test_a_12_month_return_compares_the_last_close_with_the_close_a_year_earlier() -> None:
    closes = [100.0] + [110.0] * 251 + [120.0]  # 253 closes: 252 bars back is the first one

    assert compute_return_from_prices(closes, 252, 0) == pytest.approx(0.2)


@pytest.mark.parametrize("closes", [[], [100.0] * 252])
def test_a_12_month_return_is_none_without_enough_history(closes: list[float]) -> None:
    assert compute_return_from_prices(closes, 252, 0) is None


def test_a_return_is_none_when_the_starting_close_is_not_positive() -> None:
    assert compute_return_from_prices([0.0] + [10.0] * 252, 252, 0) is None


def test_a_return_takes_a_shorter_lookback() -> None:
    assert compute_return_from_prices([10.0, 11.0, 12.0, 15.0], 3, 0) == pytest.approx(0.5)
