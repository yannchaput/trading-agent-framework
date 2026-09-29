"""Breadth overlay for cross_momentum: pure breadth/step/exposure functions, step persistence, wiring."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.utils import (
    breadth_exposure,
    breadth_share,
    load_breadth_step,
    next_breadth_step,
    save_breadth_step,
)

THRESHOLDS = (0.50, 0.30)
RISING = [float(i) for i in range(1, 101)]  # last close 100 > SMA 50.5
FALLING = list(reversed(RISING))  # last close 1 < SMA 50.5


def test_config_carries_the_agreed_breadth_settings():
    assert CONFIG["breadth_overlay"] == {
        "enabled": True,
        "sma_window": 100,
        "min_stocks": 50,
        "thresholds": (0.50, 0.30),
        "exposures": (1.0, 0.7, 0.4),
        "hysteresis": 0.05,
    }


def test_breadth_share_is_the_fraction_above_their_sma():
    closes = {"A": RISING, "B": RISING, "C": RISING, "D": FALLING}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.75)


def test_a_stock_with_too_little_history_is_excluded_from_both_sides():
    closes = {"A": RISING, "B": FALLING, "SHORT": RISING[:50]}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.5)


def test_a_non_finite_close_excludes_the_stock():
    closes = {"A": RISING, "NAN": [*RISING[:-1], float("nan")]}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(1.0)


def test_a_flat_stock_is_not_above_its_sma():
    closes = {"FLAT": [10.0] * 100, "UP": RISING}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.5)


def test_fewer_valid_stocks_than_min_stocks_gives_none():
    closes = {"A": RISING, "B": FALLING, "SHORT": RISING[:50]}

    assert breadth_share(closes, sma_window=100, min_stocks=3) is None
    assert breadth_share({}, sma_window=100, min_stocks=1) is None


@pytest.mark.parametrize(
    ("breadth", "expected"),
    [(0.60, 0), (0.50, 0), (0.49, 1), (0.30, 1), (0.29, 2), (0.0, 2)],
)
def test_the_raw_step_without_history(breadth, expected):
    assert next_breadth_step(breadth, None, THRESHOLDS, 0.05) == expected


def test_cutting_is_immediate():
    assert next_breadth_step(0.25, 0, THRESHOLDS, 0.05) == 2
    assert next_breadth_step(0.29, 1, THRESHOLDS, 0.05) == 2
    assert next_breadth_step(0.45, 0, THRESHOLDS, 0.05) == 1


@pytest.mark.parametrize(
    ("previous", "breadth", "expected"),
    [
        (2, 0.33, 2),  # 0.30 + 0.05 not cleared
        (2, 0.35, 1),
        (1, 0.52, 1),  # 0.50 + 0.05 not cleared
        (1, 0.55, 0),
    ],
)
def test_re_risking_needs_the_buffer(previous, breadth, expected):
    assert next_breadth_step(breadth, previous, THRESHOLDS, 0.05) == expected


def test_a_multi_step_jump_lands_on_the_least_defensive_step_whose_bound_is_met():
    assert next_breadth_step(0.52, 2, THRESHOLDS, 0.05) == 1
    assert next_breadth_step(0.60, 2, THRESHOLDS, 0.05) == 0


def test_breadth_exposure_maps_each_step():
    assert [breadth_exposure(step, (1.0, 0.7, 0.4)) for step in (0, 1, 2)] == [1.0, 0.7, 0.4]


def test_a_saved_step_round_trips(tmp_path):
    path = tmp_path / "breadth.json"

    save_breadth_step(path, 1, 0.42, "2026-09-29")

    assert load_breadth_step(path, n_steps=3) == 1
    assert json.loads(path.read_text()) == {"date": "2026-09-29", "step": 1, "breadth": 0.42}


def test_saving_replaces_the_previous_step(tmp_path):
    path = tmp_path / "breadth.json"
    save_breadth_step(path, 2, 0.25, "2026-09-22")

    save_breadth_step(path, 1, 0.36, "2026-09-29")

    assert load_breadth_step(path, n_steps=3) == 1


def test_a_missing_file_loads_as_none(tmp_path):
    assert load_breadth_step(tmp_path / "absent.json", n_steps=3) is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[1, 2]",
        "{}",
        '{"step": 3}',  # out of range for n_steps=3
        '{"step": -1}',
        '{"step": 1.0}',
        '{"step": "1"}',
        '{"step": true}',
    ],
)
def test_a_corrupt_or_invalid_file_loads_as_none(tmp_path, content):
    path = tmp_path / "breadth.json"
    path.write_text(content)

    assert load_breadth_step(path, n_steps=3) is None


def test_a_file_that_is_not_valid_text_loads_as_none(tmp_path):
    path = tmp_path / "breadth.json"
    path.write_bytes(b"\xff\xfe\x00garbage")

    assert load_breadth_step(path, n_steps=3) is None


def test_a_failing_write_does_not_raise(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")

    save_breadth_step(blocker / "breadth.json", 1, 0.42, "2026-09-29")  # parent is a file: the write fails


class FakeBreadthStrategy:
    """Just enough of `Strategy` for `initialize` and `_breadth_exposure`."""

    initialize = CrossMomentumStrategy.initialize
    _breadth_exposure = CrossMomentumStrategy._breadth_exposure

    def __init__(self, tmp_path, *, breadth=None, step=None, enabled=True):
        self.parameters = {"breadth_overlay": {**CONFIG["breadth_overlay"], "enabled": enabled}}
        self.vars = SimpleNamespace(
            breadth=breadth,
            breadth_step=step,
            breadth_file_path=tmp_path / "breadth.json",
            history_file_path=tmp_path / "absent_history.json",
            equity_history=[],
        )
        self.infos: list[str] = []

    def get_datetime(self):
        return datetime(2026, 9, 29, tzinfo=UTC)

    def log_info(self, message, *args, **kwargs):
        self.infos.append(message)


def test_a_breadth_of_forty_percent_caps_exposure_at_seventy_percent_and_logs_it(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=0.40)

    assert fake._breadth_exposure() == pytest.approx(0.7)
    assert fake.vars.breadth_step == 1
    assert any("40%" in message and "step 1" in message for message in fake.infos)


def test_no_reading_leaves_exposure_at_one_and_the_step_untouched(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=None, step=2)

    assert fake._breadth_exposure() == 1.0
    assert fake.vars.breadth_step == 2
    assert not fake.vars.breadth_file_path.exists()


def test_a_disabled_overlay_ignores_breadth(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=0.10, step=None, enabled=False)

    assert fake._breadth_exposure() == 1.0
    assert fake.vars.breadth_step is None
    assert not fake.vars.breadth_file_path.exists()


def test_the_stored_step_carries_hysteresis_across_consecutive_calls(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=0.25)
    assert fake._breadth_exposure() == pytest.approx(0.4)

    fake.vars.breadth = 0.33  # above 0.30 but below 0.35: stay defensive
    assert fake._breadth_exposure() == pytest.approx(0.4)

    fake.vars.breadth = 0.36
    assert fake._breadth_exposure() == pytest.approx(0.7)


def test_applying_a_reading_persists_the_step_with_the_strategy_clock_date(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=0.42)

    fake._breadth_exposure()

    assert json.loads(fake.vars.breadth_file_path.read_text()) == {"date": "2026-09-29", "step": 1, "breadth": 0.42}


def test_a_restart_resumes_on_the_persisted_step_so_hysteresis_survives(tmp_path):
    before_crash = FakeBreadthStrategy(tmp_path, breadth=0.25)
    assert before_crash._breadth_exposure() == pytest.approx(0.4)  # step 2, saved to disk

    after_restart = FakeBreadthStrategy(tmp_path, breadth=0.33)  # a new process: step starts as None
    after_restart.initialize()
    assert after_restart.vars.breadth_step == 2
    assert after_restart._breadth_exposure() == pytest.approx(0.4)  # 0.33 < 0.35: still defensive

    after_restart.vars.breadth = 0.36
    assert after_restart._breadth_exposure() == pytest.approx(0.7)


def test_a_restart_with_no_saved_step_applies_the_first_reading_directly(tmp_path):
    fake = FakeBreadthStrategy(tmp_path, breadth=0.33)

    fake.initialize()

    assert fake.vars.breadth_step is None
    assert fake._breadth_exposure() == pytest.approx(0.7)  # raw step 1, no buffer without history


class FakeScoringStrategy:
    """Just enough of `Strategy` for `compute_target_portfolio`; each ticker's indicators are canned."""

    compute_target_portfolio = CrossMomentumStrategy.compute_target_portfolio

    def __init__(self, indicators):
        self.parameters = dict(CONFIG)
        self.vars = SimpleNamespace(universe=list(indicators), target_closes={}, breadth=None, breadth_step=None)
        self._indicators = indicators
        self.errors: list[str] = []

    def _compute_indicators_for_ticker(self, ticker):
        return self._indicators[ticker]

    def log_info(self, *args, **kwargs): ...

    def log_error(self, message, *args, **kwargs):
        self.errors.append(message)


def _indicator(symbol, closes):
    return {
        "symbol": symbol,
        "price": 50.0,
        "avg_dollar_volume": 50_000_000.0,
        "volatility": 0.3,
        "trading_days": 300,
        "ret_12_1m": 0.2,
        "ret_6_1m": 0.1,
        "ret_3m": 0.05,
        "atr": None,
        "closes": closes,
    }


def test_compute_target_portfolio_sets_breadth_from_the_scored_stocks():
    rising = [float(i) for i in range(1, 301)]
    indicators = {f"UP{i}": _indicator(f"UP{i}", rising) for i in range(45)}
    indicators |= {f"DN{i}": _indicator(f"DN{i}", list(reversed(rising))) for i in range(15)}
    fake = FakeScoringStrategy(indicators)

    fake.compute_target_portfolio()

    assert fake.vars.breadth == pytest.approx(0.75)


def test_a_week_with_no_scored_stocks_resets_breadth_to_none():
    fake = FakeScoringStrategy({"AAA": None, "BBB": None})
    fake.vars.breadth = 0.40  # last week's reading must not survive

    target, ranks = fake.compute_target_portfolio()

    assert (target, ranks) == ([], {})
    assert fake.vars.breadth is None
