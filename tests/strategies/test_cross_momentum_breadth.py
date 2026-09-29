"""Breadth overlay for cross_momentum: pure breadth/step/exposure functions, step persistence, wiring."""

import json

import pytest

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
