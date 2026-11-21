"""Tests for `SectorPodsParams`."""

import re

import pytest

from trading_agent_framework.agents.tools.text import TextLimit
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.utils.errors import ConfigurationError


def test_defaults():
    p = SectorPodsParams()
    assert (p.rebalance_weekday, p.rebalance_time, p.cash_buffer, p.rebalance_band) == (1, "12:00", 0.02, 0.03)
    assert (p.min_trade_pct, p.parking_symbol, p.max_etf_weight, p.max_pod_weight) == (0.005, "SHV", 0.25, 0.40)
    assert (p.pod_tool_budget, p.pm_memory_budget, p.max_agent_steps) == (4, 3, 60)
    assert (p.agent_temperature, p.thinking_token_budget, p.max_output_tokens) == (0.3, 16384, 24576)
    assert (p.reason_max_chars, p.summary_max_chars, p.max_consecutive_abandoned) == (200, 400, 3)
    assert (p.reason_hard_max_chars, p.summary_hard_max_chars, p.max_submit_attempts) == (300, 700, 3)
    assert (p.reason_limit, p.summary_limit) == (TextLimit(200, 300), TextLimit(400, 700))
    assert (p.yahoo_retry_delays, p.min_yahoo_coverage) == ((60.0, 180.0), 0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_etf_weight": 0.5, "max_pod_weight": 0.4},
        {"rebalance_weekday": 5},
        {"rebalance_weekday": -1},
        {"rebalance_time": "noon"},
        {"max_etf_weight": 0.0},
        {"max_pod_weight": 1.5},
    ],
)
def test_params_reject_bad_values(kwargs):
    with pytest.raises(ConfigurationError):
        SectorPodsParams(**kwargs)


def test_a_cash_buffer_below_the_framework_buy_buffer_is_rejected_and_2_percent_is_accepted():
    with pytest.raises(ConfigurationError, match=re.escape("cash_buffer must be in [0.02, 1)")):
        SectorPodsParams(cash_buffer=0.01)
    assert SectorPodsParams(cash_buffer=0.02).cash_buffer == 0.02


@pytest.mark.parametrize("weekday", [0, 4])
def test_weekday_bounds_accepted(weekday):
    assert SectorPodsParams(rebalance_weekday=weekday).rebalance_weekday == weekday


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reason_hard_max_chars": 199}, "reason_hard_max_chars must be at least reason_max_chars"),
        ({"summary_hard_max_chars": 399}, "summary_hard_max_chars must be at least summary_max_chars"),
        ({"max_submit_attempts": 0}, "max_submit_attempts must be at least 1"),
        ({"reason_max_chars": 0}, "reason_max_chars and summary_max_chars must be at least 1"),
    ],
)
def test_params_reject_bad_text_limits(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        SectorPodsParams(**kwargs)
