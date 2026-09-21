from __future__ import annotations

import math

import pandas as pd

from trading_agent_framework.dashboard.components.charts import agent_calls_chart


def _calls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-05T21:00:00Z", "2026-01-06T21:00:00Z", "2026-01-07T21:00:00Z"], utc=True),
            "agent": ["trader", "trader", "researcher"],
            "model": ["qwen3-8b"] * 3,
            "input_tokens": [100.0, 200.0, None],
            "output_tokens": [20.0, 30.0, None],
            "reasoning_tokens": [None, None, None],
            "total_tokens": [120.0, 230.0, None],
            "latency_ms": [1500.0, 250.0, 2000.0],
            "tool_calls": [1, 0, 2],
        }
    )


def test_latency_is_plotted_in_seconds_against_call_time() -> None:
    fig = agent_calls_chart(_calls())

    latency = fig.data[0]
    assert list(latency.y) == [1.5, 0.25, 2.0]
    assert [pd.Timestamp(x) for x in latency.x] == list(_calls()["ts"])


def test_total_tokens_are_plotted_with_unreported_calls_left_as_gaps_not_zeros() -> None:
    fig = agent_calls_chart(_calls())

    tokens = list(fig.data[1].y)
    assert tokens[:2] == [120.0, 230.0]
    assert math.isnan(tokens[2])


def test_hovering_a_call_names_its_agent() -> None:
    fig = agent_calls_chart(_calls())

    assert list(fig.data[0].text) == ["trader", "trader", "researcher"]


def test_no_calls_gives_the_dashboards_usual_empty_chart() -> None:
    fig = agent_calls_chart(_calls().iloc[0:0])

    assert len(fig.data) == 0
    assert [a.text for a in fig.layout.annotations] == ["No data available"]
