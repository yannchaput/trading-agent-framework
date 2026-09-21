from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from trading_agent_framework.agents.telemetry import CallRecord, Usage, summarize, usage_from_message

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def _record(agent: str = "trader", **overrides: object) -> CallRecord:
    fields: dict[str, object] = {
        "ts": T0, "agent": agent, "model": "qwen3-8b", "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": 5, "total_tokens": 120, "latency_ms": 1000.0, "tool_calls": 1,
    }
    return CallRecord(**{**fields, **overrides})  # type: ignore[arg-type]


def test_usage_reads_langchain_usage_metadata_including_nested_reasoning_tokens() -> None:
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "output_token_details": {"reasoning": 7}},
    )

    assert usage_from_message(message) == Usage(input_tokens=100, output_tokens=20, reasoning_tokens=7, total_tokens=120)


def test_usage_is_unknown_not_zero_when_the_server_reports_none() -> None:
    assert usage_from_message(SimpleNamespace(usage_metadata=None)) == Usage(None, None, None, None)
    assert usage_from_message(SimpleNamespace()) == Usage(None, None, None, None)


def test_usage_without_reasoning_details_leaves_reasoning_unknown() -> None:
    message = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    assert usage_from_message(message) == Usage(input_tokens=10, output_tokens=2, reasoning_tokens=None, total_tokens=12)


def test_summarize_totals_each_agent_separately() -> None:
    records = [
        _record("trader", input_tokens=100, output_tokens=20, reasoning_tokens=5, total_tokens=120, latency_ms=1000.0, tool_calls=2),
        _record("trader", input_tokens=300, output_tokens=40, reasoning_tokens=15, total_tokens=340, latency_ms=3000.0, tool_calls=0),
        _record("researcher", model="other", input_tokens=7, output_tokens=3, reasoning_tokens=None, total_tokens=10, latency_ms=500.0, tool_calls=1),
    ]

    assert summarize(records) == {
        "trader": {
            "model": "qwen3-8b", "calls": 2, "tool_calls": 2, "input_tokens": 400, "output_tokens": 60,
            "reasoning_tokens": 20, "total_tokens": 460, "latency_ms_total": 4000.0, "latency_ms_avg": 2000.0,
        },
        "researcher": {
            "model": "other", "calls": 1, "tool_calls": 1, "input_tokens": 7, "output_tokens": 3,
            "reasoning_tokens": None, "total_tokens": 10, "latency_ms_total": 500.0, "latency_ms_avg": 500.0,
        },
    }


def test_summarize_skips_unknown_token_counts_instead_of_counting_them_as_zero() -> None:
    records = [
        _record(input_tokens=100, output_tokens=20, total_tokens=120),
        _record(input_tokens=None, output_tokens=None, reasoning_tokens=None, total_tokens=None),
    ]

    summary = summarize(records)["trader"]

    assert summary["calls"] == 2
    assert summary["input_tokens"] == 100
    assert summary["total_tokens"] == 120


def test_summarize_of_nothing_is_empty() -> None:
    assert summarize([]) == {}
