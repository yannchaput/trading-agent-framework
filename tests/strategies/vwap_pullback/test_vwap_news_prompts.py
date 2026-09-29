from __future__ import annotations

from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.prompts import CATALYSTS, build_entry_prompt, build_exit_prompt


def test_lean_headlines_keeps_the_newest_distinct_headlines() -> None:
    articles = [
        {"headline": "AAA beats estimates", "created_at": "2026-09-01T12:00:00Z", "source": "benzinga", "summary": "long text"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "", "created_at": "2026-09-01T14:00:00Z", "source": "x"},
        {"headline": "Old news", "created_at": "2026-08-30T12:00:00Z", "source": "x"},
    ]
    rows = lean_headlines(articles, 2)
    assert rows == [
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
    ]


def test_entry_prompt_names_the_tools_the_catalysts_and_the_thresholds() -> None:
    prompt = build_entry_prompt(VwapPullbackParameters())
    for word in ("enter_long", "pass_on_setup", "search_news", "M&A", "offering", *CATALYSTS):
        assert word in prompt
    assert "2.0" in prompt  # none_catalyst_min_z
    assert "at most 4" in prompt  # news_calls_per_run


def test_exit_prompt_names_every_action_and_the_flatten_time() -> None:
    prompt = build_exit_prompt(VwapPullbackParameters(), flatten_time="15:50")
    for word in ("take_partial_profit", "tighten_stop", "replace_stop_with_trailing", "exit_position", "hold", "already_stopped_out", "15:50", "0.25", "2.0"):
        assert word in prompt
    assert "it says what was and was not done" in prompt and "nothing was changed" not in prompt
