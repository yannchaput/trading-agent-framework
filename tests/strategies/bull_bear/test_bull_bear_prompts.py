from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear import prompts


@pytest.mark.parametrize(
    ("system", "tool"),
    [
        (prompts.RESEARCHER_SYSTEM, "submit_note"),
        (prompts.BULL_SYSTEM, "submit_bull_case"),
        (prompts.BEAR_SYSTEM, "submit_bear_case"),
        (prompts.JUDGE_SYSTEM, "submit_verdicts"),
    ],
)
def test_every_system_prompt_requires_english_and_names_its_submit_tool(system: str, tool: str) -> None:
    assert "English" in system and tool in system


@pytest.mark.parametrize("system", [prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM])
def test_no_prompt_states_the_review_cadence(system: str) -> None:
    assert not any(word in system.lower() for word in ("weekly", "every week", "tuesday", "daily", "every day"))


def test_only_the_bear_prompt_lists_the_concerns() -> None:
    assert "momentum_exhaustion" in prompts.BEAR_SYSTEM and "momentum_exhaustion" not in prompts.BULL_SYSTEM


def test_the_bear_may_not_call_a_strong_trend_or_a_stretched_price_a_risk() -> None:
    """Run 2026-10-10_085928: 244 `momentum_exhaustion` flags, 88% of them 'high' on ranks 1-3, returned +2.2% the next week against +1.4% overall."""
    text = prompts.BEAR_SYSTEM

    assert "far above its 200-day average" in text and "large three-month gain" in text  # named as NOT a risk
    assert "price far above its 200-day average)" not in text  # no longer offered as an example of exhaustion
    assert "one-month return far above its three-month pace" not in text


def test_momentum_exhaustion_needs_a_decline_that_ran_for_weeks() -> None:
    """MU on 2026-03-31: -21.9% over the month but -20.4% of it in the last week, a one-week drop is no exhaustion."""
    text = " ".join(prompts.BEAR_SYSTEM.split())

    # the fact sheet's figures are NEGATIVE below the high / over the month, so the thresholds are stated as negative numbers
    assert (
        "Use momentum_exhaustion only for a decline that ran for weeks: a drawdown_from_52w_high_pct of -15 or below "
        "together with a return_1m_skip_1w_pct of -10 or below" in text
    )
    assert "negative one-month return" not in text
    assert "still rising is never momentum_exhaustion" in text
    assert "strong trend is not a risk by itself" in text and "200-day average" in text and "three-month" in text


def test_every_agent_reads_the_fundamentals_fields_and_the_sources_rule() -> None:
    fields = (
        "return_1m_skip_1w_pct",
        "fundamentals_period_end",
        "fundamentals_filed",
        "period_months",
        "revenue_growth_pct",
        "operating_margin_pct",
        "operating_margin_prior_year_pct",
        "free_cash_flow_bn",
        "free_cash_flow_prior_year_bn",
        "free_cash_flow_months",
        "total_debt_bn",
        "total_debt_prior_year_bn",
        "currency",
        "fundamentals_stale",
        "fundamentals_unavailable",
    )
    # the no-news researcher reads the figures but not the sources rule, which speaks of news (it must say nothing of news)
    for field in fields:
        assert field in prompts.researcher_system(False)
    for system in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM):
        text = " ".join(system.split())
        for field in fields:
            assert field in text, field
        assert (
            "Sources can be wrong or stale: do not repeat a figure from a news item as fact when the fact sheet or a filing "
            "says otherwise." in text
        )


def test_the_bear_rates_high_only_on_fundamental_evidence() -> None:
    text = " ".join(prompts.BEAR_SYSTEM.split())

    assert "Base each case on the stock's fundamentals and financial health; news only supports it." in text
    start = text.index("Rate a stock high only on evidence in its fundamentals")
    rule = text[start : start + 900]
    for evidence in ("revenue", "operating margin", "free cash flow", "debt"):
        assert evidence in rule, evidence
    for fact in ("results", "a guidance cut", "a share issuance", "going-concern doubts", "a covenant breach", "a restatement", "an impairment"):
        assert fact in rule, fact
    assert "the company itself reports or files" in rule
    assert (
        "When the fact sheet's fundamentals are unavailable or stale, the company's own reported facts in the note are your evidence."
        in text
    )
    assert "A fallen price with intact cash flow makes a company cheaper, not riskier." in text
    for concern in ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none"):
        assert concern in text, concern


def test_the_bear_treats_news_from_others_as_support_only() -> None:
    text = " ".join(prompts.BEAR_SYSTEM.split())

    assert "News from others is never a high risk by itself, whatever consequence you argue" in text
    start = text.index("News from others is never a high risk by itself")
    rule = text[start : start + 500]
    for example in ("a competitor's technology", "an analyst action", "a lawsuit headline", "speculation about the future"):
        assert example in rule, example
    assert "lasting consequence" not in text and "already reacted to" not in text
    assert "can support a risk but never make it high" in text
    assert "do not make one" not in text


def test_the_bull_argues_from_fundamentals() -> None:
    text = " ".join(prompts.BULL_SYSTEM.split())

    assert "Base each case on the stock's fundamentals and financial health" in text
    assert "momentum_exhaustion" not in text


def test_the_judge_puts_a_stock_out_on_a_fundamental_risk_not_on_news_from_others() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())

    assert "Put a stock out when the bear names a risk shown in its fundamentals" in text
    assert "Neither is news from others, already priced or not, whatever consequence the bear argues for it." in text
    assert "lasting consequence" not in text and "already reacted to" not in text
    assert "not a reason for out by itself" in text
    assert (
        "When the fact sheet's fundamentals are unavailable or stale, the company's own reported facts in the note are the evidence."
        in text
    )


@pytest.mark.parametrize("news", [True, False])
def test_the_researcher_reads_the_figures_and_looks_for_what_explains_them(news: bool) -> None:
    text = " ".join(prompts.researcher_system(news).split())

    assert "The fact sheet already gives the latest SEC figures against a year earlier" in text
    assert "what the company itself reported or filed" in text
    assert "prior_year" in text and "Do not repeat the fact sheet's numbers" in text
    if not news:
        assert "news" not in text.lower()


@pytest.mark.parametrize("news", [True, False])
def test_the_researcher_says_who_reported_each_fact(news: bool) -> None:
    text = " ".join(prompts.researcher_system(news).split())

    assert "say who reported it: the company (a release, a filing, guidance) or others (analysts, media)" in text


def test_the_researcher_task_names_the_symbol_and_the_note_limit() -> None:
    task = prompts.researcher_task("AAA", 500)

    assert "AAA" in task and "500" in task


@pytest.mark.parametrize("task", [prompts.bull_task, prompts.bear_task, prompts.judge_task])
def test_the_debate_tasks_state_the_character_cap_they_are_given(task) -> None:  # noqa: ANN001
    text = task(234)

    assert "234 characters" in text and "rejects the whole submission" in text


def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("judge", "submit_verdicts", "11 stocks in: put at most 10 in and the others out", prompts.judge_task(300))

    assert "submit_verdicts" in text and "put at most 10 in" in text


def test_the_retry_prompt_repeats_the_task_so_the_symbol_and_the_cap_survive() -> None:
    text = prompts.retry_prompt("researcher", "submit_note", "the agent ended without calling submit_note", prompts.researcher_task("PLTR", 500))

    assert "PLTR" in text and "500 characters" in text


def test_the_retry_prompt_tells_a_researcher_that_its_research_is_gone_and_must_not_become_no_news() -> None:
    text = prompts.retry_prompt("researcher", "submit_note", "x", "task")

    assert "fact sheet alone" in text and "no news or filings were checked" in text
    assert "fact sheet alone" not in prompts.retry_prompt("judge", "submit_verdicts", "x", "task")


def test_the_judge_puts_each_stock_in_or_out_under_a_cap_and_a_trend_alone_is_no_reason_for_out() -> None:
    text = prompts.JUDGE_SYSTEM

    assert "in or out" in text and "max_in" in text
    assert "not a reason for out by itself" in text and "already reacted to" not in text
    assert "veto" not in text.lower() and "veto" not in prompts.judge_task(300).lower()


def test_no_prompt_mentions_forced_exits() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM, prompts.bull_task(300), prompts.bear_task(300), prompts.judge_task(300)):
        assert "forced" not in text.lower()


def test_no_prompt_explains_held_or_weight() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM):
        assert "held" not in text and "weight_pct" not in text


def test_the_fact_sheet_explains_the_one_week_return() -> None:
    assert "return_1w_pct" in prompts.BULL_SYSTEM and "return_1w_pct" in prompts.BEAR_SYSTEM


def test_the_researcher_reads_the_prior_year_and_is_not_asked_for_a_trend_it_cannot_see() -> None:
    assert "prior_year" in prompts.RESEARCHER_SYSTEM and "trend of revenue" not in prompts.RESEARCHER_SYSTEM


def test_the_researcher_without_news_is_not_told_about_news() -> None:
    assert "news" in prompts.researcher_system(True) and "news" not in prompts.researcher_system(False).lower()
    assert prompts.RESEARCHER_SYSTEM == prompts.researcher_system(True)


def test_neither_the_bear_nor_the_judge_has_the_lasting_consequence_escape() -> None:
    for text in (prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM):
        assert "lasting consequence" not in text and "already reacted to" not in text


def test_the_judge_starts_from_its_previous_verdict_and_says_what_changed() -> None:
    assert "previous_verdict" in prompts.JUDGE_SYSTEM and "what_changed" in prompts.JUDGE_SYSTEM


def test_max_in_is_a_ceiling_not_a_target_and_no_stock_is_put_out_for_room_alone() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())

    assert "max_in, given in the constraints, is a ceiling, not a target" in text
    assert "put a stock in only when its own case earns a place" in text
    assert "Putting fewer than max_in in is normal, and you may put none in." in text
    assert "Only when more stocks earn a place than max_in allows" in text
    assert "A stronger candidate" not in text and "needs a place under max_in" not in text
    assert "change it only for a new, material fact" in text  # the previous_verdict rule stays


def test_the_judge_is_told_how_to_read_its_recent_verdicts() -> None:
    assert "recent_verdicts" in prompts.JUDGE_SYSTEM and "in_names_return_pct" in prompts.JUDGE_SYSTEM


def test_the_judge_is_told_about_its_memory_tools() -> None:
    assert "search_memory" in prompts.JUDGE_SYSTEM and "remember_lesson" in prompts.JUDGE_SYSTEM


def test_the_bear_keeps_none_with_low_for_no_concrete_risk_and_the_fundamentals_gate_only_the_high() -> None:
    text = " ".join(prompts.BEAR_SYSTEM.split())

    assert "Use none with low when you find no concrete risk." in text
    assert "no risk in the fundamentals" not in text  # a narrowed none would cancel momentum_exhaustion, valuation, sector, news_event
