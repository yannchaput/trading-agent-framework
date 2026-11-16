"""Tests for the sector_pods prompts."""

import re
from dataclasses import replace

from trading_agent_framework.strategies.sector_pods.handoff import PM_SUBMIT, POD_SUBMIT, RISK_SUBMIT
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import PODS
from trading_agent_framework.strategies.sector_pods.prompts import PM_TASK, POD_TASK, RISK_TASK, pm_system, pod_system, risk_system

PARAMS = SectorPodsParams()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Z]{2,5}", text))


def test_pod_prompt_lists_its_sleeve():
    for pod in PODS:
        text = pod_system(pod, PARAMS)
        assert pod.label in text and pod.focus in text
        assert set(pod.sleeve) <= _words(text)
        others = {s for p in PODS if p is not pod for s in p.sleeve} - set(pod.sleeve)
        assert not others & _words(text)


def test_prompts_name_their_submit_tool():
    assert all(POD_SUBMIT in pod_system(pod, PARAMS) for pod in PODS)
    assert RISK_SUBMIT in risk_system(PARAMS)
    assert PM_SUBMIT in pm_system(PARAMS)


def test_pm_prompt_states_the_caps_from_params():
    text = pm_system(PARAMS)
    assert "25%" in text and "40%" in text and "98%" in text and "SHV" in text
    assert "20%" in pm_system(replace(PARAMS, max_etf_weight=0.2))


def test_pm_prompt_says_the_book_is_always_invested():
    text = pm_system(PARAMS)
    assert "always invested" in text and "98%" in text and "mix" in text
    assert "small book" not in text and "risk-off" not in text


def test_pm_prompt_says_the_submitted_weights_must_respect_the_caps():
    text = pm_system(PARAMS)
    assert "weights you submit must stay within these caps" in text
    assert "the caps apply to the scaled book" not in text


def test_no_prompt_contains_forced():
    texts = [pod_system(pod, PARAMS) for pod in PODS] + [risk_system(PARAMS), pm_system(PARAMS), POD_TASK, RISK_TASK, PM_TASK]
    assert not any(re.search(r"forced|\bheld\b", text, re.IGNORECASE) for text in texts)


def test_context_fields_are_named():
    assert all(f"`{f}`" in pod_system(PODS[0], PARAMS) for f in ("date", "regime", "facts", "previous"))
    assert all(f"`{f}`" in risk_system(PARAMS) for f in ("date", "regime", "holdings", "facts", "pods", "previous"))
    assert all(f"`{f}`" in pm_system(PARAMS) for f in ("date", "regime", "holdings", "unresearched", "facts", "outcome", "pods", "risk", "previous"))


def test_every_prompt_states_both_text_limits():
    # an unstated limit made the agents resubmit until they looped (run 2026-11-11_135417)
    for text in [pod_system(PODS[0], PARAMS), risk_system(PARAMS), pm_system(PARAMS)]:
        assert f"{PARAMS.reason_max_chars} characters" in text and f"{PARAMS.summary_max_chars} characters" in text


def test_every_prompt_asks_for_a_count_before_the_submit_and_never_quotes_a_hard_cap():
    cases = [(pod_system(PODS[0], PARAMS), POD_SUBMIT), (risk_system(PARAMS), RISK_SUBMIT), (pm_system(PARAMS), PM_SUBMIT)]
    for text, submit in cases:
        assert "`count_characters`" in text and f"before you call `{submit}`" in text
        assert str(PARAMS.reason_hard_max_chars) not in text and str(PARAMS.summary_hard_max_chars) not in text


def test_pm_prompt_keeps_remember_lesson_out_of_the_search_budget():
    text = pm_system(PARAMS)
    assert f"at most {PARAMS.pm_memory_budget} `search_memory` calls" in text
    assert "`remember_lesson` at most once" in text
