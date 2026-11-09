"""Tests for the sector_pods prompts."""

import re
from dataclasses import replace

from trading_agent_framework.strategies.sector_pods.handoff import PM_SUBMIT, POD_SUBMIT, RISK_SUBMIT
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import PODS
from trading_agent_framework.strategies.sector_pods.prompts import PM_TASK, POD_TASK, RISK_SYSTEM, RISK_TASK, pm_system, pod_system

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
    assert RISK_SUBMIT in RISK_SYSTEM
    assert PM_SUBMIT in pm_system(PARAMS)


def test_pm_prompt_states_the_caps_from_params():
    text = pm_system(PARAMS)
    assert "25%" in text and "40%" in text and "98%" in text and "SHV" in text
    assert "20%" in pm_system(replace(PARAMS, max_etf_weight=0.2))


def test_no_prompt_contains_forced():
    texts = [pod_system(pod, PARAMS) for pod in PODS] + [RISK_SYSTEM, pm_system(PARAMS), POD_TASK, RISK_TASK, PM_TASK]
    assert not any(re.search(r"forced|\bheld\b", text, re.IGNORECASE) for text in texts)


def test_context_fields_are_named():
    assert all(f"`{f}`" in pod_system(PODS[0], PARAMS) for f in ("date", "regime", "facts", "previous"))
    assert all(f"`{f}`" in RISK_SYSTEM for f in ("date", "regime", "holdings", "facts", "pods", "previous"))
    assert all(f"`{f}`" in pm_system(PARAMS) for f in ("date", "regime", "holdings", "unresearched", "facts", "outcome", "pods", "risk", "previous"))
