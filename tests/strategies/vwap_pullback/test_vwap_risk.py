from __future__ import annotations

from decimal import Decimal as D

import pytest
from tests.fakes import et

from trading_agent_framework.strategies.vwap_pullback import risk
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

PARAMS = VwapPullbackParameters()


def _plan(**overrides: object) -> risk.EntryPlan:
    kwargs: dict[str, object] = {
        "trigger_close": 101.85, "pullback_low": 101.1, "last_price": D("101.90"), "daily_atr": 2.0,
        "equity": D("100000"), "buying_power": D("100000"), "cash": D("100000"), "pending_sell_proceeds": D(0), "params": PARAMS,
    }
    return risk.plan_entry(**(kwargs | overrides))  # ty: ignore[invalid-argument-type]


def test_planned_stop_sits_a_tenth_of_an_atr_under_the_pullback_low() -> None:
    assert risk.planned_stop(101.1, 2.0, PARAMS) == D("100.90")


def test_plan_entry_takes_the_smallest_of_risk_size_and_cash_caps() -> None:
    plan = _plan()
    assert plan.stop_price == D("100.90")
    assert plan.r_per_share == D("0.95")
    assert plan.limit_price == D("102.00")
    assert plan.quantity == D(245)  # risk 526, 25% cap 245, cash 931


def test_plan_entry_refuses_a_stop_too_tight_for_the_band() -> None:
    with pytest.raises(risk.EntryRefused, match="outside"):
        _plan(pullback_low=101.8)


def test_plan_entry_refuses_to_chase() -> None:
    with pytest.raises(risk.EntryRefused, match="chasing"):
        _plan(last_price=D("102.20"))


def test_plan_entry_refuses_a_size_that_rounds_to_zero() -> None:
    with pytest.raises(risk.EntryRefused, match="0 shares"):
        _plan(cash=D("50"), buying_power=D("50"))


def test_entry_window() -> None:
    assert not risk.in_entry_window(et(2026, 9, 1, 9, 44), PARAMS)
    assert risk.in_entry_window(et(2026, 9, 1, 9, 45), PARAMS)
    assert risk.in_entry_window(et(2026, 9, 1, 15, 0), PARAMS)
    assert not risk.in_entry_window(et(2026, 9, 1, 15, 1), PARAMS)


def test_free_slots_and_circuit_breaker() -> None:
    assert risk.free_slots(3, 1, PARAMS) == 0
    assert risk.free_slots(1, 1, PARAMS) == 2
    assert risk.circuit_breaker_tripped(D("-1500"), D("100000"), PARAMS)
    assert not risk.circuit_breaker_tripped(D("-1499"), D("100000"), PARAMS)
