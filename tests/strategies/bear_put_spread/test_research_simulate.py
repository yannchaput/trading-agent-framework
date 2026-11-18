"""Tests for the bear put spread daily simulation."""

from datetime import date, timedelta

import pytest

from trading_agent_framework.strategies.bear_put_spread.pricing import CostModel, VolModel
from trading_agent_framework.strategies.bear_put_spread.research import SimulationResult, simulate
from trading_agent_framework.strategies.bear_put_spread.rules import SpreadParameters


def _days(count: int) -> list[date]:
    days: list[date] = []
    day = date(2026, 1, 5)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _run(spy: list[float], regimes: list[int], equity: float = 1200.0) -> tuple[list[date], SimulationResult]:
    days = _days(len(spy))
    result = simulate(days, spy, [20.0] * len(spy), dict(zip(days, regimes, strict=True)), SpreadParameters(), VolModel(), CostModel(), equity)
    return days, result


def test_a_fall_hits_the_target_then_rolls_next_session() -> None:
    days, result = _run([600 * 0.99**i for i in range(40)], [-1] * 40)
    first, second = result.trades[0], result.trades[1]
    assert first.reason == "target"
    assert first.pnl > 0
    assert first.exit is not None
    assert second.entry == days[days.index(first.exit) + 1]


def test_a_false_alarm_loses_and_exits_on_the_regime() -> None:
    days, result = _run([600.0] * 12, [-1] * 5 + [0] * 7)
    assert [t.reason for t in result.trades] == ["regime"]
    assert result.trades[0].pnl < 0
    assert result.equity[-1][1] == pytest.approx(1200.0 + result.trades[0].pnl)


def test_a_window_starting_bearish_opens_on_the_first_session() -> None:
    days, result = _run([600.0] * 3, [-1] * 3)
    assert result.trades[0].entry == days[0]


def test_too_small_an_account_skips_every_bearish_session() -> None:
    _, result = _run([600.0] * 5, [-1] * 5, equity=100.0)
    assert result.trades == []
    assert result.skipped == 5


def test_an_open_spread_at_the_end_is_marked_net_of_exit_costs() -> None:
    _, result = _run([600.0] * 3, [-1] * 3)
    trade = result.trades[-1]
    costs = CostModel()
    assert trade.reason == "open"
    assert trade.exit is None
    assert trade.costs == pytest.approx(costs.open_cost(trade.contracts) + costs.close_cost(trade.contracts))
    assert result.equity[-1][1] == pytest.approx(1200.0 + sum(t.pnl for t in result.trades))


def test_equity_has_one_point_per_session() -> None:
    days, result = _run([600.0] * 6, [-1, -1, 0, 0, 0, 0])
    assert [d for d, _ in result.equity] == days


def test_a_quiet_market_exits_on_time_before_expiry_and_cash_balances() -> None:
    days, result = _run([600.0] * 45, [-1] * 45)
    first = result.trades[0]
    assert first.reason == "time"
    assert first.exit is not None
    assert (first.expiry - first.exit).days <= SpreadParameters().exit_dte
    assert result.equity[-1][1] == pytest.approx(1200.0 + sum(t.pnl for t in result.trades))
