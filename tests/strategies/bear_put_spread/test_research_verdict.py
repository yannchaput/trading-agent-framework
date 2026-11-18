"""Tests for the SPY fall attribution, drawdown and GO / NO-GO verdict."""

from datetime import date, timedelta

import pytest

from trading_agent_framework.core.regime_scorecard import Episode
from trading_agent_framework.strategies.bear_put_spread.research import EpisodeResult, SimulationResult, Trade, episode_results, max_drawdown, verdict

DAYS = [date(2026, 1, 5) + timedelta(days=i) for i in range(8)]


def _trade(entry: date, pnl: float, reason: str = "regime") -> Trade:
    return Trade(entry, None if reason == "open" else entry, reason, 600.0, 600.0, 588.0, 583.0, entry, 1, 1.5, 1.5, 22.0, pnl, pnl / 1200)


def _episode(*, trades: int, pnl: float, open_: bool = False) -> EpisodeResult:
    return EpisodeResult(Episode(DAYS[0], DAYS[1], -0.12), trades, pnl, open_)


def _result(trades: list[Trade], curve: list[float]) -> SimulationResult:
    return SimulationResult(trades, [(DAYS[i], v) for i, v in enumerate(curve)], 0)


def test_max_drawdown() -> None:
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)
    assert max_drawdown([100]) == 0.0


def test_a_finished_fall_and_one_still_under_way() -> None:
    closes = [100.0, 95.0, 85.0, 90.0, 95.0, 100.0, 95.0, 88.0]
    first, second = episode_results(DAYS, closes, [_trade(DAYS[1], 50.0)])
    assert (first.open, first.trades, first.pnl) == (False, 1, 50.0)
    assert (second.open, second.trades, second.pnl) == (True, 0, 0.0)


def test_a_trade_counts_for_the_latest_fall_that_started_before_it() -> None:
    closes = [100.0, 95.0, 85.0, 90.0, 95.0, 100.0, 95.0, 88.0]
    first, second = episode_results(DAYS, closes, [_trade(DAYS[6], 30.0)])
    assert (first.trades, second.trades, second.pnl) == (0, 1, 30.0)


def test_verdict_ignores_the_open_fall_and_the_open_trade() -> None:
    result = _result([_trade(DAYS[1], -10.0), _trade(DAYS[2], 200.0, "open")], [1200, 1190, 1400])
    got = verdict(result, [_episode(trades=1, pnl=-10.0), _episode(trades=1, pnl=200.0, open_=True)])
    assert got.closed_pnl == pytest.approx(-10.0)
    assert (got.pnl_ok, got.go) == (False, False)
    assert (got.episodes_won, got.episodes_closed) == (0, 1)


def test_a_fall_never_traded_counts_as_not_won() -> None:
    got = verdict(_result([_trade(DAYS[1], 50.0)], [1200, 1250]), [_episode(trades=1, pnl=50.0), _episode(trades=0, pnl=0.0), _episode(trades=0, pnl=0.0)])
    assert (got.episodes_won, got.episodes_closed, got.episodes_ok) == (1, 3, False)


def test_no_closed_fall_fails_the_episode_criterion() -> None:
    assert not verdict(_result([], [1200, 1200]), [_episode(trades=0, pnl=0.0, open_=True)]).episodes_ok


def test_go_when_all_three_pass() -> None:
    got = verdict(_result([_trade(DAYS[1], 400.0)], [1000, 800, 1400]), [_episode(trades=1, pnl=5.0), _episode(trades=1, pnl=5.0), _episode(trades=1, pnl=-5.0)])
    assert got.drawdown == pytest.approx(0.20)
    assert got.go


def test_a_drawdown_over_the_bar_fails() -> None:
    got = verdict(_result([_trade(DAYS[1], 400.0)], [1000, 600, 1400]), [_episode(trades=1, pnl=5.0)])
    assert (got.drawdown_ok, got.go) == (False, False)
