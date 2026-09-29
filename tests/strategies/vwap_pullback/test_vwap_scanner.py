from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et, make_bars_frame, make_session, minute_ohlc

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState

DAY = date(2026, 9, 2)
PARAMS = dataclasses.replace(VwapPullbackParameters(), rvol_baseline_sessions=2)


def _strategy(tmp_path: Path, now: pd.Timestamp) -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap")
    return Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path), broker


def _minutes(day: date, count: int, volume: float) -> pd.DataFrame:
    return minute_ohlc(et(day.year, day.month, day.day, 9, 30), [(100, 100, 100, 100, volume)] * count)


def test_prepare_session_runs_stage_one_and_builds_rvol_baselines(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 8, 30))
    start = et(2026, 6, 1)
    broker.timestep_frames = {
        ("AAA", "day"): make_bars_frame([90.0 + 0.5 * i for i in range(71)], start=start),
        ("BBB", "day"): make_bars_frame([100.0] * 71, start=start),
        ("CHEAP", "day"): make_bars_frame([3.0] * 71, start=start),
        ("WILD", "day"): make_bars_frame([20.0] * 71, start=start),
        ("SPY", "day"): make_bars_frame([400.0] * 71, start=start),
        ("AAA", "minute"): pd.concat([
            _minutes(date(2026, 8, 31), 5, 100),
            minute_ohlc(et(2026, 9, 1, 9, 29), [(100, 100, 100, 100, 999)]),  # premarket: excluded
            _minutes(date(2026, 9, 1), 5, 100),
        ]),
    }
    state = Scanner(strategy, PARAMS, ["AAA", "BBB", "CHEAP", "WILD"]).prepare_session()
    assert state.day == DAY
    assert state.bar_stamp == "open"
    assert set(state.candidates) == {"AAA", "BBB"}
    assert state.candidates["BBB"].daily_atr == pytest.approx(2.0)
    assert state.baselines["AAA"].tolist() == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert "BBB" not in state.baselines  # no minute data: no baseline, so it cannot pass the stage-2 floor
    assert state.session_open_equity == Decimal("25000")


def _state(**candidates: CandidateInfo) -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"), candidates=dict(candidates))


def test_scan_builds_contexts_and_advances_setups(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 9, 50, 30))
    rising = [(100 + 0.06 * i, 100 + 0.06 * (i + 1) + 0.01, 100 + 0.06 * i - 0.01, 100 + 0.06 * (i + 1), 300.0) for i in range(20)]
    broker.timestep_frames = {
        ("AAA", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), rising),
        ("SPY", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), [(400, 400, 400, 400, 1000)] * 20),
    }
    broker.news = FakeNewsProvider()
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.baselines["AAA"] = pd.Series([100.0 * (m + 1) for m in range(390)])
    Scanner(strategy, PARAMS, ["AAA"]).scan(state)
    assert [c.time for c in state.contexts["AAA"]][-1] == et(2026, 9, 2, 9, 50)
    assert state.setups["AAA"].state is SetupState.IMPULSE
    assert state.setups["AAA"].impulse_high == pytest.approx(101.21)
    assert broker.news.calls == []  # headlines are only fetched for pullback/triggered setups and open trades


def test_refresh_headlines_fetches_active_setups_and_flags_new_ones(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 10, 0))
    broker.news = FakeNewsProvider({"AAA": [{"headline": "AAA beats", "created_at": "2026-09-02T07:00:00-04:00", "source": "b"}]})
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.PULLBACK)
    scanner = Scanner(strategy, PARAMS, ["AAA"])
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 0))
    assert state.headlines["AAA"][0]["headline"] == "AAA beats"
    assert state.new_headline == set()  # the first fetch is the baseline, not news
    broker.news.articles["AAA"].append({"headline": "AAA upgraded", "created_at": "2026-09-02T10:10:00-04:00", "source": "b"})
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 10))  # within the refresh interval: no call
    assert len(broker.news.calls) == 1
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 15))
    assert state.new_headline == {"AAA"}
