from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest
from tests.fakes import weekday_sessions

from trading_agent_framework.strategies.earnings_drift.reaction import bar_dates, reaction_features
from trading_agent_framework.utils.clock import MARKET_TZ

SESSIONS = weekday_sessions(date(2026, 7, 1), 31)
DAY = SESSIONS[-1].open.date()
FLAT = (100.0, 101.0, 99.0, 100.0, 1_000_000.0)


def _frame(rows: list[tuple[float, float, float, float, float]], *, stamp: str = "close") -> pd.DataFrame:
    sessions = SESSIONS[-len(rows) :]
    index = [s.close if stamp == "close" else datetime.combine(s.open.date(), time(0), tzinfo=MARKET_TZ) for s in sessions]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex(index, name="timestamp"))


STOCK = _frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 4_000_000.0)])
SPY = _frame([(400.0, 401.0, 399.0, 400.0, 1e8)] * 30 + [(400.0, 403.0, 399.0, 402.0, 1e8)])


def test_reaction_features() -> None:
    features = reaction_features(STOCK, SPY, DAY)
    assert features is not None
    assert features.gap_pct == pytest.approx(0.06)
    assert features.return_pct == pytest.approx(0.09)
    assert features.abnormal_pct == pytest.approx(0.085)
    assert features.hold_ratio == pytest.approx(0.9)
    assert features.close_location == pytest.approx(0.8)
    assert features.rel_volume == pytest.approx(4.0)
    assert features.dollar_volume_20d == pytest.approx(1e8)
    assert features.runup_20d_pct == pytest.approx(0.0)
    assert features.runup_60d_pct is None  # 31 sessions are not enough for 60
    assert features.atr14_pct == pytest.approx((13 * 2 + 10) / 14 / 109)
    assert (features.close, features.reaction_low) == (109.0, 105.0)


def test_live_open_stamped_bars_give_the_same_dates_and_features() -> None:
    assert bar_dates(_frame([FLAT] * 3, stamp="open")) == bar_dates(_frame([FLAT] * 3))
    live = reaction_features(_frame([FLAT] * 30 + [(106.0, 110.0, 105.0, 109.0, 4e6)], stamp="open"), SPY, DAY)
    assert live == reaction_features(STOCK, SPY, DAY)


def test_bars_after_the_day_are_ignored() -> None:
    assert reaction_features(STOCK, SPY, SESSIONS[-2].open.date()) is not None
    features = reaction_features(STOCK, SPY, SESSIONS[-2].open.date())
    assert features is not None and features.return_pct == pytest.approx(0.0)


def test_no_features_without_the_day_or_the_baseline() -> None:
    assert reaction_features(STOCK.iloc[:-1], SPY, DAY) is None  # no bar for the day
    assert reaction_features(STOCK.iloc[-20:], SPY, DAY) is None  # 20 rows: no full 20-session baseline before the day
    assert reaction_features(STOCK, SPY.iloc[:-1], DAY) is None  # the benchmark has no bar for the day


def test_a_down_day_has_no_hold_ratio_and_a_flat_bar_sits_mid_range() -> None:
    down = reaction_features(_frame([FLAT] * 30 + [(99.0, 99.5, 97.0, 98.0, 2e6)]), SPY, DAY)
    assert down is not None and down.hold_ratio is None
    flat = reaction_features(_frame([FLAT] * 30 + [(100.0, 100.0, 100.0, 100.0, 2e6)]), SPY, DAY)
    assert flat is not None and flat.close_location == 0.5


def test_sixty_session_runup_with_enough_history() -> None:
    sessions = weekday_sessions(date(2026, 5, 1), 70)
    closes = [50.0] * 8 + [100.0] * 61 + [104.0]
    rows = [(c, c + 1, c - 1, c, 1e6) for c in closes]
    stock = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in sessions]))
    spy = pd.DataFrame([(400.0, 401.0, 399.0, 400.0, 1e8)] * 70, columns=stock.columns, index=stock.index)
    features = reaction_features(stock, spy, sessions[-1].open.date())
    assert features is not None
    assert features.runup_60d_pct == pytest.approx(0.0)  # close before the day (100) vs 60 sessions before that (100)
    closes[-62] = 80.0
    rows = [(c, c + 1, c - 1, c, 1e6) for c in closes]
    stock = pd.DataFrame(rows, columns=stock.columns, index=stock.index)
    assert reaction_features(stock, spy, sessions[-1].open.date()).runup_60d_pct == pytest.approx(0.25)  # type: ignore[union-attr]
