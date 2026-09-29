from __future__ import annotations

import pandas as pd
import pytest
from tests.fakes import et, make_bars_frame, minute_ohlc

from trading_agent_framework.strategies.vwap_pullback.features import (
    bar_atr,
    beta,
    cumulative_volume_by_minute,
    daily_atr,
    ema_last,
    intraday_contexts,
    latest_levels,
    rvol_at,
    rvol_baseline,
    session_slice,
    vwap_series,
    zscores,
)

OPEN = et(2026, 9, 1, 9, 30)
CLOSE = et(2026, 9, 1, 16, 0)
FLAT = (1.0, 1.0, 1.0, 1.0, 1.0)


def _rising(count: int) -> list[tuple[float, float, float, float, float]]:
    return [(100 + 0.1 * i, 100 + 0.1 * i + 0.05, 100 + 0.1 * i - 0.05, 100 + 0.1 * (i + 1), 200.0) for i in range(count)]


def test_session_slice_drops_premarket_for_open_and_close_stamped_bars() -> None:
    live = minute_ohlc(et(2026, 9, 1, 9, 29), [FLAT] * 3)  # bars starting 09:29, 09:30, 09:31
    assert list(session_slice(live, OPEN, CLOSE, "open").index) == [et(2026, 9, 1, 9, 30), et(2026, 9, 1, 9, 31)]
    backtest = minute_ohlc(et(2026, 9, 1, 9, 30), [FLAT] * 3)  # bars closing 09:30 (premarket), 09:31, 09:32
    assert list(session_slice(backtest, OPEN, CLOSE, "close").index) == [et(2026, 9, 1, 9, 31), et(2026, 9, 1, 9, 32)]


def test_vwap_weights_the_typical_price_by_volume() -> None:
    df = minute_ohlc(OPEN, [(10, 12, 8, 10, 100), (20, 22, 18, 20, 300)])
    assert vwap_series(df).tolist() == pytest.approx([10.0, 17.5])


def test_cumulative_volume_carries_over_minutes_without_a_bar() -> None:
    df = pd.concat([minute_ohlc(OPEN, [(1, 1, 1, 1, 100)]), minute_ohlc(et(2026, 9, 1, 9, 33), [(1, 1, 1, 1, 50)])])
    assert cumulative_volume_by_minute(df, OPEN, "open").tolist() == [100, 100, 100, 150]


def test_rvol_baseline_averages_only_the_sessions_that_reached_the_minute() -> None:
    baseline = rvol_baseline([pd.Series([10.0, 20.0, 30.0]), pd.Series([30.0, 40.0])])
    assert baseline.tolist() == [20.0, 30.0, 30.0]
    assert rvol_at(60.0, 1, baseline) == 2.0
    assert rvol_at(60.0, 5, baseline) is None


def test_intraday_contexts_builds_completed_bars_with_rs_and_rvol() -> None:
    stock = minute_ohlc(OPEN, _rising(10))
    spy = minute_ohlc(OPEN, [(400, 400, 400, 400, 1000)] * 9 + [(400, 404, 400, 404, 1000)])
    baseline = pd.Series([100.0 * (m + 1) for m in range(390)])
    contexts = intraday_contexts(stock, spy, session_open=OPEN, now=et(2026, 9, 1, 9, 40), bar_stamp="open", beta=1.0, baseline=baseline)
    assert [c.time for c in contexts] == [et(2026, 9, 1, 9, 35), et(2026, 9, 1, 9, 40)]
    first, second = contexts
    assert first.close == pytest.approx(100.5)
    assert first.volume == 1000
    assert first.rvol == pytest.approx(2.0)  # 1000 traded vs 500 expected by minute 4
    assert first.rs == pytest.approx(0.005)
    assert second.rs == pytest.approx(0.0)  # +1% while SPY did +1%
    assert second.session_high == pytest.approx(100.95)
    assert second.session_open == 100.0


def test_intraday_contexts_waits_for_a_bucket_whose_last_minute_is_not_published_yet() -> None:
    stock = minute_ohlc(OPEN, _rising(9))  # the 09:39 bar has not landed yet
    spy = minute_ohlc(OPEN, [(400, 400, 400, 400, 1000)] * 9)
    baseline = pd.Series([100.0 * (m + 1) for m in range(390)])
    kwargs = {"session_open": OPEN, "bar_stamp": "open", "beta": 1.0, "baseline": baseline}
    assert len(intraday_contexts(stock, spy, now=et(2026, 9, 1, 9, 40, 10), **kwargs)) == 1
    assert len(intraday_contexts(stock, spy, now=et(2026, 9, 1, 9, 41), **kwargs)) == 2


def test_intraday_contexts_of_an_empty_frame_is_empty() -> None:
    empty = minute_ohlc(OPEN, [])
    assert intraday_contexts(empty, empty, session_open=OPEN, now=CLOSE, bar_stamp="open", beta=1.0, baseline=pd.Series(dtype=float)) == []


def test_daily_atr_uses_the_true_range() -> None:
    assert daily_atr(make_bars_frame([100.0] * 20), 14) == pytest.approx(2.0)
    assert daily_atr(make_bars_frame([100.0] * 10), 14) is None


def test_beta_of_a_doubled_return_series_is_two() -> None:
    moves = [0.01 if i % 2 else -0.01 for i in range(40)]
    spy, stock = [100.0], [100.0]
    for move in moves:
        spy.append(spy[-1] * (1 + move))
        stock.append(stock[-1] * (1 + 2 * move))
    index = pd.date_range(et(2026, 6, 1), periods=41, freq="1D")
    assert beta(pd.Series(stock, index=index), pd.Series(spy, index=index), 60) == pytest.approx(2.0)
    assert beta(pd.Series(stock[:5], index=index[:5]), pd.Series(spy[:5], index=index[:5]), 60) == 1.0  # too few returns


def test_zscores() -> None:
    assert zscores({"A": 1.0, "B": 1.0}) == {"A": 0.0, "B": 0.0}
    assert zscores({"A": 1.0, "B": 3.0}) == {"A": -1.0, "B": 1.0}
    assert zscores({"A": 5.0}) == {"A": 0.0}


def test_ema_bar_atr_and_latest_levels() -> None:
    assert ema_last([1.0, 1.0, 1.0], 9) == 1.0
    assert ema_last([], 9) is None
    contexts = intraday_contexts(
        minute_ohlc(OPEN, _rising(10)), minute_ohlc(OPEN, []), session_open=OPEN, now=CLOSE, bar_stamp="open", beta=1.0, baseline=pd.Series(dtype=float)
    )
    assert bar_atr(contexts, 14) == pytest.approx(0.5)  # both 5-minute bars span 0.5
    levels = latest_levels(contexts, ema_length=9, atr_length=14)
    assert levels is not None and levels.close == pytest.approx(101.0)
    assert latest_levels([], ema_length=9, atr_length=14) is None
