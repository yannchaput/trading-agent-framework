from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest
from tests.fakes import et, make_bars_frame, minute_ohlc

from trading_agent_framework.strategies.vwap_pullback.features import (
    BarContext,
    _benchmark_closes,
    _benchmark_return,
    beta,
    cumulative_volume_by_minute,
    daily_atr,
    intraday_contexts,
    minute_of_session,
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


def _reference_intraday_contexts(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    session_open: datetime,
    now: datetime,
    bar_stamp: Any,
    beta: float,
    baseline: pd.Series,
    minutes: int = 5,
) -> list[BarContext]:
    """The per-bucket-loop implementation `intraday_contexts` replaced, kept to pin that the rewrite changes nothing."""
    if df.empty:
        return []
    mos = minute_of_session(df, session_open, bar_stamp)
    buckets = mos // minutes
    vwaps = vwap_series(df)
    cumulative = df["volume"].astype(float).cumsum()
    open_price = float(df["open"].iloc[0])
    bench_closes, bench_open = _benchmark_closes(benchmark_df, session_open, bar_stamp, minutes)
    contexts: list[BarContext] = []
    session_high = -math.inf
    for bucket_value, rows in cast(Any, df.groupby(buckets.to_numpy(), sort=True)):
        bucket = int(bucket_value)
        close_time = session_open + timedelta(minutes=(bucket + 1) * minutes)
        last_minute = (bucket + 1) * minutes - 1
        label = rows.index[-1]
        has_last_minute = int(mos.loc[label]) == last_minute
        if close_time > now or (not has_last_minute and now - close_time < timedelta(minutes=1)):
            break
        high = float(rows["high"].max())
        session_high = max(session_high, high)
        close = float(rows["close"].iloc[-1])
        bench_return = _benchmark_return(bench_closes, bench_open, bucket)
        contexts.append(
            BarContext(
                time=close_time,
                open=float(rows["open"].iloc[0]),
                high=high,
                low=float(rows["low"].min()),
                close=close,
                volume=float(rows["volume"].sum()),
                vwap=float(vwaps.loc[label]),
                rs=(close / open_price - 1) - beta * bench_return,
                rvol=rvol_at(float(cumulative.loc[label]), last_minute, baseline),
                session_open=open_price,
                session_high=session_high,
            )
        )
    return contexts


def _wavy(count: int, seed: int = 0) -> list[tuple[float, float, float, float, float]]:
    rows = []
    for i in range(count):
        base = 100 + 3 * math.sin(i / 7 + seed)
        rows.append((base, base + 0.4 + (i % 5) * 0.05, base - 0.3 - (i % 3) * 0.05, base + 0.1 * ((i % 4) - 1.5), 100.0 + (i * 37 + seed) % 250))
    return rows


def _assert_same_contexts(actual: list[BarContext], expected: list[BarContext]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.time == want.time
        for name in ("open", "high", "low", "close", "volume", "vwap", "rs", "session_open", "session_high"):
            assert getattr(got, name) == pytest.approx(getattr(want, name)), name
        if want.rvol is None:
            assert got.rvol is None
        else:
            assert got.rvol == pytest.approx(want.rvol)


def _gappy_case(name: str) -> tuple[pd.DataFrame, pd.DataFrame, datetime]:
    """(stock, benchmark, now) for a named gap scenario; minute bars stamped at their open from 09:30."""
    full = minute_ohlc(OPEN, _wavy(60))
    bench = minute_ohlc(OPEN, _wavy(60, seed=2))
    end = et(2026, 9, 1, 10, 30)
    if name == "last_minute_missing_within_grace":  # bucket 1 (09:35-09:40) lacks its 09:39 bar, now < 1 min after its close
        return full.drop(full.index[9]), bench, et(2026, 9, 1, 9, 40, 30)
    if name == "last_minute_missing_beyond_grace":
        return full.drop(full.index[9]), bench, et(2026, 9, 1, 9, 41, 30)
    if name == "bucket_without_rows":  # no trade in 09:40-09:44
        return full.drop(full.index[10:15]), bench, end
    if name == "empty_benchmark":
        return full, bench.iloc[0:0], end
    if name == "now_mid_bucket":
        return full, bench, et(2026, 9, 1, 9, 52, 20)
    if name == "benchmark_missing_buckets":
        return full, bench.drop(bench.index[5:20]), end
    raise AssertionError(name)


@pytest.mark.parametrize("count", [60, 200, 390])
def test_intraday_contexts_matches_reference_on_full_sessions(count: int) -> None:
    stock = minute_ohlc(OPEN, _wavy(count))
    bench = minute_ohlc(OPEN, _wavy(count, seed=2))
    baseline = rvol_baseline([pd.Series([float(i * 150) for i in range(1, count + 1)], index=range(count))])
    kwargs: dict[str, Any] = dict(session_open=OPEN, now=et(2026, 9, 1, 16, 0), bar_stamp="open", beta=1.3, baseline=baseline)
    expected = _reference_intraday_contexts(stock, bench, **kwargs)
    assert len(expected) == count // 5
    _assert_same_contexts(intraday_contexts(stock, bench, **kwargs), expected)


@pytest.mark.parametrize(
    "name",
    ["last_minute_missing_within_grace", "last_minute_missing_beyond_grace", "bucket_without_rows", "empty_benchmark", "now_mid_bucket", "benchmark_missing_buckets"],
)
@pytest.mark.parametrize("bar_stamp", ["open", "close"])
def test_intraday_contexts_matches_reference_on_gappy_data(name: str, bar_stamp: str) -> None:
    stock, bench, now = _gappy_case(name)
    if bar_stamp == "close":  # same bars, stamped at their close
        stock, bench = stock.copy(), bench.copy()
        stock.index = stock.index + timedelta(minutes=1)
        bench.index = bench.index + timedelta(minutes=1)
    baseline = rvol_baseline([pd.Series([float(i * 150) for i in range(1, 61)], index=range(60))])
    kwargs: dict[str, Any] = dict(session_open=OPEN, now=now, bar_stamp=bar_stamp, beta=0.8, baseline=baseline)
    _assert_same_contexts(intraday_contexts(stock, bench, **kwargs), _reference_intraday_contexts(stock, bench, **kwargs))


def test_a_bucket_whose_highs_are_all_missing_does_not_poison_the_running_session_high() -> None:
    """The old loop kept the running max through a NaN high; a plain `np.maximum.accumulate` carried the NaN to every later bucket."""
    stock = minute_ohlc(OPEN, _wavy(60))
    stock.iloc[10:15, stock.columns.get_loc("high")] = float("nan")  # bucket 2 (09:40-09:44) has no usable high
    bench = minute_ohlc(OPEN, _wavy(60, seed=2))
    baseline = rvol_baseline([pd.Series([float(i * 150) for i in range(1, 61)], index=range(60))])
    kwargs: dict[str, Any] = dict(session_open=OPEN, now=et(2026, 9, 1, 10, 30), bar_stamp="open", beta=0.8, baseline=baseline)

    actual = [context.session_high for context in intraday_contexts(stock, bench, **kwargs)]
    expected = [context.session_high for context in _reference_intraday_contexts(stock, bench, **kwargs)]

    assert actual == expected
    assert not any(math.isnan(value) for value in actual)


def test_intraday_contexts_gappy_scenarios_cover_the_gates() -> None:
    """The scenarios above only mean something if they hit different gate outcomes."""
    counts = {}
    for name in ("last_minute_missing_within_grace", "last_minute_missing_beyond_grace", "bucket_without_rows", "now_mid_bucket"):
        stock, bench, now = _gappy_case(name)
        counts[name] = len(_reference_intraday_contexts(stock, bench, session_open=OPEN, now=now, bar_stamp="open", beta=1.0, baseline=pd.Series(dtype=float)))
    assert counts == {"last_minute_missing_within_grace": 1, "last_minute_missing_beyond_grace": 2, "bucket_without_rows": 11, "now_mid_bucket": 4}
