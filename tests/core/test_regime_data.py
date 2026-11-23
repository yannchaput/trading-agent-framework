"""Tests for the Yahoo SPY/VIX/VIX3M/VIXY series (no network: a fake download)."""

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from trading_agent_framework.core.regime_data import SYMBOLS, RegimeSeries, YahooRegimeData
from trading_agent_framework.utils.errors import RegimeDataError

FIRST = date(2025, 1, 6)  # a Monday


def _weekdays(count: int) -> list[date]:
    days: list[date] = []
    day = FIRST
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _frame(days: list[date], base: float, volume: float) -> pd.DataFrame:
    close = [base + i for i in range(len(days))]
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": [c - 0.5 if base == 100 else c for c in close],
            "Volume": [volume] * len(days),
            "Stock Splits": [0.0] * len(days),
        },
        index=pd.DatetimeIndex([pd.Timestamp(d) for d in days]),
    )


class FakeDownload:
    """Records calls and returns a group_by="ticker" frame; `drop` removes dates of a symbol."""

    def __init__(
        self,
        days: list[date],
        drop: dict[str, list[date]] | None = None,
        absent: tuple[str, ...] = (),
        vixy_frame: pd.DataFrame | None = None,
    ) -> None:
        self.days = days
        self.vixy_frame = vixy_frame
        self.drop = drop or {}
        self.absent = absent
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, symbols: list[str], **kwargs: Any) -> pd.DataFrame:
        self.calls.append((list(symbols), kwargs))
        parts = {}
        for symbol, base, volume in (("SPY", 100.0, 1e6), ("^VIX", 15.0, 0.0), ("^VIX3M", 17.0, 0.0), ("VIXY", 20.0, 1e6)):
            if symbol in self.absent:
                continue
            frame = self.vixy_frame if symbol == "VIXY" and self.vixy_frame is not None else _frame(self.days, base, volume)
            gone = pd.DatetimeIndex([pd.Timestamp(d) for d in self.drop.get(symbol, [])])
            parts[symbol] = frame.drop(index=gone)
        return pd.concat(parts, axis=1)


def _source(fake: Any, today: date = date(2025, 12, 31)) -> YahooRegimeData:
    return YahooRegimeData(download=fake, today=lambda: today)


def test_one_batched_download_of_the_four_symbols() -> None:
    fake = FakeDownload(_weekdays(30))
    _source(fake).series(FIRST)
    assert len(fake.calls) == 1
    symbols, kwargs = fake.calls[0]
    assert symbols == ["SPY", "^VIX", "^VIX3M", "VIXY"] == list(SYMBOLS)
    assert kwargs["auto_adjust"] is False
    assert kwargs["actions"] is True
    assert kwargs["group_by"] == "ticker"
    assert kwargs["end"] == "2026-01-01"
    assert kwargs["start"] == FIRST.isoformat()


def test_spy_closes_are_the_adjusted_closes() -> None:
    days = _weekdays(10)
    series = _source(FakeDownload(days)).series(FIRST)
    assert series.spy == tuple(99.5 + i for i in range(10))
    assert series.vix == tuple(15.0 + i for i in range(10))
    assert series.sessions == tuple(days)
    assert len(series) == 10


def test_the_series_is_aligned_on_spy_sessions_and_forward_fills_up_to_five() -> None:
    days = _weekdays(12)
    missing = days[4:7]
    series = _source(FakeDownload(days, drop={"^VIX": missing})).series(FIRST)
    assert series.sessions == tuple(days)
    assert series.vix[4:7] == (series.vix[3],) * 3


def test_a_gap_longer_than_five_sessions_fails_the_slice() -> None:
    days = _weekdays(15)
    series = _source(FakeDownload(days, drop={"^VIX": days[4:10]})).series(FIRST)
    assert any(v != v for v in series.vix)  # nan
    with pytest.raises(RegimeDataError):
        series.before(days[-1] + timedelta(days=1), 15)
    assert len(series.before(days[-1] + timedelta(days=1), 4)) == 4  # the tail is clean


def test_closes_before_excludes_the_session_itself() -> None:
    days = _weekdays(30)
    session = days[20]
    out = _source(FakeDownload(days)).closes_before(session, 3)
    assert len(out) == 3
    assert out.sessions[-1] < session
    assert out.sessions[-1] == days[19]


def test_one_download_per_day_is_reused() -> None:
    days = _weekdays(60)
    fake = FakeDownload(days)
    state = {"today": date(2025, 12, 31)}
    source = YahooRegimeData(download=fake, today=lambda: state["today"])
    source.closes_before(days[40], 5)
    source.closes_before(days[40], 5)
    source.closes_before(days[50], 5)  # a later start is served from the cache
    assert len(fake.calls) == 1
    state["today"] = date(2026, 1, 2)
    source.closes_before(days[40], 5)
    assert len(fake.calls) == 2


def test_a_failed_or_empty_download_raises_regime_data_error() -> None:
    def boom(symbols: list[str], **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("offline")

    with pytest.raises(RegimeDataError):
        _source(boom).series(FIRST)
    with pytest.raises(RegimeDataError):
        _source(lambda symbols, **kwargs: pd.DataFrame()).series(FIRST)


def test_unsorted_or_duplicate_rows_are_normalised() -> None:
    days = _weekdays(8)

    def messy(symbols: list[str], **kwargs: Any) -> pd.DataFrame:
        raw = FakeDownload(days)(symbols, **kwargs)
        return pd.concat([raw.iloc[::-1], raw.iloc[[3]]])

    series = _source(messy).series(FIRST)
    assert list(series.sessions) == sorted(set(series.sessions))
    assert len(series) == 8


def test_before_returns_a_regime_series() -> None:
    assert isinstance(_source(FakeDownload(_weekdays(10))).series(FIRST).before(date(2026, 1, 1), 2), RegimeSeries)


@pytest.mark.parametrize("symbol", ["^VIX", "^VIX3M"])
def test_a_partial_download_raises_and_is_not_cached(symbol: str) -> None:
    days = _weekdays(30)
    fake = FakeDownload(days, absent=(symbol,))
    source = _source(fake)
    with pytest.raises(RegimeDataError, match="no .* closes"):
        source.series(FIRST)
    fake.absent = ()
    assert len(source.series(FIRST)) == 30  # the retry downloads again and succeeds
    assert len(fake.calls) == 2


def test_a_series_starting_after_the_window_reports_missing_history_not_a_gap() -> None:
    days = _weekdays(30)
    series = _source(FakeDownload(days, drop={"^VIX3M": days[:3]})).series(FIRST)
    with pytest.raises(RegimeDataError, match="not enough aligned history"):
        series.before(days[-1], 29)
    assert len(series.before(days[-1], 20)) == 20


def test_vixy_is_aligned_on_spy_sessions() -> None:
    series = _source(FakeDownload(_weekdays(10))).series(FIRST)
    assert series.vixy == tuple(20.0 + i for i in range(10))


def test_a_missing_vixy_is_nan_not_an_error() -> None:
    series = _source(FakeDownload(_weekdays(10), absent=("VIXY",))).series(FIRST)
    assert len(series.vixy) == 10 and all(v != v for v in series.vixy)
    assert series.spy == tuple(99.5 + i for i in range(10))  # the regime's inputs are untouched


def test_before_slices_vixy_and_ignores_its_nan() -> None:
    days = _weekdays(30)
    series = _source(FakeDownload(days, drop={"VIXY": days[:20]})).series(FIRST)  # VIXY starts late
    out = series.before(days[-1], 25)
    assert len(out.vixy) == 25 and out.vixy[-1] == series.vixy[-2]


def test_a_series_without_vixy_keeps_an_empty_column() -> None:
    old = RegimeSeries((date(2026, 1, 2),), (100.0,), (15.0,), (17.0,))
    assert old.vixy == () and old.before(date(2026, 1, 5), 1).vixy == ()


def test_vixy_reads_adjusted_closes_so_a_reverse_split_is_no_jump() -> None:
    days = _weekdays(10)
    frame = _frame(days, 20.0, 1e6)
    frame["Adj Close"] = frame["Close"]  # smooth
    frame["Close"] = [c * 4 if i >= 5 else c for i, c in enumerate(frame["Close"])]  # 1:4 reverse split
    series = _source(FakeDownload(days, vixy_frame=frame)).series(FIRST)
    assert max(b / a for a, b in zip(series.vixy, series.vixy[1:], strict=False)) < 1.1
