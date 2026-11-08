"""Tests for the `regime` CLI (no network: a fake Yahoo source)."""

import math
from datetime import date, timedelta

import pytest

from trading_agent_framework.core import regime_cli
from trading_agent_framework.core.regime_data import RegimeSeries


def _series(count: int = 800, nan_at: tuple[int, ...] = ()) -> RegimeSeries:
    days: list[date] = []
    day = date(2020, 1, 6)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    spy = [100 + 20 * math.sin(i / 40) + i * 0.02 for i in range(count)]
    vix = [20 + 10 * math.sin(i / 25) ** 2 for i in range(count)]
    vix3m = [22.0] * count
    for i in nan_at:
        vix[i] = math.nan
    return RegimeSeries(tuple(days), tuple(spy), tuple(vix), tuple(vix3m))


def _patch(monkeypatch: pytest.MonkeyPatch, series: RegimeSeries) -> None:
    class FakeYahoo:
        def series(self, start: date) -> RegimeSeries:
            return series

    monkeypatch.setattr(regime_cli, "YahooRegimeData", FakeYahoo)


def test_the_cli_scorecard_prints_the_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    series = _series()
    _patch(monkeypatch, series)

    code = regime_cli.main(["scorecard", "--start", series.sessions[550].isoformat(), "--end", series.sessions[-1].isoformat()])

    out = capsys.readouterr().out
    assert code == 0
    for text in ("recall", "false-alarm bear days", "regime changes", "replay disagreements: 0"):
        assert text in out


def test_leading_nan_rows_are_dropped(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    series = _series(nan_at=(0, 1, 2))
    _patch(monkeypatch, series)

    code = regime_cli.main(["scorecard", "--start", series.sessions[550].isoformat(), "--end", series.sessions[-1].isoformat()])

    assert code == 0
    assert "recall" in capsys.readouterr().out


def test_a_gap_inside_the_window_fails_with_a_clear_message(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    series = _series(nan_at=(600,))
    _patch(monkeypatch, series)

    code = regime_cli.main(["scorecard", "--start", series.sessions[550].isoformat(), "--end", series.sessions[-1].isoformat()])

    assert code == 1
    assert "gap longer than the forward-fill limit" in capsys.readouterr().out


def test_a_window_without_history_fails(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    series = _series(count=100)
    _patch(monkeypatch, series)

    assert regime_cli.main(["scorecard", "--start", "2020-02-03", "--end", "2020-05-01"]) == 1
    assert "enough history" in capsys.readouterr().out
