"""Tests for the `bear-spread-research` command (no network: a fake Yahoo source)."""

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from trading_agent_framework.core.regime_data import RegimeDataError, RegimeSeries
from trading_agent_framework.strategies.bear_put_spread import research_cli

FALL_START = 600


def _series(count: int = 900) -> RegimeSeries:
    days: list[date] = []
    day = date(2020, 1, 6)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    spy = [100 + 0.05 * i if i < FALL_START else (100 + 0.05 * FALL_START) * 0.99 ** (i - FALL_START) if i < FALL_START + 30 else 0.0 for i in range(count)]
    floor = spy[FALL_START + 29]
    spy = [v if v else floor for v in spy]
    vix = [35.0 if FALL_START <= i < FALL_START + 60 else 15.0 for i in range(count)]
    return RegimeSeries(tuple(days), tuple(spy), tuple(vix), tuple([18.0] * count))


def _patch(monkeypatch: pytest.MonkeyPatch, series: RegimeSeries | None = None, error: str | None = None) -> RegimeSeries:
    data = series or _series()

    class FakeYahoo:
        def series(self, start: date) -> RegimeSeries:
            if error:
                raise RegimeDataError(error)
            return data

    monkeypatch.setattr(research_cli, "YahooRegimeData", FakeYahoo)
    return data


def _first_debit(out: Path) -> float:
    with (out / "trades.csv").open() as handle:
        return float(next(csv.DictReader(handle))["debit"])


def test_prints_a_verdict_and_writes_three_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = _patch(monkeypatch)

    code = research_cli.main(["--start", data.sessions[550].isoformat(), "--equity", "100000", "--out", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    assert ("verdict: GO" in out) or ("verdict: NO-GO" in out)
    assert out.count("PASS") + out.count("FAIL") == 3
    assert "skipped: too small" in out
    assert {p.name for p in tmp_path.iterdir()} == {"trades.csv", "episodes.csv", "equity.csv"}


def test_flags_reach_the_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _patch(monkeypatch)
    base = ["--start", data.sessions[550].isoformat(), "--equity", "100000"]

    research_cli.main([*base, "--out", str(tmp_path / "a")])
    research_cli.main([*base, "--vol-scale", "2", "--out", str(tmp_path / "b")])

    assert _first_debit(tmp_path / "a") != _first_debit(tmp_path / "b")


def test_a_download_error_exits_1_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch(monkeypatch, error="Yahoo down")

    assert research_cli.main(["--out", str(tmp_path)]) == 1

    assert "error: Yahoo down" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_an_empty_window_exits_1_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = _patch(monkeypatch)

    code = research_cli.main(["--start", data.sessions[550].isoformat(), "--end", data.sessions[10].isoformat(), "--out", str(tmp_path)])

    assert code == 1
    assert "error:" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("trigger", ["t1", "t2", "t3"])
def test_a_trigger_replaces_the_regime(trigger: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = _patch(monkeypatch)

    code = research_cli.main(["--start", data.sessions[550].isoformat(), "--equity", "100000", "--trigger", trigger, "--out", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    assert f"trigger: {trigger}" in out
    assert ("verdict: GO" in out) or ("verdict: NO-GO" in out)


def test_the_default_trigger_is_the_regime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    data = _patch(monkeypatch)

    research_cli.main(["--start", data.sessions[550].isoformat(), "--out", str(tmp_path)])

    assert "trigger: regime" in capsys.readouterr().out
