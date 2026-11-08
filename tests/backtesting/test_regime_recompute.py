"""Tests for `backtesting.regime_recompute` (tmp run directories only, never `logs/`)."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent_framework.backtesting import regime_recompute, report
from trading_agent_framework.backtesting.ledger import IndicatorLine, Ledger
from trading_agent_framework.core import regime_cli
from trading_agent_framework.core.regime_data import RegimeSeries

NY = "America/New_York"
DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


def _equity(run: Path, days: list[date] = DAYS) -> None:
    index = pd.DatetimeIndex([pd.Timestamp(d.isoformat() + " 16:00", tz=NY) for d in days], name="datetime")
    pd.DataFrame({"portfolio_value": [1.0] * len(days)}, index=index).to_parquet(run / "equity.parquet")


def _indicators(run: Path, lines: list[tuple[str, int, int, float]]) -> None:
    """Write indicators.parquet via the real writer: (name, day index, hour, value)."""
    ledger = Ledger()
    for name, i, hour, value in lines:
        d = DAYS[i]
        t = datetime(d.year, d.month, d.day, hour, 30, tzinfo=ZoneInfo(NY))
        ledger.record_line(IndicatorLine(t, name, Decimal(str(value)), "#111111", "solid", name))
    report.write_indicators(run, ledger)


def _run(tmp_path: Path, name: str = "r") -> Path:
    run = tmp_path / "s" / "backtesting" / name
    run.mkdir(parents=True)
    _equity(run)
    return run


def _regime_rows(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "indicators.parquet")
    regime: pd.DataFrame = df.loc[df["name"] == "Regime"].reset_index(drop=True)
    return regime


def test_regime_rows_are_replaced_and_other_lines_kept(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _indicators(run, [("Regime", 0, 9, 9.0), ("Other", 0, 10, 5.0), ("Other", 1, 10, 6.0)])
    before = pd.read_parquet(run / "indicators.parquet")

    result = regime_recompute.recompute_run(run, {DAYS[0]: -1, DAYS[1]: 0, DAYS[2]: 1})

    after = pd.read_parquet(run / "indicators.parquet")
    assert result.status == "rewritten"
    assert result.rows == 3
    pd.testing.assert_frame_equal(after[after["name"] == "Other"].reset_index(drop=True), before[before["name"] == "Other"].reset_index(drop=True))
    regime = _regime_rows(run)
    assert list(regime["value"]) == [-1.0, 0.0, 1.0]
    assert {ts.hour for ts in regime["datetime"]} == {9}
    assert after["datetime"].is_monotonic_increasing


def test_a_run_without_a_regime_line_gets_one_at_08_30(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _indicators(run, [("Other", 0, 10, 5.0)])

    result = regime_recompute.recompute_run(run, {DAYS[0]: 1})

    assert result.status == "added"
    row = _regime_rows(run).iloc[0]
    assert row["datetime"] == pd.Timestamp("2026-01-05 08:30", tz=NY)
    assert row["color"] == "#d1d4dc"
    assert row["plot_name"] == "Regime"
    assert row["style"] == "solid"
    assert (run / regime_recompute.BACKUP_NAME).exists()


def test_a_run_without_indicators_gets_a_new_file(tmp_path: Path) -> None:
    run = _run(tmp_path)

    result = regime_recompute.recompute_run(run, {DAYS[0]: 1, DAYS[1]: 1})

    assert result.status == "added"
    assert len(_regime_rows(run)) == 2
    assert not (run / regime_recompute.BACKUP_NAME).exists()


def test_the_written_schema_matches_write_indicators(tmp_path: Path) -> None:
    run = _run(tmp_path)
    regime_recompute.recompute_run(run, {DAYS[0]: 1})
    reference = tmp_path / "ref"
    reference.mkdir()
    _indicators(reference, [("Regime", 0, 8, 1.0)])

    written = pd.read_parquet(run / "indicators.parquet")
    expected = pd.read_parquet(reference / "indicators.parquet")
    assert list(written.dtypes) == list(expected.dtypes)
    assert list(written.columns) == list(expected.columns)


def test_recompute_is_idempotent_and_keeps_the_first_backup(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _indicators(run, [("Regime", 0, 9, 9.0), ("Other", 0, 10, 5.0)])
    original = (run / "indicators.parquet").read_bytes()
    readings = {DAYS[0]: -1, DAYS[1]: 1}

    regime_recompute.recompute_run(run, readings)
    first = _regime_rows(run)
    regime_recompute.recompute_run(run, readings)

    pd.testing.assert_frame_equal(_regime_rows(run), first)
    assert (run / regime_recompute.BACKUP_NAME).read_bytes() == original
    assert not list(run.glob("*.tmp"))


def test_a_run_without_equity_is_skipped_untouched(tmp_path: Path) -> None:
    run = tmp_path / "s" / "backtesting" / "r"
    run.mkdir(parents=True)
    _indicators(run, [("Other", 0, 10, 5.0)])
    original = (run / "indicators.parquet").read_bytes()

    result = regime_recompute.recompute_run(run, {DAYS[0]: 1})

    assert result.status == "skipped"
    assert "equity.parquet" in result.reason
    assert (run / "indicators.parquet").read_bytes() == original
    assert not (run / regime_recompute.BACKUP_NAME).exists()


def test_sessions_without_a_reading_get_no_row(tmp_path: Path) -> None:
    run = _run(tmp_path)

    result = regime_recompute.recompute_run(run, {DAYS[1]: 0})

    assert result.rows == 1
    assert list(_regime_rows(run)["datetime"]) == [pd.Timestamp("2026-01-06 08:30", tz=NY)]


def test_a_run_with_no_reading_at_all_is_skipped(tmp_path: Path) -> None:
    run = _run(tmp_path)

    result = regime_recompute.recompute_run(run, {})

    assert result.status == "skipped"
    assert not (run / "indicators.parquet").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    run = _run(tmp_path)
    _indicators(run, [("Regime", 0, 9, 9.0)])
    original = (run / "indicators.parquet").read_bytes()

    result = regime_recompute.recompute_run(run, {DAYS[0]: 1}, dry_run=True)

    assert result.status == "rewritten"
    assert result.rows == 1
    assert (run / "indicators.parquet").read_bytes() == original
    assert not (run / regime_recompute.BACKUP_NAME).exists()


def test_run_dirs_are_sorted_and_run_sessions_use_new_york_dates(tmp_path: Path) -> None:
    b = _run(tmp_path, "b")
    a = _run(tmp_path, "a")
    assert regime_recompute.run_dirs(tmp_path) == [a, b]
    assert regime_recompute.run_sessions(a) == DAYS
    assert regime_recompute.run_sessions(tmp_path) is None
    assert time(8, 30) == regime_recompute.REGIME_TIME


def test_a_failed_backup_copy_leaves_no_backup_and_the_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path)
    _indicators(run, [("Other", 0, 10, 5.0)])
    original = (run / "indicators.parquet").read_bytes()

    def broken_copy(src: Path, dst: Path) -> None:
        Path(dst).write_bytes(b"trunc")
        raise OSError("disk full")

    monkeypatch.setattr(regime_recompute.shutil, "copy2", broken_copy)

    with pytest.raises(OSError, match="disk full"):
        regime_recompute.recompute_run(run, {DAYS[0]: 1})

    assert not (run / regime_recompute.BACKUP_NAME).exists()
    assert not list(run.glob("*.tmp"))
    assert (run / "indicators.parquet").read_bytes() == original


def test_a_second_run_does_not_back_up_a_file_we_created(tmp_path: Path) -> None:
    run = _run(tmp_path)

    regime_recompute.recompute_run(run, {DAYS[0]: 1})
    regime_recompute.recompute_run(run, {DAYS[0]: -1})

    assert (run / regime_recompute.NO_ORIGINAL_MARKER).exists()
    assert not (run / regime_recompute.BACKUP_NAME).exists()
    assert list(_regime_rows(run)["value"]) == [-1.0]


def test_a_failed_parquet_write_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path)

    def broken(self: pd.DataFrame, path: Path, *args: object, **kwargs: object) -> None:
        Path(path).write_bytes(b"trunc")
        raise OSError("boom")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)

    with pytest.raises(OSError, match="boom"):
        regime_recompute.recompute_run(run, {DAYS[0]: 1})

    assert not list(run.glob("*.tmp"))
    assert not (run / "indicators.parquet").exists()


def test_an_empty_indicators_file_gets_the_write_indicators_dtypes(tmp_path: Path) -> None:
    run = _run(tmp_path)
    report.write_indicators(run, Ledger())
    reference = tmp_path / "ref"
    reference.mkdir()
    _indicators(reference, [("Regime", 0, 8, 1.0)])

    result = regime_recompute.recompute_run(run, {DAYS[0]: 1})

    assert result.status == "added"
    written = pd.read_parquet(run / "indicators.parquet")
    assert list(written.dtypes) == list(pd.read_parquet(reference / "indicators.parquet").dtypes)
    assert len(written) == 1


def test_run_sessions_drops_duplicate_dates(tmp_path: Path) -> None:
    run = tmp_path / "r"
    run.mkdir()
    index = pd.DatetimeIndex([pd.Timestamp("2026-01-05 15:00", tz=NY), pd.Timestamp("2026-01-05 16:00", tz=NY), pd.Timestamp("2026-01-06 16:00", tz=NY)])
    pd.DataFrame({"portfolio_value": [1.0] * 3}, index=index).to_parquet(run / "equity.parquet")

    assert regime_recompute.run_sessions(run) == [DAYS[0], DAYS[1]]


def test_a_corrupt_run_is_skipped_and_the_loop_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    logs, series = _cli_logs(tmp_path)
    bad = logs / "s" / "backtesting" / "a_bad"
    bad.mkdir()
    _equity(bad, list(series.sessions[-3:]))
    (bad / "indicators.parquet").write_bytes(b"not parquet")
    _patch_yahoo(monkeypatch, series)

    code = regime_cli.main(["recompute", "--logs-root", str(logs)])

    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert any(line.startswith("skipped") and "a_bad" in line for line in out)
    assert any(line.startswith("added") and line.endswith("/r") for line in out)
    assert out[-1] == "rewritten: 0, added: 1, skipped: 1"
    assert (bad / "indicators.parquet").read_bytes() == b"not parquet"


def test_unreadable_equity_files_are_skipped_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    logs, series = _cli_logs(tmp_path)
    corrupt = logs / "s" / "backtesting" / "a_corrupt"
    corrupt.mkdir()
    (corrupt / "equity.parquet").write_bytes(b"not parquet")
    naive = logs / "s" / "backtesting" / "b_naive"
    naive.mkdir()
    naive_index = pd.DatetimeIndex([pd.Timestamp(d) for d in series.sessions[-3:]], name="datetime")
    pd.DataFrame({"portfolio_value": [1.0] * 3}, index=naive_index).to_parquet(naive / "equity.parquet")
    _patch_yahoo(monkeypatch, series)

    code = regime_cli.main(["recompute", "--logs-root", str(logs)])

    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert any(line.startswith("skipped") and "a_corrupt" in line and "error:" in line for line in out)
    assert any(line.startswith("skipped") and "b_naive" in line and "error:" in line for line in out)
    assert any(line.startswith("added") and line.endswith("/r") for line in out)
    assert out[-1] == "rewritten: 0, added: 1, skipped: 2"


def _cli_logs(tmp_path: Path) -> tuple[Path, RegimeSeries]:
    logs = tmp_path / "logs"
    run = logs / "s" / "backtesting" / "r"
    run.mkdir(parents=True)
    days: list[date] = []
    d = date(2026, 1, 5)
    while len(days) < 700:
        if d.weekday() < 5:
            days.append(d)
        d = date.fromordinal(d.toordinal() + 1)
    _equity(run, days[-3:])
    n = len(days)
    return logs, RegimeSeries(tuple(days), tuple(100.0 + i * 0.01 for i in range(n)), (15.0,) * n, (18.0,) * n)


def _patch_yahoo(monkeypatch: pytest.MonkeyPatch, series: RegimeSeries) -> None:
    class FakeYahoo:
        def series(self, start: date) -> RegimeSeries:
            return series

    monkeypatch.setattr(regime_cli, "YahooRegimeData", FakeYahoo)


def test_the_cli_recompute_reports_each_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    logs, series = _cli_logs(tmp_path)
    (logs / "s" / "backtesting" / "empty").mkdir()
    _patch_yahoo(monkeypatch, series)

    code = regime_cli.main(["recompute", "--logs-root", str(logs)])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines == [
        f"skipped       0 rows  {logs}/s/backtesting/empty  (no equity.parquet)",
        f"added         3 rows  {logs}/s/backtesting/r",
        "rewritten: 0, added: 1, skipped: 1",
    ]
    assert len(_regime_rows(logs / "s" / "backtesting" / "r")) == 3
