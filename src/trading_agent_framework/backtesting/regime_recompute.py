"""Recompute the `Regime` line of existing backtest runs from the SPY/VIX regime.

Only `indicators.parquet` changes: its `Regime` rows are replaced, every other line is kept. The first time a run's file
is rewritten it is copied to `BACKUP_NAME`, which is never overwritten, so the original can always be restored.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Literal

from trading_agent_framework.core.regime import REGIME_LINE

BACKUP_NAME = "indicators.pre-vix-regime.parquet"
NO_ORIGINAL_MARKER = "indicators.pre-vix-regime.none"  # empty file: the run had no indicators.parquet, so there is nothing to back up
REGIME_TIME = time(8, 30)  # America/New_York, the time of day of the Regime lines the strategies chart
REGIME_COLOR = "#d1d4dc"
_TZ = "America/New_York"
_COLUMNS = ("datetime", "name", "value", "color", "style", "plot_name")


@dataclass(frozen=True, slots=True)
class RecomputeResult:
    """The outcome for one run.

    Attributes:
        run_dir: The run directory.
        status: "rewritten" (it had Regime rows), "added" (it had none) or "skipped" (nothing touched).
        reason: Why the run was skipped, else empty.
        rows: The number of Regime rows the run has (or would have) afterwards.
    """

    run_dir: Path
    status: Literal["rewritten", "added", "skipped"]
    reason: str = ""
    rows: int = 0


def run_dirs(logs_root: Path) -> list[Path]:
    """The run directories `logs_root/<strategy>/backtesting/<run>/`, sorted.

    Args:
        logs_root: The logs directory.

    Returns:
        The sorted run directories.
    """
    return sorted(p for p in logs_root.glob("*/backtesting/*") if p.is_dir())


def run_sessions(run_dir: Path) -> list[date] | None:
    """The run's sessions: the New York dates of `equity.parquet`'s index.

    Args:
        run_dir: The run directory.

    Returns:
        The dates, or None when the run has no `equity.parquet`.
    """
    import pandas as pd

    path = run_dir / "equity.parquet"
    if not path.exists():
        return None
    index = pd.read_parquet(path).index
    return list(dict.fromkeys(ts.date() for ts in pd.DatetimeIndex(index).tz_convert(_TZ)))


def recompute_run(run_dir: Path, readings: Mapping[date, int], *, dry_run: bool = False) -> RecomputeResult:
    """Replace the `Regime` rows of one run's `indicators.parquet`.

    Args:
        run_dir: The run directory.
        readings: The regime (-1, 0 or 1) of each session that has one.
        dry_run: Compute the result without writing anything.

    Returns:
        The outcome; a run without `equity.parquet` or without any reading is skipped and left untouched.
    """
    import pandas as pd

    sessions = run_sessions(run_dir)
    if sessions is None:
        return RecomputeResult(run_dir, "skipped", "no equity.parquet")
    dated = [(d, readings[d]) for d in sessions if d in readings]
    if not dated:
        return RecomputeResult(run_dir, "skipped", "no session with a regime reading")

    path = run_dir / "indicators.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    had_regime = False
    at = REGIME_TIME
    if old is not None:
        stamps = old.loc[old["name"] == REGIME_LINE, "datetime"].tolist()
        had_regime = bool(stamps)
        if stamps:
            at = time(stamps[0].hour, stamps[0].minute, stamps[0].second)
    new = pd.DataFrame(
        {
            "datetime": pd.Series([pd.Timestamp(f"{d.isoformat()} {at.isoformat()}", tz=_TZ) for d, _ in dated], dtype=f"datetime64[us, {_TZ}]"),
            "name": REGIME_LINE,
            "value": [float(v) for _, v in dated],
            "color": REGIME_COLOR,
            "style": "solid",
            "plot_name": REGIME_LINE,
        },
        columns=list(_COLUMNS),
    )
    status: Literal["rewritten", "added"] = "added"
    combined = new
    if old is not None:
        if had_regime:
            status = "rewritten"
        combined = pd.concat([old[old["name"] != REGIME_LINE], new], ignore_index=True)
        combined = combined.iloc[combined["datetime"].argsort(kind="mergesort")].reset_index(drop=True)
    result = RecomputeResult(run_dir, status, rows=len(new))
    if dry_run:
        return result

    backup = run_dir / BACKUP_NAME
    marker = run_dir / NO_ORIGINAL_MARKER
    if not backup.exists() and not marker.exists():
        if old is None:
            marker.touch()
        else:
            _atomic_copy(path, backup)
    temp = path.with_name(path.name + ".tmp")
    try:
        combined.to_parquet(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return result


def _atomic_copy(source: Path, target: Path) -> None:
    """Copy `source` to `target` so that `target` only ever exists complete."""
    temp = target.with_name(target.name + ".tmp")
    try:
        shutil.copy2(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
