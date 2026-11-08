"""The `regime` command: `scorecard` and `recompute`.

`uv run regime scorecard [--start YYYY-MM-DD] [--end YYYY-MM-DD]` and `uv run regime recompute [--logs-root logs] [--dry-run]`.

`scorecard` downloads SPY, VIX and VIX3M from Yahoo, computes the regime of every session in the window and
scores it against SPY's swing falls. It is read-only. `recompute` rewrites the `Regime` line of the existing backtest
runs under the logs directory (the original `indicators.parquet` is kept once as a backup).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

from trading_agent_framework.backtesting.regime_recompute import RecomputeResult, recompute_run, run_dirs, run_sessions
from trading_agent_framework.core.regime import RegimeParameters, RegimeReading, regime_by_session, regime_path
from trading_agent_framework.core.regime_data import RegimeSeries, YahooRegimeData, history_days
from trading_agent_framework.core.regime_scorecard import Scorecard, score
from trading_agent_framework.utils.errors import RegimeDataError

DEFAULT_START = date(2016, 1, 4)
DEFAULT_END = date(2026, 10, 5)
_RUN_ERRORS = (OSError, ValueError, KeyError, TypeError)  # an unreadable or malformed run is skipped, never fatal


def lookback_days() -> int:
    """Calendar days downloaded before the first scored session, enough for the regime's `min_bars`."""
    return history_days(RegimeParameters().min_bars)


def trim(series: RegimeSeries, end: date) -> RegimeSeries:
    """The rows up to `end`, minus leading rows until the first one with no `nan`.

    Raises:
        RegimeDataError: If no complete row exists or a `nan` remains after the first complete row.
    """
    rows = [i for i, d in enumerate(series.sessions) if d <= end]
    values = [(series.spy[i], series.vix[i], series.vix3m[i]) for i in rows]
    first = next((k for k, row in enumerate(values) if not any(math.isnan(v) for v in row)), None)
    if first is None:
        raise RegimeDataError("no session has SPY, VIX and VIX3M closes")
    kept = rows[first:]
    for name, column in (("spy", series.spy), ("vix", series.vix), ("vix3m", series.vix3m)):
        gaps = [series.sessions[i] for i in kept if math.isnan(column[i])]
        if gaps:
            raise RegimeDataError(f"{name} has a gap longer than the forward-fill limit in the window, first {gaps[0]}")
    return RegimeSeries(
        tuple(series.sessions[i] for i in kept),
        tuple(series.spy[i] for i in kept),
        tuple(series.vix[i] for i in kept),
        tuple(series.vix3m[i] for i in kept),
    )


def replay_disagreements(series: RegimeSeries, params: RegimeParameters, window: Sequence[date], readings: Mapping[date, RegimeReading]) -> int:
    """Window sessions where the per-session regime differs from one continuous replay of the whole series.

    Args:
        series: The aligned closes.
        params: Thresholds.
        window: The sessions counted.
        readings: The per-session readings of `series` (`regime_by_session`).

    Returns:
        The number of sessions with different regimes.
    """
    if len(series) <= params.lookback:
        return 0
    path = regime_path(series.spy, series.vix, series.vix3m, params, first=params.lookback - 1)
    position = {d: i for i, d in enumerate(series.sessions)}
    count = 0
    for d in window:
        j = position[d]  # reading i applies to session i + 1; path[0] is reading lookback - 1
        if path[j - params.lookback].regime != readings[d].regime:
            count += 1
    return count


def _print_scorecard(card: Scorecard, disagreements: int) -> None:
    """Print the per-episode table and the summary."""
    print("episode peak -> trough      depth   bear on at   coverage   overstay")
    for s in card.episodes:
        on = "never" if s.bear_on_fall is None else f"{s.bear_on_fall:+.1%}"
        print(f"{s.episode.peak} -> {s.episode.trough}  {s.episode.depth:7.1%}  {on:>10}  {s.coverage:8.0%}  {s.overstay:8d}d")
    print(f"reference episodes: {len(card.episodes)}")
    print(f"recall: {card.recall:.2f}")
    print(f"false-alarm bear days: {card.false_bear_days}")
    print(f"mean fall at switch-on: {card.mean_on_fall:+.1%}")
    print(f"median overstay: {card.median_overstay:g} sessions")
    print(f"regime changes: {card.changes}")
    print(f"replay disagreements: {disagreements}")


def run_scorecard(start: date, end: date) -> int:
    """Download the closes, score the regime over `[start, end]` and print the result.

    Args:
        start: First scored session.
        end: Last scored session.

    Returns:
        The exit code: 0, or 1 when the data cannot be used.
    """
    params = RegimeParameters()
    try:
        series = trim(YahooRegimeData().series(start - timedelta(days=lookback_days())), end)
    except RegimeDataError as exc:
        print(f"error: {exc}")
        return 1
    readings = regime_by_session(series.sessions, series.spy, series.vix, series.vix3m, params)
    close = dict(zip(series.sessions, series.spy, strict=True))
    window = [d for d in series.sessions if start <= d <= end and d in readings]
    if not window:
        print(f"error: no session between {start} and {end} has enough history")
        return 1
    card = score(window, [close[d] for d in window], {d: readings[d].regime for d in window})
    _print_scorecard(card, replay_disagreements(series, params, window, readings))
    return 0


def run_recompute(logs_root: Path, dry_run: bool) -> int:
    """Recompute the `Regime` line of every backtest run under `logs_root`.

    Args:
        logs_root: The logs directory (`<strategy>/backtesting/<run>/`).
        dry_run: Print what would change and write nothing.

    Returns:
        The exit code: 0, or 1 when the data cannot be used.
    """
    runs: list[Path] = []
    failed: dict[Path, str] = {}
    all_sessions: list[date] = []
    for run in run_dirs(logs_root):
        runs.append(run)
        try:
            all_sessions.extend(run_sessions(run) or [])
        except _RUN_ERRORS as exc:
            failed[run] = f"error: {exc}"
    if not all_sessions:
        print(f"no backtest run with an equity.parquet under {logs_root}")
        return 0
    first, last = min(all_sessions), max(all_sessions)
    try:
        series = trim(YahooRegimeData().series(first - timedelta(days=lookback_days())), last)
    except RegimeDataError as exc:
        print(f"error: {exc}")
        return 1
    params = RegimeParameters()
    readings = {d: r.regime for d, r in regime_by_session(series.sessions, series.spy, series.vix, series.vix3m, params).items()}
    totals: dict[str, int] = {"rewritten": 0, "added": 0, "skipped": 0}
    for run in runs:
        if run in failed:
            result = RecomputeResult(run, "skipped", failed[run])
        else:
            try:
                result = recompute_run(run, readings, dry_run=dry_run)
            except _RUN_ERRORS as exc:
                result = RecomputeResult(run, "skipped", f"error: {exc}")
        totals[result.status] += 1
        reason = f"  ({result.reason})" if result.reason else ""
        print(f"{result.status:9} {result.rows:5d} rows  {run}{reason}")
    suffix = " (dry run, nothing written)" if dry_run else ""
    print(f"rewritten: {totals['rewritten']}, added: {totals['added']}, skipped: {totals['skipped']}{suffix}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `regime` command (`uv run regime`).

    Args:
        argv: Arguments; the process's when None.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="regime", description="Market-regime tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    card = commands.add_parser("scorecard", help="score the regime against SPY swing falls (read-only)")
    card.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="first scored session")
    card.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END, help="last scored session")
    recompute = commands.add_parser("recompute", help="recompute the Regime line of existing backtest runs")
    recompute.add_argument("--logs-root", type=Path, default=Path("logs"), help="logs directory")
    recompute.add_argument("--dry-run", action="store_true", help="report what would change and write nothing")
    args = parser.parse_args(argv)
    if args.command == "recompute":
        return run_recompute(args.logs_root, args.dry_run)
    return run_scorecard(args.start, args.end)
