"""`uv run bull-bear-scores <run_dir> [<run_dir>...]`: the bull, the bear, the judge and its strengths of bull_bear backtest runs,
scored against forward returns (`agent_scores.py`), runs side by side.

Reads each run's `reviews.jsonl` itself (a strategy package never imports the dashboard), fetches daily opens from
Yahoo through `YahooDailyBars` (SPY's bars are the session calendar) and prints the report. Read-only: nothing is
written into a run directory. Exit 2 for a run directory without `reviews.jsonl`, 1 when Yahoo fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework.strategies.bull_bear.batch.agent_scores import (
    EMPTY_GROUP,
    HORIZONS,
    GroupStats,
    RunScores,
    Spread,
    candidates,
    forward_returns,
    score_agents,
    scored_reviews,
)
from trading_agent_framework.strategies.common.yahoo_daily_bars import DownloadFn, YahooDailyBars
from trading_agent_framework.utils.errors import YahooDataError

if TYPE_CHECKING:
    import pandas as pd

REVIEWS_FILE = "reviews.jsonl"
CALENDAR_SYMBOL = "SPY"
_WINDOW_BEFORE = timedelta(days=7)
_WINDOW_AFTER = timedelta(days=40)  # 20 sessions plus slack
_CALENDAR_DAYS_PER_BAR = 1.5  # how far back `YahooDailyBars` reaches per bar of history
_LABEL = 32
_CELL = 26
_TITLES = {
    "bull": "BULL conviction",
    "bear": "BEAR risk",
    "judge": "JUDGE verdict",
    "strength": "JUDGE strength",
    "strength_standard": "JUDGE strength",
}


@dataclass(frozen=True, slots=True)
class RunReport:
    name: str
    scores: RunScores
    reviews: int  # scored reviews
    malformed: int  # unreadable lines in reviews.jsonl


def main(argv: Sequence[str] | None = None, *, download: DownloadFn | None = None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bull-bear-scores",
        description="Score the bull, the bear, the judge and its strengths of bull_bear backtest runs against forward returns (Yahoo opens, 5 and 20 sessions).",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, metavar="run_dir", help="a backtest run directory holding reviews.jsonl")
    args = parser.parse_args(argv)
    loaded: list[tuple[str, list[dict[str, Any]], int]] = []
    for run_dir in args.run_dirs:
        path = run_dir / REVIEWS_FILE
        if not path.is_file():
            parser.error(f"{run_dir} has no {REVIEWS_FILE}")
        records, malformed = read_reviews(path)
        loaded.append((run_dir.name, records, malformed))
    try:
        opens, calendar = fetch_opens([records for _, records, _ in loaded], today=today or date.today(), download=download)
    except YahooDataError as exc:
        print(f"bull-bear-scores: {exc}", file=sys.stderr)
        return 1
    reports = [
        RunReport(name, score_agents(records, forward_returns(records, opens, calendar)), len(scored_reviews(records)), malformed)
        for name, records, malformed in loaded
    ]
    print(format_report(reports))
    return 0


def read_reviews(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Every JSON object of a `reviews.jsonl`, in file order, and how many non-blank lines were not one."""
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            malformed += 1
    return records, malformed


def fetch_opens(runs: Sequence[Sequence[Mapping[str, Any]]], *, today: date, download: DownloadFn | None = None) -> tuple[dict[str, dict[date, float]], list[date]]:
    """{symbol: {session date: open}} for every candidate of every run plus SPY, from one Yahoo download spanning 7 days
    before the first review to 40 days after the last (capped at today), and SPY's sessions as the calendar.

    Raises YahooDataError when the download fails or has no SPY."""
    reviews = [review for records in runs for review in scored_reviews(records)]
    symbols = sorted({symbol for review in reviews for symbol in candidates(review)} | {CALENDAR_SYMBOL})
    days = [date.fromisoformat(review["date"]) for review in reviews] or [today]
    first, end = min(days) - _WINDOW_BEFORE, min(max(days) + _WINDOW_AFTER, today)
    history = math.ceil((end - first).days / _CALENDAR_DAYS_PER_BAR) + 1
    bars = YahooDailyBars(download=download, history=history).bars(symbols, end)
    opens = {symbol: _session_opens(bar.df) for symbol, bar in bars.items()}
    if CALENDAR_SYMBOL not in opens:
        raise YahooDataError(f"Yahoo returned no {CALENDAR_SYMBOL} bars, the session calendar")
    return opens, sorted(opens[CALENDAR_SYMBOL])


def _session_opens(df: pd.DataFrame) -> dict[date, float]:
    """{session date: open} of a `Bars.df`, whose index stamps each session at its close."""
    import pandas as pd

    sessions = [stamp.date() for stamp in pd.DatetimeIndex(df.index)]
    return {session: float(value) for session, value in zip(sessions, df["open"].to_numpy(dtype=float), strict=True)}


def format_report(runs: Sequence[RunReport], horizons: Sequence[int] = HORIZONS) -> str:
    """One block per agent (bull, bear, judge, strength, strength_standard), each run a column group with one column per horizon."""
    width = _CELL * len(horizons) + len(horizons) - 1
    header = " " * _LABEL + "".join(f" | {run.name:<{width}}" for run in runs)
    subheader = " " * _LABEL + "".join(" | " + " ".join(f"{horizon}d".rjust(_CELL) for horizon in horizons) for _ in runs)
    lines: list[str] = []
    for index, agent in enumerate(runs[0].scores.agents):  # every run lists the agents in the same order
        scores = [[run.scores.agents[index].by_horizon[horizon] for horizon in horizons] for run in runs]
        lines += ["", f"{_TITLES[agent.agent]}, spread {agent.spread_label}", header, subheader]
        for group in agent.by_horizon[horizons[0]].groups:
            lines.append(_row(f"  {group}", [[score.groups[group] for score in run] for run in scores], _group_cell))
        for concern in sorted({concern for run in scores for score in run for concern in score.concerns}):
            lines.append(_row(f"  concern {concern}", [[score.concerns.get(concern, EMPTY_GROUP) for score in run] for run in scores], _group_cell))
        lines.append(_row("  spread", [[score.spread for score in run] for run in scores], _spread_cell))
    lines.append("")
    for run in runs:
        lines.append(f"{run.name}: {run.reviews} reviews scored, {run.malformed} malformed lines")
        lines.append(f"  candidate x horizon pairs left out: {run.scores.dropped} without prices, {run.scores.immature} whose exit is past the end of the data")
        skipped = "; ".join(f"{agent.agent} " + ", ".join(f"{horizon}d {agent.by_horizon[horizon].spread.skipped}" for horizon in horizons) for agent in run.scores.agents)
        lines.append(f"  reviews without a spread (a group missing): {skipped}")
    return "\n".join(lines)


def _row(label: str, cells: Sequence[Sequence[Any]], render: Any) -> str:
    return f"{label:<{_LABEL}}" + "".join(" | " + " ".join(render(cell) for cell in run) for run in cells)


def _group_cell(stats: GroupStats) -> str:
    if stats.n == 0 or stats.mean_excess is None or stats.hit_rate is None:
        return "n=0".rjust(_CELL)
    return f"n={stats.n} {stats.mean_excess:+.2%} hit {stats.hit_rate:.0%}".rjust(_CELL)


def _spread_cell(spread: Spread) -> str:
    if spread.mean is None:
        return f"- ({spread.reviews} rev)".rjust(_CELL)
    t = "-" if spread.t is None else f"{spread.t:+.2f}"
    return f"{spread.mean:+.2%} t {t} ({spread.reviews} rev)".rjust(_CELL)


if __name__ == "__main__":
    sys.exit(main())
