"""Run cross_momentum once per rebalance weekday, or compare two such sets.

    uv run python scripts/experiments/cross_momentum_weekdays.py run --label shv-baseline
    uv run python scripts/experiments/cross_momentum_weekdays.py compare BASELINE.json CANDIDATE.json

`run` records each backtest in logs/cross_momentum/experiments/<label>.json and resumes a set interrupted
mid-way. Days run one after another, each in its own process: every backtest rewrites
data/cross_momentum_*_backtesting.json, so two at once would corrupt each other. Run it from the project root.
The verdict rules are in trading_agent_framework.strategies.cross_momentum.weekday_protocol.
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from trading_agent_framework.config import TradingMode, find_project_root, load_strategy_env
from trading_agent_framework.strategies.cross_momentum.weekday_protocol import (
    DAY_NAMES,
    Manifest,
    RunEntry,
    compare,
    days_to_run,
    load_manifest,
    mismatches,
    read_run_metrics,
    render,
    save_manifest,
)

STRATEGY = "cross_momentum"
UNIVERSE_FILE = Path("data/universe/us_stock_universe.json")
EXPERIMENTS_DIR = Path("logs/cross_momentum/experiments")
RUN_DIR_PREFIX = "RUN_DIR="


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _days(value: str) -> list[int]:
    days = [int(part) for part in value.split(",")]
    if any(day not in range(5) for day in days) or len(set(days)) != len(days):
        raise argparse.ArgumentTypeError("days are distinct integers from 0 (Monday) to 4 (Friday)")
    return days


def _require_project_root() -> Path:
    root = find_project_root()
    if Path.cwd().resolve() != root.resolve():
        sys.exit(f"Run this from the project root ({root}): the strategy reads data/ relative to the working directory.")
    return root


def _one(args: argparse.Namespace) -> int:
    """One backtest with `day_of_week = args.day`; prints its run directory on the last line."""
    from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
    from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy
    from trading_agent_framework.strategies.cross_momentum.utils import load_cross_momentum_universe

    root = _require_project_root()
    logging.basicConfig(level=logging.INFO)
    load_strategy_env(STRATEGY, TradingMode.BACKTESTING.value, root)
    universe = load_cross_momentum_universe()
    if not universe:
        sys.exit("Universe file not found: run `uv run batch-universe` first.")
    strategy = CrossMomentumStrategy(
        broker=PlaceholderBroker(STRATEGY),
        mode=TradingMode.BACKTESTING,
        universe=universe,
        parameters={"day_of_week": args.day},
    )
    result = strategy.run_backtesting(slippage=Decimal(str(args.slippage)))
    print(f"{RUN_DIR_PREFIX}{Path(result.run_dir).resolve()}", flush=True)
    return 0


def _run(args: argparse.Namespace) -> int:
    from trading_agent_framework.strategies.cross_momentum import CrossMomentumStrategy

    _require_project_root()
    if not UNIVERSE_FILE.exists():
        sys.exit("Universe file not found: run `uv run batch-universe` first.")
    fresh = Manifest(
        label=args.label,
        slippage=args.slippage,
        start=CrossMomentumStrategy.parameters["backtesting_start"].isoformat(),
        end=CrossMomentumStrategy.parameters["backtesting_end"].isoformat(),
        universe_sha256=hashlib.sha256(UNIVERSE_FILE.read_bytes()).hexdigest(),
    )
    path = EXPERIMENTS_DIR / f"{args.label}.json"
    manifest = fresh
    if path.exists():
        manifest = load_manifest(path)
        differing = mismatches(manifest, fresh)
        if differing:
            details = ", ".join(f"{name} {getattr(manifest, name)!r} -> {getattr(fresh, name)!r}" for name in differing)
            sys.exit(f"{path} was recorded with a different setup ({details}); use a new label.")

    days = days_to_run(manifest, args.days)
    if not days:
        print(f"{path}: every requested day is already recorded")
        return 0
    for day in days:
        print(f"[{args.label}] {DAY_NAMES[day]}: running", flush=True)
        command = [sys.executable, __file__, "one", "--day", str(day), "--slippage", str(args.slippage)]
        completed = subprocess.run(command, capture_output=True, text=True)
        run_dirs = [line[len(RUN_DIR_PREFIX):] for line in completed.stdout.splitlines() if line.startswith(RUN_DIR_PREFIX)]
        if completed.returncode != 0 or not run_dirs:
            sys.stderr.write(completed.stderr[-4000:])
            sys.exit(f"[{args.label}] {DAY_NAMES[day]}: the backtest failed (exit {completed.returncode}); the days before it are saved in {path}")
        manifest.runs[day] = RunEntry(run_dir=run_dirs[-1], commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")))
        save_manifest(path, manifest)
        print(f"[{args.label}] {DAY_NAMES[day]}: {run_dirs[-1]}", flush=True)
    print(f"Manifest: {path.resolve()}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        baseline = load_manifest(Path(args.baseline))
        candidate = load_manifest(Path(args.candidate))
        baseline_metrics = {day: read_run_metrics(Path(entry.run_dir)) for day, entry in baseline.runs.items()}
        candidate_metrics = {day: read_run_metrics(Path(entry.run_dir)) for day, entry in candidate.runs.items()}
        comparison = compare(baseline, candidate, baseline_metrics, candidate_metrics)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot compare: {exc}", file=sys.stderr)
        return 2
    print(render(comparison))
    return 0 if comparison.verdict == "KEEP" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run (or resume) a set of weekday backtests")
    run.add_argument("--label", required=True, help="the set's name: logs/cross_momentum/experiments/<label>.json")
    run.add_argument("--slippage", type=float, default=0.0, help="per-trade slippage passed to run_backtesting (default 0)")
    run.add_argument("--days", type=_days, default=[0, 1, 2, 3, 4], help="comma-separated weekdays, 0 = Monday (default 0,1,2,3,4)")
    run.set_defaults(handler=_run)

    one = commands.add_parser("one", help=argparse.SUPPRESS)
    one.add_argument("--day", type=int, required=True)
    one.add_argument("--slippage", type=float, default=0.0)
    one.set_defaults(handler=_one)

    comparison = commands.add_parser("compare", help="judge a candidate set against a baseline set")
    comparison.add_argument("baseline", help="the baseline manifest (JSON path)")
    comparison.add_argument("candidate", help="the candidate manifest (JSON path)")
    comparison.set_defaults(handler=_compare)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
