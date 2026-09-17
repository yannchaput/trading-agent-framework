"""Recursively scan logs/ for backtesting run directories."""

from __future__ import annotations

import glob
import os

from trading_agent_framework.dashboard.models import RunIndex, RunRef


def scan_runs(base_path: str = "logs") -> RunIndex:
    """Discover all runs by finding *_tearsheet_metrics.json files under base_path."""
    pattern = os.path.join(base_path, "**", "*_tearsheet_metrics.json")
    matches = glob.glob(pattern, recursive=True)

    runs: list[RunRef] = []
    seen_dirs: set[str] = set()

    for match in matches:
        run_dir = os.path.dirname(match)
        norm = os.path.normpath(run_dir)
        if norm in seen_dirs:
            continue
        seen_dirs.add(norm)

        try:
            ref = RunRef.from_path(norm)
            runs.append(ref)
        except ValueError:
            continue

    # Sort by strategy name, then by timestamp descending within each group
    runs.sort(key=lambda r: (r.strategy_name, r.run_ts))

    return RunIndex(runs=runs)
