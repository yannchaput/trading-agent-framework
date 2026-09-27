"""Read-only access to vLLM benchmark results (sibling repo benchmark-vllm-models).

Layout: <results>/<YYYYMMDD-HHMMSS>/ with meta.json, summary.json and one <model key>.jsonl per model.
A run is complete once summary.json exists; until then it is not listed. Streamlit-free, like
reader.py: the Models page adds the caching. Nothing here ever writes into the results directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import find_project_root
from trading_agent_framework.dashboard.models import (
    BenchmarkModel,
    BenchmarkRun,
    BenchmarkRunRef,
    ScenarioScore,
)

BENCHMARK_DIR_FLAG = "--benchmark-dir"
DEFAULT_RESULTS_DIR = Path("..") / "benchmark-vllm-models" / "results"


class BenchmarkReadError(Exception):
    """A benchmark run's files are missing or malformed."""


def split_benchmark_dir(argv: Sequence[str]) -> tuple[list[str], str | None]:
    """Remove ``--benchmark-dir PATH`` / ``--benchmark-dir=PATH`` from argv; return (the rest, PATH)."""
    rest: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == BENCHMARK_DIR_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{BENCHMARK_DIR_FLAG} needs a path")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(BENCHMARK_DIR_FLAG + "="):
            value = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    return rest, value


def resolve_results_dir(argv: Sequence[str], start: Path | None = None) -> Path:
    """``--benchmark-dir`` if given, else benchmark-vllm-models/results next to the project root."""
    _, value = split_benchmark_dir(argv)
    if value is not None:
        return Path(value).expanduser().resolve()
    return (find_project_root(start) / DEFAULT_RESULTS_DIR).resolve()


def scan_benchmark_runs(base: Path) -> list[BenchmarkRunRef]:
    """Complete runs (meta.json and summary.json present) under `base`, newest first."""
    if not base.is_dir():
        return []
    refs: list[BenchmarkRunRef] = []
    for child in base.iterdir():
        if not (child.is_dir() and (child / "meta.json").is_file() and (child / "summary.json").is_file()):
            continue
        try:
            refs.append(BenchmarkRunRef.from_path(child))
        except ValueError:
            continue
    refs.sort(key=lambda ref: ref.started_at, reverse=True)
    return refs


def load_benchmark_run(ref: BenchmarkRunRef) -> BenchmarkRun:
    """Join meta.json and summary.json on the model key."""
    meta = _read_json(ref.path / "meta.json")
    summary = _read_json(ref.path / "summary.json")
    try:
        return _build_run(ref, meta, summary)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BenchmarkReadError(f"Malformed benchmark run {ref.run_id}: {exc!r}") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReadError(f"Cannot read {path.name}: {exc}") from exc


def _build_run(ref: BenchmarkRunRef, meta: dict[str, Any], summary: Any) -> BenchmarkRun:
    if not isinstance(summary, list):
        raise TypeError("summary.json is not a list of models")
    meta_models = {entry["key"]: entry for entry in meta["models"]}
    versions = meta.get("vllm_versions") or {}
    return BenchmarkRun(
        ref=ref,
        started_at=_opt_datetime(meta.get("started_at")) or ref.started_at,
        finished_at=_opt_datetime(meta.get("finished_at")),
        repeats=int(meta["repeats"]),
        timeout_s=float(meta["timeout_s"]),
        scenarios=tuple(str(scenario) for scenario in meta["scenarios"]),
        models=tuple(
            _build_model(entry, meta_models.get(entry["key"], {}), versions.get(entry["key"])) for entry in summary
        ),
    )


def _build_model(entry: dict[str, Any], meta_model: dict[str, Any], vllm_version: str | None) -> BenchmarkModel:
    return BenchmarkModel(
        key=str(entry["key"]),
        display_name=str(entry.get("display_name") or meta_model.get("display_name") or entry["key"]),
        served_name=meta_model.get("served_name"),
        vllm_version=vllm_version,
        ran=bool(entry.get("ran", True)),
        error=entry.get("error"),
        overall=_opt_float(entry.get("overall")),
        mean_partial=_opt_float(entry.get("mean_partial")),
        categories={str(name): float(score) for name, score in (entry.get("categories") or {}).items()},
        scenarios={
            str(scenario_id): ScenarioScore(
                passed=int(score["passed"]), runs=int(score["runs"]), mean_partial=float(score["mean_partial"])
            )
            for scenario_id, score in (entry.get("scenarios") or {}).items()
        },
        runs_passed=_opt_int(entry.get("runs_passed")),
        runs_total=_opt_int(entry.get("runs_total")),
        text_tool_calls=_opt_int(entry.get("text_tool_calls")),
        avg_tool_calls=_opt_float(entry.get("avg_tool_calls")),
        median_run_s=_opt_float(entry.get("median_run_s")),
        median_tokens_per_s=_opt_float(entry.get("median_tokens_per_s")),
        timeouts=_opt_int(entry.get("timeouts")),
        errors=_opt_int(entry.get("errors")),
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _opt_datetime(value: Any) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
