"""Hand-built vLLM benchmark results trees, shaped like the real
../benchmark-vllm-models/results/20260927-132243 run (meta.json, summary.json, <key>.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LATEST = "20260927-132243"
OLDER = "20260925-233646"
IN_PROGRESS = "20260928-090000"  # newer than LATEST but has no summary.json yet
SWITCHED = "20260926-100000"  # a third, standalone run for the run-switch regression test below
SCENARIOS = ["reasoning.rsi_signal", "reasoning.headline_trap", "tools.limit_order"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _meta(models: list[tuple[str, str, str]], started: str, finished: str) -> dict[str, Any]:
    return {
        "started_at": started,
        "finished_at": finished,
        "models": [{"key": key, "display_name": name, "start_function": f"vllmStart{key}", "port": 8000 + i, "served_name": served} for i, (key, name, served) in enumerate(models)],
        "vllm_versions": {key: "0.30.0" for key, _, _ in models},
        "repeats": 2,
        "timeout_s": 360.0,
        "scenarios": SCENARIOS,
        "cli_args": {"only": None, "repeats": 2, "scenarios": "*", "timeout": 360.0},
    }


def _summary(key: str, name: str, **fields: Any) -> dict[str, Any]:
    entry = {
        "key": key,
        "display_name": name,
        "ran": True,
        "error": None,
        "overall": 0.5,
        "mean_partial": 0.9,
        "categories": {"reasoning": 0.5, "tools": 0.5},
        # Keys sorted alphabetically, like the real file -- NOT in meta.json's scenario order.
        "scenarios": {
            "reasoning.headline_trap": {"passed": 2, "runs": 2, "mean_partial": 1.0},
            "reasoning.rsi_signal": {"passed": 0, "runs": 2, "mean_partial": 0.8},
            "tools.limit_order": {"passed": 1, "runs": 2, "mean_partial": 0.9},
        },
        "runs_passed": 3,
        "runs_total": 6,
        "text_tool_calls": 0,
        "avg_tool_calls": 3.24,
        "median_run_s": 4.9,
        "median_tokens_per_s": 131.7,
        "timeouts": 0,
        "errors": 0,
    }
    entry.update(fields)
    return entry


def record(
    scenario_id: str,
    repeat: int,
    *,
    passed: bool,
    partial: float,
    status: str = "ok",
    error: str | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One <key>.jsonl line, trace included (the reader must drop it)."""
    if checks is None:
        checks = [
            {"type": "called", "passed": passed, "reason": "found" if passed else "no call matching submit_order{'symbol': 'DOCU'}"},
            {"type": "max_tool_calls", "passed": True, "reason": "4 tool calls (max 10)"},
        ]
    return {
        "scenario_id": scenario_id,
        "category": scenario_id.split(".")[0],
        "repeat": repeat,
        "status": status,
        "passed": passed,
        "partial": partial,
        "checks": checks,
        "metrics": {
            "total_s": 6.906,
            "tool_calls": 4,
            "model_calls": 5,
            "tool_arg_errors": 0,
            "text_tool_calls": 0,
            "completion_tokens": 882,
            "reasoning_chars": 2506,
            "tokens_per_s": 121.35,
        },
        "error": error,
        "trace": {"sessions": [{"index": 1, "prompt": "p", "messages": [{"role": "human", "content": "x" * 50}]}]},
    }


def _write_jsonl(path: Path, lines: list[dict[str, Any] | str]) -> None:
    path.write_text("\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def build_results_tree(base: Path) -> Path:
    """LATEST (glm + qwen3627b, complete, one malformed .jsonl line, one timeout),
    OLDER (glm + gptoss where gptoss did not run), IN_PROGRESS (no summary.json),
    a non-timestamp folder and a stray file. Returns `base`."""
    latest = base / LATEST
    write_json(
        latest / "meta.json",
        _meta(
            [("glm", "GLM-4.7-Flash", "glm-4.7-flash"), ("qwen3627b", "Qwen3.6-27B-AWQ", "qwen3.6-27b-awq")],
            "2026-09-27T13:22:43",
            "2026-09-27T16:08:15",
        ),
    )
    write_json(
        latest / "summary.json",
        [
            _summary("glm", "GLM-4.7-Flash"),
            _summary(
                "qwen3627b",
                "Qwen3.6-27B-AWQ",
                overall=0.83,
                mean_partial=0.99,
                categories={"reasoning": 0.75, "tools": 1.0},
                scenarios={
                    "reasoning.headline_trap": {"passed": 2, "runs": 2, "mean_partial": 1.0},
                    "reasoning.rsi_signal": {"passed": 1, "runs": 2, "mean_partial": 0.96},
                    "tools.limit_order": {"passed": 2, "runs": 2, "mean_partial": 1.0},
                },
                runs_passed=5,
                text_tool_calls=1,
                avg_tool_calls=3.51,
                median_run_s=29.7,
                median_tokens_per_s=44.5,
                timeouts=1,
            ),
        ],
    )
    _write_jsonl(
        latest / "qwen3627b.jsonl",
        [
            record("reasoning.rsi_signal", 2, passed=False, partial=0.0, status="timeout", error="run exceeded 360s", checks=[]),
            record("reasoning.headline_trap", 1, passed=True, partial=1.0),
            "{not json",
            record("reasoning.rsi_signal", 1, passed=True, partial=1.0),
            record("tools.limit_order", 1, passed=True, partial=1.0),
        ],
    )
    _write_jsonl(
        latest / "glm.jsonl",
        [
            record("reasoning.rsi_signal", 1, passed=False, partial=0.8),
            record("reasoning.rsi_signal", 2, passed=False, partial=0.8),
        ],
    )
    (latest / "glm.vllm.log").write_text("vllm log\n", encoding="utf-8")

    older = base / OLDER
    write_json(
        older / "meta.json",
        _meta(
            [("glm", "GLM-4.7-Flash", "glm-4.7-flash"), ("gptoss", "Gpt-OSS-20b", "gpt-oss-20b")],
            "2026-09-25T23:36:46",
            "2026-09-26T00:46:00",
        ),
    )
    write_json(
        older / "summary.json",
        [
            _summary("glm", "GLM-4.7-Flash"),
            {
                "key": "gptoss",
                "display_name": "Gpt-OSS-20b",
                "ran": False,
                "error": "vLLM server failed to start",
                "overall": None,
                "mean_partial": None,
                "categories": {},
                "scenarios": {},
                "runs_passed": 0,
                "runs_total": 0,
                "text_tool_calls": 0,
                "avg_tool_calls": None,
                "median_run_s": None,
                "median_tokens_per_s": None,
                "timeouts": 0,
                "errors": 0,
            },
        ],
    )
    _write_jsonl(older / "glm.jsonl", [record("reasoning.rsi_signal", 1, passed=False, partial=0.8)])

    write_json(base / IN_PROGRESS / "meta.json", _meta([("glm", "GLM-4.7-Flash", "glm-4.7-flash")], "2026-09-28T09:00:00", ""))
    write_json(base / "notes" / "meta.json", {})
    write_json(base / "notes" / "summary.json", [])
    (base / "README.txt").write_text("not a run\n", encoding="utf-8")
    return base


def add_switched_run(base: Path) -> Path:
    """A third, standalone complete run (older than LATEST, newer than OLDER) where BOTH
    models ran and `gptoss` -- not `glm` -- wins overall. Added on top of `build_results_tree`
    only by the test that needs it, so a run-switch test can assert the drill-down's default
    model is genuinely recomputed for the newly-selected run (not the only valid choice, as
    OLDER's single ran model would be, and not LATEST's winner either)."""
    run_dir = base / SWITCHED
    write_json(
        run_dir / "meta.json",
        _meta(
            [("glm", "GLM-4.7-Flash", "glm-4.7-flash"), ("gptoss", "Gpt-OSS-20b", "gpt-oss-20b")],
            "2026-09-26T10:00:00",
            "2026-09-26T11:00:00",
        ),
    )
    write_json(
        run_dir / "summary.json",
        [
            _summary("glm", "GLM-4.7-Flash", overall=0.40, mean_partial=0.5),
            _summary("gptoss", "Gpt-OSS-20b", overall=0.88, mean_partial=0.95),
        ],
    )
    _write_jsonl(run_dir / "glm.jsonl", [record("reasoning.rsi_signal", 1, passed=False, partial=0.4)])
    _write_jsonl(run_dir / "gptoss.jsonl", [record("reasoning.rsi_signal", 1, passed=True, partial=0.95)])
    return run_dir
