"""Throwaway replay of the balance-based bear (and the judge without a veto) against the logged 5Y bull_bear run.

Not a repo command; kept for the record. Spec: docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-design.md,
section "Validation: replay first, keep/drop rule fixed now". The sample, the arms, the anti-goal rule and the five
keep/drop tests are fixed there; nothing here is tuned on the results.

Subcommands (run from the repository root, `uv run python scripts/replays/bull_bear_bear_balance.py <command>`):

- `prepare` (removed): ran once at v1's commit `407a310` (tag `archive/bull-bear-bear-balance-v1`); it needed the
  v1 fact-sheet code, which was not merged. Its outputs are committed: `replay_fields.json` (arm B's four fields for
  every candidate of every review) and `replay_plan.json` (the sample and the anti-goal list).
- `run [--arms A,B] [--reviews D1,D2]`: per sample review and arm, a fresh bull, bear and judge on the logged inputs
  (arm A: the run's prompts and fact sheets; arm B: the prompts at `407a310`, the fact sheets plus the four fields). One JSON
  line per (review, arm) in `replay_results.jsonl`; a (review, arm) already there is skipped (resume).
- `anti-goal`: arm B's bear alone on each anti-goal review outside the sample (one line each, same file, same resume).
- `score [--out PATH]`: the five keep/drop tests and the non-gating tables, written as the replay report.

Inputs are read by absolute path from the MAIN checkout (logs, SEC cache, env file) and never written to.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import statistics
import subprocess
import tempfile
import time
import tokenize
import types
import urllib.request
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

MAIN = Path("/mnt/data/projets/trading-agent-framework")
RUN_DIR = MAIN / "logs/bull_bear/backtesting/2026-11-05_012455_backtesting"
ENV_FILE = MAIN / "env/.env.bull_bear.backtesting"
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs/superpowers/replays/data"
PLAN_FILE = DATA / "replay_plan.json"
FIELDS_FILE = DATA / "replay_fields.json"
RESULTS_FILE = DATA / "replay_results.jsonl"
REPORT_FILE = REPO / "docs/superpowers/replays/2026-11-08-bull-bear-bear-balance.md"

PROMPTS_PATH = "src/trading_agent_framework/strategies/bull_bear/prompts.py"
OLD_PROMPTS_REV = "d852eac~1"  # the last commit before the prompt change; its prompts.py equals the one the 5Y run used
NEW_PROMPTS_REV = "407a310"  # arm B's prompts (the v1 change); kept reachable by the tag archive/bull-bear-bear-balance-v1

ANTI_GOAL_MIN = 6  # fewer listed stocks: test 3 cannot be run and does not pass
REVIEWS_EXPECTED = 262
HORIZON = 20  # sessions, the scored horizon

STAGES = {"bull": "Make the bull case", "bear": "Make the bear case", "judge": "Give your verdict"}
NEW_FIELDS = ("operating_margin_change_pts", "debt_growth_pct", "days_since_filed", "return_since_filed_pct")
ARMS = ("A", "B")
LEVELS = ("low", "medium", "high")


# --- small pure helpers (no I/O) ------------------------------------------------------------------


def qualifies_anti_goal(sheet: Mapping[str, Any]) -> bool:
    """The spec's anti-goal rule on one arm-B fact sheet: a large, recent deterioration the market has not bid up."""
    revenue = sheet.get("revenue_growth_pct")
    margin_change = sheet.get("operating_margin_change_pts")
    margin_before = sheet.get("operating_margin_prior_year_pct")
    days = sheet.get("days_since_filed")
    since = sheet.get("return_since_filed_pct")
    large = (revenue is not None and revenue <= -10) or (margin_change is not None and margin_change <= -10 and margin_before is not None and margin_before >= -50)
    return bool(large and days is not None and days <= 45 and since is not None and since < 0)


def with_new_fields(sheet: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The arm-B fact sheet: the logged sheet with the four fields where the live code puts them.

    `fundamentals_fields` writes `operating_margin_change_pts` right after `operating_margin_prior_year_pct` and
    `debt_growth_pct` right after `total_debt_prior_year_bn`; the pipeline appends `days_since_filed` and
    `return_since_filed_pct` after every other fundamentals key, so they close the sheet.
    """
    out: dict[str, Any] = {}
    for key, value in sheet.items():
        if key in NEW_FIELDS:
            continue
        out[key] = value
        if key == "operating_margin_prior_year_pct" and "operating_margin_change_pts" in fields:
            out["operating_margin_change_pts"] = fields["operating_margin_change_pts"]
        if key == "total_debt_prior_year_bn" and "debt_growth_pct" in fields:
            out["debt_growth_pct"] = fields["debt_growth_pct"]
    for key in ("days_since_filed", "return_since_filed_pct"):
        if key in fields:
            out[key] = fields[key]
    return out


def margin_change(sheet: Mapping[str, Any]) -> float | None:
    """`operating_margin_change_pts` from the logged sheet: the live code subtracts the same two rounded margins."""
    margin, before = sheet.get("operating_margin_pct"), sheet.get("operating_margin_prior_year_pct")
    return None if margin is None or before is None else round(margin - before, 1) + 0.0


def share(part: int, whole: int) -> float | None:
    """`part / whole`, or None for an empty whole."""
    return part / whole if whole else None


def keep_drop_tests(
    *,
    bear: Mapping[str, Sequence[str]],
    out_rate_b: tuple[int, int],
    anti_goal: tuple[int, int],
    differ: Mapping[str, tuple[int, int]],
    flips: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """The spec's five keep/drop tests, each with its numbers and pass flag.

    Args:
        bear: the bear's risk levels per arm ("A", "B") over the valid sample reviews.
        out_rate_b: (outs, stocks) in arm B among bull medium/high with bear high.
        anti_goal: (rated high by arm B's bear, stocks listed).
        differ: (differing, compared) for "B_vs_A" and "A_vs_logged".
        flips: the 20d excess of "B_in_A_out" and "A_in_B_out" stocks (an empty group counts as 0).
    """
    tests: list[dict[str, Any]] = []
    b_levels, a_levels = Counter(bear["B"]), Counter(bear["A"])
    high_b = share(b_levels["high"], sum(b_levels.values()))
    low_b, low_a = share(b_levels["low"], sum(b_levels.values())), share(a_levels["low"], sum(a_levels.values()))
    ok1 = high_b is not None and low_b is not None and low_a is not None and 0.04 <= high_b <= 0.15 and round(low_b - low_a, 12) <= 0.15  # round: 0.65 - 0.5 is 0.15000000000000002
    tests.append({"test": 1, "name": "the bear still argues", "high_share_B": high_b, "low_share_B": low_b, "low_share_A": low_a, "pass": ok1})
    outs, stocks = out_rate_b
    rate = share(outs, stocks)
    tests.append(
        {
            "test": 2,
            "name": "no veto",
            "out_rate_B": rate,
            "outs": outs,
            "stocks": stocks,
            "pass": rate is not None and rate < 0.75,
            "note": None if stocks else "no bull medium/high with bear high in arm B: cannot be computed, does not pass",
        }
    )
    high, listed = anti_goal
    testable = listed >= ANTI_GOAL_MIN
    tests.append(
        {
            "test": 3,
            "name": "real risks still flagged",
            "rated_high": high,
            "listed": listed,
            "testable": testable,
            "pass": testable and 2 * high >= listed,
            "note": None if testable else f"only {listed} anti-goal stocks in the whole run (< {ANTI_GOAL_MIN}): the test cannot be run and does not pass",
        }
    )
    b_vs_a, a_vs_logged = share(*differ["B_vs_A"]), share(*differ["A_vs_logged"])
    tests.append(
        {
            "test": 4,
            "name": "an effect above noise",
            "differ_B_vs_A": b_vs_a,
            "differ_A_vs_logged": a_vs_logged,
            "counts": {key: list(value) for key, value in differ.items()},
            "pass": b_vs_a is not None and a_vs_logged is not None and b_vs_a > a_vs_logged,
        }
    )
    b_in = statistics.fmean(flips["B_in_A_out"]) if flips["B_in_A_out"] else 0.0
    a_in = statistics.fmean(flips["A_in_B_out"]) if flips["A_in_B_out"] else 0.0
    tests.append(
        {
            "test": 5,
            "name": "no harm in the flips",
            "mean_B_in_A_out": b_in,
            "n_B_in_A_out": len(flips["B_in_A_out"]),
            "mean_A_in_B_out": a_in,
            "n_A_in_B_out": len(flips["A_in_B_out"]),
            "pass": b_in >= a_in,
        }
    )
    return tests


# --- inputs: the run log, the review records --------------------------------------------------------


def _string_literal(text: str) -> str:
    """The Python string literal at the start of `text`, evaluated (the repr may use single or double quotes)."""
    token = next(tokenize.generate_tokens(io.StringIO(text).readline))
    return ast.literal_eval(token.string)


def parse_log(path: Path = RUN_DIR / "backtest.log") -> dict[str, dict[str, dict[str, Any]]]:
    """{stage: {review date: {"message": the logged task message, "context": its context}}} for the bull, bear and judge."""
    marker = "HumanMessage(content="
    found: dict[str, dict[str, dict[str, Any]]] = {stage: {} for stage in STAGES}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[AgentRunResult]" not in line or marker not in line:
                continue
            message = _string_literal(line[line.index(marker) + len(marker) :])
            stage = next((name for name, prefix in STAGES.items() if message.startswith(prefix)), None)
            if stage is None:
                continue
            context = ast.literal_eval(message.split("Context:\n", 1)[1])
            day = context["current_datetime"][:10]
            if day in found[stage] and found[stage][day]["message"] != message:
                raise SystemExit(f"two different {stage} messages for {day}")
            found[stage][day] = {"message": message, "context": context}
    for stage, by_day in found.items():
        if len(by_day) != REVIEWS_EXPECTED:
            raise SystemExit(f"{len(by_day)} {stage} messages in the log, expected {REVIEWS_EXPECTED}")
    return found


def read_reviews(path: Path = RUN_DIR / "reviews.jsonl") -> list[dict[str, Any]]:
    """The run's review records, in file order."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != REVIEWS_EXPECTED:
        raise SystemExit(f"{len(records)} records in reviews.jsonl, expected {REVIEWS_EXPECTED}")
    return records


def load_prompts(rev: str | None) -> types.ModuleType:
    """The prompts module at a git revision (exec'd into a fresh namespace), or the current one for None."""
    if rev is None:
        from trading_agent_framework.strategies.bull_bear import prompts

        return prompts
    source = subprocess.run(["git", "show", f"{rev}:{PROMPTS_PATH}"], cwd=REPO, check=True, capture_output=True, text=True).stdout
    module = types.ModuleType(f"bull_bear_prompts_{rev}")
    exec(compile(source, f"{rev}:{PROMPTS_PATH}", "exec"), module.__dict__)
    return module


def git_sha(rev: str) -> str:
    """The full SHA of a revision."""
    return subprocess.run(["git", "rev-parse", rev], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _symbols(context: Mapping[str, Any]) -> list[str]:
    return [stock["fact_sheet"]["symbol"] for stock in context["stocks"]]


# --- run ------------------------------------------------------------------------------------------


class ServerDown(Exception):
    """The LLM server does not answer: stop the replay rather than record a failure that is not the model's."""


def _server_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)


def _results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.is_file():
        return []
    return [json.loads(line) for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(line: Mapping[str, Any]) -> None:
    with RESULTS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


class _Replay:
    """One (review, arm) run of the debate outside a strategy, with the run's settings."""

    def __init__(self, arm: str, day: str, prompts: types.ModuleType) -> None:
        from trading_agent_framework.agents.config import LLMCredentials
        from trading_agent_framework.agents.manager import AgentManager
        from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
        from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

        self.arm, self.day, self.prompts = arm, day, prompts
        self.params = BullBearParams()
        self.credentials = LLMCredentials.from_env()
        self.manager = AgentManager(LLMCredentials.from_env)
        self.manager.enable_telemetry(now=lambda: datetime.now(UTC))
        self.recorder = HandoffRecorder(self.params)
        self.submit = submit_tools(self.recorder)
        self._memory_dir: tempfile.TemporaryDirectory[str] | None = None

    def create(self, stages: Sequence[str], review_time: datetime) -> None:
        """The agents of `stages`, built as `BullBearStrategy.initialize` builds them, with this arm's prompts."""
        params, prompts = self.params, self.prompts
        cap, budget, temperature = params.max_output_tokens, params.thinking_token_budget, params.agent_temperature
        for name, system, tool in (("bull", prompts.BULL_SYSTEM, "submit_bull_case"), ("bear", prompts.BEAR_SYSTEM, "submit_bear_case")):
            if name in stages:
                self.manager.create(name=name, system_prompt=system, tools=[self.submit[tool]], temperature=temperature, final_tools=[tool], max_output_tokens=cap, thinking_token_budget=budget)
        if "judge" in stages:
            from trading_agent_framework.agents.tools import only
            from trading_agent_framework.memory.store import MemoryStore
            from trading_agent_framework.memory.tools import memory_tools

            self._memory_dir = tempfile.TemporaryDirectory(prefix="bull_bear_replay_memory_")
            store = MemoryStore(Path(self._memory_dir.name) / "memory.sqlite", strategy_name="bull_bear", now=lambda: review_time, fresh=True)  # empty: the run's memory is gone
            self.manager.create(
                name="judge",
                system_prompt=prompts.JUDGE_SYSTEM,
                tools=[*only(memory_tools(store), {"search_memory", "remember_lesson"}), self.submit["submit_verdicts"]],
                temperature=temperature,
                exempt_tools=["submit_verdicts"],
                final_tools=["submit_verdicts"],
                compact_history=True,
                max_output_tokens=params.judge_max_output_tokens,
                thinking_token_budget=params.judge_thinking_token_budget,
            )

    def close(self) -> None:
        if self._memory_dir is not None:
            self._memory_dir.cleanup()

    def stage(self, name: str, tool: str, task: str, context: dict[str, Any], tool_budget: int | None = None) -> dict[str, Any]:
        """`BullBearPipeline._run_stage`: one run, then one fresh run with the submit tool forced; a loop fails at once."""
        from trading_agent_framework.utils.errors import AgentError, AgentLoopError

        agent = self.manager[name]
        run_id = uuid.uuid4().hex
        prompt, force = task, None
        info: dict[str, Any] = {"ok": False, "attempts": 0, "errors": []}
        first_call = len(self.manager._calls)
        started = time.monotonic()
        for attempt in range(2):
            self.recorder.last_error, error = None, ""
            info["attempts"] = attempt + 1
            try:
                agent.run(prompt, context=context, run_id=run_id, force_tool=force, tool_budget=tool_budget, max_steps=self.params.max_agent_steps)
            except AgentLoopError as exc:
                info["errors"].append(f"loop: {exc}")
                break
            except AgentError as exc:
                if not _server_up(self.credentials.base_url):
                    raise ServerDown(str(exc)) from exc
                error = str(exc)
            if self.recorder.submitted:
                info["ok"] = True
                break
            error = self.recorder.last_error or error or f"the agent ended without calling {tool}"
            info["errors"].append(error[:1000])
            prompt, force = self.prompts.retry_prompt(name, tool, error, task), tool
        calls = self.manager._calls[first_call:]
        info.update(
            seconds=round(time.monotonic() - started, 1),
            calls=len(calls),
            input_tokens=[call.input_tokens for call in calls],
            output_tokens=sum(call.output_tokens or 0 for call in calls),
            reasoning_tokens=sum(call.reasoning_tokens or 0 for call in calls),
        )
        return info


def _arm_sheet(arm: str, day: str, sheet: Mapping[str, Any], fields: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    return dict(sheet) if arm == "A" else with_new_fields(sheet, fields[day][sheet["symbol"]])


def _evidence(arm: str, day: str, context: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The bull's or bear's logged context, with arm B's sheets."""
    return {**context, "stocks": [{**stock, "fact_sheet": _arm_sheet(arm, day, stock["fact_sheet"], fields)} for stock in context["stocks"]]}


def _shown(context: Mapping[str, Any]) -> dict[str, int]:
    """How many stocks of a context show each new field (0 everywhere for arm A)."""
    return {key: sum(key in stock["fact_sheet"] for stock in context["stocks"]) for key in NEW_FIELDS}


def replay_review(arm: str, entry: Mapping[str, Any], log: Mapping[str, Any], fields: Mapping[str, Any], prompts: types.ModuleType, *, bear_only: bool = False) -> dict[str, Any]:
    """Bull, bear, then judge on that arm's bull and bear (or the bear alone); one result line."""
    day = entry["date"]
    bull_ctx, bear_ctx, judge_ctx = (log[stage][day]["context"] for stage in STAGES)
    review_time = datetime.fromisoformat(judge_ctx["current_datetime"])
    symbols = _symbols(bull_ctx)
    replay = _Replay(arm, day, prompts)
    params = replay.params
    line: dict[str, Any] = {"date": day, "index": entry.get("index"), "arm": arm, "kind": "anti_goal_bear" if bear_only else "full", "ok": False, "failed_stage": None, "stocks": len(symbols)}
    started = datetime.now(UTC)
    stages: dict[str, Any] = {}
    try:
        replay.create(("bear",) if bear_only else tuple(STAGES), review_time)
        evidence_bear = _evidence(arm, day, bear_ctx, fields)
        line["bear_context_fields"] = _shown(evidence_bear)
        bull: dict[str, Any] = {}
        if not bear_only:
            replay.recorder.expect_bull(symbols)
            stages["bull"] = replay.stage("bull", "submit_bull_case", prompts.bull_task(params.argument_max_chars), _evidence(arm, day, bull_ctx, fields))
            if not stages["bull"]["ok"]:
                line["failed_stage"] = "bull"
                return line
            bull = {case.symbol: case for case in replay.recorder.submission}
            line["bull"] = [asdict(case) for case in bull.values()]
        replay.recorder.expect_bear(symbols)
        stages["bear"] = replay.stage("bear", "submit_bear_case", prompts.bear_task(params.argument_max_chars), evidence_bear)
        if not stages["bear"]["ok"]:
            line["failed_stage"] = "bear"
            return line
        bear = {case.symbol: case for case in replay.recorder.submission}
        line["bear"] = [asdict(case) for case in bear.values()]
        if bear_only:
            line["ok"] = True
            return line
        previous = {stock["fact_sheet"]["symbol"]: stock["previous_verdict"] for stock in judge_ctx["stocks"] if "previous_verdict" in stock}
        context = {
            "current_datetime": judge_ctx["current_datetime"],
            "stocks": [
                {
                    "fact_sheet": _arm_sheet(arm, day, stock["fact_sheet"], fields),
                    "note": stock["note"],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                    **({"previous_verdict": previous[symbol]} if symbol in previous else {}),
                }
                for stock in judge_ctx["stocks"]
                for symbol in [stock["fact_sheet"]["symbol"]]
            ],
            "constraints": judge_ctx["constraints"],
        }
        replay.recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})
        stages["judge"] = replay.stage("judge", "submit_verdicts", prompts.judge_task(params.reason_max_chars), context, tool_budget=params.judge_memory_budget)
        tokens = [value for value in stages["judge"]["input_tokens"] if value is not None]
        line["judge_prompt_tokens"] = tokens[0] if tokens else None  # the first judge call: system prompt, tools and the task with its context
        line["judge_max_input_tokens"] = max(tokens) if tokens else None
        if not stages["judge"]["ok"]:
            line["failed_stage"] = "judge"
            return line
        line["verdicts"] = [asdict(verdict) for verdict in replay.recorder.submission.verdicts]
        line["ok"] = True
        return line
    finally:
        replay.close()
        line["stages"] = stages
        line["started"] = started.isoformat(timespec="seconds")
        line["seconds"] = round((datetime.now(UTC) - started).total_seconds(), 1)


def _load_plan_and_fields() -> tuple[dict[str, Any], dict[str, Any]]:
    if not PLAN_FILE.is_file() or not FIELDS_FILE.is_file():
        raise SystemExit(f"{PLAN_FILE.name} or {FIELDS_FILE.name} is missing: they are committed with this script")
    return json.loads(PLAN_FILE.read_text(encoding="utf-8")), json.loads(FIELDS_FILE.read_text(encoding="utf-8"))["fields"]


def _replay_all(jobs: Sequence[tuple[str, dict[str, Any], bool]]) -> None:
    """Run each (arm, review entry, bear_only) not already in the results file."""
    _load_env()
    _, fields = _load_plan_and_fields()
    log = parse_log()
    prompts = {"A": load_prompts(OLD_PROMPTS_REV), "B": load_prompts(NEW_PROMPTS_REV)}
    from trading_agent_framework.agents.config import LLMCredentials

    base_url = LLMCredentials.from_env().base_url
    done = {(line["date"], line["arm"], line["kind"]) for line in _results()}
    for arm, entry, bear_only in jobs:
        key = (entry["date"], arm, "anti_goal_bear" if bear_only else "full")
        if key in done:
            print(f"skip {key}: already in {RESULTS_FILE.name}")
            continue
        if not _server_up(base_url):
            raise SystemExit(f"the LLM server at {base_url} does not answer: stopped before {key}")
        print(f"{datetime.now().strftime('%H:%M:%S')} {key} ...", flush=True)
        try:
            line = replay_review(arm, entry, log, fields, prompts[arm], bear_only=bear_only)
        except ServerDown as exc:
            raise SystemExit(f"the LLM server went down during {key}: nothing recorded for it ({exc})") from exc
        _append(line)
        done.add(key)
        stage_times = ", ".join(f"{name} {info['seconds']}s x{info['attempts']}" for name, info in line["stages"].items())
        print(
            f"  ok={line['ok']} failed={line['failed_stage']} {line['seconds']}s ({stage_times}); judge prompt {line.get('judge_prompt_tokens')} "
            f"tokens; bear fields shown {line.get('bear_context_fields')}",
            flush=True,
        )


def run(args: argparse.Namespace) -> None:
    """Steps 2-3: the full replay of the sample (resumable)."""
    plan, _ = _load_plan_and_fields()
    arms = [arm.strip() for arm in args.arms.split(",")]
    if not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms takes A and/or B, got {args.arms}")
    sample = plan["sample"]
    if args.reviews:
        wanted = {day.strip() for day in args.reviews.split(",")}
        unknown = wanted - {entry["date"] for entry in sample}
        if unknown:
            raise SystemExit(f"not sample reviews: {', '.join(sorted(unknown))}")
        sample = [entry for entry in sample if entry["date"] in wanted]
    _replay_all([(arm, entry, False) for entry in sample for arm in arms])


def anti_goal(_args: argparse.Namespace) -> None:
    """Arm B's bear alone on each anti-goal review outside the sample."""
    plan, _ = _load_plan_and_fields()
    _replay_all([("B", {"date": day, "index": None}, True) for day in plan["anti_goal"]["extra_reviews"]])


# --- score ----------------------------------------------------------------------------------------


def _calls(cases: Sequence[Mapping[str, Any]] | None, key: str) -> dict[str, str]:
    return {case["symbol"]: case[key] for case in cases or []}


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def score(args: argparse.Namespace) -> None:
    """Step 4: the five tests and the non-gating tables, written as the report."""
    from trading_agent_framework.strategies.bull_bear.batch.agent_scores import forward_returns
    from trading_agent_framework.strategies.bull_bear.batch.scores_cli import fetch_opens

    plan, _ = _load_plan_and_fields()
    records = {record["date"]: record for record in read_reviews()}
    lines = _results()
    full = {(line["date"], line["arm"]): line for line in lines if line["kind"] == "full"}
    bear_only = {line["date"]: line for line in lines if line["kind"] == "anti_goal_bear"}
    sample_days = [entry["date"] for entry in plan["sample"]]

    missing = [f"{day} {arm}" for day in sample_days for arm in ARMS if (day, arm) not in full]
    failed = {arm: [day for day in sample_days if (day, arm) in full and not full[(day, arm)]["ok"]] for arm in ARMS}
    valid = [day for day in sample_days if all((day, arm) in full and full[(day, arm)]["ok"] for arm in ARMS)]
    missing_extra = [day for day in plan["anti_goal"]["extra_reviews"] if day not in bear_only]
    complete = not missing and not missing_extra

    # Arms and the logged run, per valid review: {arm: {day: {symbol: call}}}.
    sources: dict[str, dict[str, dict[str, Any]]] = {"A": {}, "B": {}, "logged": {}}
    for day in valid:
        for arm in ARMS:
            line = full[(day, arm)]
            sources[arm][day] = {"bull": _calls(line["bull"], "conviction"), "bear": _calls(line["bear"], "risk"), "verdict": _calls(line["verdicts"], "verdict")}
        record = records[day]
        sources["logged"][day] = {"bull": _calls(record["bull"], "conviction"), "bear": _calls(record["bear"], "risk"), "verdict": _calls(record["verdicts"], "verdict")}

    def stocks(day: str) -> list[str]:
        return [entry["symbol"] for entry in records[day]["debate_set"]]

    # 20-session excess per (review, symbol), from the logged review records (the excess is over the review's candidates).
    excess: dict[tuple[str, str], float] = {}
    returns_note = ""
    if valid:
        chosen = [records[day] for day in valid]
        opens, calendar = fetch_opens([chosen], today=date.today())
        returns = forward_returns(chosen, opens, calendar, horizons=(HORIZON,))
        excess = {(row.review.isoformat(), row.symbol): row.excess for row in returns.rows}
        returns_note = f"{len(returns.rows)} stock x review pairs with a {HORIZON}-session excess; {returns.dropped} dropped (no open), {returns.immature} immature"

    # Test inputs.
    bear_levels = {arm: [sources[arm][day]["bear"][symbol] for day in valid for symbol in stocks(day)] for arm in ARMS}

    def out_rate(arm: str) -> tuple[int, int]:
        group = [(day, symbol) for day in valid for symbol in stocks(day) if sources[arm][day]["bull"].get(symbol) in ("medium", "high") and sources[arm][day]["bear"].get(symbol) == "high"]
        return sum(sources[arm][day]["verdict"].get(symbol) == "out" for day, symbol in group), len(group)

    listed = plan["anti_goal"]["listed"]
    anti_rows = []
    for entry in listed:
        day, symbol = entry["date"], entry["symbol"]
        line = full.get((day, "B")) if day in sample_days else bear_only.get(day)
        rating = _calls(line.get("bear"), "risk").get(symbol) if line and line.get("ok") else None
        anti_rows.append({**entry, "B_bear": rating, "logged_bear": _calls(records[day]["bear"], "risk").get(symbol)})
    anti_high = sum(row["B_bear"] == "high" for row in anti_rows)
    anti_missing = sum(row["B_bear"] is None for row in anti_rows)  # a failed or missing bear run: not rated high

    def differ(left: str, right: str) -> tuple[int, int]:
        pairs = [(day, symbol) for day in valid for symbol in stocks(day)]
        return sum(sources[left][day]["verdict"].get(symbol) != sources[right][day]["verdict"].get(symbol) for day, symbol in pairs), len(pairs)

    flips: dict[str, list[float]] = {"B_in_A_out": [], "A_in_B_out": []}
    flip_counts = Counter()
    for day in valid:
        for symbol in stocks(day):
            a, b = sources["A"][day]["verdict"].get(symbol), sources["B"][day]["verdict"].get(symbol)
            group = "B_in_A_out" if (b, a) == ("in", "out") else "A_in_B_out" if (a, b) == ("in", "out") else None
            if group is None:
                continue
            flip_counts[group] += 1
            if (day, symbol) in excess:
                flips[group].append(excess[(day, symbol)])

    tests = keep_drop_tests(bear=bear_levels, out_rate_b=out_rate("B"), anti_goal=(anti_high, len(listed)), differ={"B_vs_A": differ("B", "A"), "A_vs_logged": differ("A", "logged")}, flips=flips)
    if not complete:
        decision = f"NONE: the replay is incomplete ({len(missing)} sample (review, arm) lines and {len(missing_extra)} anti-goal bear reviews missing); no decision is taken on partial data"
    elif all(test["pass"] for test in tests):
        decision = "KEEP: all five tests pass"
    else:
        decision = "DROP: " + "; ".join(f"test {test['test']} ({test['name']}) fails" for test in tests if not test["pass"])

    # Non-gating tables.
    pooled = sorted(excess[(day, symbol)] for day in valid for symbol in stocks(day) if (day, symbol) in excess)
    cuts = statistics.quantiles(pooled, n=10) if len(pooled) >= 10 else None

    def level_table(source: str) -> list[str]:
        rows = []
        for level in LEVELS:
            group = [(day, symbol) for day in valid for symbol in stocks(day) if sources[source][day]["bear"].get(symbol) == level]
            values = [excess[key] for key in group if key in excess]
            worst = share(sum(value <= cuts[0] for value in values), len(values)) if cuts else None
            best = share(sum(value >= cuts[-1] for value in values), len(values)) if cuts else None
            outs = share(sum(sources[source][day]["verdict"].get(symbol) == "out" for day, symbol in group), len(group))
            rows.append(f"| {level} | {len(group)} | {_signed(statistics.fmean(values) if values else None)} | {_pct(worst)} | {_pct(best)} | {_pct(outs, 0)} |")
        return rows

    def cross_table(source: str) -> list[str]:
        rows = ["| bull \\ bear | low | medium | high |", "|---|---|---|---|"]
        for bull_level in LEVELS:
            cells = []
            for bear_level in LEVELS:
                group = [(day, symbol) for day in valid for symbol in stocks(day) if sources[source][day]["bull"].get(symbol) == bull_level and sources[source][day]["bear"].get(symbol) == bear_level]
                outs = sum(sources[source][day]["verdict"].get(symbol) == "out" for day, symbol in group)
                cells.append(f"{_pct(share(outs, len(group)), 0)} of {len(group)}")
            rows.append(f"| {bull_level} | " + " | ".join(cells) + " |")
        return rows

    prompt_sizes = {
        arm: [
            (full[(day, arm)]["judge_max_input_tokens"], full[(day, arm)]["judge_prompt_tokens"], full[(day, arm)]["stocks"], day)
            for day in sample_days
            if (day, arm) in full and full[(day, arm)].get("judge_max_input_tokens")
        ]
        for arm in ARMS
    }
    timings = {arm: [full[(day, arm)]["seconds"] for day in sample_days if (day, arm) in full] for arm in ARMS}

    report = [
        "# bull_bear bear-balance replay",
        "",
        f"Generated by `scripts/replays/bull_bear_bear_balance.py score` on {datetime.now(UTC).isoformat(timespec='seconds')}. Spec: "
        "`docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-design.md`. Run replayed: `" + str(RUN_DIR.relative_to(MAIN)) + "`.",
        "",
        f"**Decision: {decision}**",
        "",
        "## Coverage",
        "",
        f"- Sample: {len(sample_days)} reviews; valid in both arms: {len(valid)}; missing lines: {', '.join(missing) or 'none'}.",
        f"- Failed (no valid submission after the retry, or a loop; excluded from both arms): A {failed['A'] or 'none'}, B {failed['B'] or 'none'}.",
        f"- Anti-goal bear-only reviews: {len(plan['anti_goal']['extra_reviews']) - len(missing_extra)} of {len(plan['anti_goal']['extra_reviews'])} done.",
        f"- Outcomes: {returns_note or 'none (no valid review)'}.",
        "",
        "## Keep/drop tests (fixed in the spec before the run)",
        "",
        "| # | test | numbers | result |",
        "|---|---|---|---|",
    ]
    t1, t2, t3, t4, t5 = tests
    lr_a, lr_b = out_rate("A"), out_rate("logged")
    report += [
        f"| 1 | {t1['name']} | B high {_pct(t1['high_share_B'])} (needs 4-15%); low B {_pct(t1['low_share_B'])} vs A {_pct(t1['low_share_A'])} "
        f"(B may exceed A by 15 points at most) | {'PASS' if t1['pass'] else 'FAIL'} |",
        f"| 2 | {t2['name']} | B: out on bull medium/high + bear high {_pct(t2['out_rate_B'], 0)} ({t2['outs']} of {t2['stocks']}; "
        f"needs < 75%); A {_pct(share(*lr_a), 0)} ({lr_a[0]} of {lr_a[1]}); logged {_pct(share(*lr_b), 0)} ({lr_b[0]} of {lr_b[1]}) "
        f"| {'PASS' if t2['pass'] else 'FAIL'} |",
        f"| 3 | {t3['name']} | B's bear rates high {t3['rated_high']} of {t3['listed']} listed (needs at least half; {anti_missing} "
        f"without a B rating, counted as not high){'' if t3['testable'] else '; NOT TESTABLE'} | {'PASS' if t3['pass'] else 'FAIL'} "
        "|",
        f"| 4 | {t4['name']} | verdicts B vs A differ {_pct(t4['differ_B_vs_A'])} ({t4['counts']['B_vs_A'][0]} of {t4['counts']['B_vs_A'][1]}); "
        f"A vs logged {_pct(t4['differ_A_vs_logged'])} ({t4['counts']['A_vs_logged'][0]} of {t4['counts']['A_vs_logged'][1]}) | {'PASS' if t4['pass'] else 'FAIL'} "
        "|",
        f"| 5 | {t5['name']} | B in / A out: {_signed(t5['mean_B_in_A_out'])} mean 20d excess (n={t5['n_B_in_A_out']} of {flip_counts['B_in_A_out']} "
        f"flips); A in / B out: {_signed(t5['mean_A_in_B_out'])} (n={t5['n_A_in_B_out']} of {flip_counts['A_in_B_out']}); empty group "
        f"= 0 | {'PASS' if t5['pass'] else 'FAIL'} |",
        "",
    ]
    for test in tests:
        if test.get("note"):
            report.append(f"- Test {test['test']}: {test['note']}.")
    report += [
        "",
        "## Anti-goal list",
        "",
        "| review | symbol | revenue % | margin chg pts | margin prior % | days | since filed % | in sample | logged bear | B bear |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in anti_rows:
        report.append(
            f"| {row['date']} | {row['symbol']} | {row['revenue_growth_pct']} | {row['operating_margin_change_pts']} | {row['operating_margin_prior_year_pct']} "
            f"| {row['days_since_filed']} | {row['return_since_filed_pct']} | {'yes' if row['in_sample'] else 'no'} | {row['logged_bear']} "
            f"| {row['B_bear'] or 'missing'} |"
        )
    report += ["", f"## Not gating (too few stocks on {len(valid)} reviews)", "", "Deciles are over the pooled 20d excess of every candidate of the valid reviews.", ""]
    for source in ("B", "A", "logged"):
        report += [f"Bear level, {source}:", "", "| bear risk | n | 20d excess | in worst 10% | in best 10% | judge out |", "|---|---|---|---|---|---|", *level_table(source), ""]
    for source in ("B", "A", "logged"):
        report += [f"Judge out rate by bull x bear, {source}:", "", *cross_table(source), ""]
    report += ["## Judge prompt size and timings", ""]
    for arm in ARMS:
        if prompt_sizes[arm]:
            tokens, first, count, day = max(prompt_sizes[arm])
            report.append(
                f"- Arm {arm}: largest judge prompt {tokens} tokens ({count} stocks, {day}; its first call {first}), as reported by the server: "
                "system prompt, tools and task included, later calls also carry the memory round trips."
            )
        if timings[arm]:
            report.append(f"- Arm {arm}: {len(timings[arm])} reviews, median {statistics.median(timings[arm]) / 60:.1f} min, total {sum(timings[arm]) / 3600:.2f} h.")
    out = Path(args.out) if args.out else REPORT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {out}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="replay the sample reviews (resumable)")
    run_parser.add_argument("--arms", default="A,B")
    run_parser.add_argument("--reviews", default="", help="comma-separated sample review dates (default: all 24)")
    commands.add_parser("anti-goal", help="arm B's bear on the anti-goal reviews outside the sample (resumable)")
    score_parser = commands.add_parser("score", help="the keep/drop tests and the report")
    score_parser.add_argument("--out", default="", help=f"report path (default {REPORT_FILE.relative_to(REPO)})")
    args = parser.parse_args(argv)
    {"run": run, "anti-goal": anti_goal, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
