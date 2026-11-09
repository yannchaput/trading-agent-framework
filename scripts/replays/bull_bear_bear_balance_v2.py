"""Throwaway v2 replay of the balance-based bear (and the judge without a veto) on reviews v1 never used.

Not a repo command; kept for the record. Spec: docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-v2-design.md.
v1's prompts and fields are re-tested byte for byte; only the samples and the six keep/drop tests are new. The sample
seeds, sizes and thresholds below are fixed in the spec; nothing here is tuned on the results.

This script reuses v1's (`bull_bear_bear_balance.py`, loaded by file path) log parsing, fields and agent runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import types
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_V1_PATH = Path(__file__).with_name("bull_bear_bear_balance.py")
_V1_SPEC = importlib.util.spec_from_file_location("bull_bear_bear_balance", _V1_PATH)
assert _V1_SPEC is not None and _V1_SPEC.loader is not None
v1 = importlib.util.module_from_spec(_V1_SPEC)
_V1_SPEC.loader.exec_module(v1)

FULL_SEED = 20261109
FULL_SIZE = 36
JUDGE_SEED = 20261110
JUDGE_SIZE = 60
POOL_EXPECTED = 230
FULL_ARMS = ("A", "B")
JUDGE_ARMS = ("J-old", "J-new")
MIN_VETO = 30  # stocks in test 2's group
MIN_RUNWAY = 10  # stocks in test 3's and test 5's groups
MIN_HIGH = 20  # high stocks in test 4
NEW_PROMPTS_REV = "407a310"  # the v1 commit whose prompts v2 re-tests
RUNWAY_MONTHS = 12


# --- pure helpers (no I/O) --------------------------------------------------------------------------


def fresh_pool(days: Sequence[str], v1_plan: Mapping[str, Any]) -> list[str]:
    """The run's review days minus v1's sample reviews and v1's anti-goal-only reviews, sorted."""
    used = {entry["date"] for entry in v1_plan["sample"]} | set(v1_plan["anti_goal"]["extra_reviews"])
    return sorted(set(days) - used)


def full_sample(pool: Sequence[str]) -> list[str]:
    """The 36 reviews of arms A and B: a fixed-seed draw of the sorted pool."""
    if len(pool) < FULL_SIZE:
        raise ValueError(f"{len(pool)} reviews in the pool, need {FULL_SIZE}")
    return sorted(random.Random(FULL_SEED).sample(sorted(pool), FULL_SIZE))


def judge_eligible(record: Mapping[str, Any]) -> bool:
    """True when some stock has a logged bull conviction of medium or high and a logged bear risk of high."""
    bull = {case["symbol"]: case["conviction"] for case in record["bull"]}
    return any(case["risk"] == "high" and bull.get(case["symbol"]) in ("medium", "high") for case in record["bear"])


def judge_sample(pool: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The 60 reviews of the judge-only arms: a fixed-seed draw of the pool reviews with an eligible stock."""
    eligible = sorted(day for day in pool if judge_eligible(records[day]))
    if len(eligible) < JUDGE_SIZE:
        raise ValueError(f"{len(eligible)} eligible reviews in the pool, need {JUDGE_SIZE}")
    return sorted(random.Random(JUDGE_SEED).sample(eligible, JUDGE_SIZE))


def judge_context(arm: str, day: str, judge_ctx: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The logged judge input; for J-new, each fact sheet also carries the four fields (nothing else changes)."""
    if arm == "J-old":
        return {**judge_ctx, "stocks": [dict(stock) for stock in judge_ctx["stocks"]]}
    stocks = []
    for stock in judge_ctx["stocks"]:
        sheet = stock["fact_sheet"]
        stocks.append({**stock, "fact_sheet": v1.with_new_fields(sheet, fields[day][sheet["symbol"]])})
    return {**judge_ctx, "stocks": stocks}


def valid_pairs(lines: Sequence[Mapping[str, Any]], days: Sequence[str], arms: tuple[str, str], kind: str) -> tuple[list[str], dict[str, list[str]]]:
    """(days where both arms have a valid line of `kind`, {arm: days where that arm's line failed}); a failure drops the day from both."""
    by_key = {(line["date"], line["arm"]): line for line in lines if line["kind"] == kind}
    failed = {arm: [day for day in days if (day, arm) in by_key and not by_key[(day, arm)]["ok"]] for arm in arms}
    valid = [day for day in days if all((day, arm) in by_key and by_key[(day, arm)]["ok"] for arm in arms)]
    return valid, failed


def done_key(line: Mapping[str, Any]) -> tuple[str, str, str]:
    """The resume key of a result line: (review, arm, kind), kind "full" or "judge"."""
    return (line["date"], line["arm"], line["kind"])


def runway_symbols(sheets: Sequence[Mapping[str, Any]]) -> set[str]:
    """Symbols whose fact sheet shows a runway under 12 months (an absent runway does not count)."""
    return {sheet["symbol"] for sheet in sheets if sheet.get("runway_months") is not None and sheet["runway_months"] < RUNWAY_MONTHS}


def _mean(values: Sequence[float]) -> float:
    """The mean, or 0 for an empty group (the excess is relative to the review's mean: 0 is average)."""
    return statistics.fmean(values) if values else 0.0


def keep_drop_tests_v2(
    *,
    bear: Mapping[str, Sequence[str]],
    veto: Mapping[str, tuple[int, int]],
    runway_out_new: tuple[int, int],
    excess_by_level_b: Mapping[str, Sequence[float]],
    runway_high_b: tuple[int, int],
    flips: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    """The spec's six keep/drop tests, each with its numbers, `testable` (where a minimum applies), `pass` and `note`.

    Args:
        bear: the bear's risk levels per arm ("A", "B") over the valid full reviews.
        veto: (outs, stocks) per judge arm ("J-old", "J-new") among logged bull medium/high with logged bear high.
        runway_out_new: (outs, stocks) of J-new among stocks with a runway under 12 months.
        excess_by_level_b: arm B's 20d excess per bear level ("high", "medium"), only the stocks that have one.
        runway_high_b: (rated high by arm B's bear, stocks) among stocks with a runway under 12 months.
        flips: the 20d excess of "B_in_A_out" and "A_in_B_out" stocks (an empty group counts as 0).
    """
    tests: list[dict[str, Any]] = []
    b_levels, a_levels = Counter(bear["B"]), Counter(bear["A"])
    high_b = v1.share(b_levels["high"], sum(b_levels.values()))
    low_b, low_a = v1.share(b_levels["low"], sum(b_levels.values())), v1.share(a_levels["low"], sum(a_levels.values()))
    ok1 = high_b is not None and low_b is not None and low_a is not None and 0.04 <= high_b <= 0.15 and round(low_b - low_a, 12) <= 0.15
    tests.append({"test": 1, "name": "the bear still argues", "high_share_B": high_b, "low_share_B": low_b, "low_share_A": low_a, "testable": True, "pass": ok1, "note": None})

    (old_outs, old_stocks), (new_outs, new_stocks) = veto["J-old"], veto["J-new"]
    old_rate, new_rate = v1.share(old_outs, old_stocks), v1.share(new_outs, new_stocks)
    testable2 = min(old_stocks, new_stocks) >= MIN_VETO
    ok2 = testable2 and old_rate is not None and new_rate is not None and new_rate < 0.75 and round(old_rate - new_rate, 12) >= 0.15
    tests.append(
        {
            "test": 2,
            "name": "no veto",
            "out_rate_old": old_rate,
            "out_rate_new": new_rate,
            "stocks": (old_stocks, new_stocks),
            "testable": testable2,
            "pass": ok2,
            "note": None if testable2 else f"fewer than {MIN_VETO} stocks in the group: the test cannot be run and does not pass",
        }
    )

    out3, stocks3 = runway_out_new
    testable3 = stocks3 >= MIN_RUNWAY
    tests.append(
        {
            "test": 3,
            "name": "the judge still removes real risks",
            "outs": out3,
            "stocks": stocks3,
            "testable": testable3,
            "pass": testable3 and 5 * out3 >= 4 * stocks3,
            "note": None if testable3 else f"fewer than {MIN_RUNWAY} runway-under-12 stocks: the test cannot be run and does not pass",
        }
    )

    high_excess, medium_excess = list(excess_by_level_b["high"]), list(excess_by_level_b["medium"])
    testable4 = len(high_excess) >= MIN_HIGH and bool(medium_excess)
    tests.append(
        {
            "test": 4,
            "name": "high means risk",
            "mean_high": _mean(high_excess),
            "mean_medium": _mean(medium_excess),
            "n_high": len(high_excess),
            "n_medium": len(medium_excess),
            "testable": testable4,
            "pass": testable4 and _mean(high_excess) < _mean(medium_excess),
            "note": None if testable4 else f"fewer than {MIN_HIGH} high stocks with an outcome (or no medium stock): the test cannot be run and does not pass",
        }
    )

    high5, stocks5 = runway_high_b
    testable5 = stocks5 >= MIN_RUNWAY
    tests.append(
        {
            "test": 5,
            "name": "real risks still flagged",
            "rated_high": high5,
            "stocks": stocks5,
            "testable": testable5,
            "pass": testable5 and 2 * high5 >= stocks5,
            "note": None if testable5 else f"fewer than {MIN_RUNWAY} runway-under-12 stocks: the test cannot be run and does not pass",
        }
    )

    b_in, a_in = _mean(flips["B_in_A_out"]), _mean(flips["A_in_B_out"])
    tests.append(
        {
            "test": 6,
            "name": "no harm in the flips",
            "mean_B_in_A_out": b_in,
            "n_B_in_A_out": len(flips["B_in_A_out"]),
            "mean_A_in_B_out": a_in,
            "n_A_in_B_out": len(flips["A_in_B_out"]),
            "testable": True,
            "pass": b_in >= a_in,
            "note": None,
        }
    )
    return tests


def decision(tests: Sequence[Mapping[str, Any]], complete: bool) -> str:
    """The report's decision line: NONE while the replay is incomplete, KEEP when all six pass, else DROP with each failure."""
    if not complete:
        return "NONE: the replay is incomplete; no decision is taken on partial data"
    if all(test["pass"] for test in tests):
        return "KEEP: all six tests pass"
    return "DROP: " + "; ".join(f"test {test['test']} ({test['name']}) fails" for test in tests if not test["pass"])


# --- paths and I/O -------------------------------------------------------------------------------------

PLAN_V2 = v1.DATA / "replay_v2_plan.json"
RESULTS_V2 = v1.DATA / "replay_v2_results.jsonl"
REPORT_V2 = v1.REPO / "docs/superpowers/replays/2026-11-08-bull-bear-bear-balance-v2.md"


def _results() -> list[dict[str, Any]]:
    """The v2 result lines written so far."""
    if not RESULTS_V2.is_file():
        return []
    return [json.loads(line) for line in RESULTS_V2.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(line: Mapping[str, Any]) -> None:
    """Append one result line (the resume record)."""
    with RESULTS_V2.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def _load_plan() -> dict[str, Any]:
    """The committed v2 plan file (written by `prepare`)."""
    if not PLAN_V2.is_file():
        raise SystemExit("run `prepare` first")
    return json.loads(PLAN_V2.read_text(encoding="utf-8"))


def prepare(args: argparse.Namespace) -> None:
    """Pin the prompts, fix both samples and write the plan file. No agent call."""
    del args  # no options
    log = v1.parse_log()
    records = {record["date"]: record for record in v1.read_reviews()}
    if sorted(records) != sorted(log["bull"]):
        raise SystemExit("the review dates of reviews.jsonl and of the log differ")
    current, pinned = v1.load_prompts(None), v1.load_prompts(NEW_PROMPTS_REV)
    if not Path(current.__file__).resolve().is_relative_to(v1.REPO):
        raise SystemExit(f"the current prompts are read from {current.__file__}, not from this worktree: launch from the worktree root")
    same = all(getattr(current, name) == getattr(pinned, name) for name in ("BULL_SYSTEM", "BEAR_SYSTEM", "JUDGE_SYSTEM"))
    same = same and all(getattr(current, name)(300) == getattr(pinned, name)(300) for name in ("bull_task", "bear_task", "judge_task"))
    if not same:
        raise SystemExit(f"the current prompts differ from v1's at {NEW_PROMPTS_REV}: v2 re-tests them byte for byte")
    v1_plan = json.loads(v1.PLAN_FILE.read_text(encoding="utf-8"))
    pool = fresh_pool(sorted(records), v1_plan)
    if len(pool) != POOL_EXPECTED:
        raise SystemExit(f"{len(pool)} reviews in the fresh pool, expected {POOL_EXPECTED}")
    full = full_sample(pool)
    eligible = [day for day in pool if judge_eligible(records[day])]
    judge_days = judge_sample(pool, records)
    plan = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "head": v1.git_sha("HEAD"),
        "prompts_rev": NEW_PROMPTS_REV,
        "old_prompts_rev": v1.OLD_PROMPTS_REV,
        "pool_size": len(pool),
        "full_sample": full,
        "judge_eligible_count": len(eligible),
        "judge_sample": judge_days,
        "thresholds": {
            "test1": "B high share 4-15%, B low share <= A + 15 points",
            "test2": f"J-new out rate < 75% and >= 15 points below J-old, min {MIN_VETO} stocks",
            "test3": f"J-new out rate on runway < {RUNWAY_MONTHS} >= 80%, min {MIN_RUNWAY} stocks",
            "test4": f"mean 20d excess of B high < B medium, min {MIN_HIGH} high stocks",
            "test5": f"B rates high >= half of runway < {RUNWAY_MONTHS} stocks, min {MIN_RUNWAY}",
            "test6": "mean 20d excess B in/A out >= A in/B out, empty group = 0",
        },
    }
    v1.DATA.mkdir(parents=True, exist_ok=True)
    PLAN_V2.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"pool {len(pool)}; full sample {len(full)}; judge eligible {len(eligible)}; judge sample {len(judge_days)}")
    print("full sample:", ", ".join(full))
    print("judge sample:", ", ".join(judge_days))
    print(f"wrote {PLAN_V2}")


def replay_judge(arm: str, day: str, log: Mapping[str, Any], fields: Mapping[str, Any], prompts: types.ModuleType) -> dict[str, Any]:
    """The judge alone on the logged bull and bear arguments of a review (J-new adds the four fields); one result line."""
    judge_ctx = log["judge"][day]["context"]
    review_time = datetime.fromisoformat(judge_ctx["current_datetime"])
    symbols = [stock["fact_sheet"]["symbol"] for stock in judge_ctx["stocks"]]
    replay = v1._Replay(arm, day, prompts)
    params = replay.params
    line: dict[str, Any] = {"date": day, "index": None, "arm": arm, "kind": "judge", "ok": False, "failed_stage": None, "stocks": len(symbols)}
    started = datetime.now(UTC)
    stages: dict[str, Any] = {}
    try:
        replay.create(("judge",), review_time)
        previous = {stock["fact_sheet"]["symbol"]: stock["previous_verdict"] for stock in judge_ctx["stocks"] if "previous_verdict" in stock}
        replay.recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})
        context = judge_context(arm, day, judge_ctx, fields)
        line["fields_shown"] = v1._shown(context)
        stages["judge"] = replay.stage("judge", "submit_verdicts", prompts.judge_task(params.reason_max_chars), context, tool_budget=params.judge_memory_budget)
        tokens = [value for value in stages["judge"]["input_tokens"] if value is not None]
        line["judge_prompt_tokens"] = tokens[0] if tokens else None
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


def _execute(jobs: Sequence[tuple[str, str, str]]) -> None:
    """Run each (arm, review day, kind) job not already in the results file; stop if the server goes down."""
    v1._load_env()
    _, fields = v1._load_plan_and_fields()
    log = v1.parse_log()
    old, new = v1.load_prompts(v1.OLD_PROMPTS_REV), v1.load_prompts(NEW_PROMPTS_REV)
    prompts = {"A": old, "B": new, "J-old": old, "J-new": new}
    from trading_agent_framework.agents.config import LLMCredentials

    base_url = LLMCredentials.from_env().base_url
    done = {done_key(line) for line in _results()}
    for arm, day, kind in jobs:
        key = (day, arm, kind)
        if key in done:
            print(f"skip {key}: already in {RESULTS_V2.name}")
            continue
        if not v1._server_up(base_url):
            raise SystemExit(f"the LLM server at {base_url} does not answer: stopped before {key}")
        print(f"{datetime.now().strftime('%H:%M:%S')} {key} ...", flush=True)
        try:
            if kind == "full":
                line = v1.replay_review(arm, {"date": day, "index": None}, log, fields, prompts[arm])
            else:
                line = replay_judge(arm, day, log, fields, prompts[arm])
        except v1.ServerDown as exc:
            raise SystemExit(f"the LLM server went down during {key}: nothing recorded for it ({exc})") from exc
        _append(line)
        done.add(key)
        stage_times = ", ".join(f"{name} {info['seconds']}s x{info['attempts']}" for name, info in line["stages"].items())
        print(f"  ok={line['ok']} failed={line['failed_stage']} {line['seconds']}s ({stage_times}); judge prompt {line.get('judge_prompt_tokens')} tokens", flush=True)


def _pick(plan_days: Sequence[str], wanted: str, label: str) -> list[str]:
    """The plan's days, or the comma-separated subset in `wanted` (an unknown day stops the run)."""
    if not wanted:
        return list(plan_days)
    chosen = [day.strip() for day in wanted.split(",")]
    unknown = sorted(set(chosen) - set(plan_days))
    if unknown:
        raise SystemExit(f"not {label} reviews: {', '.join(unknown)}")
    return [day for day in plan_days if day in chosen]


def run(args: argparse.Namespace) -> None:
    """Arms A and B on the full sample (resumable)."""
    arms = [arm.strip() for arm in args.arms.split(",")]
    if not set(arms) <= set(FULL_ARMS):
        raise SystemExit(f"--arms takes A and/or B, got {args.arms}")
    days = _pick(_load_plan()["full_sample"], args.reviews, "full-sample")
    _execute([(arm, day, "full") for day in days for arm in arms])


def judge(args: argparse.Namespace) -> None:
    """Arms J-old and J-new on the judge sample (resumable)."""
    arms = [arm.strip() for arm in args.arms.split(",")]
    if not set(arms) <= set(JUDGE_ARMS):
        raise SystemExit(f"--arms takes J-old and/or J-new, got {args.arms}")
    days = _pick(_load_plan()["judge_sample"], args.reviews, "judge-sample")
    _execute([(arm, day, "judge") for day in days for arm in arms])


# --- score ----------------------------------------------------------------------------------------


def score(args: argparse.Namespace) -> None:
    """The six tests and the non-gating tables, written as the report."""
    from trading_agent_framework.strategies.bull_bear.batch.agent_scores import forward_returns
    from trading_agent_framework.strategies.bull_bear.batch.scores_cli import fetch_opens

    plan = _load_plan()
    _, fields = v1._load_plan_and_fields()
    records = {record["date"]: record for record in v1.read_reviews()}
    log = v1.parse_log()
    lines = _results()
    full_days, judge_days = plan["full_sample"], plan["judge_sample"]
    by_line = {(line["date"], line["arm"]): line for line in lines}
    valid_full, failed_full = valid_pairs(lines, full_days, FULL_ARMS, "full")
    valid_judge, failed_judge = valid_pairs(lines, judge_days, JUDGE_ARMS, "judge")
    missing = [f"{day} {arm}" for days, arms in ((full_days, FULL_ARMS), (judge_days, JUDGE_ARMS)) for day in days for arm in arms if (day, arm) not in by_line]
    complete = not missing

    def calls(cases: Sequence[Mapping[str, Any]] | None, key: str) -> dict[str, str]:
        return {case["symbol"]: case[key] for case in cases or []}

    def logged(day: str) -> dict[str, Any]:
        record = records[day]
        return {"bull": calls(record["bull"], "conviction"), "bear": calls(record["bear"], "risk"), "verdict": calls(record["verdicts"], "verdict")}

    # Per source and review: {source: {day: {"bull", "bear", "verdict": {symbol: call}}}}.
    sources: dict[str, dict[str, dict[str, Any]]] = {"A": {}, "B": {}, "logged": {}, "J-old": {}, "J-new": {}}
    for day in valid_full:
        for arm in FULL_ARMS:
            line = by_line[(day, arm)]
            sources[arm][day] = {"bull": calls(line["bull"], "conviction"), "bear": calls(line["bear"], "risk"), "verdict": calls(line["verdicts"], "verdict")}
    for day in valid_full + valid_judge:
        sources["logged"][day] = logged(day)
    for day in valid_judge:
        for arm in JUDGE_ARMS:
            sources[arm][day] = {**logged(day), "verdict": calls(by_line[(day, arm)]["verdicts"], "verdict")}

    def stocks(day: str) -> list[str]:
        return [entry["symbol"] for entry in records[day]["debate_set"]]

    def runway(day: str) -> set[str]:
        return runway_symbols([stock["fact_sheet"] for stock in log["judge"][day]["context"]["stocks"]])

    # 20-session excess per (review, symbol), over every valid review of either kind.
    union = sorted(set(valid_full) | set(valid_judge))
    excess: dict[tuple[str, str], float] = {}
    returns_note = ""
    if union:
        chosen = [records[day] for day in union]
        opens, calendar = fetch_opens([chosen], today=date.today())
        returns = forward_returns(chosen, opens, calendar, horizons=(v1.HORIZON,))
        excess = {(row.review.isoformat(), row.symbol): row.excess for row in returns.rows}
        returns_note = f"{len(returns.rows)} stock x review pairs with a {v1.HORIZON}-session excess; {returns.dropped} dropped (no open), {returns.immature} immature"

    # Test inputs.
    bear_levels = {arm: [sources[arm][day]["bear"][symbol] for day in valid_full for symbol in stocks(day)] for arm in FULL_ARMS}

    def veto_group(day: str) -> list[str]:
        return [symbol for symbol in stocks(day) if sources["logged"][day]["bull"].get(symbol) in ("medium", "high") and sources["logged"][day]["bear"].get(symbol) == "high"]

    def outs(source: str, pairs: Sequence[tuple[str, str]]) -> tuple[int, int]:
        return sum(sources[source][day]["verdict"].get(symbol) == "out" for day, symbol in pairs), len(pairs)

    veto_pairs = [(day, symbol) for day in valid_judge for symbol in veto_group(day)]
    runway_pairs_judge = [(day, symbol) for day in valid_judge for symbol in stocks(day) if symbol in runway(day)]
    runway_pairs_full = [(day, symbol) for day in valid_full for symbol in stocks(day) if symbol in runway(day)]
    excess_by_level = {
        level: [excess[(day, symbol)] for day in valid_full for symbol in stocks(day) if sources["B"][day]["bear"].get(symbol) == level and (day, symbol) in excess] for level in ("high", "medium")
    }
    runway_high = (sum(sources["B"][day]["bear"].get(symbol) == "high" for day, symbol in runway_pairs_full), len(runway_pairs_full))

    def flip_groups(left: str, right: str, days: Sequence[str]) -> tuple[dict[str, list[float]], Counter[str]]:
        groups: dict[str, list[float]] = {f"{left}_in_{right}_out": [], f"{right}_in_{left}_out": []}
        counts: Counter[str] = Counter()
        for day in days:
            for symbol in stocks(day):
                a, b = sources[left][day]["verdict"].get(symbol), sources[right][day]["verdict"].get(symbol)
                group = f"{left}_in_{right}_out" if (a, b) == ("in", "out") else f"{right}_in_{left}_out" if (a, b) == ("out", "in") else None
                if group is None:
                    continue
                counts[group] += 1
                if (day, symbol) in excess:
                    groups[group].append(excess[(day, symbol)])
        return groups, counts

    b_a_flips, b_a_counts = flip_groups("B", "A", valid_full)
    flips = {"B_in_A_out": b_a_flips["B_in_A_out"], "A_in_B_out": b_a_flips["A_in_B_out"]}
    tests = keep_drop_tests_v2(
        bear=bear_levels,
        veto={"J-old": outs("J-old", veto_pairs), "J-new": outs("J-new", veto_pairs)},
        runway_out_new=outs("J-new", runway_pairs_judge),
        excess_by_level_b=excess_by_level,
        runway_high_b=runway_high,
        flips=flips,
    )
    verdict = decision(tests, complete)

    def differ(left: str, right: str, days: Sequence[str]) -> tuple[int, int]:
        pairs = [(day, symbol) for day in days for symbol in stocks(day)]
        return sum(sources[left][day]["verdict"].get(symbol) != sources[right][day]["verdict"].get(symbol) for day, symbol in pairs), len(pairs)

    # Non-gating tables.
    pooled = sorted(excess[(day, symbol)] for day in union for symbol in stocks(day) if (day, symbol) in excess)
    cuts = statistics.quantiles(pooled, n=10) if len(pooled) >= 10 else None

    def level_table(source: str, days: Sequence[str]) -> list[str]:
        rows = []
        for level in v1.LEVELS:
            group = [(day, symbol) for day in days for symbol in stocks(day) if sources[source][day]["bear"].get(symbol) == level]
            values = [excess[key] for key in group if key in excess]
            worst = v1.share(sum(value <= cuts[0] for value in values), len(values)) if cuts else None
            best = v1.share(sum(value >= cuts[-1] for value in values), len(values)) if cuts else None
            out_share = v1.share(outs(source, group)[0], len(group))
            rows.append(f"| {level} | {len(group)} | {v1._signed(statistics.fmean(values) if values else None)} | {v1._pct(worst)} | {v1._pct(best)} | {v1._pct(out_share, 0)} |")
        return rows

    def cross_table(source: str, days: Sequence[str]) -> list[str]:
        rows = ["| bull \\ bear | low | medium | high |", "|---|---|---|---|"]
        for bull_level in v1.LEVELS:
            cells = []
            for bear_level in v1.LEVELS:
                group = [(day, symbol) for day in days for symbol in stocks(day) if sources[source][day]["bull"].get(symbol) == bull_level and sources[source][day]["bear"].get(symbol) == bear_level]
                cells.append(f"{v1._pct(v1.share(outs(source, group)[0], len(group)), 0)} of {len(group)}")
            rows.append(f"| {bull_level} | " + " | ".join(cells) + " |")
        return rows

    def share_text(pair: tuple[int, int]) -> str:
        return f"{v1._pct(v1.share(*pair), 0)} ({pair[0]} of {pair[1]})"

    def flip_text(values: Sequence[float], flips: int) -> str:
        return f"{v1._signed(statistics.fmean(values) if values else None)} mean 20d excess (n={len(values)} of {flips} flips)"

    def result(test: Mapping[str, Any]) -> str:
        return "PASS" if test["pass"] else "NOT TESTABLE" if not test["testable"] else "FAIL"

    t1, t2, t3, t4, t5, t6 = tests
    report = [
        "# bull_bear bear-balance replay, v2",
        "",
        f"Generated by `scripts/replays/bull_bear_bear_balance_v2.py score` on {datetime.now(UTC).isoformat(timespec='seconds')}. Spec: "
        "`docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-v2-design.md`. Run replayed: `" + str(v1.RUN_DIR.relative_to(v1.MAIN)) + "`. Prompts re-tested: v1 at `" + NEW_PROMPTS_REV + "`.",
        "",
        f"**Decision: {verdict}**",
        "",
        "## Coverage",
        "",
        f"- Full sample (arms A, B): {len(full_days)} reviews; valid in both: {len(valid_full)}. Failed (excluded from both arms): A {failed_full['A'] or 'none'}, B {failed_full['B'] or 'none'}.",
        f"- Judge sample (J-old, J-new): {len(judge_days)} reviews; valid in both: {len(valid_judge)}. Failed: J-old {failed_judge['J-old'] or 'none'}, J-new {failed_judge['J-new'] or 'none'}.",
        f"- Missing lines: {', '.join(missing) or 'none'}.",
        f"- Outcomes: {returns_note or 'none (no valid review)'}.",
        "",
        "## Keep/drop tests (fixed in the spec before the run)",
        "",
        "| # | test | numbers | result |",
        "|---|---|---|---|",
        f"| 1 | {t1['name']} | B high {v1._pct(t1['high_share_B'])} (needs 4-15%); low B {v1._pct(t1['low_share_B'])} vs A {v1._pct(t1['low_share_A'])} "
        f"(B may exceed A by 15 points at most) | {result(t1)} |",
        f"| 2 | {t2['name']} | out rate on logged bull medium/high + bear high: J-old {v1._pct(t2['out_rate_old'], 0)}, J-new {v1._pct(t2['out_rate_new'], 0)} "
        f"(needs J-new < 75% and 15 points below J-old; {t2['stocks'][0]} stocks, min {MIN_VETO}) | {result(t2)} |",
        f"| 3 | {t3['name']} | J-new puts out {share_text((t3['outs'], t3['stocks']))} of runway < {RUNWAY_MONTHS} stocks (needs 80%, min {MIN_RUNWAY}) | {result(t3)} |",
        f"| 4 | {t4['name']} | B high {v1._signed(t4['mean_high'])} (n={t4['n_high']}) vs B medium {v1._signed(t4['mean_medium'])} (n={t4['n_medium']}); "
        f"high must be lower, min {MIN_HIGH} high | {result(t4)} |",
        f"| 5 | {t5['name']} | B rates high {share_text((t5['rated_high'], t5['stocks']))} of runway < {RUNWAY_MONTHS} stocks (needs half, min {MIN_RUNWAY}) | {result(t5)} |",
        f"| 6 | {t6['name']} | B in / A out {v1._signed(t6['mean_B_in_A_out'])} (n={t6['n_B_in_A_out']} of {b_a_counts['B_in_A_out']} flips); "
        f"A in / B out {v1._signed(t6['mean_A_in_B_out'])} (n={t6['n_A_in_B_out']} of {b_a_counts['A_in_B_out']}); empty group = 0 | {result(t6)} |",
        "",
    ]
    report += [f"- Test {test['test']}: {test['note']}." for test in tests if test["note"]]
    b_vs_a, a_vs_logged, old_vs_logged = differ("B", "A", valid_full), differ("A", "logged", valid_full), differ("J-old", "logged", valid_judge)
    judge_flips, judge_flip_counts = flip_groups("J-new", "J-old", valid_judge)
    report += [
        "",
        "## Not gating",
        "",
        f"- Verdicts B vs A differ {share_text(b_vs_a)}; A vs logged {share_text(a_vs_logged)} (v1's 'effect above noise').",
        f"- J-old vs logged verdicts differ {share_text(old_vs_logged)} (judge rerun noise).",
        f"- J-new in / J-old out: {flip_text(judge_flips['J-new_in_J-old_out'], judge_flip_counts['J-new_in_J-old_out'])}; "
        f"J-old in / J-new out: {flip_text(judge_flips['J-old_in_J-new_out'], judge_flip_counts['J-old_in_J-new_out'])}.",
        "",
        "Deciles are over the pooled 20d excess of every candidate of the valid reviews of both kinds.",
        "",
    ]
    for source, days in (("B", valid_full), ("A", valid_full), ("logged", valid_full)):
        report += [f"Bear level, {source} (full sample):", "", "| bear risk | n | 20d excess | in worst 10% | in best 10% | judge out |", "|---|---|---|---|---|---|", *level_table(source, days), ""]
    for source, days in (("B", valid_full), ("A", valid_full), ("logged", valid_full), ("J-old", valid_judge), ("J-new", valid_judge)):
        report += [f"Judge out rate by bull x bear, {source}:", "", *cross_table(source, days), ""]
    report += [
        "## v1's anti-goal rule on the full sample",
        "",
        "| review | symbol | revenue % | margin chg pts | days | since filed % | logged bear | B bear |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for day in valid_full:
        for stock in log["bull"][day]["context"]["stocks"]:
            sheet = v1.with_new_fields(stock["fact_sheet"], fields[day][stock["fact_sheet"]["symbol"]])
            if v1.qualifies_anti_goal(sheet):
                symbol = sheet["symbol"]
                shown = [sheet.get(key) for key in ("revenue_growth_pct", "operating_margin_change_pts", "days_since_filed", "return_since_filed_pct")]
                ratings = [sources["logged"][day]["bear"].get(symbol), sources["B"][day]["bear"].get(symbol)]
                report.append("| " + " | ".join(str(cell) for cell in [day, symbol, *shown, *ratings]) + " |")
    report += ["", "## Judge prompt size and timings", ""]
    for arm in (*FULL_ARMS, *JUDGE_ARMS):
        arm_lines = [line for (_, name), line in by_line.items() if name == arm and line.get("judge_max_input_tokens")]
        if arm_lines:
            top = max(arm_lines, key=lambda line: line["judge_max_input_tokens"])
            report.append(
                f"- Arm {arm}: largest judge prompt {top['judge_max_input_tokens']} tokens ({top['stocks']} stocks, {top['date']}; "
                f"its first call {top['judge_prompt_tokens']}), as reported by the server."
            )
        seconds = [line["seconds"] for (_, name), line in by_line.items() if name == arm]
        if seconds:
            report.append(f"- Arm {arm}: {len(seconds)} reviews, median {statistics.median(seconds) / 60:.1f} min, total {sum(seconds) / 3600:.2f} h.")
    out = Path(args.out) if args.out else REPORT_V2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nwrote {out}")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="pin the prompts, fix both samples, write the plan file (no agent call)")
    run_parser = commands.add_parser("run", help="arms A and B on the full sample (resumable)")
    run_parser.add_argument("--arms", default="A,B")
    run_parser.add_argument("--reviews", default="", help="comma-separated full-sample review dates (default: all)")
    judge_parser = commands.add_parser("judge", help="arms J-old and J-new on the judge sample (resumable)")
    judge_parser.add_argument("--arms", default="J-old,J-new")
    judge_parser.add_argument("--reviews", default="", help="comma-separated judge-sample review dates (default: all)")
    score_parser = commands.add_parser("score", help="the six keep/drop tests and the report")
    score_parser.add_argument("--out", default="", help=f"report path (default {REPORT_V2.relative_to(v1.REPO)})")
    args = parser.parse_args(argv)
    {"prepare": prepare, "run": run, "judge": judge, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
