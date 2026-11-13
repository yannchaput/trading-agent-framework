"""Throwaway judge-only replay: does telling the judge which grounds for an out are strong and which are weak help?

Not a repo command; kept for the record. Background: over the 5Y run 2026-11-05_012455 the judge's outs for dilution, a short
runway or a pending binary event were losers at 20 sessions (-2.2% to -3.6% against the review's mean), while its outs for
debt, a revenue decline or the bear's "high" label were not (+0.9%, +1.8%, +6.1%). Those themes were read off the 5Y results,
so the test runs on the COVID run 2026-11-09_224910 (73 reviews, 2020-01 to 2021-05), which that analysis never saw.

Three arms replay the judge alone on the logged fact sheets, notes, bull and bear cases of every review:
`old-1` and `old-2` (today's `JUDGE_SYSTEM`, twice: the second is the rerun-noise arm) and `new` (`new_judge_system`).

The keep/drop rule, fixed on 2026-11-12 before any replay result existed. KEEP only if all three hold, else DROP; a gate
that cannot be measured fails:
1. Weak grounds: among the stocks `old-1` put out for debt, a revenue decline or the bear's label (and for no strong
   ground), `new` puts out at most two thirds of the share `old-2` puts out. At least `MIN_GROUP` stocks.
2. Strong grounds: among the stocks `old-1` put out for runway, dilution or a pending event, the share `new` puts out is
   within `STRONG_TOLERANCE` points of `old-2`'s or above. At least `MIN_GROUP` stocks.
3. No harm: the mean over reviews of (new book - old-1 book) at 20 sessions is not below minus the absolute mean of
   (old-2 book - old-1 book). A book's return is `stock_weight` times the sum of its stocks' forward returns.

Reuses `bull_bear_bear_balance.py` (loaded by file path) for the log parsing and the agent runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import types
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

RUN_DIR = v1.MAIN / "logs/bull_bear/backtesting/2026-11-09_224910_backtesting"
REVIEWS_EXPECTED = 73
vars(v1)["REVIEWS_EXPECTED"] = REVIEWS_EXPECTED  # v1 checks the count of its own run (log and reviews.jsonl)
ARMS = ("old-1", "old-2", "new")
REPORT = v1.REPO / "docs/superpowers/replays/2026-11-12-bull-bear-judge-veto-grounds.md"
HORIZON = 20  # sessions, the gated horizon
MIN_GROUP = 30  # stocks a gate's group needs
WEAK_RATIO = 2 / 3  # gate 1: new's out rate over old-2's, at most
STRONG_TOLERANCE = 0.10  # gate 2: new's out rate may be this far below old-2's
MAX_FAILED = 3  # reviews an arm may fail before the replay is void

WEAK = re.compile(r"debt|leverage|revenue (fell|down|declin|drop|contract)|revenue -\d|bear.{0,20}high", re.IGNORECASE)
STRONG = re.compile(r"runway|burn|dilut|offering|raise|going.concern|catalyst|binary|fda|trial", re.IGNORECASE)

_ANCHOR = "the note) that outweighs the bull's case. "
_ADDITION = (
    "Weigh what the risk means for the company's funding. A short runway_months, a share issuance, going-concern "
    "doubts or a decisive event still pending that the company reported (a trial readout, a ruling, a vote) are strong "
    "grounds for out. A rise in debt or a revenue decline, taken alone, is weak grounds: it puts a stock out only when "
    "the bull's case is thin as well. The bear's rating is the bear's opinion: judge the facts behind it, not the rating. "
)


# --- pure helpers (no I/O) --------------------------------------------------------------------------


def new_judge_system(old: str) -> str:
    """The candidate judge prompt: `old` with the grounds paragraph after its out rule."""
    if old.count(_ANCHOR) != 1:
        raise ValueError("the judge prompt no longer has the out rule this replay extends")
    return old.replace(_ANCHOR, _ANCHOR + _ADDITION)


def ground(reason: str) -> str | None:
    """`strong`, `weak` (and no strong ground) or None: what an out's reason rests on."""
    if STRONG.search(reason):
        return "strong"
    return "weak" if WEAK.search(reason) else None


def book_return(verdicts: Mapping[str, str], returns: Mapping[str, float], weight: float) -> float:
    """`weight` times the forward returns of the stocks put in; a stock with no return counts for nothing."""
    return weight * sum(returns[symbol] for symbol, verdict in verdicts.items() if verdict == "in" and symbol in returns)


def out_rate(symbols: Sequence[tuple[str, str]], arm: Mapping[str, Mapping[str, str]]) -> float | None:
    """The share of (day, symbol) pairs an arm put out; None for an empty group."""
    return statistics.fmean(arm[day][symbol] == "out" for day, symbol in symbols) if symbols else None


def decide(
    weak: Sequence[tuple[str, str]], strong: Sequence[tuple[str, str]], arms: Mapping[str, Mapping[str, Mapping[str, str]]], spread_new: float | None, spread_rerun: float | None
) -> dict[str, Any]:
    """The three gates and the verdict; `arms[name][day][symbol]` is that arm's verdict."""
    weak_new, weak_old = out_rate(weak, arms["new"]), out_rate(weak, arms["old-2"])
    strong_new, strong_old = out_rate(strong, arms["new"]), out_rate(strong, arms["old-2"])
    gate1 = len(weak) >= MIN_GROUP and weak_new is not None and weak_old is not None and weak_new <= WEAK_RATIO * weak_old
    gate2 = len(strong) >= MIN_GROUP and strong_new is not None and strong_old is not None and strong_new >= strong_old - STRONG_TOLERANCE
    gate3 = spread_new is not None and spread_rerun is not None and spread_new >= -abs(spread_rerun)
    return {
        "weak": {"n": len(weak), "new": weak_new, "old-2": weak_old, "pass": gate1},
        "strong": {"n": len(strong), "new": strong_new, "old-2": strong_old, "pass": gate2},
        "no_harm": {"new_minus_old": spread_new, "rerun_minus_old": spread_rerun, "pass": gate3},
        "verdict": "KEEP" if gate1 and gate2 and gate3 else "DROP",
    }


# --- run --------------------------------------------------------------------------------------------


def _results_file(arm: str) -> Path:
    return v1.DATA / f"judge_vetoes_{arm}.jsonl"


def _lines(arm: str) -> list[dict[str, Any]]:
    path = _results_file(arm)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def _parse_log() -> dict[str, dict[str, dict[str, Any]]]:
    return v1.parse_log(RUN_DIR / "backtest.log")


def _prompts(arm: str) -> types.SimpleNamespace:
    from trading_agent_framework.strategies.bull_bear import prompts

    system = new_judge_system(prompts.JUDGE_SYSTEM) if arm == "new" else prompts.JUDGE_SYSTEM
    return types.SimpleNamespace(BULL_SYSTEM=prompts.BULL_SYSTEM, BEAR_SYSTEM=prompts.BEAR_SYSTEM, JUDGE_SYSTEM=system, judge_task=prompts.judge_task, retry_prompt=prompts.retry_prompt)


def replay_judge(arm: str, day: str, context: dict[str, Any]) -> dict[str, Any]:
    """The judge alone on one review's logged context; one result line."""
    symbols = [stock["fact_sheet"]["symbol"] for stock in context["stocks"]]
    prompts = _prompts(arm)
    replay = v1._Replay(arm, day, prompts)
    params = replay.params
    replay.create(("judge",))
    previous = {stock["fact_sheet"]["symbol"]: stock["previous_verdict"]["verdict"] for stock in context["stocks"] if "previous_verdict" in stock}
    replay.recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous=previous)
    stage = replay.stage("judge", "submit_verdicts", prompts.judge_task(params.reason_max_chars), context)
    line: dict[str, Any] = {"date": day, "arm": arm, "ok": stage["ok"], "stocks": len(symbols), "stage": stage, "at": datetime.now(UTC).isoformat(timespec="seconds")}
    if stage["ok"]:
        line["verdicts"] = [asdict(verdict) for verdict in replay.recorder.submission.verdicts]
    return line


def run(args: argparse.Namespace) -> None:
    """One arm over every review, or `--reviews` (resumable: a review already in the arm's file is skipped)."""
    if args.arm not in ARMS:
        raise SystemExit(f"--arm takes one of {', '.join(ARMS)}")
    v1._load_env()
    from trading_agent_framework.agents.config import LLMCredentials

    base_url = LLMCredentials.from_env().base_url
    log = _parse_log()["judge"]
    days = [day.strip() for day in args.reviews.split(",")] if args.reviews else sorted(log)
    done = {line["date"] for line in _lines(args.arm)}
    v1.DATA.mkdir(parents=True, exist_ok=True)
    for day in days:
        if day in done:
            continue
        if not v1._server_up(base_url):
            raise SystemExit(f"the LLM server at {base_url} does not answer: stopped before {day}")
        try:
            line = replay_judge(args.arm, day, log[day]["context"])
        except v1.ServerDown as exc:
            raise SystemExit(f"the LLM server went down during {day}: nothing recorded ({exc})") from exc
        with _results_file(args.arm).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        info = line["stage"]
        print(f"{datetime.now().strftime('%H:%M:%S')} {args.arm} {day} ok={line['ok']} {info['seconds']}s x{info['attempts']} input {info['input_tokens']}", flush=True)


# --- score ------------------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def score(_args: argparse.Namespace) -> None:
    """The three gates and the non-gating tables, printed and written as the report."""
    from trading_agent_framework.strategies.bull_bear.batch.agent_scores import forward_returns, newey_west_t
    from trading_agent_framework.strategies.bull_bear.batch.scores_cli import fetch_opens
    from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

    weight = BullBearParams().stock_weight
    reviews = v1.read_reviews(RUN_DIR / "reviews.jsonl")
    logged = {review["date"]: {v["symbol"]: v["verdict"] for v in review.get("verdicts", [])} for review in reviews}
    lines = {arm: {line["date"]: line for line in _lines(arm)} for arm in ARMS}
    failed = {arm: sorted(day for day, line in lines[arm].items() if not line["ok"]) for arm in ARMS}
    days = sorted(day for day in logged if all(lines[arm].get(day, {}).get("ok") for arm in ARMS))
    arms = {arm: {day: {v["symbol"]: v["verdict"] for v in lines[arm][day]["verdicts"]} for day in days} for arm in ARMS}
    reasons = {day: {v["symbol"]: v["reason"] for v in lines["old-1"][day]["verdicts"]} for day in days}

    opens, calendar = fetch_opens([reviews], today=date.today())
    rows = forward_returns(reviews, opens, calendar, horizons=(5, HORIZON)).rows
    raw: dict[int, dict[str, dict[str, float]]] = {5: {}, HORIZON: {}}
    for row in rows:
        raw[row.horizon].setdefault(row.review.isoformat(), {})[row.symbol] = row.raw

    def spreads(arm: str, horizon: int) -> list[float]:
        return [book_return(arms[arm][day], raw[horizon][day], weight) - book_return(arms["old-1"][day], raw[horizon][day], weight) for day in days if day in raw[horizon]]

    def mean(values: Sequence[float]) -> float | None:
        return statistics.fmean(values) if values else None

    outs = [(day, symbol) for day in days for symbol, verdict in arms["old-1"][day].items() if verdict == "out"]
    weak = [pair for pair in outs if ground(reasons[pair[0]][pair[1]]) == "weak"]
    strong = [pair for pair in outs if ground(reasons[pair[0]][pair[1]]) == "strong"]
    result = decide(weak, strong, arms, mean(spreads("new", HORIZON)), mean(spreads("old-2", HORIZON)))

    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    def gate(name: str, group: str, new: str, rerun: str, passed: bool) -> str:
        return f"| {name} | {group} | {new} | {rerun} | {passed} |"

    def book_size(arm: str) -> str:
        return f"{statistics.fmean(sum(v == 'in' for v in arms[arm][day].values()) for day in days):.1f}"

    weak_gate, strong_gate, harm = result["weak"], result["strong"], result["no_harm"]

    def flips(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]]) -> str:
        pairs = [(a[day][symbol], b[day][symbol]) for day in days for symbol in a[day] if symbol in b[day]]
        return f"{sum(x != y for x, y in pairs)} of {len(pairs)} ({statistics.fmean(x != y for x, y in pairs):.1%})" if pairs else "n/a"

    def moved(now: str, before: str) -> str:  # stocks `new` has `now` and old-1 has `before`: their 20-session return against the review's mean
        values = []
        for day in days:
            day_raw = raw[HORIZON].get(day, {})
            if day_raw:
                base = statistics.fmean(day_raw.values())
                values += [day_raw[s] - base for s, v in arms["new"][day].items() if v == now and arms["old-1"][day].get(s) == before and s in day_raw]
        return f"n={len(values)}, {_pct(mean(values))}"

    out = [
        "# bull_bear judge: strong and weak grounds for an out (replay on the COVID run)",
        "",
        f"Run replayed: `{RUN_DIR.relative_to(v1.MAIN)}`, judge only, arms {', '.join(ARMS)}. Script: `scripts/replays/bull_bear_judge_vetoes.py` (the rule is in its docstring).",
        f"Reviews scored (all three arms recorded): {len(days)} of {len(logged)}. Failed: " + "; ".join(f"{arm} {len(days_)}" for arm, days_ in failed.items()) + ".",
        "",
        f"## Verdict: {result['verdict']}",
        "",
        "| Gate | Group | new | old-2 | Pass |",
        "|---|---|---|---|---|",
        gate("1. Weak grounds: out rate at most 2/3 of the rerun's", f"{weak_gate['n']} stocks", rate(weak_gate["new"]), rate(weak_gate["old-2"]), weak_gate["pass"]),
        gate("2. Strong grounds: out rate within 10 points of the rerun's", f"{strong_gate['n']} stocks", rate(strong_gate["new"]), rate(strong_gate["old-2"]), strong_gate["pass"]),
        gate("3. No harm: book against old-1, 20 sessions, mean per review", f"{len(spreads('new', HORIZON))} reviews", _pct(harm["new_minus_old"]), _pct(harm["rerun_minus_old"]), harm["pass"]),
        "",
        "## Not gated",
        "",
        "| | new | old-2 (rerun) |",
        "|---|---|---|",
    ]
    for horizon in (5, HORIZON):
        cells = []
        for arm in ("new", "old-2"):
            values = spreads(arm, horizon)
            t = newey_west_t(values, lags=3 if horizon == HORIZON else 0) if len(values) > 3 else None
            cells.append(f"{_pct(mean(values))} (t {t:.2f})" if t is not None else _pct(mean(values)))
        out.append(f"| Book minus old-1 book, {horizon} sessions | {cells[0]} | {cells[1]} |")
    out += (
        [
            f"| Verdicts that differ from old-1 | {flips(arms['new'], arms['old-1'])} | {flips(arms['old-2'], arms['old-1'])} |",
            f"| Stocks in per review | {book_size('new')} | {book_size('old-2')} |",
            "",
            f"- old-1 against the logged run (other quantisation, memory tools): {flips(arms['old-1'], {day: logged[day] for day in days})} verdicts differ.",
            f"- Stocks new puts in and old-1 out (20 sessions, against the review's mean): {moved('in', 'out')}.",
            f"- Stocks new puts out and old-1 in: {moved('out', 'in')}.",
            "- Seconds per review: " + ", ".join(f"{arm} {statistics.fmean(lines[arm][d]['stage']['seconds'] for d in days):.0f}" for arm in ARMS) + ".",
            "",
        ]
        if days
        else ["no review scored"]
    )
    text = "\n".join(out)
    print(text)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    """Parse the command line and run `run` or `score`."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="replay one arm (resumable)")
    run_parser.add_argument("--arm", required=True)
    run_parser.add_argument("--reviews", default="", help="comma-separated review dates (default: all)")
    run_parser.set_defaults(func=run)
    commands.add_parser("score", help="the gates and the report").set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
