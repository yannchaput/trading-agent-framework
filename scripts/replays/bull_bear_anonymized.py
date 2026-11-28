"""Throwaway spike: does bull_bear's debate lean on the model's memory of how a stock turned out?

Not a repo command; kept for the record. TODO "Bull Bear" item 2, from the 2026-11-24 external assessment. Item 1 (split
at Qwen 3.8's knowledge edge, ~2025-04) found no memory signature, except a doubt on 2024-09..2025-04: the most recent
training months, and the strongest stretch of every run. This replay tests that stretch directly.

Source run: the news-off 2Y run 2026-11-27_085328 (`research_news` off: its notes are SEC facts only, so hiding the
ticker and the company name really hides the identity; a news note names CEOs, products and analysts). Sample: its 35
reviews from 2024-09-01 to 2025-04-30. Bull, bear and judge rerun on the logged fact sheets, notes and previous verdicts,
with today's prompts and settings, in three arms:

- `named-1`, `named-2`: as logged (the second is the rerun-noise arm);
- `anon`: each stock is "STOCK NN" (NN = its rank in the debate set), the fact sheet loses `company`, and the ticker, the
  company name and its first word are replaced by the label in every note and previous reason. Numbers are left exact
  (shortcut: a model could still recognise a famous company from exact revenue; round them if this replay flags a leak).

The rule, fixed on 2026-11-27 before any replay result existed. Per review, A = mean forward return of the judge's `in`
stocks minus that of the debate set's top 10 by rank, 20 sessions from the open after the review. `named` = the mean of
`named-1` and `named-2`; d = named - anon. Only reviews all three arms completed count; more than 3 failed reviews in an
arm voids the replay. LEAK if all four hold, else NO LEAK DETECTED:
1. mean(named) > 0;
2. mean(d) >= 0.3 x mean(named) (anon keeps under 70% of the judge's edge over the rank, as the assessment's gate);
3. Newey-West t of d >= 2 (3 lags: weekly reviews, 20-session returns);
4. mean(d) > |mean(named-1 - named-2)| (beyond rerun noise).
Not gating: the same at 5 sessions, the bull's high - low and the judge's in - out spreads, verdict agreement.

Run from the repository root: `uv run python scripts/replays/bull_bear_anonymized.py run [--arms ...]` (resumable: a
(review, arm) already in the results file is skipped), then `... score`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_V1_SPEC = importlib.util.spec_from_file_location("bull_bear_bear_balance", Path(__file__).with_name("bull_bear_bear_balance.py"))
assert _V1_SPEC is not None and _V1_SPEC.loader is not None
v1 = importlib.util.module_from_spec(_V1_SPEC)
_V1_SPEC.loader.exec_module(v1)  # reused: parse_log, _server_up, _load_env (env/.env.bull_bear.backtesting)

RUN_DIR = v1.MAIN / "logs/bull_bear/backtesting/2026-11-27_085328_backtesting"
vars(v1)["REVIEWS_EXPECTED"] = 115  # parse_log checks the count of this run's reviews
FIRST, LAST = date(2024, 9, 1), date(2025, 4, 30)
ARMS = ("named-1", "named-2", "anon")
RESULTS = v1.MAIN / "docs/superpowers/replays/data/anonymized_results.jsonl"  # docs/ is local only (.git/info/exclude): kept in the main checkout
REPORT = v1.MAIN / "docs/superpowers/replays/2026-11-27-bull-bear-anonymized.md"
MAX_FAILED = 3
# A company name's first word is scrubbed too ("Carvana" for "CARVANA CO."), unless it is a common word.
SUFFIXES = ("inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings", "holding", "group", "sa", "nv", "n.v", "ag", "lp")
COMMON = {"the", "american", "first", "united", "general", "national", "international", "global", "new", "great", "south", "north", "west", "east", "royal", "energy", "applied"}


# --- anonymization (pure) -----------------------------------------------------------------------


def labels(sheets: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """{symbol: "STOCK NN"}, NN the stock's place in the debate set (rank order); upper case, as `common/handoff.key` reads a symbol."""
    return {sheet["symbol"]: f"STOCK {index:02d}" for index, sheet in enumerate(sheets, start=1)}


def scrubber(sheets: Sequence[Mapping[str, Any]], label: Mapping[str, str]) -> Any:
    """A function replacing every stock's ticker, company name and the name's first word with its label, in any text.

    The rest of the name and a legal suffix right after a label go too ("Carvana Co." -> "STOCK 03", not "STOCK 03 Co.").
    """
    pairs: list[tuple[str, str]] = []
    tails: list[re.Pattern[str]] = []
    for sheet in sheets:
        name = str(sheet.get("company") or "").strip()
        pairs.append((rf"\b{re.escape(sheet['symbol'])}\b", label[sheet["symbol"]]))
        if name:
            pairs.append((rf"\b{re.escape(name)}", label[sheet["symbol"]]))
            first = re.split(r"[\s,.]+", name)[0]
            if len(first) >= 4 and first.lower() not in COMMON:
                pairs.append((rf"\b{re.escape(first)}\b", label[sheet["symbol"]]))
            words = sorted({word for word in re.split(r"[\s,]+", name.lower().rstrip(".")) if word} | set(SUFFIXES), key=len, reverse=True)
            alternatives = "|".join(re.escape(word) + r"\.?" for word in words)
            tails.append(re.compile(rf"({re.escape(label[sheet['symbol']])})(?:,?\s+(?:{alternatives}))+(?![\w-])", re.IGNORECASE))
    pairs.sort(key=lambda pair: -len(pair[0]))  # the full name before its first word
    compiled = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in pairs]

    def scrub(text: str | None) -> str | None:
        if text is None:
            return None
        for pattern, repl in compiled:
            text = pattern.sub(repl, text)
        for tail in tails:
            text = tail.sub(r"\1", text)
        return text

    return scrub


def anonymize(stocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A logged judge context's stocks (fact sheet, note, previous verdict) with every identity replaced by its label."""
    sheets = [stock["fact_sheet"] for stock in stocks]
    label = labels(sheets)
    scrub = scrubber(sheets, label)
    out = []
    for stock in stocks:
        sheet = {key: value for key, value in stock["fact_sheet"].items() if key != "company"}
        sheet["symbol"] = label[sheet["symbol"]]
        entry: dict[str, Any] = {"fact_sheet": sheet, "note": scrub(stock["note"])}
        if "previous_verdict" in stock:
            previous = dict(stock["previous_verdict"])
            previous["reason"] = scrub(previous.get("reason"))
            if previous.get("what_changed"):
                previous["what_changed"] = scrub(previous["what_changed"])
            entry["previous_verdict"] = previous
        out.append(entry)
    return out


# --- run --------------------------------------------------------------------------------------------


class _Debate:
    """Bull, bear and judge built as `BullBearStrategy.initialize` builds them, outside a strategy."""

    def __init__(self) -> None:
        from trading_agent_framework.agents import ToolBudget
        from trading_agent_framework.agents.config import LLMCredentials
        from trading_agent_framework.agents.manager import AgentManager
        from trading_agent_framework.agents.tools.text import text_tools
        from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
        from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
        from trading_agent_framework.strategies.bull_bear.prompts import BEAR_SYSTEM, BULL_SYSTEM, JUDGE_SYSTEM

        params = self.params = BullBearParams()
        self.base_url = LLMCredentials.from_env().base_url
        self.manager = AgentManager(LLMCredentials.from_env)
        self.recorder = recorder = HandoffRecorder(params)
        submit = submit_tools(recorder, params)
        for name, system, tool in (("bull", BULL_SYSTEM, "submit_bull_case"), ("bear", BEAR_SYSTEM, "submit_bear_case")):
            self.manager.create(
                name=name,
                system_prompt=system,
                tools=[*text_tools(lambda: recorder.gate), submit[tool]],
                temperature=params.agent_temperature,
                budget=ToolBudget(exempt=(tool, "count_characters")),
                final_tools=[tool],
                max_output_tokens=params.max_output_tokens,
                thinking_token_budget=params.thinking_token_budget,
            )
        self.manager.create(
            name="judge",
            system_prompt=JUDGE_SYSTEM,
            tools=[*text_tools(lambda: recorder.gate), submit["submit_verdicts"]],
            temperature=params.agent_temperature,
            budget=ToolBudget(exempt=("submit_verdicts", "count_characters")),
            final_tools=["submit_verdicts"],
            max_output_tokens=params.judge_max_output_tokens,
            thinking_token_budget=params.judge_thinking_token_budget,
        )

    def stage(self, name: str, tool: str, task: str, context: dict[str, Any]) -> bool:
        """`BullBearReview._run_stage`: one run, then one fresh run with the submit tool forced. True once a submission is recorded."""
        from trading_agent_framework.strategies.bull_bear.prompts import retry_prompt
        from trading_agent_framework.utils.errors import AgentError, AgentLoopError

        run_id, prompt, force = uuid.uuid4().hex, task, None
        for _ in range(2):
            self.recorder.last_error, error = None, ""
            try:
                self.manager[name].run(prompt, context=context, run_id=run_id, force_tool=force, max_steps=self.params.max_agent_steps)
            except AgentLoopError:
                return False
            except AgentError as exc:
                if not v1._server_up(self.base_url):
                    raise v1.ServerDown(str(exc)) from exc
                error = str(exc)
            if self.recorder.submitted:
                return True
            error = self.recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force = retry_prompt(name, tool, error, task), tool
        return False


def replay(arm: str, day: str, judge_ctx: Mapping[str, Any]) -> dict[str, Any]:
    """One (review, arm): bull, bear, then judge on them. Submissions are stored under the real symbols."""
    from trading_agent_framework.strategies.bull_bear.prompts import bear_task, bull_task, judge_task

    logged = judge_ctx["stocks"]
    stocks = anonymize(logged) if arm == "anon" else [{key: stock[key] for key in ("fact_sheet", "note", "previous_verdict") if key in stock} for stock in logged]
    real = {stock["fact_sheet"]["symbol"]: original["fact_sheet"]["symbol"] for stock, original in zip(stocks, logged, strict=True)}
    shown = list(real)
    debate = _Debate()
    params, recorder = debate.params, debate.recorder
    evidence = [{"fact_sheet": stock["fact_sheet"], "note": stock["note"]} for stock in stocks]
    base = {"current_datetime": judge_ctx["current_datetime"], "stocks": evidence, "constraints": {"argument_max_chars": params.argument_max_chars}}
    line: dict[str, Any] = {"date": day, "arm": arm, "ok": False, "failed_stage": None, "labels": real if arm == "anon" else None}
    started = time.monotonic()
    try:
        recorder.expect_bull(shown)
        if not debate.stage("bull", "submit_bull_case", bull_task(params.argument_max_chars), base):
            line["failed_stage"] = "bull"
            return line
        bull = {case.symbol: case for case in recorder.submission}
        recorder.expect_bear(shown)
        if not debate.stage("bear", "submit_bear_case", bear_task(params.argument_max_chars), base):
            line["failed_stage"] = "bear"
            return line
        bear = {case.symbol: case for case in recorder.submission}
        previous = {stock["fact_sheet"]["symbol"]: stock["previous_verdict"] for stock in stocks if "previous_verdict" in stock}
        context = {
            "current_datetime": judge_ctx["current_datetime"],
            "stocks": [
                {
                    **evidence[index],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                    **({"previous_verdict": previous[symbol]} if symbol in previous else {}),
                }
                for index, symbol in enumerate(shown)
            ],
            "constraints": {"max_in": params.max_book_size, "reason_max_chars": params.reason_max_chars},
        }
        recorder.expect_verdicts(shown, max_in=params.max_book_size, previous={symbol: entry["verdict"] for symbol, entry in previous.items()})
        if not debate.stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context):
            line["failed_stage"] = "judge"
            return line
        line["bull"] = [{**asdict(case), "symbol": real[case.symbol]} for case in bull.values()]
        line["bear"] = [{**asdict(case), "symbol": real[case.symbol]} for case in bear.values()]
        line["verdicts"] = [{**asdict(verdict), "symbol": real[verdict.symbol]} for verdict in recorder.submission.verdicts]
        line["ok"] = True
        return line
    finally:
        line["seconds"] = round(time.monotonic() - started, 1)


def _results() -> list[dict[str, Any]]:
    if not RESULTS.is_file():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample() -> list[dict[str, Any]]:
    records = [json.loads(line) for line in (RUN_DIR / "reviews.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [record for record in records if not record.get("abandoned") and record.get("debate_set") and FIRST <= date.fromisoformat(record["date"]) <= LAST]


def run(args: argparse.Namespace) -> None:
    """Every (review, arm) of the sample not yet in the results file, review by review."""
    v1._load_env()
    from trading_agent_framework.agents.config import LLMCredentials

    base_url = LLMCredentials.from_env().base_url
    arms = [arm.strip() for arm in args.arms.split(",")]
    if not set(arms) <= set(ARMS):
        raise SystemExit(f"--arms takes {', '.join(ARMS)}")
    judge_log = v1.parse_log(RUN_DIR / "backtest.log")["judge"]
    done = {(line["date"], line["arm"]) for line in _results()}
    sample = _sample()
    print(f"{len(sample)} reviews {sample[0]['date']}..{sample[-1]['date']}, arms {arms}")
    for record in sample:
        for arm in arms:
            key = (record["date"], arm)
            if key in done:
                continue
            if not v1._server_up(base_url):
                raise SystemExit(f"the LLM server at {base_url} does not answer: stopped before {key}")
            print(f"{datetime.now().strftime('%H:%M:%S')} {key} ...", flush=True)
            try:
                line = replay(arm, record["date"], judge_log[record["date"]]["context"])
            except v1.ServerDown as exc:
                raise SystemExit(f"the LLM server went down during {key}: nothing recorded ({exc})") from exc
            line["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            ins = [verdict["symbol"] for verdict in line.get("verdicts", []) if verdict["verdict"] == "in"]
            print(f"  ok={line['ok']} failed={line['failed_stage']} {line['seconds']}s in={ins}", flush=True)


# --- score ------------------------------------------------------------------------------------------


def _nw_t(values: Sequence[float], lags: int) -> float | None:
    from trading_agent_framework.strategies.bull_bear.batch.agent_scores import newey_west_t

    return newey_west_t(values, lags=lags) if len(values) > 2 else None


def _measures(line: Mapping[str, Any], record: Mapping[str, Any], rows: Mapping[str, Any]) -> dict[str, float | None]:
    """A (judge in minus top 10 by rank, raw), B (judge in - out, excess), C (bull high - low, excess) for one arm's review."""

    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    verdict = {entry["symbol"]: entry["verdict"] for entry in line["verdicts"]}
    conviction = {entry["symbol"]: entry["conviction"] for entry in line["bull"]}
    top10 = [entry["symbol"] for entry in sorted(record["debate_set"], key=lambda entry: entry["rank"])[:10] if entry["symbol"] in rows]
    ins = mean([rows[s].raw for s, v in verdict.items() if v == "in" and s in rows])
    top = mean([rows[s].raw for s in top10])
    spread = {}
    for name, calls, good, bad in (("B", verdict, "in", "out"), ("C", conviction, "high", "low")):
        g = mean([rows[s].excess for s, c in calls.items() if c == good and s in rows])
        b = mean([rows[s].excess for s, c in calls.items() if c == bad and s in rows])
        spread[name] = None if g is None or b is None else g - b
    return {"A": None if ins is None or top is None else ins - top, **spread}


def score(_args: argparse.Namespace) -> None:
    """The keep/drop rule and the non-gating tables, printed and written to the report."""
    from trading_agent_framework.strategies.bull_bear.batch.agent_scores import forward_returns
    from trading_agent_framework.strategies.bull_bear.batch.scores_cli import fetch_opens

    sample = {record["date"]: record for record in _sample()}
    lines = {(line["date"], line["arm"]): line for line in _results()}
    failed = {arm: sum(1 for day in sample if (day, arm) in lines and not lines[day, arm]["ok"]) for arm in ARMS}
    days = [day for day in sample if all(lines.get((day, arm), {}).get("ok") for arm in ARMS)]
    out = [
        f"# bull_bear anonymized replay ({date.today()})",
        "",
        f"Source `{RUN_DIR.name}`, {FIRST}..{LAST}: {len(sample)} reviews, {len(days)} done by all arms; failed per arm {failed}.",
        "",
    ]
    void, gate = max(failed.values()) > MAX_FAILED, "No completed review: nothing to score."
    opens, calendar = fetch_opens([list(sample.values())], today=date.today())
    for horizon in (20, 5):
        rets = forward_returns([sample[day] for day in days], opens, calendar, horizons=(horizon,))
        rows: dict[str, dict[str, Any]] = {}
        for row in rets.rows:
            rows.setdefault(row.review.isoformat(), {})[row.symbol] = row
        per = {arm: [_measures(lines[day, arm], sample[day], rows.get(day, {})) for day in days] for arm in ARMS}
        out += [f"## Horizon {horizon} sessions", "", "| measure | named-1 | named-2 | anon | named - anon (t) |", "|---|---|---|---|---|"]
        for key, label in (("A", "A judge in - top 10"), ("B", "B judge in - out"), ("C", "C bull high - low")):
            triples = [(a, b, c) for i in range(len(days)) for a, b, c in [(per["named-1"][i][key], per["named-2"][i][key], per["anon"][i][key])] if a is not None and b is not None and c is not None]
            if len(triples) < 3:
                out.append(f"| {label} | - | - | - | too few reviews |")
                continue
            n1, n2, an = ([t[j] for t in triples] for j in range(3))
            d = [(a + b) / 2 - c for a, b, c in triples]
            t = _nw_t(d, 3 if horizon == 20 else 0)
            out.append(f"| {label} | {statistics.mean(n1):+.2%} | {statistics.mean(n2):+.2%} | {statistics.mean(an):+.2%} | {statistics.mean(d):+.2%} ({t:+.2f}) n={len(d)} |")
            if key == "A" and horizon == 20:
                named = statistics.mean([(a + b) / 2 for a, b, _ in triples])
                noise = abs(statistics.mean(n1) - statistics.mean(n2))
                tests = [named > 0, statistics.mean(d) >= 0.3 * named, t is not None and t >= 2, statistics.mean(d) > noise]
                verdict = "VOID (too many failed reviews)" if void else ("LEAK" if all(tests) else "NO LEAK DETECTED")
                gate = f"Gate (A, 20 sessions): named {named:+.2%}, d {statistics.mean(d):+.2%}, t {t}, rerun noise {noise:.2%}; tests {tests} -> **{verdict}**"
        out.append("")
    agree = {}
    for arm in ("named-2", "anon"):
        same = total = 0
        for day in days:
            base = {v["symbol"]: v["verdict"] for v in lines[day, "named-1"]["verdicts"]}
            for v in lines[day, arm]["verdicts"]:
                total += 1
                same += base.get(v["symbol"]) == v["verdict"]
        agree[arm] = f"{same / total:.0%} of {total}" if total else "-"
    out += [f"Verdict agreement with named-1: named-2 {agree['named-2']}, anon {agree['anon']}.", "", gate]
    text = "\n".join(out) + "\n"
    print(text)
    REPORT.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description="bull_bear anonymized replay (spike)")
    sub = parser.add_subparsers(required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--arms", default=",".join(ARMS))
    p_run.set_defaults(func=run)
    sub.add_parser("score").set_defaults(func=score)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
