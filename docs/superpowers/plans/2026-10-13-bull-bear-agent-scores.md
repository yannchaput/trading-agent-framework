# bull_bear agent forward-return scores Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI, `uv run bull-bear-scores <run_dir>...`, that scores the bull's conviction, the bear's risk and the judge's in/out calls of bull_bear backtest runs against forward returns, runs side by side.

**Architecture:** A pure module (`strategies/bull_bear/batch/agent_scores.py`) turns review records plus daily opens into forward returns and per-agent scores; a thin CLI (`strategies/bull_bear/batch/scores_cli.py`) reads `reviews.jsonl`, fetches opens through the existing `YahooDailyBars`, and prints the report. Nothing in the strategy changes or imports the package.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff (line length 200), pyright (standard, `src` only), pandas only inside the CLI (via `Bars.df`), yfinance through `YahooDailyBars` (injectable `download`).

**Spec:** `docs/superpowers/specs/2026-10-13-bull-bear-agent-scores-design.md`

## Global Constraints

- Work in the worktree `/home/yann/projets/trading-agent-framework-agent-scores`, branch `feat/bull-bear-agent-scores`; never `cd` to the main checkout.
- The automated tests never touch the network: the CLI tests inject a fake `download`.
- Horizons: 5 and 20 sessions. Entry: the open of the first SPY session strictly after the review date. Exit: the open `h` sessions after the entry session.
- Excess return: raw minus the equal-weight mean raw return of that review's candidates that have a return at the same horizon.
- Spreads (positive = the agent was right): bull `high - low`, bear `low - high`, judge `in - out`. Computed per review, averaged across reviews.
- t-stat across reviews: plain at 5 sessions, Newey-West with 3 lags (Bartlett weights) at 20 sessions.
- A review counts when it is not `abandoned` and has a `debate_set`. A picker record (no `verdicts`, has `picks`) puts its `picks` in and the rest of the `debate_set` out.
- Exit codes: 2 for a run directory without `reviews.jsonl` (argparse error), 1 for `YahooDataError`, 0 otherwise.
- Download window: 7 calendar days before the first review to 40 calendar days after the last, capped at today.
- The strategy package never imports `dashboard/`.
- Test files have unique basenames (no `__init__.py` under `tests/`): `tests/strategies/bull_bear/test_bull_bear_agent_scores.py`, `tests/strategies/bull_bear/test_bull_bear_scores_cli.py`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- A run from another strategy (bill_ackman's `reviews.jsonl` has no `debate_set`): the report prints with 0 reviews scored and exits 0, no crash. Test in Task 4.
- An execution-stage abandoned review still carries `verdicts`, `bull` and `bear`: none of its calls count. Test in Task 3.
- A verdict on a symbol outside the review's debate set: ignored, never scored against another review's returns. Test in Task 3.
- Yahoo returns no SPY: exit 1 with a message naming SPY, never a `KeyError`. Test in Task 4.
- A run whose last reviews are within 20 sessions of the data's end (or of today): those exits are dropped and counted, and the download window stops at today. Tests in Task 2 (dropped) and Task 4 (window).

---

### Task 1: Pin bull_bear's backtest slippage at 0

`c061013` (on `main`) already set `BullBearStrategy.parameters["slippage"]` to `Decimal("0")`, the picker run's baseline. This task only pins it, so the test passes on first run: that is expected here (the production change predates the plan). To prove the test can fail, Step 2 flips the value temporarily.

**Files:**
- Test: `tests/strategies/bull_bear/test_bull_bear_strategy.py` (add after `test_run_backtesting_passes_the_class_window_the_daily_yahoo_source_and_the_preloads`, line 233)

**Interfaces:**
- Consumes: `BullBearStrategy.parameters["slippage"]`, `BullBearStrategy.run_backtesting(**overrides)` (passes `slippage=self.parameters["slippage"]`, `agent_bull_bear.py:198`).
- Produces: nothing used by later tasks.

- [ ] **Step 1: Write the test**

Add to `tests/strategies/bull_bear/test_bull_bear_strategy.py`, after line 233:

```python
def test_run_backtesting_charges_no_slippage_so_runs_compare_with_the_picker_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["slippage"] == BullBearStrategy.parameters["slippage"] == Decimal(0)  # the picker run 2026-10-10_085928 had none
```

- [ ] **Step 2: Run it, then prove it can fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_strategy.py -k no_slippage -v`
Expected: PASS.

Then temporarily change `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py:57` to `"slippage": Decimal("0.0005"),`, run the same command, expect FAIL (`assert Decimal('0.0005') == ...`), and restore `"slippage": Decimal("0"),`. Confirm with `git diff --stat` that only the test file changed.

- [ ] **Step 3: Commit**

```bash
git add tests/strategies/bull_bear/test_bull_bear_strategy.py
git commit -m "test(bull_bear): pin the backtest slippage at 0, the picker run's baseline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Forward returns of each review's candidates

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/batch/__init__.py`
- Create: `src/trading_agent_framework/strategies/bull_bear/batch/agent_scores.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_agent_scores.py`

**Interfaces:**
- Consumes: review records as parsed from `reviews.jsonl` (dicts with `date` ISO string, `abandoned` bool, `debate_set` list of `{"symbol", "rank", "held"}`).
- Produces (used by Tasks 3 and 4):
  - `HORIZONS: tuple[int, ...] = (5, 20)`
  - `@dataclass(frozen=True, slots=True) class ForwardReturn: review: date; symbol: str; horizon: int; raw: float; excess: float`
  - `@dataclass(frozen=True, slots=True) class Returns: rows: list[ForwardReturn]; dropped: int`
  - `def scored_reviews(reviews: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]`
  - `def candidates(review: Mapping[str, Any]) -> list[str]`
  - `def forward_returns(reviews: Sequence[Mapping[str, Any]], opens: Mapping[str, Mapping[date, float]], calendar: Sequence[date], horizons: Sequence[int] = HORIZONS) -> Returns`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/bull_bear/test_bull_bear_agent_scores.py`:

```python
"""bull_bear's agent scores: forward returns of each review's candidates, then each agent's calls against them."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.batch.agent_scores import ForwardReturn, forward_returns


def _weekdays(start: date, count: int) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


SESSIONS = _weekdays(date(2026, 1, 5), 40)  # Monday 2026-01-05 onwards
TUESDAY = date(2026, 1, 6)  # SESSIONS[1]; its orders fill at SESSIONS[2]'s open


def _review(day: date, symbols: list[str], **fields: Any) -> dict[str, Any]:
    return {"date": day.isoformat(), "abandoned": False, "debate_set": [{"symbol": s, "rank": i + 1, "held": False} for i, s in enumerate(symbols)], **fields}


# --- forward returns ---------------------------------------------------------------------------------------


def test_a_forward_return_runs_from_the_open_after_the_review_to_the_open_h_sessions_later() -> None:
    entry, exit_ = SESSIONS[2], SESSIONS[7]
    opens = {"AAA": {entry: 100.0, exit_: 110.0}, "BBB": {entry: 50.0, exit_: 50.0}}

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB"])], opens, SESSIONS, horizons=(5,))

    by_symbol = {row.symbol: row for row in result.rows}
    assert by_symbol["AAA"] == ForwardReturn(TUESDAY, "AAA", 5, pytest.approx(0.10), pytest.approx(0.05))  # excess: +10% against a +5% mean
    assert by_symbol["BBB"].raw == pytest.approx(0.0) and by_symbol["BBB"].excess == pytest.approx(-0.05)
    assert result.dropped == 0


def test_a_candidate_without_an_open_is_dropped_counted_and_left_out_of_the_mean() -> None:
    entry, exit_ = SESSIONS[2], SESSIONS[7]
    opens = {"AAA": {entry: 100.0, exit_: 110.0}, "BBB": {entry: 50.0, exit_: 50.0}, "CCC": {entry: 10.0}}  # CCC has no exit open

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB", "CCC", "DDD"])], opens, SESSIONS, horizons=(5,))  # DDD: no data at all

    assert sorted(row.symbol for row in result.rows) == ["AAA", "BBB"]
    assert {row.symbol: row.excess for row in result.rows} == {"AAA": pytest.approx(0.05), "BBB": pytest.approx(-0.05)}
    assert result.dropped == 2


def test_a_review_whose_exit_is_past_the_last_session_is_dropped_and_counted() -> None:
    late = SESSIONS[-3]  # entry SESSIONS[-2]; 5 sessions later is past the calendar
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}

    result = forward_returns([_review(late, ["AAA"])], opens, SESSIONS, horizons=(5, 20))

    assert result.rows == [] and result.dropped == 2  # one candidate x two horizons


def test_each_horizon_gets_its_own_rows_and_its_own_candidate_mean() -> None:
    entry, exit5, exit20 = SESSIONS[2], SESSIONS[7], SESSIONS[22]
    opens = {"AAA": {entry: 100.0, exit5: 102.0, exit20: 120.0}, "BBB": {entry: 100.0, exit5: 100.0, exit20: 100.0}}

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB"])], opens, SESSIONS)

    assert {(row.symbol, row.horizon): row.excess for row in result.rows} == {
        ("AAA", 5): pytest.approx(0.01),
        ("BBB", 5): pytest.approx(-0.01),
        ("AAA", 20): pytest.approx(0.10),
        ("BBB", 20): pytest.approx(-0.10),
    }


def test_abandoned_reviews_and_reviews_without_a_debate_set_give_no_returns() -> None:
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}
    reviews = [_review(TUESDAY, ["AAA"], abandoned=True), {"date": TUESDAY.isoformat(), "abandoned": False, "holdings": ["AAA"]}]

    result = forward_returns(reviews, opens, SESSIONS)

    assert result.rows == [] and result.dropped == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_agent_scores.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bull_bear.batch'`.

- [ ] **Step 3: Write the package and the forward returns**

Create `src/trading_agent_framework/strategies/bull_bear/batch/__init__.py`:

```python
"""bull_bear batch tools: run by hand over finished runs, never imported by the strategy."""
```

Create `src/trading_agent_framework/strategies/bull_bear/batch/agent_scores.py`:

```python
"""Score the bull, the bear and the judge of a bull_bear run against forward returns (pure: no I/O, no network).

A review's candidates are its `debate_set`, the stocks every agent was shown. Each candidate's forward return runs
from the open of the first session after the review date (the fill: a Tuesday 12:00 review's market orders fill at
the next bar's open, `backtesting/fills.py`) to the open `h` sessions later; its excess return subtracts the mean raw
return of that review's candidates at the same horizon, so it measures discrimination, not the market.
Spec: docs/superpowers/specs/2026-10-13-bull-bear-agent-scores-design.md.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

HORIZONS = (5, 20)  # sessions: about one and four weeks


@dataclass(frozen=True, slots=True)
class ForwardReturn:
    review: date
    symbol: str
    horizon: int  # sessions
    raw: float
    excess: float  # raw minus the review's candidate mean at this horizon


@dataclass(frozen=True, slots=True)
class Returns:
    rows: list[ForwardReturn]
    dropped: int  # candidate x horizon pairs with no entry or no exit open


def scored_reviews(reviews: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The reviews whose calls count: not abandoned (an execution-stage abandonment keeps its verdicts, but its book
    never fully traded) and with a debate set (another strategy's records have none)."""
    return [review for review in reviews if not review.get("abandoned") and review.get("debate_set")]


def candidates(review: Mapping[str, Any]) -> list[str]:
    """The review's debate set, in rank order, each symbol once."""
    return list(dict.fromkeys(entry["symbol"] for entry in review["debate_set"]))


def forward_returns(
    reviews: Sequence[Mapping[str, Any]],
    opens: Mapping[str, Mapping[date, float]],
    calendar: Sequence[date],
    horizons: Sequence[int] = HORIZONS,
) -> Returns:
    """Each scored review's candidates, at each horizon: entry at the open of the first `calendar` session strictly
    after the review date, exit at the open `horizon` sessions later. A candidate with no open on either day, or a
    review whose exit is past the calendar, is dropped and counted."""
    sessions = sorted(calendar)
    rows: list[ForwardReturn] = []
    dropped = 0
    for review in scored_reviews(reviews):
        day = date.fromisoformat(review["date"])
        entry_index = bisect_right(sessions, day)  # the first session strictly after the review date
        symbols = candidates(review)
        for horizon in horizons:
            exit_index = entry_index + horizon
            if exit_index >= len(sessions):
                dropped += len(symbols)
                continue
            entry_day, exit_day = sessions[entry_index], sessions[exit_index]
            raw: dict[str, float] = {}
            for symbol in symbols:
                series = opens.get(symbol, {})
                entry, exit_ = series.get(entry_day), series.get(exit_day)
                if entry is None or exit_ is None or entry <= 0:
                    dropped += 1
                    continue
                raw[symbol] = exit_ / entry - 1
            if raw:
                mean = sum(raw.values()) / len(raw)
                rows.extend(ForwardReturn(day, symbol, horizon, value, value - mean) for symbol, value in raw.items())
    return Returns(rows, dropped)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_agent_scores.py -v`
Expected: 5 passed.

Run: `uv run ruff check src/trading_agent_framework/strategies/bull_bear/batch tests/strategies/bull_bear/test_bull_bear_agent_scores.py && uv run pyright src/trading_agent_framework/strategies/bull_bear/batch`
Expected: `All checks passed!` and `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/batch tests/strategies/bull_bear/test_bull_bear_agent_scores.py
git commit -m "feat(bull_bear): forward returns of each review's candidates, from the fill open

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Score each agent's calls

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/batch/agent_scores.py` (imports, then append after `forward_returns`)
- Test: `tests/strategies/bull_bear/test_bull_bear_agent_scores.py` (append; extend the import line)

**Interfaces:**
- Consumes: `ForwardReturn`, `Returns`, `HORIZONS`, `scored_reviews`, `candidates` (Task 2).
- Produces (used by Task 4):
  - `@dataclass(frozen=True, slots=True) class GroupStats: n: int; mean_excess: float | None; hit_rate: float | None`
  - `EMPTY_GROUP: GroupStats`
  - `@dataclass(frozen=True, slots=True) class Spread: mean: float | None; t: float | None; reviews: int; skipped: int`
  - `@dataclass(frozen=True, slots=True) class HorizonScore: groups: dict[str, GroupStats]; spread: Spread; concerns: dict[str, GroupStats]`
  - `@dataclass(frozen=True, slots=True) class AgentScore: agent: str; spread_label: str; by_horizon: dict[int, HorizonScore]`
  - `@dataclass(frozen=True, slots=True) class RunScores: agents: list[AgentScore]; dropped: int` (agents always in the order bull, bear, judge)
  - `def newey_west_t(values: Sequence[float], *, lags: int) -> float | None`
  - `def score_agents(reviews: Sequence[Mapping[str, Any]], returns: Returns, horizons: Sequence[int] = HORIZONS) -> RunScores`

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bull_bear/test_bull_bear_agent_scores.py`, replace the import line with:

```python
from trading_agent_framework.strategies.bull_bear.batch.agent_scores import ForwardReturn, Returns, forward_returns, newey_west_t, score_agents
```

Append:

```python
# --- the t-stat ---------------------------------------------------------------------------------------------


def test_with_no_lag_the_t_stat_is_the_textbook_one() -> None:
    assert newey_west_t([1.0, 2.0, 3.0, 4.0], lags=0) == pytest.approx(2.5 / (1.2909944 / 2))  # mean / (sample sd / root n)


def test_newey_west_adds_the_bartlett_weighted_autocovariances() -> None:
    # demeaned [-1.5, 0.5, -0.5, 1.5]: gamma0 = 1.25, gamma1 = -0.4375; 1.25 + 2 x 0.5 x -0.4375 = 0.8125, x 4/3 = 1.083333
    assert newey_west_t([1.0, 3.0, 2.0, 4.0], lags=1) == pytest.approx(2.5 / (1.0833333 / 4) ** 0.5)


def test_the_t_stat_needs_two_values_and_some_variance() -> None:
    assert newey_west_t([0.01], lags=0) is None
    assert newey_west_t([0.02, 0.02, 0.02], lags=0) is None


# --- agent scores -------------------------------------------------------------------------------------------

NEXT_TUESDAY = date(2026, 1, 13)


def _returns(*rows: tuple[date, str, float], horizon: int = 5) -> Returns:
    """Rows with a given excess return (raw = excess: only the excess is scored)."""
    return Returns([ForwardReturn(day, symbol, horizon, excess, excess) for day, symbol, excess in rows], dropped=0)


def _bull(**convictions: str) -> list[dict[str, str]]:
    return [{"symbol": s, "conviction": c, "argument": "x"} for s, c in convictions.items()]


def _bear(**risks: tuple[str, str]) -> list[dict[str, str]]:
    return [{"symbol": s, "risk": risk, "concern": concern, "argument": "x"} for s, (risk, concern) in risks.items()]


def _verdicts(**verdicts: str) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": v, "reason": "x", "what_changed": None, "previous_verdict": None} for s, v in verdicts.items()]


def test_the_bull_scores_high_against_low_conviction_review_by_review() -> None:
    reviews = [
        _review(TUESDAY, ["AAA", "BBB", "CCC"], bull=_bull(AAA="high", BBB="low", CCC="medium")),
        _review(NEXT_TUESDAY, ["AAA", "BBB", "CCC"], bull=_bull(AAA="high", BBB="low", CCC="medium")),
    ]
    returns = _returns(
        (TUESDAY, "AAA", 0.04), (TUESDAY, "BBB", -0.02), (TUESDAY, "CCC", -0.02),
        (NEXT_TUESDAY, "AAA", 0.01), (NEXT_TUESDAY, "BBB", 0.03), (NEXT_TUESDAY, "CCC", -0.04),
    )  # fmt: skip

    bull = score_agents(reviews, returns, horizons=(5,)).agents[0]

    assert bull.agent == "bull" and bull.spread_label == "high - low"
    score = bull.by_horizon[5]
    assert list(score.groups) == ["low", "medium", "high"]
    high, low, medium = score.groups["high"], score.groups["low"], score.groups["medium"]
    assert (high.n, high.mean_excess, high.hit_rate) == (2, pytest.approx(0.025), 1.0)
    assert (low.n, low.mean_excess, low.hit_rate) == (2, pytest.approx(0.005), 0.5)
    assert (medium.n, medium.mean_excess, medium.hit_rate) == (2, pytest.approx(-0.03), 0.0)
    # per review: 0.04 - (-0.02) = 0.06, then 0.01 - 0.03 = -0.02; mean 0.02, t = 0.02 / (0.0565685 / root 2) = 0.5
    assert score.spread.mean == pytest.approx(0.02) and score.spread.t == pytest.approx(0.5)
    assert (score.spread.reviews, score.spread.skipped) == (2, 0)


def test_the_bear_is_right_when_low_risk_beats_high_risk_and_its_concerns_are_broken_down() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], bear=_bear(AAA=("low", "none"), BBB=("high", "valuation"), CCC=("high", "valuation")))]
    returns = _returns((TUESDAY, "AAA", 0.03), (TUESDAY, "BBB", -0.01), (TUESDAY, "CCC", -0.02))

    bear = score_agents(reviews, returns, horizons=(5,)).agents[1]

    assert bear.agent == "bear" and bear.spread_label == "low - high"
    score = bear.by_horizon[5]
    assert score.spread.mean == pytest.approx(0.03 - (-0.015)) and score.spread.t is None  # one review: no t-stat
    assert score.concerns["valuation"].n == 2 and score.concerns["valuation"].mean_excess == pytest.approx(-0.015)
    assert score.concerns["none"].n == 1


def test_the_judge_scores_in_against_out() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], verdicts=_verdicts(AAA="in", BBB="in", CCC="out"))]
    returns = _returns((TUESDAY, "AAA", 0.02), (TUESDAY, "BBB", 0.0), (TUESDAY, "CCC", -0.02))

    judge = score_agents(reviews, returns, horizons=(5,)).agents[2]

    assert judge.agent == "judge" and judge.spread_label == "in - out"
    score = judge.by_horizon[5]
    assert (score.groups["in"].n, score.groups["out"].n) == (2, 1)
    assert score.spread.mean == pytest.approx(0.01 - (-0.02))


def test_a_picker_record_puts_its_picks_in_and_the_rest_of_its_debate_set_out() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], picks=[{"symbol": "AAA", "reason": "x"}])]
    returns = _returns((TUESDAY, "AAA", 0.02), (TUESDAY, "BBB", 0.0), (TUESDAY, "CCC", -0.02))

    score = score_agents(reviews, returns, horizons=(5,)).agents[2].by_horizon[5]

    assert (score.groups["in"].n, score.groups["out"].n) == (1, 2)
    assert score.spread.mean == pytest.approx(0.02 - (-0.01))


def test_a_review_missing_one_group_is_skipped_for_the_spread_and_counted() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB"], bull=_bull(AAA="high", BBB="high"))]  # no low conviction: no spread
    returns = _returns((TUESDAY, "AAA", 0.01), (TUESDAY, "BBB", -0.01))

    score = score_agents(reviews, returns, horizons=(5,)).agents[0].by_horizon[5]

    assert score.groups["high"].n == 2
    assert (score.spread.mean, score.spread.reviews, score.spread.skipped) == (None, 0, 1)


def test_an_abandoned_review_keeps_its_verdicts_but_none_of_its_calls_count() -> None:
    abandoned = _review(
        TUESDAY,
        ["AAA", "BBB"],
        abandoned=True,
        bull=_bull(AAA="high", BBB="low"),
        bear=_bear(AAA=("low", "none"), BBB=("high", "fundamentals")),
        verdicts=_verdicts(AAA="in", BBB="out"),
    )
    returns = _returns((TUESDAY, "AAA", 0.05), (TUESDAY, "BBB", -0.05))

    scores = score_agents([abandoned], returns, horizons=(5,))

    for agent in scores.agents:
        score = agent.by_horizon[5]
        assert all(group.n == 0 for group in score.groups.values()) and score.spread.reviews == 0 and score.spread.skipped == 0


def test_a_verdict_on_a_symbol_outside_the_debate_set_is_ignored() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB"], verdicts=_verdicts(AAA="in", BBB="out", ZZZ="in"))]
    returns = _returns((TUESDAY, "AAA", 0.01), (TUESDAY, "BBB", -0.01), (NEXT_TUESDAY, "ZZZ", 0.50))  # ZZZ only has a return for another review

    score = score_agents(reviews, returns, horizons=(5,)).agents[2].by_horizon[5]

    assert score.groups["in"].n == 1 and score.spread.mean == pytest.approx(0.02)


def test_the_four_week_spread_uses_newey_west_with_three_lags() -> None:
    days = [TUESDAY + timedelta(weeks=week) for week in range(6)]
    excesses = [(0.03, -0.01), (0.00, 0.02), (0.04, 0.01), (-0.02, 0.01), (0.05, -0.03), (0.01, 0.00)]
    reviews = [_review(day, ["AAA", "BBB"], bull=_bull(AAA="high", BBB="low")) for day in days]
    returns = Returns([ForwardReturn(day, s, 20, x, x) for day, (a, b) in zip(days, excesses, strict=True) for s, x in (("AAA", a), ("BBB", b))], dropped=0)

    spread = score_agents(reviews, returns, horizons=(20,)).agents[0].by_horizon[20].spread

    assert spread.t == pytest.approx(newey_west_t([a - b for a, b in excesses], lags=3))
    assert spread.t != pytest.approx(newey_west_t([a - b for a, b in excesses], lags=0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_agent_scores.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'newey_west_t'` (Task 2's names exist; Task 3's do not yet).

- [ ] **Step 3: Write the scoring**

In `agent_scores.py`, replace the import block with:

```python
from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any
```

Below `HORIZONS = (5, 20)`, add:

```python
LEVELS = ("low", "medium", "high")
VERDICTS = ("in", "out")
_SESSIONS_PER_REVIEW = 5  # one review a week
```

Append after `forward_returns`:

```python
@dataclass(frozen=True, slots=True)
class GroupStats:
    n: int
    mean_excess: float | None  # None when n == 0
    hit_rate: float | None  # share with a positive excess return


EMPTY_GROUP = GroupStats(0, None, None)


@dataclass(frozen=True, slots=True)
class Spread:
    mean: float | None  # mean across reviews of (better group's mean excess - worse group's); None without a review
    t: float | None  # across reviews; None under 2 reviews or with zero variance
    reviews: int  # reviews with both groups
    skipped: int  # scored reviews missing a group


@dataclass(frozen=True, slots=True)
class HorizonScore:
    groups: dict[str, GroupStats]  # in display order
    spread: Spread
    concerns: dict[str, GroupStats] = field(default_factory=dict)  # the bear's, by concern; empty for the others


@dataclass(frozen=True, slots=True)
class AgentScore:
    agent: str  # "bull", "bear" or "judge"
    spread_label: str  # e.g. "high - low"
    by_horizon: dict[int, HorizonScore]


@dataclass(frozen=True, slots=True)
class RunScores:
    agents: list[AgentScore]  # bull, bear, judge
    dropped: int  # from `Returns.dropped`


def bull_calls(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: case["conviction"] for case in review.get("bull") or [] if case.get("conviction") in LEVELS}


def bear_calls(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: case["risk"] for case in review.get("bear") or [] if case.get("risk") in LEVELS}


def bear_concerns(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: str(case.get("concern") or "none") for case in review.get("bear") or []}


def judge_calls(review: Mapping[str, Any]) -> dict[str, str]:
    """The in/out judge's `verdicts`; a picker record's `picks` are in and the rest of its debate set out."""
    if "verdicts" in review:
        return {verdict["symbol"]: verdict["verdict"] for verdict in review["verdicts"] or [] if verdict.get("verdict") in VERDICTS}
    if "picks" in review:
        picked = {pick["symbol"] for pick in review["picks"] or []}
        return {symbol: "in" if symbol in picked else "out" for symbol in candidates(review)}
    return {}


# (agent, its groups in display order, (the group that should do better, the one that should do worse), its calls)
_AGENTS: tuple[tuple[str, tuple[str, ...], tuple[str, str], Callable[[Mapping[str, Any]], dict[str, str]]], ...] = (
    ("bull", LEVELS, ("high", "low"), bull_calls),
    ("bear", LEVELS, ("low", "high"), bear_calls),
    ("judge", VERDICTS, ("in", "out"), judge_calls),
)


def score_agents(reviews: Sequence[Mapping[str, Any]], returns: Returns, horizons: Sequence[int] = HORIZONS) -> RunScores:
    """Each agent's calls against the excess returns: pooled group stats, and a spread computed per review (a review
    missing either group is skipped and counted), averaged across reviews, with a t-stat across reviews."""
    excess = {(row.review, row.symbol, row.horizon): row.excess for row in returns.rows}
    counted = scored_reviews(reviews)
    agents: list[AgentScore] = []
    for name, groups, (better, worse), calls_of in _AGENTS:
        by_horizon: dict[int, HorizonScore] = {}
        for horizon in horizons:
            pooled: dict[str, list[float]] = {group: [] for group in groups}
            concerns: dict[str, list[float]] = {}
            spreads: list[float] = []
            skipped = 0
            for review in counted:
                day = date.fromisoformat(review["date"])
                in_review: dict[str, list[float]] = {group: [] for group in groups}
                for symbol, group in calls_of(review).items():
                    value = excess.get((day, symbol, horizon))
                    if value is not None:
                        in_review[group].append(value)
                        pooled[group].append(value)
                if in_review[better] and in_review[worse]:
                    spreads.append(_mean(in_review[better]) - _mean(in_review[worse]))
                else:
                    skipped += 1
                if name == "bear":
                    for symbol, concern in bear_concerns(review).items():
                        value = excess.get((day, symbol, horizon))
                        if value is not None:
                            concerns.setdefault(concern, []).append(value)
            spread = Spread(_mean(spreads) if spreads else None, newey_west_t(spreads, lags=_overlap_lags(horizon)), len(spreads), skipped)
            by_horizon[horizon] = HorizonScore(
                groups={group: _stats(values) for group, values in pooled.items()},
                spread=spread,
                concerns={concern: _stats(values) for concern, values in sorted(concerns.items())},
            )
        agents.append(AgentScore(name, f"{better} - {worse}", by_horizon))
    return RunScores(agents, returns.dropped)


def newey_west_t(values: Sequence[float], *, lags: int) -> float | None:
    """t-stat of the mean of `values`, with a Newey-West (Bartlett) long-run variance over `lags` lags scaled by
    n / (n - 1): at 0 lags this is the textbook t, the mean over (sample standard deviation / root n). None under two
    values or with no variance."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    demeaned = [value - mean for value in values]
    variance = sum(x * x for x in demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * sum(demeaned[i] * demeaned[i - lag] for i in range(lag, n)) / n
    variance *= n / (n - 1)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance / n)


def _overlap_lags(horizon: int) -> int:
    """The later reviews a horizon's window overlaps: 0 at 5 sessions, 3 at 20."""
    return max(0, math.ceil(horizon / _SESSIONS_PER_REVIEW) - 1)


def _stats(values: Sequence[float]) -> GroupStats:
    if not values:
        return EMPTY_GROUP
    return GroupStats(len(values), _mean(values), sum(1 for value in values if value > 0) / len(values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_agent_scores.py -v`
Expected: 16 passed.

Run: `uv run ruff check src/trading_agent_framework/strategies/bull_bear/batch tests/strategies/bull_bear/test_bull_bear_agent_scores.py && uv run pyright src/trading_agent_framework/strategies/bull_bear/batch`
Expected: `All checks passed!` and `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/batch/agent_scores.py tests/strategies/bull_bear/test_bull_bear_agent_scores.py
git commit -m "feat(bull_bear): score the bull, the bear and the judge against forward returns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The `bull-bear-scores` CLI, its entry point and docs

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/batch/scores_cli.py`
- Modify: `pyproject.toml:32` (add the entry after `batch-universe`)
- Modify: `CLAUDE.md:15` (Commands, after the `batch-universe` line)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (a bullet after the Architecture bullet, before `## Gotchas`)
- Test: `tests/strategies/bull_bear/test_bull_bear_scores_cli.py`

**Interfaces:**
- Consumes: `HORIZONS`, `EMPTY_GROUP`, `GroupStats`, `Spread`, `RunScores`, `candidates`, `scored_reviews`, `forward_returns`, `score_agents` (Tasks 2-3); `YahooDailyBars(*, download: DownloadFn | None = None, history: int)` with `.bars(symbols: Sequence[str], today: date) -> dict[str, Bars]` and `DownloadFn` (`strategies/common/yahoo_daily_bars.py`); `Bars.df["open"]` indexed by tz-aware session-close timestamps; `YahooDataError` (`utils/errors.py`).
- Produces: `main(argv: Sequence[str] | None = None, *, download: DownloadFn | None = None, today: date | None = None) -> int`; `read_reviews(path: Path) -> tuple[list[dict[str, Any]], int]`; `fetch_opens(runs: Sequence[Sequence[Mapping[str, Any]]], *, today: date, download: DownloadFn | None = None) -> tuple[dict[str, dict[date, float]], list[date]]`; `format_report(runs: Sequence[RunReport], horizons: Sequence[int] = HORIZONS) -> str`; `RunReport(name: str, scores: RunScores, reviews: int, malformed: int)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/bull_bear/test_bull_bear_scores_cli.py`:

```python
"""`uv run bull-bear-scores <run_dir>...`: bull_bear runs scored side by side; a fake download, never the network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_agent_framework.strategies.bull_bear.batch.scores_cli import main

TODAY = date(2026, 4, 30)
SESSIONS = 60  # business days from Friday 2026-01-02


def _yahoo_frame(opens: list[float]) -> pd.DataFrame:
    """One ticker as yfinance returns it: Open/High/Low/Close/Volume, a naive session-date index."""
    index = pd.bdate_range(start="2026-01-02", periods=len(opens))
    return pd.DataFrame({"Open": opens, "High": [o + 1 for o in opens], "Low": [o - 1 for o in opens], "Close": opens, "Volume": [1e6] * len(opens)}, index=index)


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """What `yfinance.download(symbols, group_by="ticker")` returns: (ticker, field) columns."""
    return pd.concat(frames, axis=1)


class FakeDownload:
    def __init__(self, result: pd.DataFrame | None = None, raises: Exception | None = None) -> None:
        self.result, self.raises = result, raises
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, symbols: list[str], **kwargs: Any) -> pd.DataFrame | None:
        self.calls.append((list(symbols), kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


def _prices(*, spy: bool = True) -> pd.DataFrame:
    frames = {"AAA": _yahoo_frame([100.0 * 1.01**i for i in range(SESSIONS)]), "BBB": _yahoo_frame([100.0 * 0.99**i for i in range(SESSIONS)])}
    if spy:
        frames["SPY"] = _yahoo_frame([500.0] * SESSIONS)
    return _batch(frames)


def _judge_review(day: str) -> dict[str, Any]:
    return {
        "date": day,
        "abandoned": False,
        "debate_set": [{"symbol": "AAA", "rank": 1, "held": False}, {"symbol": "BBB", "rank": 2, "held": False}],
        "bull": [{"symbol": "AAA", "conviction": "high", "argument": "x"}, {"symbol": "BBB", "conviction": "low", "argument": "x"}],
        "bear": [{"symbol": "AAA", "risk": "low", "concern": "none", "argument": "x"}, {"symbol": "BBB", "risk": "high", "concern": "valuation", "argument": "x"}],
        "verdicts": [{"symbol": "AAA", "verdict": "in", "reason": "x"}, {"symbol": "BBB", "verdict": "out", "reason": "x"}],
    }


def _picker_review(day: str) -> dict[str, Any]:
    record = _judge_review(day)
    del record["verdicts"]
    return {**record, "picks": [{"symbol": "AAA", "reason": "x"}]}


def _run(tmp_path: Path, name: str, lines: list[str]) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_two_runs_print_the_bull_the_bear_and_the_judge_side_by_side(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    judge = _run(tmp_path, "judge_run", [json.dumps(_judge_review("2026-01-06")), json.dumps(_judge_review("2026-01-13"))])
    picker = _run(tmp_path, "picker_run", [json.dumps(_picker_review("2026-01-06")), "not json"])

    assert main([str(picker), str(judge)], download=FakeDownload(_prices()), today=TODAY) == 0

    out = capsys.readouterr().out
    for title in ("BULL conviction, spread high - low", "BEAR risk, spread low - high", "JUDGE verdict, spread in - out"):
        assert title in out
    assert "concern valuation" in out
    assert "picker_run: 1 reviews scored" in out and "1 malformed lines" in out
    assert "judge_run: 2 reviews scored" in out
    spread_rows = [line for line in out.splitlines() if line.startswith("  spread")]
    assert len(spread_rows) == 3 and all("+" in row for row in spread_rows)  # AAA rises, BBB falls: every agent is right


def test_the_download_covers_every_run_from_a_week_before_the_first_review_to_forty_days_after_the_last(tmp_path: Path) -> None:
    first = _run(tmp_path, "a", [json.dumps(_judge_review("2026-01-06"))])
    second = _run(tmp_path, "b", [json.dumps(_judge_review("2026-01-13"))])
    download = FakeDownload(_prices())

    main([str(first), str(second)], download=download, today=TODAY)

    ((symbols, kwargs),) = download.calls
    assert sorted(symbols) == ["AAA", "BBB", "SPY"]
    assert kwargs["start"] <= "2025-12-30" and kwargs["end"] == "2026-02-23"  # 2026-01-13 + 40 days, end exclusive


def test_the_download_window_stops_at_today(tmp_path: Path) -> None:
    run = _run(tmp_path, "running", [json.dumps(_judge_review("2026-01-13"))])
    download = FakeDownload(_prices())

    main([str(run)], download=download, today=date(2026, 1, 20))

    assert download.calls[0][1]["end"] == "2026-01-21"


def test_a_run_directory_without_reviews_exits_2(tmp_path: Path) -> None:
    empty = tmp_path / "no_reviews"
    empty.mkdir()

    with pytest.raises(SystemExit) as exit_info:
        main([str(empty)], download=FakeDownload(_prices()), today=TODAY)

    assert exit_info.value.code == 2


def test_a_failing_download_exits_1_with_a_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path, "r", [json.dumps(_judge_review("2026-01-06"))])

    assert main([str(run)], download=FakeDownload(raises=RuntimeError("rate limited")), today=TODAY) == 1

    assert "rate limited" in capsys.readouterr().err


def test_no_spy_in_the_download_exits_1_naming_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path, "r", [json.dumps(_judge_review("2026-01-06"))])

    assert main([str(run)], download=FakeDownload(_prices(spy=False)), today=TODAY) == 1

    assert "SPY" in capsys.readouterr().err


def test_a_run_from_another_strategy_prints_a_report_with_no_review_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ackman = _run(tmp_path, "ackman_run", [json.dumps({"date": "2026-01-06", "abandoned": False, "holdings": ["AAA"], "targets": {"AAA": 0.5}})])

    assert main([str(ackman)], download=FakeDownload(_prices()), today=TODAY) == 0

    assert "ackman_run: 0 reviews scored" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_scores_cli.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bull_bear.batch.scores_cli'`.

- [ ] **Step 3: Write the CLI**

Create `src/trading_agent_framework/strategies/bull_bear/batch/scores_cli.py`:

```python
"""`uv run bull-bear-scores <run_dir> [<run_dir>...]`: the bull, the bear and the judge of bull_bear backtest runs,
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
from typing import Any

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

REVIEWS_FILE = "reviews.jsonl"
CALENDAR_SYMBOL = "SPY"
_WINDOW_BEFORE = timedelta(days=7)
_WINDOW_AFTER = timedelta(days=40)  # 20 sessions plus slack
_CALENDAR_DAYS_PER_BAR = 1.5  # how far back `YahooDailyBars` reaches per bar of history
_LABEL = 32
_CELL = 26
_TITLES = {"bull": "BULL conviction", "bear": "BEAR risk", "judge": "JUDGE verdict"}


@dataclass(frozen=True, slots=True)
class RunReport:
    name: str
    scores: RunScores
    reviews: int  # scored reviews
    malformed: int  # unreadable lines in reviews.jsonl


def main(argv: Sequence[str] | None = None, *, download: DownloadFn | None = None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bull-bear-scores",
        description="Score the bull, the bear and the judge of bull_bear backtest runs against forward returns (Yahoo opens, 5 and 20 sessions).",
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
    opens = {symbol: {stamp.date(): float(value) for stamp, value in bar.df["open"].items()} for symbol, bar in bars.items()}
    if CALENDAR_SYMBOL not in opens:
        raise YahooDataError(f"Yahoo returned no {CALENDAR_SYMBOL} bars, the session calendar")
    return opens, sorted(opens[CALENDAR_SYMBOL])


def format_report(runs: Sequence[RunReport], horizons: Sequence[int] = HORIZONS) -> str:
    """One block per agent (bull, bear, judge), each run a column group with one column per horizon."""
    width = _CELL * len(horizons) + len(horizons) - 1
    header = " " * _LABEL + "".join(f" | {run.name:<{width}}" for run in runs)
    subheader = " " * _LABEL + "".join(" | " + " ".join(f"{horizon}d".rjust(_CELL) for horizon in horizons) for _ in runs)
    lines: list[str] = []
    for index, agent in enumerate(runs[0].scores.agents):  # every run lists bull, bear, judge in that order
        scores = [[run.scores.agents[index].by_horizon[horizon] for horizon in horizons] for run in runs]
        lines += ["", f"{_TITLES[agent.agent]}, spread {agent.spread_label}", header, subheader]
        for group in agent.by_horizon[horizons[0]].groups:
            lines.append(_row(f"  {group}", [[score.groups[group] for score in run] for run in scores], _group_cell))
        for concern in sorted({concern for run in scores for score in run for concern in score.concerns}):
            lines.append(_row(f"  concern {concern}", [[score.concerns.get(concern, EMPTY_GROUP) for score in run] for run in scores], _group_cell))
        lines.append(_row("  spread", [[score.spread for score in run] for run in scores], _spread_cell))
    lines.append("")
    lines += [f"{run.name}: {run.reviews} reviews scored, {run.scores.dropped} candidate x horizon pairs without prices, {run.malformed} malformed lines" for run in runs]
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_scores_cli.py -v`
Expected: 7 passed.

- [ ] **Step 5: Register the entry point**

In `pyproject.toml`, after line 32 (`batch-universe = ...`), add:

```toml
bull-bear-scores = "trading_agent_framework.strategies.bull_bear.batch.scores_cli:main"
```

Run: `uv sync -q --extra dashboard && uv run bull-bear-scores --help`
Expected: the argparse usage line `usage: bull-bear-scores [-h] run_dir [run_dir ...]`.

- [ ] **Step 6: Document it**

In `CLAUDE.md`, after line 15 (`uv run batch-universe ...`), add:

```bash
uv run bull-bear-scores <run_dir> [<run_dir>...]  # score bull_bear's bull/bear/judge calls against forward returns, runs side by side (fetches Yahoo opens; read-only)
```

In `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`, insert this bullet on the line before `## Gotchas` (after the Architecture bullet and its blank line):

```markdown
- `bull_bear/batch/` (run by hand over finished runs; the strategy never imports it): `agent_scores.py` (**pure**: `forward_returns` from the open of the first session after each review, the fill, to the open 5 and 20 sessions later, with the excess over that review's candidate mean; `score_agents`: the bull's high - low conviction, the bear's low - high risk with a concern breakdown, the judge's in - out (a picker record's `picks` are in, the rest of its debate set out), each spread computed per review and averaged, with a t-stat across reviews, Newey-West with 3 lags at 20 sessions; abandoned reviews count for nothing) and `scores_cli.py` (`uv run bull-bear-scores <run_dir>...`: reads `reviews.jsonl` itself, never the dashboard's reader, fetches opens through `YahooDailyBars` with SPY as the calendar, prints runs side by side; exit 2 without `reviews.jsonl`, 1 on a Yahoo failure).
```

- [ ] **Step 7: Full verification**

Run: `uv run pytest -q`
Expected: every test passes (3428 before this branch, plus 1 + 16 + 7 = 24 new: 3452 passed).

Run: `uv run ruff check src tests && uv run pyright src/trading_agent_framework/strategies/bull_bear/batch`
Expected: `All checks passed!` for `src tests` (the pre-existing `scripts/tests/smoke_congress_trades.py` import-order finding is outside these paths and out of scope) and `0 errors`.

Manual, network (by hand, like `scripts/tests/`): `uv run bull-bear-scores /home/yann/projets/trading-agent-framework/logs/bull_bear/backtesting/2026-10-10_085928_backtesting /home/yann/projets/trading-agent-framework/logs/bull_bear/backtesting/2026-10-12_005438_backtesting`
Expected: the three blocks with both runs side by side; 53 reviews scored for each; the judge's 5d spread is negative in both (the session's full-run analysis found chosen minus rejected at -0.40 pt a week, on closes; opens will differ slightly).

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/batch/scores_cli.py tests/strategies/bull_bear/test_bull_bear_scores_cli.py pyproject.toml CLAUDE.md src/trading_agent_framework/strategies/bull_bear/CLAUDE.md
git commit -m "feat(bull_bear): bull-bear-scores CLI, runs scored side by side

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
