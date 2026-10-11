# bull_bear veto judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn bull_bear's judge from a picker (5-10 of the top 15) into a judge that gives a keep/veto verdict on every debated stock, and have code build a book of exactly `book_size` stocks from what it keeps.

**Architecture:** A new pure module `book.py` builds the book (surviving holdings first, then surviving new names, each in rank order, cut at `book_size`). `handoff.py` gains `submit_verdicts` (one verdict per debated stock, at most a per-review veto cap). `BullBearParams` gets `book_size` and `max_vetoes`; the debate size is derived (`book_size + max_vetoes`). The pipeline computes the cap so that `book_size` stocks always survive, then sizes the book with the unchanged `capped_inverse_volatility` and trades through the unchanged `Rebalancer`. The picker (`submit_picks`, `min_picks`, `max_picks`, `shortlist_size`) is removed last, once nothing uses it.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pyright. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-11-bull-bear-veto-judge-design.md`

## Global Constraints

- Work in the worktree `/home/yann/projets/trading-agent-framework-veto-judge`, branch `feat/bull-bear-veto-judge`. Never switch branches in `/home/yann/projets/trading-agent-framework` (the user works on `main` there).
- Touch only `src/trading_agent_framework/strategies/bull_bear/`, `tests/strategies/bull_bear/` and docs. Never touch `strategies/common/`, `strategies/cross_momentum/` or `scoring.py`.
- All four agents stay: researcher, bull, bear, judge. The researcher, bull and bear prompts and tools are unchanged.
- The judge decides on merit only: its context carries no `in_book` flag and no queue position; the fact sheet is unchanged.
- Defaults: `book_size` = 10, `max_vetoes` = 5, so `debate_size` = 15.
- `handoff.py` has NO `from __future__ import annotations` (the agent layer builds tool schemas from real annotations).
- Tests use hand-written fakes, never `MagicMock`; the suite never touches the network.
- Ruff line length is 200. Run ruff on the bull_bear paths only: `uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear` (an unrelated stale smoke script trips ruff's import sort in this worktree).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- The judge keeps every stock (no veto at all): the book must be the top `book_size` in holdings-then-rank order and `judge_verdicts` must read `{"keep": n}` (Task 2 `max_vetoes=0` test, Task 5 happy path).
- A debate set between `book_size` and `debate_size` stocks (thin data): the cap must shrink to `len(debate) - book_size`, and a judge over the cap must be refused and retried, never trusted (Task 2 cap refusal, Task 5 cap and retry tests).
- The local model writes `"Veto "`, `"KEEP"` or `" aaa "`: verdicts and symbols must be normalised and accepted, exactly as the bull's and bear's cases are (Task 2 normalisation test).
- A paper/live state file written before this change (with `last_picks`): it must still load with its `last_completed_review` and streak, otherwise a restart on a completed Tuesday would review again (Task 3).
- More surviving holdings than `book_size` (the user lowers `book_size` while holding more): the best-ranked `book_size` holdings form the book and the rest are sold (Task 1).

---

### Task 1: `book.py`, the book from the judge's survivors

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/book.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_book.py`

**Interfaces:**
- Consumes: `DebateSet`, `DebateStock` from `strategies/bull_bear/debate_set.py` (unchanged: `DebateSet.stocks` is in rank order; `DebateStock.symbol`, `.rank`, `.held`).
- Produces: `build_book(debate: DebateSet, vetoed: Collection[str], book_size: int) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

Create `tests/strategies/bull_bear/test_bull_bear_book.py`:

```python
from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.book import build_book
from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet, DebateStock
from trading_agent_framework.strategies.common.scoring import MomentumRow, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _row(symbol: str) -> MomentumRow:
    row = score_stock(symbol, [100.0 + i for i in range(300)], [1e6] * 300, CONFIG)
    assert row is not None
    return row


def _debate(ranks: list[int], held: frozenset[int] = frozenset()) -> DebateSet:
    """The debated stocks S01, S02, ... in rank order, as `build_debate_set` returns them; `held` are the ranks owned."""
    return DebateSet(stocks=tuple(DebateStock(rank, _row(f"S{rank:02d}"), rank in held) for rank in sorted(ranks)), forced_exits=())


def _symbols(ranks: list[int]) -> tuple[str, ...]:
    return tuple(f"S{rank:02d}" for rank in ranks)


def test_without_a_veto_the_book_is_the_top_book_size_ranks() -> None:
    assert build_book(_debate(list(range(1, 16))), [], 10) == _symbols(list(range(1, 11)))


def test_a_vetoed_stock_is_replaced_by_the_next_ranked_survivor() -> None:
    book = build_book(_debate(list(range(1, 16))), ["S01", "S05", "S07"], 10)

    assert book == _symbols([2, 3, 4, 6, 8, 9, 10, 11, 12, 13])


def test_kept_holdings_come_first_then_new_names_in_rank_order() -> None:
    """The spec's example: holdings at ranks 2, 9 and 22; the judge vetoes rank 9 and ranks 1, 5 and 7."""
    debate = _debate([*range(1, 16), 22], held=frozenset({2, 9, 22}))

    book = build_book(debate, ["S09", "S01", "S05", "S07"], 10)

    assert book == _symbols([2, 22, 3, 4, 6, 8, 10, 11, 12, 13])


def test_more_surviving_holdings_than_book_size_keeps_the_best_ranked_holdings() -> None:
    debate = _debate(list(range(1, 16)), held=frozenset({2, 4, 6, 8}))

    assert build_book(debate, [], 3) == _symbols([2, 4, 6])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_book.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.bull_bear.book'`

- [ ] **Step 3: Write minimal implementation**

Create `src/trading_agent_framework/strategies/bull_bear/book.py`:

```python
"""The book: the stocks bull_bear holds after a review, built by code from the judge's verdicts (pure).

The survivors are the debated stocks the judge did not veto. Holdings come first, so a kept holding ranked within
`retention_rank` keeps its slot (cross_momentum's hysteresis), then new names; each group in rank order. The first
`book_size` survivors form the book. The pipeline caps the vetoes so that at least `book_size` stocks survive.
"""

from __future__ import annotations

from collections.abc import Collection

from trading_agent_framework.strategies.bull_bear.debate_set import DebateSet


def build_book(debate: DebateSet, vetoed: Collection[str], book_size: int) -> tuple[str, ...]:
    """The book's symbols in buy order: kept holdings, then kept new names, each by rank, cut at `book_size`.

    More kept holdings than `book_size` (after `book_size` was lowered) keeps the best-ranked holdings.
    """
    vetoes = set(vetoed)
    survivors = [stock for stock in debate.stocks if stock.symbol not in vetoes]  # `debate.stocks` is in rank order
    holdings = [stock.symbol for stock in survivors if stock.held]
    newcomers = [stock.symbol for stock in survivors if not stock.held]
    return tuple([*holdings, *newcomers][:book_size])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_book.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git add src/trading_agent_framework/strategies/bull_bear/book.py tests/strategies/bull_bear/test_bull_bear_book.py
git commit -m "feat(bull_bear): build_book, the book from the judge's survivors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `submit_verdicts`, the judge's new hand-off (added beside `submit_picks`)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`

**Interfaces:**
- Consumes: the existing private helpers `_items`, `_symbol`, `_text`, `_one_of`, `_covering`; `BullBearParams.reason_max_chars` (300).
- Produces:
  - `VERDICTS = ("keep", "veto")`, stage constant `VERDICT = "verdict"`, tool name `"submit_verdicts"`;
  - `Verdict(symbol: str, verdict: str, reason: str)` (frozen dataclass);
  - `Verdicts(verdicts: tuple[Verdict, ...])` with property `vetoed -> list[str]` (vetoed symbols, submission order);
  - `validate_verdicts(raw: Any, *, expected: Sequence[str], max_vetoes: int, max_chars: int) -> Verdicts`;
  - `HandoffRecorder.expect_verdicts(symbols: Sequence[str], *, max_vetoes: int) -> None`;
  - `submit_tools(recorder)["submit_verdicts"](verdicts: list[dict[str, Any]]) -> dict[str, Any]`.
  - `submit_picks` and everything picker-related stay until Task 6.

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bull_bear/test_bull_bear_handoff.py`, change the import line to:

```python
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, Choice, HandoffRecorder, Note, Picks, Verdict, submit_tools
```

Add this helper after `_choices`:

```python
def _verdicts(keep: list[str], veto: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": "keep", "reason": "bull wins"} for s in keep] + [{"symbol": s, "verdict": "veto", "reason": "bear wins"} for s in veto or []]
```

Add this section just before `# --- the recorder ---`:

```python
# --- verdicts -----------------------------------------------------------------------------------


def test_a_verdict_for_every_stock_is_recorded_with_its_vetoes() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=2)

    result = tools["submit_verdicts"](_verdicts(["AAA", "BBB", "CCC", "DDD"], ["EEE", "FFF"]))

    assert result == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "keep", "bull wins")
    assert recorder.submission.vetoed == ["EEE", "FFF"]


def test_keeping_every_stock_is_valid_even_with_no_veto_allowed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=0)

    assert tools["submit_verdicts"](_verdicts(DEBATE)) == {"status": "recorded"}
    assert recorder.submission.vetoed == []


def test_verdicts_and_symbols_are_normalised_like_the_debaters_cases() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=1)

    raw = [{"symbol": " aaa ", "verdict": "Veto ", "reason": "  lawsuit filed  "}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "veto", "lawsuit filed")


@pytest.mark.parametrize(
    ("verdicts", "message"),
    [
        ({"AAA": "keep"}, "verdicts must be a list of objects"),
        (_verdicts(DEBATE[:5]), "no case for FFF"),
        (_verdicts([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_verdicts([*DEBATE, "AAA"]), "AAA appears twice"),
        ([{"symbol": "AAA", "verdict": "sell", "reason": "x"}, *_verdicts(DEBATE[1:])], "the verdict for AAA must be one of keep, veto"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": " "}, *_verdicts(DEBATE[1:])], "AAA has no reason"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": "x" * 301}, *_verdicts(DEBATE[1:])], "the reason for AAA is 301 characters"),
        (_verdicts(DEBATE[:4], DEBATE[4:]), "2 vetoes: veto at most 1 stocks and keep the others"),
    ],
)
def test_bad_verdicts_are_refused(verdicts: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=1)

    result = tools["submit_verdicts"](verdicts)

    assert message in result["error"] and not recorder.submitted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'Verdict'`

- [ ] **Step 3: Write minimal implementation**

In `src/trading_agent_framework/strategies/bull_bear/handoff.py`:

Replace the constants block:

```python
LEVELS = ("low", "medium", "high")
CONCERNS = ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none")
VERDICTS = ("keep", "veto")

NOTE, BULL, BEAR, PICKS, VERDICT = "note", "bull", "bear", "picks", "verdict"
_TOOL_NAMES = {NOTE: "submit_note", BULL: "submit_bull_case", BEAR: "submit_bear_case", PICKS: "submit_picks", VERDICT: "submit_verdicts"}
```

Add after the `Picks` dataclass:

```python
@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str  # one of VERDICTS
    reason: str


@dataclass(frozen=True, slots=True)
class Verdicts:
    verdicts: tuple[Verdict, ...]  # one per debated stock, in submission order

    @property
    def vetoed(self) -> list[str]:
        """The vetoed symbols, in submission order."""
        return [verdict.symbol for verdict in self.verdicts if verdict.verdict == "veto"]
```

Give `_covering` the name of what it validates (the bull's and bear's messages are unchanged):

```python
def _covering(raw: Any, expected: Sequence[str], what: str = "cases") -> list[tuple[str, Mapping[str, Any]]]:
    """(symbol, item) for exactly one item per symbol in `expected`, in submission order."""
    items = _items(raw, what)
```

(the rest of `_covering` is unchanged).

Add after `validate_picks`:

```python
def validate_verdicts(raw: Any, *, expected: Sequence[str], max_vetoes: int, max_chars: int) -> Verdicts:
    """One verdict (keep or veto) with a reason per `expected` symbol, and at most `max_vetoes` vetoes."""
    verdicts = tuple(Verdict(symbol, _one_of(item, "verdict", symbol, VERDICTS), _text(item, "reason", symbol, max_chars)) for symbol, item in _covering(raw, expected, "verdicts"))
    vetoes = sum(1 for verdict in verdicts if verdict.verdict == "veto")
    if vetoes > max_vetoes:
        raise HandoffError(f"{vetoes} vetoes: veto at most {max_vetoes} stocks and keep the others")
    return Verdicts(verdicts)
```

In `HandoffRecorder`, add after `expect_picks`:

```python
    def expect_verdicts(self, symbols: Sequence[str], *, max_vetoes: int) -> None:
        self._arm(VERDICT, {"expected": list(symbols), "max_vetoes": max_vetoes})
```

Update the `submission` docstring to: `"""The armed stage's valid submission (`Note`, `list[BullCase]`, `list[BearCase]`, `Picks` or `Verdicts`), else None."""`

In `HandoffRecorder.submit`, replace the final `else:` branch (the `validate_picks` call) with:

```python
            elif stage == PICKS:
                value = validate_picks(
                    args[0],
                    args[1],
                    allowed=context["allowed"],
                    held=context["held"],
                    min_picks=params.min_picks,
                    max_picks=params.max_picks,
                    max_chars=params.reason_max_chars,
                )
            else:
                value = validate_verdicts(args[0], expected=context["expected"], max_vetoes=context["max_vetoes"], max_chars=params.reason_max_chars)
```

In `submit_tools`, add before the `return`, and add the tool to the returned dict:

```python
    def submit_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one verdict per stock: objects with symbol, verdict (keep or veto) and one short reason."""
        return recorder.submit(VERDICT, verdicts)

    return {
        "submit_note": submit_note,
        "submit_bull_case": submit_bull_case,
        "submit_bear_case": submit_bear_case,
        "submit_picks": submit_picks,
        "submit_verdicts": submit_verdicts,
    }
```

Update the `submit_tools` docstring to: `"""The submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -v`
Expected: all pass (the new verdict tests included; nothing else changed behaviour)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git add src/trading_agent_framework/strategies/bull_bear/handoff.py tests/strategies/bull_bear/test_bull_bear_handoff.py
git commit -m "feat(bull_bear): submit_verdicts, a keep/veto verdict per debated stock

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: state remembers `last_book`; an old `last_picks` file still loads

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/state.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (the one `BullBearState(...)` call in step 8)
- Test: `tests/strategies/bull_bear/test_bull_bear_state.py`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Produces: `BullBearState.last_book: list[dict[str, str]]` (`[{symbol, reason, date}]`), replacing `last_picks`. `STATE_VERSION` stays 1 (bumping it would make every existing state invalid and lose `last_completed_review`).

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bull_bear/test_bull_bear_state.py`:

In `test_a_saved_state_loads_back`, replace `last_picks=[...]` with `last_book=[{"symbol": "AAA", "reason": "bull wins", "date": "2026-10-06"}]`.

In the parametrized invalid contents, replace `json.dumps({"version": STATE_VERSION, "last_picks": [{"symbol": "AAA"}]}),` with `json.dumps({"version": STATE_VERSION, "last_book": [{"symbol": "AAA"}]}),`.

Add after `test_a_saved_state_loads_back`:

```python
def test_a_state_saved_before_the_veto_judge_loads_without_its_last_picks(tmp_path: Path) -> None:
    """A paper/live file from the picker keeps its date and streak: a completed Tuesday is not reviewed again."""
    path = tmp_path / "state.json"
    old = {"version": STATE_VERSION, "last_completed_review": "2026-10-06", "abandoned_streak": 1, "last_picks": [{"symbol": "AAA", "reason": "won", "date": "2026-10-06"}]}
    path.write_text(json.dumps(old))

    assert StateStore(path).load() == BullBearState(last_completed_review="2026-10-06", abandoned_streak=1)
```

In `tests/strategies/bull_bear/test_bull_bear_pipeline.py`, in `test_a_review_debates_the_top_15_and_buys_the_judges_picks`, replace the last line with:

```python
    assert [entry["symbol"] for entry in state.last_book] == UNIVERSE[:5]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_state.py tests/strategies/bull_bear/test_bull_bear_pipeline.py -v`
Expected: FAIL: `TypeError: BullBearState.__init__() got an unexpected keyword argument 'last_book'`, and `AttributeError: 'BullBearState' object has no attribute 'last_book'`

- [ ] **Step 3: Write minimal implementation**

In `src/trading_agent_framework/strategies/bull_bear/state.py`:

Module docstring, first paragraph:

```python
"""What bull_bear remembers between reviews, and the per-review log.

`StateStore` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a
restart), the abandoned-review streak and the last book, in one small JSON file per mode, written atomically. A
missing or invalid file is an empty state; a file from before the veto judge (with `last_picks`) loads, the old key
ignored. `ReviewLog` appends one JSON line per review to the run directory. Neither ever raises on an I/O problem:
bookkeeping must not stop a review.
"""
```

Replace `_PICK_KEYS = {"symbol", "reason", "date"}` with `_BOOK_KEYS = {"symbol", "reason", "date"}`.

In `BullBearState`, replace the `last_picks` field with:

```python
    last_book: list[dict[str, str]] = field(default_factory=list)  # [{symbol, reason, date}] of the last completed review
```

In `_valid`, replace `picks = raw.get("last_picks", [])` and the two lines that check it with:

```python
    book = raw.get("last_book", [])
    return (
        (last is None or isinstance(last, str))
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
        and isinstance(book, list)
        and all(isinstance(entry, dict) and set(entry) == _BOOK_KEYS and all(isinstance(value, str) for value in entry.values()) for entry in book)
    )
```

In `StateStore.load`, replace `last_picks=[dict(pick) for pick in raw.get("last_picks", [])],` with `last_book=[dict(entry) for entry in raw.get("last_book", [])],`.

In `src/trading_agent_framework/strategies/bull_bear/pipeline.py`, step 8, replace `last_picks=[...]` in the `BullBearState(...)` call with:

```python
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, last_book=[{"symbol": pick.symbol, "reason": pick.reason, "date": today} for pick in picks.picks]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -v`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git add src/trading_agent_framework/strategies/bull_bear/state.py src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/bull_bear/test_bull_bear_state.py tests/strategies/bull_bear/test_bull_bear_pipeline.py
git commit -m "feat(bull_bear): the state remembers the last book; picker state files still load

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `book_size`, `max_vetoes`, the derived `debate_size`; `build_debate_set(debate_size=...)`

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/debate_set.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (the `build_debate_set` call)
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `tests/strategies/bull_bear/test_bull_bear_debate_set.py`

**Interfaces:**
- Produces: `BullBearParams.book_size: int = 10`, `BullBearParams.max_vetoes: int = 5`, property `BullBearParams.debate_size -> int` (`book_size + max_vetoes`); `build_debate_set(ranked, holdings, *, debate_size: int, retention_rank: int) -> DebateSet` (keyword `shortlist_size` renamed).
- The picker fields stay until Task 6 (the defaults satisfy both sets of rules).

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bull_bear/test_bull_bear_params.py`, add after `test_the_defaults_are_the_specs_and_valid`:

```python
def test_the_book_defaults_debate_the_top_15() -> None:
    params = BullBearParams()

    assert (params.book_size, params.max_vetoes, params.debate_size) == (10, 5, 15)
    assert BullBearParams(book_size=8, max_vetoes=7).debate_size == 15
```

and add these rows to the `test_an_invalid_value_is_refused_with_a_reason` parameter list:

```python
        ({"book_size": 30, "retention_rank": 40}, "book_size x min_weight must fit"),
        ({"book_size": 4}, "book_size x max_weight must reach"),
        ({"max_vetoes": 25}, "retention_rank must be above debate_size"),
        ({"book_size": 0}, "book_size and max_vetoes must be at least 1"),
        ({"max_vetoes": 0}, "book_size and max_vetoes must be at least 1"),
```

In `tests/strategies/bull_bear/test_bull_bear_debate_set.py`, replace every `shortlist_size=15` with `debate_size=15` (4 calls).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_debate_set.py -v`
Expected: FAIL: `AttributeError: 'BullBearParams' object has no attribute 'book_size'`, `TypeError: ... unexpected keyword argument 'book_size'`, and `TypeError: build_debate_set() got an unexpected keyword argument 'debate_size'`

- [ ] **Step 3: Write minimal implementation**

In `src/trading_agent_framework/strategies/bull_bear/parameters.py`, add after `max_picks: int = 10`:

```python
    book_size: int = 10  # stocks held after every completed review
    max_vetoes: int = 5  # most stocks the judge may veto in one review
```

Add these entries to the `problems` dict, after `"1 <= min_picks <= max_picks <= shortlist_size": ...`:

```python
            "book_size and max_vetoes must be at least 1": self.book_size < 1 or self.max_vetoes < 1,
            "retention_rank must be above debate_size (book_size + max_vetoes)": self.retention_rank <= self.book_size + self.max_vetoes,
            "book_size x min_weight must fit in 1 - cash_buffer": self.book_size * self.min_weight > investable + _EPS,
            "book_size x max_weight must reach 1 - cash_buffer (the book is always fully invested)": self.book_size * self.max_weight < investable - _EPS,
```

Add after the `investable` property:

```python
    @property
    def debate_size(self) -> int:
        """The top ranks debated: enough for `book_size` stocks to survive `max_vetoes` vetoes."""
        return self.book_size + self.max_vetoes
```

In `src/trading_agent_framework/strategies/bull_bear/debate_set.py`:

Module docstring, first sentence: replace "The top `shortlist_size` by momentum rank" with "The top `debate_size` by momentum rank".

Replace the function signature and its filter line:

```python
def build_debate_set(ranked: Sequence[RankedRow], holdings: Collection[str], *, debate_size: int, retention_rank: int) -> DebateSet:
```

```python
        if entry.rank <= debate_size or (entry.row.symbol in held and entry.rank <= retention_rank)  # hysteresis: held stocks still ranked within the retention threshold are debated
```

In `src/trading_agent_framework/strategies/bull_bear/pipeline.py`, step 2:

```python
        debate = build_debate_set(ranked, holdings, debate_size=params.debate_size, retention_rank=params.retention_rank)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -v`
Expected: all pass (the pipeline still debates the top 15: `debate_size` is 15 by default)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git add src/trading_agent_framework/strategies/bull_bear/parameters.py src/trading_agent_framework/strategies/bull_bear/debate_set.py src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_debate_set.py
git commit -m "feat(bull_bear): book_size and max_vetoes; the debate size is derived

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: the review runs on verdicts and builds the book

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py`, `tests/strategies/bull_bear/test_bull_bear_prompts.py`, `tests/strategies/bull_bear/test_bull_bear_strategy.py`, `tests/strategies/bull_bear/test_bull_bear_backtest.py`

**Interfaces:**
- Consumes: `build_book` (Task 1); `Verdicts`, `HandoffRecorder.expect_verdicts`, `submit_verdicts` (Task 2); `BullBearState.last_book` (Task 3); `BullBearParams.book_size`, `.max_vetoes`, `.debate_size` (Task 4).
- Produces: `reviews.jsonl` lines with `verdicts`, `judge_verdicts`, `veto_cap`, `book` (no `picks`/`drops`); the judge's context `constraints == {"max_vetoes": cap, "reason_max_chars": n}` and no `held` key; the judge agent's only tool is `submit_verdicts`.

- [ ] **Step 1: Write the failing pipeline tests**

In `tests/strategies/bull_bear/test_bull_bear_pipeline.py`:

Change `from collections.abc import Callable` to `from collections.abc import Callable, Collection`.

Replace `picks_first` with:

```python
def verdicts(vetoed: Collection[str] = ()) -> Script:
    """The judge keeps every stock but `vetoed`."""

    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        symbols = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]]
        tools["submit_verdicts"]([{"symbol": s, "verdict": "veto" if s in vetoed else "keep", "reason": "bear wins" if s in vetoed else "bull wins"} for s in symbols])

    return step
```

In `_harness`, replace `"judge": FakeAgent(picks_first(5))` with `"judge": FakeAgent(verdicts())`.

Replace the happy-path test:

```python
def test_a_review_debates_the_top_15_and_buys_the_top_10_the_judge_keeps(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=True, abandoned_streak=0)
    research = h.agents["researcher"].calls
    assert [call["context"]["fact_sheet"]["symbol"] for call in research] == UNIVERSE[:15]
    assert all(call["tool_budget"] == 3 for call in research)
    bull_context = h.agents["bull"].calls[0]["context"]
    assert [stock["fact_sheet"]["symbol"] for stock in bull_context["stocks"]] == UNIVERSE[:15]
    assert bull_context["stocks"][0]["note"] == "2026-09-30: quarterly revenue up 12% y/y"
    assert h.agents["bear"].calls[0]["context"] == bull_context  # the same evidence, and not the bull's case
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in UNIVERSE[:10]]  # in the book's order
    state = h.store.load()
    assert state.last_completed_review == "2026-10-06" and state.abandoned_streak == 0
    assert [entry["symbol"] for entry in state.last_book] == UNIVERSE[:10]
    assert {entry["reason"] for entry in state.last_book} == {"bull wins"}
```

In `test_the_debaters_are_told_the_character_caps_that_reject_a_submission`, replace the `judge["context"]["constraints"]` assertion with:

```python
    assert judge["context"]["constraints"] == {"max_vetoes": 5, "reason_max_chars": 77}
```

Replace `test_the_review_log_records_the_whole_review_as_one_line`'s last lines (from the `picks` assertion on) with:

```python
    assert "picks" not in line and "drops" not in line
    assert line["verdicts"][0] == {"symbol": "S00", "verdict": "keep", "reason": "bull wins"} and len(line["verdicts"]) == 15
    assert line["judge_verdicts"] == {"keep": 15} and line["veto_cap"] == 5
    assert line["book"] == UNIVERSE[:10]
    stock_weights = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert list(stock_weights) == UNIVERSE[:10]
    assert sum(stock_weights.values()) == pytest.approx(0.98)
    assert all(0.04 <= weight <= 0.20 for weight in stock_weights.values())
    assert [order["symbol"] for order in line["orders"]] == UNIVERSE[:10]
```

Replace `test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out`:

```python
def test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20, "S19": 20, "OLD": 20}, params=BullBearParams(retention_rank=18))
    h.agents["judge"].script = verdicts(vetoed={"S16"})

    outcome = h.run()

    assert outcome.completed
    (line,) = h.log_lines()
    assert [stock["symbol"] for stock in line["debate_set"]] == [*UNIVERSE[:15], "S16"]
    assert line["debate_set"][-1] == {"symbol": "S16", "rank": 17, "held": True}
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}, {"symbol": "S19", "reason": "rank 20"}]
    assert "held" not in h.agents["judge"].calls[0]["context"]  # the fact sheet says held; the judge decides on merit
    assert {"symbol": "S16", "verdict": "veto", "reason": "bear wins"} in line["verdicts"]
    assert line["book"] == UNIVERSE[:10]
    assert {("OLD", "sell", 20.0), ("S19", "sell", 20.0), ("S16", "sell", 20.0)} <= set(h.orders)  # a vetoed holding is sold
    assert h.bars.calls == [(UNIVERSE, ["OLD", "S16", "S19"])]


def test_a_kept_holding_takes_its_slot_before_better_ranked_new_names(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20}, params=BullBearParams(retention_rank=18))

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == ["S16", *UNIVERSE[:9]]
    assert not any(order[0] == "S16" for order in h.orders)  # held at 10%, target ~9.8%: inside the band


def test_vetoed_stocks_are_replaced_by_the_next_ranked_survivors(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].script = verdicts(vetoed={"S00", "S03"})

    assert h.run().completed

    (line,) = h.log_lines()
    assert line["book"] == [symbol for symbol in UNIVERSE[:12] if symbol not in {"S00", "S03"}]
    assert line["judge_verdicts"] == {"keep": 13, "veto": 2}
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in line["book"]]


def test_the_veto_cap_shrinks_so_that_book_size_stocks_always_survive(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:12]})

    assert h.run().completed

    assert h.agents["judge"].calls[0]["context"]["constraints"]["max_vetoes"] == 2
    (line,) = h.log_lines()
    assert line["veto_cap"] == 2 and len(line["book"]) == 10


def test_a_judge_over_the_veto_cap_is_refused_and_corrects_itself_on_the_retry(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [verdicts(vetoed=set(UNIVERSE[:6]))]

    outcome = h.run()

    assert outcome.completed
    retry = h.agents["judge"].calls[1]
    assert retry["force_tool"] == "submit_verdicts" and "6 vetoes: veto at most 5" in retry["task"]
    assert h.log_lines()[0]["book"] == UNIVERSE[:10]
```

In the parametrize of `test_a_debater_without_a_valid_submission_abandons_the_review_with_no_order`, replace `("judge", "submit_picks")` with `("judge", "submit_verdicts")`.

Replace `test_a_debate_set_smaller_than_min_picks_abandons_the_review`:

```python
def test_a_debate_set_smaller_than_book_size_abandons_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:9]})

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == []
    (line,) = h.log_lines()
    assert line["stage"] == "data" and "fewer than book_size (10)" in line["error"]
```

In `test_a_broker_error_during_the_rebalance_logs_the_orders_already_sent`, replace the decision assertions with:

```python
    assert line["book"] == UNIVERSE[:10] and line["judge_verdicts"] == {"keep": 15}
    stock_targets = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert sorted(stock_targets) == UNIVERSE[:10] and "SHV" in line["targets"]
```

In `test_the_parking_instrument_is_never_forced_out_and_a_position_outside_the_universe_is`, replace `assert h.agents["judge"].calls[0]["context"]["held"] == []` with `assert "held" not in h.agents["judge"].calls[0]["context"]`.

- [ ] **Step 2: Write the failing prompt, strategy and backtest tests**

In `tests/strategies/bull_bear/test_bull_bear_prompts.py`:
- in the first parametrize, replace `(prompts.JUDGE_SYSTEM, "submit_picks")` with `(prompts.JUDGE_SYSTEM, "submit_verdicts")`;
- replace the retry test with:

```python
def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("submit_verdicts", "6 vetoes: veto at most 5 stocks and keep the others")

    assert "submit_verdicts" in text and "veto at most 5" in text
```

- add:

```python
def test_the_judge_gives_a_verdict_per_stock_and_a_trend_alone_is_no_reason_to_veto() -> None:
    assert "keep or veto" in prompts.JUDGE_SYSTEM and "max_vetoes" in prompts.JUDGE_SYSTEM
    assert "not a reason to veto" in prompts.JUDGE_SYSTEM
    assert "pick" not in prompts.JUDGE_SYSTEM.lower() and "pick" not in prompts.judge_task(300).lower()
```

In `tests/strategies/bull_bear/test_bull_bear_strategy.py`, in `test_initialize_creates_the_four_agents_with_only_their_own_tools`, replace `"judge": {"submit_picks"},` with `"judge": {"submit_verdicts"},`.

In `tests/strategies/bull_bear/test_bull_bear_backtest.py`:
- add the import `from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams`;
- replace `_judge` with:

```python
def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_verdicts"]([{"symbol": s["fact_sheet"]["symbol"], "verdict": "keep", "reason": "bull wins"} for s in context["stocks"]])
```

- in `_run`, add `settings=BullBearParams(book_size=5, max_vetoes=1),  # six stocks: debate the top 6, hold 5` to the `BullBearStrategy(...)` call, after `sector_of=...`;
- rename `test_a_backtest_reviews_on_the_two_tuesdays_and_buys_the_judges_picks` to `test_a_backtest_reviews_on_the_two_tuesdays_and_buys_the_book`;
- in `test_the_first_reviews_orders_fill_and_the_second_review_trades_nothing`, change the comments `# same picks, ...` to `# same book, ...` and `# the judge's five picks, each filled once` to `# the book's five stocks, each filled once`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/ -q`
Expected: FAIL in the pipeline, prompts, strategy and backtest tests (e.g. `submit_verdicts is not expected at this point of the review`, `KeyError: 'submit_verdicts'`, constraints still `min_picks`/`max_picks`)

- [ ] **Step 4: Implement the pipeline**

In `src/trading_agent_framework/strategies/bull_bear/pipeline.py`:

Module docstring, first two paragraphs:

```python
"""One review: data → debate set → researcher (one run per stock) → bull → bear → judge → book → sizing → rebalance.

The judge gives a verdict, keep or veto, on every debated stock; code builds the book (the kept holdings, then the
kept new names, by rank, up to `book_size`), validates (through the `HandoffRecorder`), sizes, remembers
(`StateStore`) and calls the `Rebalancer`. A debater stage that never yields a valid submission abandons the review:
no order at all, forced exits included. One stock's failed research only gives it the note "research unavailable".
Free text from an agent is logged and otherwise ignored. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md
and docs/superpowers/specs/2026-10-11-bull-bear-veto-judge-design.md.
"""
```

Imports: add `from trading_agent_framework.strategies.bull_bear.book import build_book` and replace `Picks` with `Verdicts` in the handoff import:

```python
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdicts
```

Replace the `min_picks` guard after the debate-set log line with:

```python
        if len(symbols) < params.book_size:
            raise ReviewAbandoned("data", f"only {len(symbols)} stocks in the debate set, fewer than book_size ({params.book_size})")
        veto_cap = min(params.max_vetoes, len(symbols) - params.book_size)  # at least book_size stocks always survive
```

Replace everything from `# 6. Judge.` down to (not including) `# A forced exit is sold by code, never debated` with:

```python
        # 6. Judge: a verdict, keep or veto, on every debated stock, on merit (no hint of which stocks code would hold).
        self._recorder.expect_verdicts(symbols, max_vetoes=veto_cap)
        context = {
            "current_datetime": moment,
            "stocks": [
                {
                    "fact_sheet": sheets[symbol],
                    "note": notes[symbol],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                }
                for symbol in symbols
            ],
            "constraints": {"max_vetoes": veto_cap, "reason_max_chars": params.reason_max_chars},
        }
        self._run_stage("judge", "submit_verdicts", judge_task(params.reason_max_chars), context)
        verdicts: Verdicts = self._recorder.submission
        vetoed = verdicts.vetoed
        vetoes = ", ".join(f"{verdict.symbol} ({verdict.reason})" for verdict in verdicts.verdicts if verdict.verdict == "veto")
        strategy.log_info(f"[judge] vetoes: {vetoes or 'none'}; kept {len(verdicts.verdicts) - len(vetoed)}")

        # 7. The book: the kept holdings, then the kept new names, by rank, up to book_size.
        book = build_book(debate, vetoed, params.book_size)
        strategy.log_info(f"[bull_bear] book: {', '.join(book)}")

        # 8. Sizing and execution.
        volatilities = {symbol: debate.stock(symbol).row.volatility for symbol in book}
        target_weights = capped_inverse_volatility(volatilities, total=params.investable, min_weight=params.min_weight, max_weight=params.max_weight)
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        # The decision goes into the record before the first order: an abandonment during execution still says why.
        record.update(
            verdicts=[asdict(verdict) for verdict in verdicts.verdicts],
            judge_verdicts=dict(Counter(verdict.verdict for verdict in verdicts.verdicts)),
            veto_cap=veto_cap,
            book=list(book),
            targets={**target.weights, params.parking_symbol: target.parking_weight},
        )
```

Replace `# 8. Remember and log.` and the `self._state.save(...)` line with:

```python
        # 9. Remember and log.
        reasons = {verdict.symbol: verdict.reason for verdict in verdicts.verdicts}
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, last_book=[{"symbol": symbol, "reason": reasons[symbol], "date": today} for symbol in book]))
```

- [ ] **Step 5: Implement the prompts and the agent's tool**

In `src/trading_agent_framework/strategies/bull_bear/prompts.py`, replace `JUDGE_SYSTEM` and `judge_task`:

```python
JUDGE_SYSTEM = (
    "You are the judge. For every stock, weigh the bull's case against the bear's and give your verdict: keep or veto. "
    "Do not trade: code builds the portfolio from the stocks you keep, sizes the positions and places the orders.\n\n"
    f"{_FACT_SHEET}\n"
    "Each stock has the researcher's note, the bull's conviction and argument, and the bear's risk, concern and argument. "
    "Veto a stock only when the bear's risk is concrete and specific to it (an event, a filing, guidance, a lawsuit, a "
    "balance-sheet problem) and outweighs the bull's case. A strong or extended price trend is not a reason to veto by "
    "itself: it is the momentum this portfolio buys. A held stock you veto is sold. You may veto none, and at most "
    "max_vetoes, given in the constraints.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_verdicts exactly once, with one object per stock: the symbol, the verdict (keep or "
    f"veto) and one short reason. Give a verdict for every stock in stocks and no other. {_SUBMIT_RULE}"
)
```

```python
def judge_task(max_chars: int) -> str:
    return (
        "Give your verdict, keep or veto, on every stock in the context and submit them. "
        f"Each reason is at most {max_chars} characters; a longer one rejects the whole submission. The current datetime is in the context."
    )
```

In `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py`:
- module docstring, replace "builds the fact sheets, sizes the judge's picks by inverse volatility and places every order" with "builds the fact sheets, builds a book of `book_size` stocks from those the judge keeps, sizes it by inverse volatility and places every order";
- in `initialize`, replace `tools=[submit["submit_picks"]]` with `tools=[submit["submit_verdicts"]]` for the judge.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q`
Expected: all pass

Run: `uv run pytest -q`
Expected: all pass (3,268 + the tests added in Tasks 1-5)

- [ ] **Step 7: Lint, type-check and commit**

```bash
uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
uv run pyright src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git add src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): the judge gives verdicts and code builds a book of book_size

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected from pyright: `0 errors, 0 warnings, 0 informations`.

---

### Task 6: remove the picker; docs

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/sizing.py` (docstring)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `tests/strategies/bull_bear/test_bull_bear_handoff.py`

**Interfaces:**
- Removes: `BullBearParams.shortlist_size`, `.min_picks`, `.max_picks` and their rules; `handoff.Choice`, `Picks`, `validate_picks`, `PICKS`, `HandoffRecorder.expect_picks`, the `submit_picks` tool.
- Nothing else changes.

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bull_bear/test_bull_bear_params.py`:

Replace the first assertion of `test_the_defaults_are_the_specs_and_valid` with:

```python
    assert (params.book_size, params.max_vetoes, params.debate_size, params.retention_rank) == (10, 5, 15, 35)
```

Replace the parameter list of `test_an_invalid_value_is_refused_with_a_reason` with:

```python
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"book_size": 30, "retention_rank": 40}, "book_size x min_weight must fit"),
        ({"book_size": 4}, "book_size x max_weight must reach"),
        ({"max_vetoes": 25}, "retention_rank must be above debate_size"),
        ({"retention_rank": 15}, "retention_rank must be above debate_size"),
        ({"book_size": 0}, "book_size and max_vetoes must be at least 1"),
        ({"max_vetoes": 0}, "book_size and max_vetoes must be at least 1"),
        ({"rebalance_time": "9:00"}, "rebalance_time"),
        ({"rebalance_weekday": 5}, "rebalance_weekday must be a weekday"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
        ({"parking_symbol": " "}, "parking_symbol must not be blank"),
        ({"min_yahoo_coverage": 0.0}, "min_yahoo_coverage must be in (0, 1]"),
        ({"yahoo_retry_delays": (60.0, -1.0)}, "yahoo_retry_delays"),
        ({"research_tool_budget": 0}, "research_tool_budget must be at least 1"),
    ],
```

Add:

```python
@pytest.mark.parametrize("field", ["shortlist_size", "min_picks", "max_picks"])
def test_the_pickers_parameters_are_gone(field: str) -> None:
    with pytest.raises(TypeError):
        BullBearParams(**{field: 5})
```

In `tests/strategies/bull_bear/test_bull_bear_handoff.py`:
- change the import to `from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdict, submit_tools`;
- delete `_choices` and the whole `# --- picks ---` section (`test_picks_within_the_bounds_with_every_unpicked_holding_dropped_are_recorded`, `test_drops_may_be_omitted_when_nothing_held_is_left_out`, `test_bad_picks_are_refused` and its parametrize);
- add:

```python
def test_the_judge_has_submit_verdicts_and_no_picker_tool_is_left() -> None:
    _, tools = _tools()

    assert set(tools) == {"submit_note", "submit_bull_case", "submit_bear_case", "submit_verdicts"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_handoff.py -v`
Expected: FAIL in `test_the_pickers_parameters_are_gone` (`DID NOT RAISE TypeError`) and `test_the_judge_has_submit_verdicts_and_no_picker_tool_is_left` (`submit_picks` still offered)

- [ ] **Step 3: Remove the picker**

In `src/trading_agent_framework/strategies/bull_bear/parameters.py`:
- delete the fields `shortlist_size`, `min_picks`, `max_picks` (the fields now start with `retention_rank`, then `book_size` and `max_vetoes`); change the `retention_rank` comment to `# a holding ranked worse (or unranked) is a forced exit; above debate_size`;
- delete these `problems` entries: `"shortlist_size must be at least 1"`, `"retention_rank must be above shortlist_size"`, `"1 <= min_picks <= max_picks <= shortlist_size"`, `"max_picks x min_weight must fit in 1 - cash_buffer"`, `"min_picks x max_weight must reach 1 - cash_buffer (the book is always fully invested)"`;
- reword the band rule's key to `"min_weight must be above rebalance_band (a new stock below the band would never be bought)"`.

In `src/trading_agent_framework/strategies/bull_bear/handoff.py`:
- constants: `NOTE, BULL, BEAR, VERDICT = "note", "bull", "bear", "verdict"` and `_TOOL_NAMES = {NOTE: "submit_note", BULL: "submit_bull_case", BEAR: "submit_bear_case", VERDICT: "submit_verdicts"}`;
- delete the `Choice` and `Picks` dataclasses, `validate_picks`, `HandoffRecorder.expect_picks`, the `elif stage == PICKS:` branch in `submit`, and the `submit_picks` tool and its dict entry;
- change the import to `from collections.abc import Callable, Mapping, Sequence` (`Collection` is no longer used);
- `submission` docstring: `"""The armed stage's valid submission (`Note`, `list[BullCase]`, `list[BearCase]` or `Verdicts`), else None."""`;
- `submit_tools` docstring: `"""The four submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""`.

In `src/trading_agent_framework/strategies/bull_bear/sizing.py`, first docstring line: `"""Target weights for the book (pure): inverse volatility, bounded per stock, summing to what is investable.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q`
Expected: all pass

Run: `grep -rn -E "picks|Picks|Choice|shortlist_size" src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear --include=*.py`
Expected: matches ONLY in these deliberate places, and nowhere else (anything else is a leftover to remove now):
- `state.py` and `test_bull_bear_state.py`: `last_picks`, the old state key that must still load;
- `test_bull_bear_params.py`: the parametrize of `test_the_pickers_parameters_are_gone`;
- `test_bull_bear_pipeline.py`: `assert "picks" not in line and "drops" not in line`.

- [ ] **Step 5: Update `strategies/bull_bear/CLAUDE.md`**

Make these seven replacements in the Architecture bullet (each OLD text appears exactly once), then append one sentence.

1. OLD:
```text
debates the top 15 plus the holdings ranked 16-35 (
```
NEW:
```text
debates the top `debate_size` = `book_size` + `max_vetoes` (10 + 5 = 15) plus the holdings ranked 16-35 (
```

2. OLD:
```text
`submit_note`, `submit_bull_case`, `submit_bear_case`, `submit_picks`;
```
NEW:
```text
`submit_note`, `submit_bull_case`, `submit_bear_case`, `submit_verdicts`;
```

3. OLD:
```text
then the judge (5-10 picks; `drops` must name exactly the held debate stocks not picked). Code sizes the picks by inverse volatility within [4%, 20%] summing to 98%
```
NEW:
```text
then the judge (a verdict, keep or veto, with a reason, on EVERY debated stock, on merit: no hint of which stocks code would hold; at most `min(max_vetoes, len(debate set) - book_size)` vetoes, so `book_size` stocks always survive). Code builds the book (`book.build_book`: the kept holdings, then the kept new names, each by rank, cut at `book_size`; a vetoed holding is sold), sizes it by inverse volatility within [4%, 20%] summing to 98%
```

4. OLD:
```text
or a debate set under `min_picks` abandons the review
```
NEW:
```text
or a debate set under `book_size` abandons the review
```

5. OLD:
```text
logged in `reviews.jsonl` with the judge's picks, drops and targets)
```
NEW:
```text
logged in `reviews.jsonl` with the judge's verdicts, the book and the targets)
```

6. OLD:
```text
last completed review date, abandoned streak, last picks, kept for the log
```
NEW:
```text
last completed review date, abandoned streak, last book (`last_book`; a file from the picker, with `last_picks`, still loads), kept for the log
```

7. OLD:
```text
`ReviewLog` appends one line per review (forced exits included) to `reviews.jsonl` in the run directory.
```
NEW:
```text
`ReviewLog` appends one line per review (forced exits included) to `reviews.jsonl` in the run directory: `verdicts`, `judge_verdicts` (keep/veto counts: a judge that keeps everything turns bull_bear into a code-only top-10 book), `veto_cap`, `book`.
```

Append at the end of the Architecture bullet:
```text
 The judge was a picker (5-10 of the top 15) until 2026-10-11: its entries added nothing in run `2026-10-10_085928` while its drops of held stocks worked (spec `docs/superpowers/specs/2026-10-11-bull-bear-veto-judge-design.md`).
```

Run: `grep -n -E "submit_picks|min_picks|5-10 picks" src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`
Expected: only the appended history sentence ("5-10 of the top 15") may mention the picker; no `submit_picks`, no `min_picks`.

- [ ] **Step 6: Final verification**

Run: `uv run pytest -q`
Expected: all pass, 0 failures

Run: `uv run ruff check src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear`
Expected: `All checks passed!`

Run: `uv run pyright src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear`
Expected: `0 errors, 0 warnings, 0 informations`

Run: `git diff --stat main...HEAD`
Expected: only paths under `src/trading_agent_framework/strategies/bull_bear/`, `tests/strategies/bull_bear/` and `docs/superpowers/`; nothing under `strategies/common/` or `strategies/cross_momentum/`.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "refactor(bull_bear): remove the picker; document the veto judge

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
