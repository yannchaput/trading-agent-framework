# search_news Instructions Apart From Facts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `search_news` answers in two blocks (instructions, facts), GDELT says titles-only on any call, and an opted-in agent's budgeted calls are labelled "<tool> call k of N"; bull_bear's researcher opts in and is told its budget.

**Architecture:** A frozen `ToolBudget` dataclass replaces `AgentManager.create(exempt_tools=...)`. The manager's `ToolBudgetLimit` middleware numbers allowed counted calls in `after_model` (one thread, before tools run) and adds the label to each answer in a `wrap_tool_call` hook. `GdeltNewsProvider` marks its articles; `search_news` turns the mark into an instruction and strips it from the facts.

**Tech Stack:** Python 3.14, LangChain agents middleware (`ToolCallLimitMiddleware`), pandas/pyarrow, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-25-news-search-instructions-design.md`

## Global Constraints

- Answer shape (non-error): `{"instructions": {...}, "facts": {"count": N, "articles": [...]}}`, `instructions` first and left out when empty; inside it `tool_call` comes first when present.
- Error answers keep `error` as a top-level key; a labelled error is `{"instructions": {"tool_call": "..."}, "error": "..."}`. The budget refusal is unchanged.
- Label text, verbatim: `f"{tool_name} call {k} of {limit}"`, e.g. `search_news call 3 of 7`.
- Titles-only instruction, verbatim, under `instructions.source`: `Titles only: no article text exists for these dates, and include_content cannot add any. Do not search again for the text of these articles: use the titles.`
- The titles-only instruction appears only when the answer has at least one article and every article is marked by GDELT; never on Benzinga articles, never on an empty answer.
- `ToolBudget` fields, verbatim: `exempt: tuple[str, ...] = ()`, `label_calls: bool = False`. The per-run limit stays `AgentHandle.run(tool_budget=N)`; `final_tools` stays its own `create` argument.
- Only bull_bear's researcher sets `label_calls=True`.
- Gate per the user's CLAUDE.md: ruff, ruff format, pyright, pytest --cov >= 80%; no `noqa`/`type: ignore` added. `uv sync --extra dashboard` in the worktree.

## Review Focus

- Two counted calls in one model turn get distinct, consecutive numbers in request order: `test_parallel_calls_in_one_turn_are_numbered_in_request_order` (Task 2).
- A memoized repeat search gets its own number while the tool's cached dict stays unlabelled: `test_a_repeated_identical_search_gets_its_own_label_and_the_cached_answer_is_untouched` (Task 3).
- A mixed answer across the 2026-04-29 cutoff (GDELT titles + Benzinga articles) gets no titles-only instruction: `test_a_mixed_answer_gets_no_titles_only_instruction` (Task 3).
- A labelled error answer still reads as a failure to `final_tools` (`_tool_succeeded`) and to `"error" in result` checks: `test_an_error_answer_is_labelled_beside_the_error` (Task 2) plus the unchanged final-tools tests.
- An exempt call in the same turn as counted calls is neither numbered nor labelled: `test_exempt_calls_are_not_labelled` (Task 2).

---

### Task 1: `ToolBudget` replaces `exempt_tools`

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (`create`, `_tool_budget_middleware` signature), `src/trading_agent_framework/agents/__init__.py` (export)
- Modify (call sites, `exempt_tools=[...]` -> `budget=ToolBudget(exempt=(...))`): `strategies/bull_bear/agent_bull_bear.py` (3 sites), `strategies/sector_pods/agent_sector_pods.py`, `strategies/bill_ackman/agent_bill_ackman.py`, `strategies/earnings_drift/agent_earnings_drift.py`
- Modify docs: `agents/CLAUDE.md`, `strategies/{bull_bear,bill_ackman,sector_pods}/CLAUDE.md` where they name `exempt_tools`
- Test: `tests/agents/test_agent_manager_tool_budget.py`, and the four strategy tests that pass or assert `exempt_tools` (`tests/strategies/{bill_ackman/test_ackman_strategy.py, bull_bear/test_bull_bear_strategy.py, earnings_drift/test_drift_strategy.py, sector_pods/test_sector_pods_strategy.py}`)

**Interfaces:**
- Produces: `@dataclass(frozen=True) class ToolBudget` in `agents/manager.py` (fields above, Google docstring), exported as `trading_agent_framework.agents.ToolBudget`; `AgentManager.create(..., budget: ToolBudget | None = None, final_tools=..., ...)` with `exempt_tools` removed.

- [ ] **Step 1: Write the failing test.** In `test_agent_manager_tool_budget.py`, change `_agent(..., exempt_tools=...)` to build `budget=ToolBudget(exempt=tuple(exempt_tools or ()))`, and add:

```python
def test_tool_budget_defaults_count_every_tool_and_label_nothing() -> None:
    assert ToolBudget() == ToolBudget(exempt=(), label_calls=False)
```

- [ ] **Step 2: Run** `uv run pytest tests/agents/test_agent_manager_tool_budget.py -q`. Expected: FAIL, `ImportError` on `ToolBudget`.
- [ ] **Step 3: Implement** `ToolBudget` and `create(budget=...)`; `_tool_budget_middleware(tool_names, budget, final_tools)` reads `budget.exempt`. Update the 6 strategy call sites (bull_bear has 3 `create` calls with `exempt_tools`) and the four strategy tests' expectations, and the CLAUDE.md mentions. No behaviour change.
- [ ] **Step 4: Run** `uv run pytest tests/agents tests/strategies -q`. Expected: all pass.
- [ ] **Step 5: Commit** `refactor(agents): ToolBudget packs the budget settings of AgentManager.create`.

### Task 2: Opt-in call labels in the budget middleware

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (`ToolBudgetLimit`)
- Test: `tests/agents/test_agent_manager_tool_budget.py`

**Interfaces:**
- Consumes: `ToolBudget.label_calls` (Task 1).
- Produces: with `label_calls=True` and a run budget, each allowed counted call's `ToolMessage` content, when it parses to a JSON object, gets `"instructions": {"tool_call": "<tool> call k of N", ...existing instructions}` as its first key. `k` = the run's count before the turn (base `run_tool_call_count`) + position among the turn's allowed counted calls. Numbers are stored by `tool_call["id"]` in `after_model` and applied in a `wrap_tool_call(self, request, handler)` method of `ToolBudgetLimit`; answers that are not JSON objects, exempt calls, refusals, and runs with `run_limit is None` are left unchanged.

- [ ] **Step 1: Write the failing tests** (a dict-returning tool `def quote(symbol: str) -> dict` returning `{"facts": {"symbol": symbol}}`, and one returning `{"error": "bad"}`; scripted turns as in the file's `_calls`):

```python
def test_counted_calls_are_labelled_with_their_number_and_the_budget() -> None:
    # two turns, one quote each, budget 3, label_calls=True
    assert [json.loads(c.result)["instructions"]["tool_call"] for c in quote_calls] == ["quote call 1 of 3", "quote call 2 of 3"]
    assert list(json.loads(quote_calls[0].result)) == ["instructions", "facts"]  # instructions first

def test_parallel_calls_in_one_turn_are_numbered_in_request_order() -> None:
    # one turn: quote A, quote B, quote C; budget 7
    assert labels_by_symbol == {"A": "quote call 1 of 7", "B": "quote call 2 of 7", "C": "quote call 3 of 7"}

def test_an_error_answer_is_labelled_beside_the_error() -> None:
    assert json.loads(result) == {"instructions": {"tool_call": "broken call 1 of 2"}, "error": "bad"}

def test_exempt_calls_are_not_labelled() -> None:
    # same turn: quote (counted) + submit (exempt) -> submit's answer has no "instructions"

def test_labels_are_opt_in() -> None:
    # label_calls=False (default): answers are unchanged

def test_no_label_without_a_run_budget() -> None:
    # label_calls=True, run() without tool_budget: answers unchanged
```

- [ ] **Step 2: Run** the file. Expected: the new tests FAIL (no `instructions` key).
- [ ] **Step 3: Implement** numbering in `ToolBudgetLimit.after_model` and the `wrap_tool_call` hook (parse `ToolMessage.content` with `json.loads`; on a dict, rebuild it with `instructions` first; leave the message alone otherwise). The middleware takes `label_calls` from `_tool_budget_middleware`.
- [ ] **Step 4: Run** `uv run pytest tests/agents -q`. Expected: all pass, including `test_agent_manager_final_tools.py` unchanged.
- [ ] **Step 5: Commit** `feat(agents): opt-in "<tool> call k of N" labels on budgeted calls`.

### Task 3: `search_news` answers in two blocks; GDELT says titles-only

**Files:**
- Modify: `src/trading_agent_framework/backtesting/news.py` (`_article`), `src/trading_agent_framework/agents/tools/news.py` (`TITLES_ONLY_NOTE` text, `_search` result), `scripts/tests/smoke_news.py` (`result["facts"]["articles"]`)
- Test: `tests/backtesting/test_gdelt_news.py`, `tests/backtesting/test_cutoff_news.py` (if it asserts article dicts), `tests/agents/tools/test_news_tools.py`

**Interfaces:**
- Produces: GDELT article dict = `{"id", "headline", "source", "created_at", "symbols", "titles_only": True}` (no `summary`). `search_news` answer per Global Constraints; the `titles_only` key never appears in `facts`. `TITLES_ONLY_NOTE` holds the new instruction text.

- [ ] **Step 1: Write the failing tests.**
  - `test_gdelt_news.py`: `test_an_article_is_marked_titles_only_and_has_no_summary` — `set(article) == {"id", "headline", "source", "created_at", "symbols", "titles_only"}` and `article["titles_only"] is True`.
  - `test_news_tools.py` (replace PR #43's five note tests): `test_marked_articles_get_the_titles_only_instruction_on_any_call` (include_content False and True both give `result["instructions"] == {"source": TITLES_ONLY_NOTE}`), `test_facts_hold_count_and_articles_without_the_marker`, `test_unmarked_articles_get_no_instruction_even_with_an_empty_summary`, `test_a_mixed_answer_gets_no_titles_only_instruction`, `test_an_empty_answer_has_facts_only` (`== {"facts": {"count": 0, "articles": []}}`), `test_errors_stay_flat` (`{"error": ...}` unchanged), `test_a_repeated_identical_search_gets_its_own_label_and_the_cached_answer_is_untouched` (run the tool through an `AgentManager` agent with `ToolBudget(label_calls=True)` and a scripted model calling the same search twice: labels "search_news call 1 of 7" and "... 2 of 7"; the dict returned by the tool has no `tool_call`). Update the other existing assertions on `result["count"]`/`result["articles"]` to `result["facts"][...]`.
- [ ] **Step 2: Run** `uv run pytest tests/agents/tools/test_news_tools.py tests/backtesting -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** `_article` adds the marker and drops `summary`; `_search` builds `facts` from the articles with `titles_only` popped (on copies; never mutate the provider's dicts), `instructions = {"source": TITLES_ONLY_NOTE}` when `articles` and all were marked, and returns `{"instructions": ..., "facts": ...}` or `{"facts": ...}`. Update `smoke_news.py`. Update `backtesting/CLAUDE.md`/`agents/CLAUDE.md` if they describe the answer or the note.
- [ ] **Step 4: Run** `uv run pytest tests -q`. Expected: all pass (fix any other test that reads `result["articles"]` from `search_news`).
- [ ] **Step 5: Commit** `feat(news): search_news answers in instructions and facts; GDELT says titles-only on any call`.

### Task 4: bull_bear's researcher opts in and is told its budget

**Files:**
- Modify: `strategies/bull_bear/pinned_news.py` (`symbols_searched` into `instructions`), `strategies/bull_bear/prompts.py` (`researcher_system(news: bool, search_budget: int)`), `strategies/bull_bear/agent_bull_bear.py` (researcher: `budget=ToolBudget(exempt=(...), label_calls=True)`, prompt gets `self.settings.research_tool_budget`), `strategies/bull_bear/CLAUDE.md`
- Test: `tests/strategies/bull_bear/test_bull_bear_pinned_news.py`, `test_bull_bear_prompts.py`, `test_bull_bear_strategy.py`

**Interfaces:**
- Consumes: Task 1 `ToolBudget`, Task 3 answer shape.
- Produces: `researcher_system(news: bool, search_budget: int) -> str`; `RESEARCHER_SYSTEM` built with the default `BullBearParams().research_tool_budget`.

- [ ] **Step 1: Write the failing tests.**
  - pinned: `test_a_rewritten_search_says_so_in_instructions` — `result["instructions"]["symbols_searched"] == "VKTX"`, `"symbols_searched" not in result["facts"]`, and the wrapped tool's returned dict is not mutated.
  - prompts: `test_the_researcher_is_told_its_search_budget_and_the_two_blocks` — `"at most 7" in researcher_system(True, 7)`, mentions `instructions` and `facts`; `researcher_system(False, 7)` has no budget sentence.
  - strategy: the researcher is created with `ToolBudget(exempt=("submit_note", "count_characters", "get_income_statement", "get_balance_sheet"), label_calls=True)`; the bull/bear/judge agents keep `label_calls=False`.
- [ ] **Step 2: Run** `uv run pytest tests/strategies/bull_bear -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** (copy the answer, put `symbols_searched` inside `instructions`, keep `instructions` first; prompt sentences in the file's style).
- [ ] **Step 4: Run** `uv run pytest tests/strategies/bull_bear -q`, then the full gate: `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov -q`. Expected: all pass, coverage >= 80%.
- [ ] **Step 5: Commit** `feat(bull_bear): the researcher opts in to call labels and is told its search budget`.
