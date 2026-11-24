# search_news Titles-Only Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `search_news` tells the model when the returned articles carry no text, so it stops digging for text that does not exist.

**Architecture:** One check at the end of `_search` in `agents/tools/news.py`, on the returned articles only (no knowledge of GDELT or backtesting). bull_bear's `pin_search_news` passes extra result keys through unchanged, so it needs no edit.

**Tech Stack:** Python 3.14, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-24-gdelt-titles-only-note-design.md`

## Global Constraints

- Note text, verbatim: `Titles only: no summary or article text exists for these dates. include_content or a narrower window will not add any.`
- Result key: `"note"`.
- Note only when at least one article is returned and none has a non-empty `summary` or `content`.
- No change to the tool budget, the prompts, the tool docstring or `GdeltNewsProvider`.

## Review Focus

- An article with `summary: None` (key present, value None) counts as no text: covered by using truthiness, pinned in test `test_search_news_notes_titles_only_results` (one article has `summary: None`).
- An empty result (`count: 0`) gets no note: `test_search_news_adds_no_note_without_articles`.
- Error results (`{"error": ...}`) are untouched: they return before the check, no new test needed.
- A mix (one GDELT title, one Benzinga article with text across the cutoff) gets no note: `test_search_news_adds_no_note_when_any_article_has_text`.

---

### Task 1: Titles-only note in `search_news`

**Files:**
- Modify: `src/trading_agent_framework/agents/tools/news.py` (end of `_search`, plus one module constant)
- Test: `tests/agents/tools/test_news_tools.py`

**Interfaces:**
- Produces: module constant `TITLES_ONLY_NOTE: str` in `agents/tools/news.py`; `search_news` result may carry `"note": TITLES_ONLY_NOTE`.

- [ ] **Step 1: Write the failing tests**

Add a stub provider returning fixed articles, next to `_StubProvider`:

```python
class _ArticlesProvider:
    def __init__(self, articles: list[dict[str, object]]) -> None:
        self.articles = articles

    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
        return self.articles
```

Tests (import `TITLES_ONLY_NOTE` from `trading_agent_framework.agents.tools.news`):

```python
def test_search_news_notes_titles_only_results() -> None:
    articles = [{"id": "u1", "headline": "A", "summary": ""}, {"id": "u2", "headline": "B", "summary": None}]
    result = _tool(_provider_strategy(_ArticlesProvider(articles)))(symbols="VKTX", include_content=True)
    assert result["note"] == TITLES_ONLY_NOTE
    assert result["count"] == 2


def test_search_news_adds_no_note_when_any_article_has_text() -> None:
    articles = [{"id": "u1", "headline": "A", "summary": ""}, {"id": "b2", "headline": "B", "summary": "Text."}]
    assert "note" not in _tool(_provider_strategy(_ArticlesProvider(articles)))(symbols="VKTX")


def test_search_news_counts_content_as_text() -> None:
    articles = [{"id": "b1", "headline": "A", "summary": "", "content": "<p>Body</p>"}]
    assert "note" not in _tool(_provider_strategy(_ArticlesProvider(articles)))(symbols="VKTX", include_content=True)


def test_search_news_adds_no_note_without_articles() -> None:
    assert _tool(_provider_strategy(_ArticlesProvider([])))(symbols="VKTX") == {"count": 0, "articles": []}
```

Existing test `test_search_news_works_with_any_broker_that_provides_news` asserts the exact result of `_StubProvider`, whose article has no text, so it would now get the note. Its intent is "any broker's provider works", not "no note": give `_StubProvider`'s article `"summary": "Stub summary"` and update that test's expected dict to match. No assertion is weakened.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_news_tools.py -v`
Expected: import error on `TITLES_ONLY_NOTE` (all tests in the file fail to collect).

- [ ] **Step 3: Implement**

In `agents/tools/news.py`, add the constant `TITLES_ONLY_NOTE` (Global Constraints text) with a one-line comment: GDELT, the backtest source before 2026-04-29, has titles only. At the end of `_search`, build the result dict; when `articles` is non-empty and `not any(a.get("summary") or a.get("content") for a in articles)`, add `"note": TITLES_ONLY_NOTE`. Leave the `search_news` docstring as it is: it is the tool description the model sees, and the spec keeps it unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_news_tools.py tests/strategies/bull_bear -q`
Expected: all pass.

- [ ] **Step 5: Quality gate**

Run: `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov -q && uv run interrogate -v`
Expected: exit 0 each, 0 pyright errors, coverage >= 80%, interrogate >= 80%.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/news.py tests/agents/tools/test_news_tools.py docs/superpowers/specs/2026-11-24-gdelt-titles-only-note-design.md docs/superpowers/plans/2026-11-24-gdelt-titles-only-note.md
git commit -m "fix(news): search_news notes titles-only results so the agent stops digging for text"
```

(Per the user's workflow: `/simplify`, then `/ponytail:ponytail-review` on the staged diff, before this commit.)

### Afterwards (not code)

Judge the effect with 2-3 bull_bear 2Y runs per arm on the same window; the budget-refusal rate (5.4% of researcher runs on 11-24) is the first number to compare.
