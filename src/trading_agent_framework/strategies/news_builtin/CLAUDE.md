# strategies/news_builtin/ -- news_binary LLM agent

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- `news_builtin/` (`NewsBinaryStrategy` in `agent_news_binary.py`, registered as `"news_binary"` in `utils/strategy_factory.py`: an LLM agent built from `PrebuiltTools.all(self)` + `news_tools(self)`; system prompt is inline in `agent_news_binary.py`, grounding logic in `grounding.py`).

## Gotchas

- **`news_binary`'s grounding gate (`grounding.py`) grounds on the attempt, not on the content.** `search_news` must have been called with `include_content=True` this run (`_requested_full_content` binds args against the real signature via `inspect.signature`, so positional or keyword both count), but an empty-content response still grounds the run (there is no `_received_full_content` check). Both stricter and looser gates fail: requiring only that some `search_news` succeeded let the model decide straight off headlines without ever attempting a read, while requiring an article with non-empty `content` is sometimes impossible -- some days' news for the queried symbols is only terse Benzinga data-print wires that never carry a body, so the model retried article after article (100+ tool calls in one run) until it exhausted its turn/token budget with no decision recorded, which the tool-call-repair middleware and the retry-once mechanism cannot fix (no decision was ever attempted). The prompt (step 3) mirrors this: the call is mandatory, but "if this call comes back with no content... proceed to decide using the headline alone" rather than keep searching for content that may not exist that day. The refusal message says outright that an empty result counts and that the call itself is what's required; keep it that way: its earlier wording, "grounded in a specific article", read literally as requiring the article's substance, and glm-4.7-flash re-derived a "body-grounded article" rule that exists nowhere in this codebase and wrote it into `remember_decision`, which then resurfaced through `search_memory` in later runs.
