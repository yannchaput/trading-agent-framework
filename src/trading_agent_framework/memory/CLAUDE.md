# memory/ -- agent memory (SQLite)

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- `memory/` -- agent memory (lumibot's `strategy.memory`): `records.py` (**pure**: ids, JSON, lean items, search scoring), `store.py` (`MemoryStore`, the only SQLite code: append-only `memory_events`, the `memory_index` projection, `memory_retrievals`; one DB per strategy and mode at `memory/<strategy>/<mode>/memory.sqlite`), `tools.py` (lumibot's 9 memory tools as plain typed functions, plus `agent_call_context` for provenance). No LangChain here: the agent layer wraps the tools.

## Gotchas

- **Memory tools are token-budgeted.** They return lean payloads (`{id, kind, status}`, or lean search items without metadata) and have one-line docstrings, because both reach the LLM on every call. Weigh the token cost before adding a field. Validation problems come back as `{"error": ...}`; database failures raise `MemoryStoreError`. `tools.py` deliberately has no `from __future__ import annotations` (the agent layer reads real annotations).
- **`MemoryStore` never knows `Strategy`.** It gets time from injected `now` / `wall_clock` callables (the strategy passes `clock.now`) and held positions as `HeldPosition` arguments to `compact_state`. Keep it that way so the store stays testable and backtest-clock friendly.
- **Backtesting memory is wiped** at the first `strategy.memory` access of every backtest run (`fresh=True`), so one run's lessons can't leak into the next. Paper and live memory are never wiped.
