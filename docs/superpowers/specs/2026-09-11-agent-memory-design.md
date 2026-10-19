# Agent memory — design

- **Date:** 2026-09-11
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md` ("Memory management" bullet); lumibot reference
  [agents_memory.html](https://lumibot.lumiwealth.com/agents_memory.html).

## 1. Context

Lumibot gives trading agents a persistent, agent-writable memory: notes, research proposals, risk notes,
decisions, lessons and investment theses, stored in SQLite and searchable by the agent through built-in
tools. This spec ports that memory layer to this framework.

Reference implementation (read in full while designing):
- `lumibot/components/memory/store.py` — `MemoryStore` (event log + projection + retrievals).
- `lumibot/components/agents/builtins.py` — the 9 memory tool bindings (`_bind_memory_*`,
  `_bind_remember_*`, `_bind_*_thesis`) and `_agent_memory_context_kwargs`.
- `lumibot/components/agents/manager.py` — `_memory_state` (compact state injected into prompts) and the
  `position_order_without_memory_thesis` warning.

**Not covered by this spec** (explicitly out of scope, per discussion):
- LangChain wrapping of the tools. No agent layer exists yet ("Agentic framework" TODO item). Tools ship as
  plain typed Python functions that the agent layer will wrap with `@tool` / `StructuredTool.from_function`.
  No `langchain` dependency is added.
- Injecting `compact_state()` into agent prompts, the `position_order_without_memory_thesis` check, and
  calling `record_order_submitted` from an order tool. These belong to the agent-layer / tools TODO items;
  the store APIs they need are delivered here.
- Parquet exports (lumibot rewrites 3 Parquet files after every write; needs pyarrow). SQLite is the only
  store.
- Lumibot's BotSpot provenance fields (`deployment_id`, `run_id`) — lumibot-cloud specific.
- Full-text search (FTS5). Lumibot's term-count scoring is kept (see §5).

## 2. Decisions taken during brainstorming

| Topic | Decision |
|---|---|
| DB location | `<project_root>/memory/<strategy>/<mode>/memory.sqlite` — mirrors `logs/<strategy>/<mode>/`. |
| Scope | `MemoryStore` + the 9 tools as plain functions. |
| Backtesting | Fresh (empty) DB on every backtest run. Paper/live DBs are never wiped. |
| Tool payloads | Lean: same tool names and parameters as lumibot, trimmed return values. |
| Approach | Faithful port of lumibot's schema/event model, with the fixes listed in §3.3. |

## 3. Architecture

### 3.1 Modules

New package `src/trading_agent_framework/memory/`, following the repo's pure / I/O split:

| Module | Role | Depends on |
|---|---|---|
| `memory/records.py` | **Pure**, no I/O: `safe_name`, `normalize_symbol`, `new_memory_id`, JSON encode/decode (`Decimal` → `str`), `compact_text`, row → lean item, search scoring. | stdlib only |
| `memory/store.py` | `MemoryStore` — the only module touching SQLite (stdlib `sqlite3`). Schema, event append, projection upsert, search + retrieval log, `compact_state`. Also `memory_db_path()` and the `HeldPosition` dataclass. | `records.py`, `entities`, `utils/errors.py` |
| `memory/tools.py` | `memory_tools(store) -> list[Callable]` and the `agent_call_context()` context manager. | `store.py` |
| `memory/__init__.py` | Exports `MemoryStore`, `HeldPosition`, `memory_db_path`, `memory_tools`, `agent_call_context`. | — |

Plus:
- `utils/errors.py`: `MemoryStoreError(TradingFrameworkError)` and
  `MemoryValidationError(MemoryStoreError, ValueError)`.
- `core/strategy.py`: lazy `Strategy.memory` property (§7).
- `.gitignore`: `/memory/`.

### 3.2 Boundaries

- **`MemoryStore` does not know `Strategy`.** Lumibot's store holds the strategy and calls
  `get_positions` / `get_last_price` / `get_datetime` itself. Ours receives:
  - `now: Callable[[], datetime]` — the event `timestamp` (the strategy passes `strategy.clock.now`, so
    backtests stamp simulated time);
  - `wall_clock: Callable[[], datetime]` — lumibot's `wall_time` column; defaults to real UTC time;
  - held positions as an explicit argument to `compact_state` (§5.3).
- **Tools never touch SQLite**; they only call `MemoryStore` methods.
- **Provenance through a context variable.** Like lumibot's `current_agent_tool_context()`, tools read
  `agent_name` / `model_call_id` from a `contextvars.ContextVar` set by
  `with agent_call_context(agent_name=..., model_call_id=...):`. The future agent runtime wraps each model
  call in it; until then both fields are `NULL`. The model never sees these parameters.
- **Error wrapping.** Every `sqlite3.Error` is re-raised as `MemoryStoreError` at the store boundary
  (repo rule: no raw library exception escapes). Bad input from the agent (unknown/closed thesis id, empty
  text, non-list tags) raises `MemoryValidationError`.
- **Threading.** Memory is used only on the strategy thread (the Alpaca stream thread never touches it).
  Each store operation opens a connection, runs **one** transaction, commits, and closes the connection.
  WAL journal mode (as lumibot) so the DB can be inspected with `sqlite3` while the bot runs.

### 3.3 Fixes relative to lumibot

1. **Atomic writes.** Lumibot appends the event and upserts the projection on two separate connections and
   computes `sequence` with `MAX(sequence)+1` outside a transaction. Here event insert, `sequence`
   allocation and projection upsert are one transaction.
2. **Thesis-id validation.** Lumibot's `update_thesis` / `close_thesis` on an unknown id silently create a
   projection row with that id. Here they raise `MemoryValidationError` unless the id is an index row with
   `kind == "thesis"` and `status == "open"` (so a closed thesis, or a `remember(kind="thesis")` note, is
   rejected).
3. **Search de-duplication.** Lumibot returns both an index row and the event row that produced it (their
   seen-ids use different prefixes, so they never collide). Here the event row that is an index row's
   `latest_event_id` is excluded (§5.2).
4. **Connections are closed.** Lumibot's `with sqlite3.connect(...) as conn:` only commits; it never closes.
5. **Short memory ids.** `<kind>_<8 hex>` (e.g. `thesis_7f3a9c21`) instead of `<kind>_<32 hex>`: the model
   must copy thesis ids back into `update_thesis` / `close_thesis`. On the (negligible) collision, a new id
   is drawn. Event and retrieval ids keep full `uuid4().hex` (see §10.3).

## 4. Storage

### 4.1 Location

`memory_db_path(project_root, strategy_name, mode) -> Path` returns
`project_root / "memory" / safe_name(strategy_name) / mode.value / "memory.sqlite"`.

`safe_name` is lumibot's `_safe_name` (`[^A-Za-z0-9_.=-]+` → `_`, stripped of `_`), which also blocks
`../` path traversal. The raw strategy name is stored in the `strategy` columns.

### 4.2 Schema

Identical to lumibot (same tables, columns, indexes, `SCHEMA_VERSION = 1`) so existing SQL against lumibot
DBs keeps working:

- `memory_events` — append-only ledger: `event_id` PK, `sequence` UNIQUE, `timestamp`, `wall_time`,
  `strategy`, `event_type`, `subject_type`, `subject_id`, `symbol`, `agent_name`, `model_call_id`,
  `retrieval_id`, `text`, `tags_json`, `metadata_json`, `payload_json`, `payload_hash` (sha256 of
  `payload_json`), `schema_version`. Indexes on `(subject_type, subject_id)`, `symbol`, `event_type`.
- `memory_index` — projection (current state per memory): `memory_id` PK, `latest_event_id`, `kind`,
  `status`, `symbol`, `text`, `tags_json`, `metadata_json`, `created_at`, `updated_at`, `schema_version`.
  Indexes on `kind`, `symbol`, `status`. Upsert keeps the original `created_at`.
- `memory_retrievals` — provenance of every search: `retrieval_id` PK, `timestamp`, `wall_time`,
  `strategy`, `agent_name`, `model_call_id`, `query`, `kind`, `symbol`, `status`, `result_limit`,
  `candidate_ids_json`, `selected_ids_json`, `rendered_text`, `schema_version`. Indexes on `symbol`,
  `agent_name`.

Timestamps are ISO-8601 strings (`timestamp` from `now()`, `wall_time` from `wall_clock()` in UTC with a
`Z` suffix, as lumibot).

### 4.3 Fresh reset

`MemoryStore(db_path, *, strategy_name, now, wall_clock=<utc now>, fresh=False)`. The constructor creates
the parent directory; with `fresh=True` it first deletes `memory.sqlite`, `memory.sqlite-wal` and
`memory.sqlite-shm` if present. Then it creates the schema (`CREATE ... IF NOT EXISTS`).

## 5. Events and the Python API

### 5.1 Events

Same event types as lumibot, produced by the same `MemoryStore` methods. "Projected" events also upsert
`memory_index`; "history-only" events are written to `memory_events` alone (still searchable, §5.2).

| Method | `event_type` | kind / status | Projected |
|---|---|---|---|
| `remember(text, *, kind="memory", tags=None, metadata=None)` | `memory.created` | `<kind>` / `active` | yes |
| `remember_proposal(text, *, symbol=None, action=None, tags=None, metadata=None)` | `proposal.recorded` | `proposal` / `proposed` | yes |
| `remember_risk_note(text, *, symbol=None, tags=None, metadata=None)` | `risk_note.recorded` | `risk_note` / `active` | yes |
| `remember_decision(text, *, symbol=None, action=None, evidence=None, tags=None)` | `decision.recorded` | `decision` / `recorded` | yes |
| `remember_lesson(text, *, symbol=None, outcome=None, tags=None)` | `lesson.validated` if `outcome.get("validated") is True`, else `lesson.proposed` | `lesson` / `validated` or `proposed` | yes |
| `open_thesis(text, *, symbol=None, tags=None, metadata=None)` | `thesis.opened` | `thesis` / `open` | yes |
| `update_thesis(thesis_id, text, *, metadata=None)` | `thesis.updated` | `thesis` / `open` | yes (same `memory_id`) |
| `close_thesis(thesis_id, text, *, outcome=None)` | `thesis.closed` | `thesis` / `closed` | yes (same `memory_id`) |
| `compact_state(...)` side effect | `thesis.outcome_observed` | `thesis_outcome` / `observed` | no |
| `record_order_submitted(order, *, metadata=None)` | `order.submitted` | `order` / `submitted` | no |
| `record_warning(text, *, kind="agent_warning", symbol=None, metadata=None)` | `<kind>` (e.g. `position_order_without_memory_thesis`) | `warning` / none | no |

For history-only events the kind / status are those stored in the event payload (lumibot parity), which is
what search filters on (§5.2). `record_warning`'s payload is `{"warning": text, "metadata": ...}`, so its
kind falls back to `subject_type = "warning"`.

Every write method also takes keyword-only `agent_name=None, model_call_id=None, retrieval_id=None`
(lumibot parity); tools fill the first two from `agent_call_context`.

Metadata follows lumibot:
- `remember_proposal`: `{"symbol", "action", **metadata}`; `remember_risk_note`: `{"symbol", **metadata}`;
  `remember_decision`: `{"symbol", "action", "evidence": evidence or {}}`;
  `remember_lesson`: `{"symbol", "outcome": outcome or {}}`.
- `open_thesis`: `{"symbol", **metadata, "status": "open"}`. `update_thesis`:
  `{"thesis_id", **metadata, "status": "open"}`, symbol inherited from the existing thesis when not given.
  `close_thesis`: `{"thesis_id", "outcome": outcome or {}, "status": "closed"}`, symbol inherited.
- `symbol` is upper-cased everywhere; when not passed explicitly it is taken from `metadata["symbol"]`.
- Tags: `list[str]` or `None`.

`record_order_submitted(order: Order, ...)` takes this repo's `Order` entity (lumibot takes a loose bag of
kwargs). `subject_id = f"order_{order.identifier}"`. Payload:
`{"kind": "order", "status": "submitted", "symbol", "side", "quantity", "order_type", "asset_type",
"order": {identifier, client_order_id, status, time_in_force, limit_price, stop_price, stop_limit_price,
trail_price, trail_percent} (non-None only), "metadata", "decision_id"}`. Text as lumibot:
`"Submitted order buy 10 SPY as market"`. **Decision provenance:** when `model_call_id` is set, the store
looks up the latest `decision.recorded` event with the same `model_call_id` (and `agent_name`, if set) and
stores its `subject_id` as `decision_id` — this replaces lumibot's separate `decision_provenance()` call.
`Decimal` values serialise as strings.

`get(memory_id) -> dict | None` returns the full projection row (all columns, JSON decoded) — for strategy
code and tests, not for the model.

### 5.2 Search

`search(query, *, limit=10, kind=None, symbol=None, status=None, agent_name=None, model_call_id=None)`,
lumibot's semantics:

- **Candidates:** every `memory_index` row matching the `kind` / `symbol` / `status` filters, plus every
  `memory_events` row with a `subject_id` that matches `symbol` and whose `event_id` is **not** the
  `latest_event_id` of an index row (i.e. superseded thesis texts, orders, warnings, outcome observations).
  Event candidates take `kind` / `status` from their payload (falling back to `subject_type`) and are
  filtered on them.
- **Score:** number of whitespace-separated query terms (lower-cased) found as substrings of the item's
  JSON (lower-cased). Empty query → score 1 for every candidate. Candidates with score 0 are dropped.
- **Order:** `(score, timestamp)` descending; the first `limit` are selected. `limit` is clamped to ≥ 1 in
  the store (lumibot parity); the tool clamps further (§6).
- **Retrieval log:** one `memory_retrievals` row per call (all candidate ids, selected ids, rendered text
  `"<kind> <symbol> <status>: <text>"` joined by blank lines, as lumibot).
- **Returns** `{"count": <number of scored candidates>, "retrieval_id": ..., "results": [lean items]}`.

Scoring loads candidates into Python. Per-strategy memory stays in the hundreds-to-thousands of rows, where
a search costs a few milliseconds; FTS5 is the upgrade path if that ever changes.

### 5.3 Compact state

`compact_state(held: Sequence[HeldPosition] = (), *, max_theses=8, max_lessons=8,
max_chars_per_item=900, update_open_thesis_outcomes=True) -> dict`.

`HeldPosition(symbol: str, quantity: Decimal, last_price: Decimal | None)` is a frozen dataclass; the caller
(the future agent layer) builds it from `strategy.get_positions()` / `get_last_prices()`, skipping
zero-quantity positions.

Returns:
```python
{
    "as_of": <now() ISO>,
    "held_symbols": [sorted, upper-cased],
    "open_theses": [lean items],        # kind=thesis, status=open, newest first, ≤ max_theses
    "validated_lessons": [lean items],  # kind=lesson, status=validated, newest first, ≤ max_lessons
    "retrieval_policy": "If you hold a symbol and plan to add, reduce, or sell it, "
                        "call search_memory for its open thesis first.",
}
```

Lumibot's `current_position_rationales` is dropped: it is `open_theses` filtered to held symbols, i.e. the
same items twice. `schema_version` / `strategy` are dropped from the dict (constant per DB).

Side effect (lumibot parity): for each open thesis whose symbol is held, if no `thesis.outcome_observed`
event exists for that thesis with a `timestamp` on the current `now()` date, append one with text
`"Observed open thesis for SPY: quantity=10 last_price=512.3 market_value=5123.0"` and metadata
`{"thesis_id", "symbol", "status": "observed", "position": {symbol, quantity, last_price, market_value},
"observed_date"}`. Lumibot swallows every error here; ours lets `MemoryStoreError` propagate — a failing
DB is not something to hide from the caller.

### 5.4 Lean item shape

What the model sees (search results, compact state):
`{"id", "kind", "status", "symbol", "updated_at", "text"}`, plus `"tags"` only when non-empty.
`updated_at` is truncated to the minute (`2026-09-11T14:32`). `text` is truncated by `compact_text`
(lumibot's `"... [truncated]"` suffix) to 900 chars in compact state and 500 chars in search results.
Event-sourced search items use the `event_id` as `id` and add `"event_type"`.

## 6. Tools

`memory_tools(store: MemoryStore) -> list[Callable[..., dict]]` returns 9 closures. Each has lumibot's tool
name as `__name__`, lumibot's parameters (names, order, defaults), full type hints, and a **one-line
docstring** (the future LangChain layer sends it as the tool description on every call).

| Tool | Parameters (identical to lumibot) | Docstring |
|---|---|---|
| `remember` | `text: str, kind: str = "memory", tags: list[str] \| None = None` | Store a memory or note. |
| `search_memory` | `query: str, limit: int = 10, kind: str \| None = None, symbol: str \| None = None, status: str \| None = None` | Search memories, lessons, decisions and theses; filter by symbol/status to find a held position's open thesis. |
| `remember_proposal` | `text: str, symbol: str \| None = None, action: str \| None = None, tags: list[str] \| None = None` | Record a non-final trade idea (not an executed decision). |
| `remember_risk_note` | `text: str, symbol: str \| None = None, tags: list[str] \| None = None` | Record a compact risk note or bear case. |
| `remember_decision` | `text: str, symbol: str \| None = None, action: str \| None = None` | Record an actual trading decision. |
| `remember_lesson` | `text: str, symbol: str \| None = None` | Record a compact lesson for future runs. |
| `open_thesis` | `text: str, symbol: str \| None = None, tags: list[str] \| None = None` | Open an investment thesis. |
| `update_thesis` | `thesis_id: str, text: str` | Update an open thesis. |
| `close_thesis` | `thesis_id: str, text: str` | Close a thesis and record its outcome. |

Behaviour:
- **Write tools return** `{"id", "kind", "status"}`.
- **`search_memory` returns** the store's `{"count", "retrieval_id", "results"}` (lumibot also echoes
  `query` / `kind` / `symbol` / `status` back; dropped). `limit` is clamped to `1..20`.
- **Validation errors** (`MemoryValidationError`: unknown/closed thesis id, empty `text`, non-list `tags`)
  are returned as `{"error": "<message>"}` so the model can correct itself. Any other exception
  (`MemoryStoreError` from a disk/DB failure, programming errors) propagates.
- **`remember_decision.mutates_trading = True`** — the function attribute mirrors lumibot's
  `metadata={"mutates_trading": True}`. Which agents receive which tools is the agent layer's decision.
- Every tool passes `agent_name` / `model_call_id` from `agent_call_context` to the store.

`agent_call_context(agent_name: str | None = None, model_call_id: str | None = None)` is a context manager
over a module-level `ContextVar`; it restores the previous value on exit (nesting-safe).

## 7. Strategy wiring

```python
@property
def memory(self) -> MemoryStore:
    """This strategy's agent memory for the current trading mode (lumibot's `strategy.memory`)."""
    if self._memory is None or self._memory_mode is not self.trading_mode:
        root = self.project_root if self.project_root is not None else find_project_root()
        self._memory = MemoryStore(
            memory_db_path(root, self.name, self.trading_mode),
            strategy_name=self.name,
            now=self.clock.now,
            fresh=self.is_backtesting,
        )
        self._memory_mode = self.trading_mode
    return self._memory
```

- **Lazy:** a strategy that never touches memory creates no DB file.
- **Per mode:** `_run_trading` reassigns `trading_mode`; if memory was opened under another mode, the
  property reopens the right DB (paper memories can never land in the live DB).
- **Backtesting:** each backtest run constructs a new `Strategy`, so its first `.memory` access wipes and
  recreates `memory/<strategy>/backtesting/`. The previous run's DB remains on disk for inspection until
  the next run. `run_backtesting` is not implemented yet; the behaviour is covered by tests on
  `is_backtesting` directly.
- `find_project_root` comes from `config/env.py` (the same helper the logging setup uses).

## 8. Testing

pytest, `tmp_path`, fake `now` / `wall_clock` callables; no network, no `MagicMock` (repo convention).

- `tests/memory/test_records.py` — `safe_name`, `normalize_symbol`, id format, JSON with `Decimal`,
  `compact_text`, lean item shape (minute truncation, tags only when non-empty), scoring (empty query,
  case-insensitivity, substring match).
- `tests/memory/test_store.py` —
  - each method's `event_type` / kind / status / projected-or-not, metadata shapes, symbol upper-casing and
    fallback from `metadata["symbol"]`;
  - projection upsert keeps `created_at`, updates `updated_at` / `latest_event_id`; `sequence` increments;
  - `update_thesis` / `close_thesis` reject unknown ids, closed theses and `remember(kind="thesis")`
    notes; symbol inheritance;
  - `remember_lesson` validated vs proposed;
  - search: candidate set, de-duplication, filters, ordering, empty query, limit, `memory_retrievals` row;
  - `compact_state`: content, limits, truncation, one `thesis.outcome_observed` per thesis per day, none
    for unheld symbols;
  - `record_order_submitted`: payload, text, `decision_id` lookup by `model_call_id`;
  - `record_warning` event type from `kind`;
  - `fresh=True` deletes all three files; `fresh=False` preserves data across instances;
  - `sqlite3.Error` surfaces as `MemoryStoreError` (e.g. DB path is a directory).
- `tests/memory/test_tools.py` — the 9 names; parameter names/order/defaults equal the table in §6;
  single-line docstrings; full annotations; `mutates_trading` only on `remember_decision`; lean return
  shapes; error dicts; `limit` clamp; `agent_call_context` values stored on events, restored after exit.
- `tests/core/test_strategy.py` (additions) — DB path per mode under `project_root`, lazy creation, the
  backtesting wipe, paper/live preserved, reopen on mode change, events stamped with `strategy.clock.now()`.

No smoke script: memory is local-only, fully covered by the automated suite.

## 9. Documentation

- `CLAUDE.md`: `memory/` bullet in the architecture list; gotcha "memory tools return lean payloads — weigh
  the token cost before adding fields"; note that `MemoryStore` must stay independent of `Strategy`.
- `TODO.md`: strike "Memory" in the MIGRATION list once implemented.

## 10. Amendments made while planning

These override the sections they name.

1. **§5.2 / §5.4 — superseded history.** An event candidate whose `event_type` is a projected type
   (`memory.created`, `proposal.recorded`, `risk_note.recorded`, `decision.recorded`, `lesson.proposed`,
   `lesson.validated`, `thesis.opened`, `thesis.updated`, `thesis.closed`) is by construction an older
   version of a memory (the latest one is excluded), so its search `status` is `"superseded"` instead of
   its payload status. Otherwise `search_memory(status="open")` would return the original `thesis.opened`
   event of a thesis that has since been closed. Lean event items carry `"event_type"` and `"memory_id"`
   (the `subject_id`) so the model can tie history back to its memory.
2. **§5.1 — thesis tags.** `update_thesis` / `close_thesis` keep the thesis's existing tags (lumibot writes
   `tags=None`, wiping them from the projection).
3. **§3.3 #5 — id wording.** Event and retrieval ids keep full `uuid4().hex` because no tool takes them as
   input (they are not "never shown": `search_memory` returns a `retrieval_id`, and event-sourced search
   items use their `event_id` as `id`).
4. **§4.1 — `safe_name`.** A name that is empty or made only of dots (`""`, `"."`, `".."`) becomes
   `"strategy"`, so no path segment can be `..`.
5. **§5.1 — `remember`.** An empty `kind` raises `MemoryValidationError`.
6. **§4.2 — JSON.** Stored with `ensure_ascii=False` (UTF-8 text stays readable and searchable as typed).
7. **§5.3 — `HeldPosition`.** `last_price` defaults to `None`; `market_value` is a property
   (`quantity * last_price`, or `None`).
8. **§6 — thesis error message.** Unknown id (or not a thesis): `unknown thesis_id '<id>'`. Existing but not
   open: `thesis '<id>' is not open (status: <status>)`.
9. **§8 — test layout.** Test files are `tests/memory/test_memory_records.py`, `test_memory_store.py`,
   `test_memory_tools.py` and `tests/core/test_strategy_memory.py` (a new file, mirroring
   `test_strategy_market_data.py`, rather than additions to `test_strategy.py`). Shared helpers
   `make_memory_store` / `memory_rows` go in `tests/fakes.py`.
10. **§9 — `TODO.md`.** Not edited by the implementation: the file carries the user's own uncommitted
    changes. The user strikes "Memory" themselves.
