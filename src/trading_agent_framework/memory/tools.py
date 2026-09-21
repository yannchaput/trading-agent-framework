"""Lumibot's 9 memory tools as plain typed functions; the agent layer wraps them as LLM tools.

Each docstring is a single line on purpose: it becomes the tool description sent to the model
on every call. Validation errors come back as `{"error": ...}` so the model can correct itself.
"""

import threading
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypedDict

from trading_agent_framework.memory.records import normalize_symbol
from trading_agent_framework.memory.store import MemoryStore
from trading_agent_framework.utils.errors import MemoryValidationError

MAX_SEARCH_LIMIT = 20


@dataclass(frozen=True, slots=True)
class _AgentCall:
    agent_name: str | None = None
    model_call_id: str | None = None
    run_id: str | None = None


# `_AgentCall` is frozen, so the shared default holds no mutable state; ContextVar still wants
# `None` (ruff B039) so a stray write can't be confused with a real context.
_CURRENT_CALL: ContextVar[_AgentCall | None] = ContextVar("memory_agent_call", default=None)


@contextmanager
def agent_call_context(
    agent_name: str | None = None, model_call_id: str | None = None, run_id: str | None = None
) -> Iterator[None]:
    """Attribute memory tool calls made inside this block to `agent_name` / `model_call_id`.

    `run_id` scopes one agent run: `remember_decision` records an identical decision only once per run.
    """
    token = _CURRENT_CALL.set(_AgentCall(agent_name, model_call_id, run_id))
    try:
        yield
    finally:
        _CURRENT_CALL.reset(token)


def current_run_id() -> str | None:
    """The id of the agent run this code is executing in, or `None` outside `agent_call_context(run_id=...)`."""
    call = _CURRENT_CALL.get()
    return call.run_id if call is not None else None


class _Provenance(TypedDict):
    agent_name: str | None
    model_call_id: str | None


def _provenance() -> _Provenance:
    # A precise TypedDict (not `dict[str, str | None]`) so a static checker knows exactly which
    # two keyword-only parameters `**_provenance()` fills on each `store.*` call below, instead
    # of conservatively checking `str | None` against every keyword-only parameter (including
    # `Mapping`-typed ones like `metadata`/`outcome`/`evidence`).
    call = _CURRENT_CALL.get() or _AgentCall()
    return {"agent_name": call.agent_name, "model_call_id": call.model_call_id}


def _write(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        item = action()
    except MemoryValidationError as exc:
        return {"error": str(exc)}
    return {"id": item["id"], "kind": item["kind"], "status": item["status"]}


class RunMemo:
    """Results of the tool calls already made in the current agent run, keyed by what makes two calls identical.

    A local LLM re-sends a call it already made (112 identical `remember_decision` copies in one backtest
    run, the same `search_news` scan twice), so a repeat within a run gets the first result -- or, with
    `on_repeat`, an answer built from it -- instead of running again. Only the latest run is kept, in
    memory: nothing is persisted, and a run id that has been superseded starts empty. The lock spans the
    check and the call: identical calls sent in ONE model response run on LangGraph worker threads, and
    they all passed an unlocked check before any of them had run.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._results: dict[Hashable, dict[str, Any]] = {}

    def once(
        self,
        run_id: str | None,
        key: Hashable,
        call: Callable[[], dict[str, Any]],
        on_repeat: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if run_id is None:
            return call()
        with self._lock:
            if run_id != self._run_id:
                self._run_id, self._results = run_id, {}
            if key in self._results:
                first = self._results[key]
                return first if on_repeat is None else on_repeat(first)
            result = call()
            if "error" not in result:  # a refused call did nothing, so a corrected retry must go through
                self._results[key] = result
            return result


def memory_tools(store: MemoryStore) -> list[Callable[..., dict[str, Any]]]:
    """Lumibot's memory tools bound to `store`, in lumibot's registration order."""
    decisions = RunMemo()

    def remember(text: str, kind: str = "memory", tags: list[str] | None = None) -> dict[str, Any]:
        """Store a memory or note."""
        return _write(lambda: store.remember(text, kind=kind, tags=tags, **_provenance()))

    def search_memory(
        query: str,
        limit: int = 10,
        kind: str | None = None,
        symbol: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Search memories, lessons, decisions and theses; filter by symbol/status to find a held position's open thesis."""
        return store.search(
            query,
            limit=min(max(int(limit), 1), MAX_SEARCH_LIMIT),
            kind=kind,
            symbol=symbol,
            status=status,
            **_provenance(),
        )

    def remember_proposal(
        text: str,
        symbol: str | None = None,
        action: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a non-final trade idea (not an executed decision)."""
        return _write(
            lambda: store.remember_proposal(
                text, symbol=symbol, action=action, tags=tags, **_provenance()
            )
        )

    def remember_risk_note(
        text: str, symbol: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Record a compact risk note or bear case."""
        return _write(
            lambda: store.remember_risk_note(text, symbol=symbol, tags=tags, **_provenance())
        )

    def remember_decision(
        text: str, symbol: str | None = None, action: str | None = None
    ) -> dict[str, Any]:
        """Record an actual trading decision."""
        call = _CURRENT_CALL.get() or _AgentCall()
        key = (text.strip(), normalize_symbol(symbol), action)
        return decisions.once(
            call.run_id,
            key,
            lambda: _write(
                lambda: store.remember_decision(text, symbol=symbol, action=action, **_provenance())
            ),
        )

    def remember_lesson(text: str, symbol: str | None = None) -> dict[str, Any]:
        """Record a compact lesson for future runs."""
        return _write(lambda: store.remember_lesson(text, symbol=symbol, **_provenance()))

    def open_thesis(
        text: str, symbol: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Open an investment thesis."""
        return _write(lambda: store.open_thesis(text, symbol=symbol, tags=tags, **_provenance()))

    def update_thesis(thesis_id: str, text: str) -> dict[str, Any]:
        """Update an open thesis."""
        return _write(lambda: store.update_thesis(thesis_id, text, **_provenance()))

    def close_thesis(thesis_id: str, text: str) -> dict[str, Any]:
        """Close a thesis and record its outcome."""
        return _write(lambda: store.close_thesis(thesis_id, text, **_provenance()))

    # Lumibot's `mutates_trading` tool metadata: the agent layer decides which agents get it.
    vars(remember_decision)["mutates_trading"] = True

    return [
        remember,
        search_memory,
        remember_proposal,
        remember_risk_note,
        remember_decision,
        remember_lesson,
        open_thesis,
        update_thesis,
        close_thesis,
    ]
