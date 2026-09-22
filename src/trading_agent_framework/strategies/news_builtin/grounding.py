"""Requires `search_news` to have succeeded before `remember_decision` or `submit_order` may run.

The local LLM sometimes recorded a decision -- occasionally citing report details it could not have
read -- or opened a decide-then-research-then-decide-again loop, all without ever calling `search_news`
first. The prompt's own workflow already puts "scan broad-market news" before deciding or trading (steps
2 and 5-7), but wording alone did not reliably get it followed. This wraps the assembled tool list to
enforce that ordering in code: a call to either gated tool is refused, without running, until
`search_news` has returned a non-error result earlier in the same agent run (by `current_run_id()`,
mirroring `agents/tools/trading.py`'s `_OrdersThisRun`). Outside an agent run nothing is gated, same
convention as every other per-run tool tracker in this codebase.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from trading_agent_framework.memory.tools import current_run_id

_GATED = ("remember_decision", "submit_order")


class _NewsGroundedRuns:
    """Whether `search_news` has succeeded in the current agent run; only the latest run is kept."""

    def __init__(self) -> None:
        self._run_id: str | None = None
        self._grounded = False

    def mark(self) -> None:
        run_id = current_run_id()
        if run_id is None:
            return
        if run_id != self._run_id:
            self._run_id, self._grounded = run_id, False
        self._grounded = True

    def satisfied(self) -> bool:
        run_id = current_run_id()
        return run_id is None or (run_id == self._run_id and self._grounded)


def require_search_news_before(tools: list[Callable[..., dict[str, Any]]]) -> list[Callable[..., dict[str, Any]]]:
    """`tools` with `remember_decision`/`submit_order` (whichever are present) gated on a prior `search_news`.

    Every other tool, including `search_news` itself in name and signature, passes through unchanged
    from the caller's point of view. Raises `ValueError` if `tools` has no `search_news` -- gating
    against a tool that can never ground the run would silently lock the other two forever.
    """
    by_name = {tool.__name__: tool for tool in tools}
    if "search_news" not in by_name:
        raise ValueError("require_search_news_before needs a 'search_news' tool in the list")

    grounded = _NewsGroundedRuns()
    search_news = by_name["search_news"]

    @functools.wraps(search_news)
    def wrapped_search_news(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = search_news(*args, **kwargs)
        if "error" not in result:
            grounded.mark()
        return result

    def _guard(name: str, inner: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if not grounded.satisfied():
                return {"error": f"call search_news at least once in this run before calling {name}, so it is grounded in this run's news"}
            return inner(*args, **kwargs)

        return wrapped

    replacements: dict[str, Callable[..., dict[str, Any]]] = {"search_news": wrapped_search_news}
    for name in _GATED:
        if name in by_name:
            replacements[name] = _guard(name, by_name[name])
    return [replacements.get(tool.__name__, tool) for tool in tools]
