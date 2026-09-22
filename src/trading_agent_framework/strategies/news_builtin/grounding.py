"""Requires a full-content `search_news` read before `remember_decision` or `submit_order` may run.

The local LLM sometimes recorded a decision -- occasionally citing report details it could not have
read -- or opened a decide-then-research-then-decide-again loop, all without ever calling `search_news`
first. The prompt's own workflow already puts "scan broad-market news" before deciding or trading (steps
2 and 5-7), but wording alone did not reliably get it followed. This wraps the assembled tool list to
enforce that ordering in code: a call to either gated tool is refused, without running, until
`search_news` has returned a non-error result earlier in the same agent run (by `current_run_id()`,
mirroring `agents/tools/trading.py`'s `_OrdersThisRun`). Outside an agent run nothing is gated, same
convention as every other per-run tool tracker in this codebase.

A broad `include_content=False` scan (prompt step 2) alone does not ground the run: the prompt also
requires reading the single most relevant article in full (`include_content=True`, step 3) before
deciding or trading, and the local LLM was observed skipping straight from the broad scan to a decision
without ever reading an article -- effectively judging "bearish" from headlines alone. Only a
`search_news` call that both succeeds and resolves `include_content=True` (whether passed positionally
or by keyword; resolved against the real tool's signature via `inspect.signature`) grounds the run.

Requesting content is not the same as receiving it, so grounding also requires the response to carry
some: at least one returned article with a non-empty `content` field (the shape `parse_news` produces).
Without this check, a zero-result search (`{"count": 0, "articles": []}`, e.g. from a degenerate
start==end window) or a result of articles that happen to have no source content both pass with no
`"error"` key, and the local LLM was observed constructing exactly such empty-window calls to ground a
run for free without ever reading an article.

A content-less result is also a legitimate outcome, not just a gaming attempt: some articles (terse
Benzinga data-print/quote wires such as "USA ISM Manufacturing PMI For December 47.9 Vs 48.3 Est.")
never carry a body on the provider's side, no matter how the window is narrowed. The local LLM was
observed retrying the exact same doomed pick many turns in a row instead of following the prompt's own
instruction to choose a different on-topic article -- in two backtest iterations this ran the run's
turn budget down to where the model gave up mid-loop and emitted a malformed pseudo tool-call as plain
text instead of a real one, silently ending the run with no decision recorded. The refusal now names
the article ids that already came back with no content this run, so the prompt's "pick a different
one" instruction has something concrete to act on instead of a bare retry-me error.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any

from trading_agent_framework.memory.tools import current_run_id

_GATED = ("remember_decision", "submit_order")


class _NewsGroundedRuns:
    """Whether a full-content `search_news` read has succeeded in the current agent run; only the latest run is kept.

    Also tracks the ids of articles that were requested with `include_content=True` but came back
    with no content this run, so a refusal can name them instead of inviting a repeat of the same
    doomed pick.
    """

    def __init__(self) -> None:
        self._run_id: str | None = None
        self._grounded = False
        self._empty_content_ids: set[Any] = set()

    def _start_if_new_run(self, run_id: str) -> None:
        if run_id != self._run_id:
            self._run_id, self._grounded, self._empty_content_ids = run_id, False, set()

    def mark(self) -> None:
        run_id = current_run_id()
        if run_id is None:
            return
        self._start_if_new_run(run_id)
        self._grounded = True

    def note_empty_content(self, article_ids: Iterable[Any]) -> None:
        run_id = current_run_id()
        if run_id is None:
            return
        self._start_if_new_run(run_id)
        self._empty_content_ids.update(article_ids)

    def satisfied(self) -> bool:
        run_id = current_run_id()
        return run_id is None or (run_id == self._run_id and self._grounded)

    def empty_content_ids(self) -> set[Any]:
        run_id = current_run_id()
        if run_id != self._run_id:
            return set()
        return set(self._empty_content_ids)


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
    search_news_signature = inspect.signature(search_news)

    def _requested_full_content(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        try:
            bound = search_news_signature.bind_partial(*args, **kwargs)
        except TypeError:
            return False
        bound.apply_defaults()
        return bool(bound.arguments.get("include_content", False))

    def _received_full_content(result: dict[str, Any]) -> bool:
        articles = result.get("articles")
        if not isinstance(articles, list):
            return False
        return any(isinstance(article, dict) and article.get("content") for article in articles)

    def _content_less_article_ids(result: dict[str, Any]) -> Iterable[Any]:
        articles = result.get("articles")
        if not isinstance(articles, list):
            return ()
        return (
            article["id"]
            for article in articles
            if isinstance(article, dict) and not article.get("content") and "id" in article
        )

    @functools.wraps(search_news)
    def wrapped_search_news(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = search_news(*args, **kwargs)
        if "error" not in result and _requested_full_content(args, kwargs):
            if _received_full_content(result):
                grounded.mark()
            else:
                grounded.note_empty_content(_content_less_article_ids(result))
        return result

    def _guard(name: str, inner: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(inner)
        def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            if not grounded.satisfied():
                message = (
                    f"call search_news with include_content=True at least once in this run before "
                    f"calling {name}, so it is grounded in a specific article, not just the headline scan"
                )
                empty_ids = sorted(grounded.empty_content_ids(), key=str)
                if empty_ids:
                    message += f"; these article ids already came back with no content this run, pick a different one: {empty_ids}"
                return {"error": message}
            return inner(*args, **kwargs)

        return wrapped

    replacements: dict[str, Callable[..., dict[str, Any]]] = {"search_news": wrapped_search_news}
    for name in _GATED:
        if name in by_name:
            replacements[name] = _guard(name, by_name[name])
    return [replacements.get(tool.__name__, tool) for tool in tools]
