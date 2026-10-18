"""`output_cap_middleware`: caps the output tokens of every agent turn, so one runaway generation cannot fill the window.

A reasoning model can think until the context window is full (no text, no tool call): tens of thousands of tokens
and minutes of generation for nothing. Compaction cannot help, it only runs between model calls. The cap sends
`max_tokens` (LangChain's OpenAI client sends it as `max_completion_tokens`) with every agent turn, so such a call
stops at `finish_reason="length"` once the cap is spent. Not capped: compaction's summary call, which calls the model
directly, outside every `wrap_model_call` middleware.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "OutputCap")

# vLLM (and OpenAI) refuse a request whose prompt plus `max_tokens` is larger than the window with this phrase.
_WINDOW_REFUSAL = "maximum context length"


def output_cap_middleware(max_tokens: int) -> Any:
    """A `wrap_model_call` that sends `max_tokens` with every agent turn.

    vLLM refuses a call whose prompt plus `max_tokens` is larger than the window instead of shortening it; such a
    call is sent again without the cap, since the room left in the window is then already below it. A prompt that
    does not fit the window at all is refused again, and that error propagates.
    """
    from langchain.agents.middleware import wrap_model_call

    @wrap_model_call
    def cap_output(request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            return handler(request.override(model_settings={**request.model_settings, "max_tokens": max_tokens}))
        except Exception as exc:
            if _WINDOW_REFUSAL not in str(exc):
                raise
            logger.log_info(f"the output cap of {max_tokens} tokens does not fit what is left of the window: sent without it")
            return handler(request)

    return cap_output
