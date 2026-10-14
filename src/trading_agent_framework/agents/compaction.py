"""History compaction: LangChain's `SummarizationMiddleware`, set up for this framework's agents.

When an agent's history reaches `TRIGGER_FRACTION` of the model's input window, the older messages are
replaced by one summary written by the same chat model, and the most recent `KEEP_FRACTION` of the window
is kept as is (tool calls stay paired with their results). The window is the chat model's
`profile["max_input_tokens"]`: `AgentManager` sets it from `LLM_MAX_INPUT_TOKENS` for a model it builds.

LangChain is imported inside the function, like everywhere in `agents/`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trading_agent_framework.utils.errors import ConfigurationError

TRIGGER_FRACTION = 0.8  # compact once the history reaches 80% of the input window
# Kept after compacting: a message-count default (LangChain's 20) would keep a whole researcher run.
KEEP_FRACTION = 0.3


def compaction_middleware(chat_model: Any, *, trigger_fraction: float = TRIGGER_FRACTION, keep_fraction: float = KEEP_FRACTION) -> Any:
    """A `SummarizationMiddleware` on `chat_model`; `ConfigurationError` when its input window is unknown.

    Its summary call goes straight to the chat model, not through the agent's middleware: telemetry does
    not record it. Once it has run, the run's result no longer holds the summarized messages, so its
    `tool_calls` and `output` cover only what was kept.
    """
    profile = getattr(chat_model, "profile", None)
    if not isinstance(profile, Mapping) or not isinstance(profile.get("max_input_tokens"), int):
        raise ConfigurationError("history compaction needs the model's input window: set LLM_MAX_INPUT_TOKENS (the server's max_model_len)")
    from langchain.agents.middleware import SummarizationMiddleware

    return SummarizationMiddleware(chat_model, trigger=("fraction", trigger_fraction), keep=("fraction", keep_fraction))
