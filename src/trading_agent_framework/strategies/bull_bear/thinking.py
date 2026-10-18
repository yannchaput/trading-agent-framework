"""Two middlewares that keep a bull_bear agent from thinking its context window full: they turn thinking off.

A reasoning model sometimes thinks until no room is left, with no text and no tool call (seen: 19k-31.5k reasoning
tokens on a 32k window, where every call that worked used at most 11k). Two turns are where that happened:

- any turn, once the output cap (`BullBearParams.max_output_tokens`) stops it at `finish_reason="length"` with no
  tool call: `length_recovery_middleware` asks that same turn again, without thinking, its submit tool forced;
- the turn after a rejected submission (a note 537 characters long against a 500 cap), a mere edit:
  `rejected_submission_middleware` runs it without thinking from the start, its submit tool forced.

Both work inside the run, so the research the agent did is kept (the pipeline's forced retry is a fresh run).
"Without thinking" is Qwen's chat template switch (`chat_template_kwargs.enable_thinking`, sent by vLLM); a model
whose template ignores it thinks anyway, bounded by the output cap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trading_agent_framework.strategies.bull_bear.pipeline import tool_failed


def thinking_guards(agent_name: str, submit_tool: str, log: Callable[[str], object]) -> list[Any]:
    """Both middlewares, for `AgentManager.create(middleware=...)`, in the order they need.

    The rejection guard comes first (outermost): the length guard must see the request it turned thinking off in,
    or a correction stopped at the cap would be asked again for nothing.
    """
    return [rejected_submission_middleware(agent_name, submit_tool, log), length_recovery_middleware(agent_name, submit_tool, log)]


def _without_thinking(request: Any, submit_tool: str) -> Any:
    """`request` with Qwen's thinking turned off and `submit_tool` forced."""
    extra_body = request.model_settings.get("extra_body") or {}
    template = {**extra_body.get("chat_template_kwargs", {}), "enable_thinking": False}
    settings = {**request.model_settings, "extra_body": {**extra_body, "chat_template_kwargs": template}}
    return request.override(model_settings=settings, tool_choice=submit_tool)


def _thinking_off(request: Any) -> bool:
    extra_body = request.model_settings.get("extra_body") or {}
    return extra_body.get("chat_template_kwargs", {}).get("enable_thinking") is False


def length_recovery_middleware(agent_name: str, submit_tool: str, log: Callable[[str], object]) -> Any:
    """A `wrap_model_call`: a turn stopped at the output cap with no tool call is asked again without thinking.

    The new answer replaces the stopped one (both calls are recorded by telemetry). A turn that already ran without
    thinking is not asked again: it would only write the same thing.
    """
    from langchain.agents.middleware import wrap_model_call
    from langchain_core.messages import AIMessage

    @wrap_model_call
    def answer_without_thinking(request: Any, handler: Callable[[Any], Any]) -> Any:
        response = handler(request)
        ai_message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        if ai_message is None or ai_message.tool_calls or _thinking_off(request):
            return response
        if (ai_message.response_metadata or {}).get("finish_reason") != "length":
            return response
        log(f"[{agent_name}] stopped at the output cap without calling a tool: asked again without thinking, {submit_tool} forced")
        return handler(_without_thinking(request, submit_tool))

    return answer_without_thinking


MAX_REJECTIONS = 3  # rejected submissions that end a run: the first try and two corrections


def rejected_submission_middleware(agent_name: str, submit_tool: str, log: Callable[[str], object]) -> Any:
    """A `wrap_model_call`: the turn after a rejected `submit_tool` call runs without thinking, `submit_tool` forced.

    Rejected: the handoff's `{"error": ...}`, or a call the tool could not take (`status="error"`). Only the tool
    results since the model's last turn count; a failed research tool leaves thinking on.

    Forced and without thinking, a valid submission is the model's only way out of the run, so the run ends (an empty
    assistant message, no model call) at `MAX_REJECTIONS` rejected submissions in its history: with nothing recorded,
    the pipeline's forced retry, then abandonment, take over. Without the limit, a submission the model cannot get right
    would loop until the history overflows, or, for an agent whose history is compacted, up to LangGraph's 9,999 steps.
    """
    from langchain.agents.middleware import ModelResponse, wrap_model_call
    from langchain_core.messages import AIMessage

    def rejected(message: Any) -> bool:
        return getattr(message, "type", None) == "tool" and message.name == submit_tool and (message.status == "error" or tool_failed(str(message.content)))

    @wrap_model_call
    def resubmit_without_thinking(request: Any, handler: Callable[[Any], Any]) -> Any:
        rejections = sum(1 for message in request.messages if rejected(message))
        if rejections >= MAX_REJECTIONS:
            log(f"[{agent_name}] {submit_tool} rejected {rejections} times: the run ends without a submission")
            return ModelResponse(result=[AIMessage(content="")])
        for message in reversed(request.messages):  # the tool results since the model's last turn
            if getattr(message, "type", None) != "tool":
                break
            if rejected(message):
                log(f"[{agent_name}] {submit_tool} was rejected: the correction runs without thinking, {submit_tool} forced")
                return handler(_without_thinking(request, submit_tool))
        return handler(request)

    return resubmit_without_thinking
