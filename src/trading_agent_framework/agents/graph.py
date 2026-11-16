"""Plumbing for LangGraph shared-state graphs whose nodes are agents.

Import-light on purpose: nodes are plain callables and reducers plain functions, so nothing here
imports LangGraph or LangChain.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.utils.errors import AgentError, AgentLoopError
from trading_agent_framework.utils.log import ColorLogger

if TYPE_CHECKING:
    from trading_agent_framework.agents.tools.text import LengthGate

logger = ColorLogger(logging.getLogger(__name__), "AgentNode")


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for a state slot written by parallel nodes: right-hand keys win, `None` counts as empty."""
    return {**(left or {}), **(right or {})}


class SubmitSlot:
    """Holds the validated payload an agent's submit tool recorded during one run.

    The submit tool calls `submit`; the owning `AgentNode` calls `arm` before each run. One slot per
    agent, so it must not be shared by nodes that run concurrently.
    """

    def __init__(self, validate: Callable[[Any, Mapping[str, Any]], str | None], gate: LengthGate | None = None) -> None:
        """`validate(payload, state)` returns an error message, or `None` when the payload is acceptable.

        `gate` is the agent's length gate, for its submit tool to put texts through before `submit`; `arm` resets it.
        """
        self._validate = validate
        self._gate = gate
        self._state: Mapping[str, Any] = {}
        self._value: Any | None = None

    def arm(self, state: Mapping[str, Any]) -> None:
        """Clear the recorded value, give the gate its attempts back and keep `state` for the validator."""
        self._state = state
        self._value = None
        if self._gate is not None:
            self._gate.reset()

    @property
    def gate(self) -> LengthGate | None:
        """The length gate the submit tool uses, or `None`."""
        return self._gate

    def submit(self, payload: Any) -> dict[str, Any]:
        """Validate and record `payload`; a later valid submit in the same arming replaces the earlier one."""
        message = self._validate(payload, self._state)
        if message is not None:
            return {"error": message}
        self._value = payload
        return {"status": "recorded"}

    @property
    def value(self) -> Any | None:
        """The last valid payload since `arm`, or `None`."""
        return self._value


class AgentLike(Protocol):
    """What `AgentNode` needs from an agent: the `run` signature of `AgentHandle`."""

    def run(
        self,
        task_prompt: str,
        *,
        context: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        force_tool: str | None = None,
        tool_budget: int | None = None,
        max_steps: int | None = None,
    ) -> AgentRunResult:
        """Run the agent once."""
        ...


@dataclass(frozen=True)
class AgentNode:
    """A LangGraph node that runs one agent against slots of the shared state.

    Attributes:
        name: Node name, used in logs and in `errors` entries.
        agent: The agent to run.
        slot: The slot the agent's submit tool records into.
        submit_tool: Name of that submit tool (forced on the retry).
        task: The task prompt.
        reads: Picks the state slots the agent sees, passed as `context=`.
        write: Turns the submission (`None` when unavailable) into the state update.
        tool_budget: Per-run tool-call cap, passed to `agent.run`.
        max_steps: Per-run superstep cap, passed to `agent.run`.
    """

    name: str
    agent: AgentLike
    slot: SubmitSlot
    submit_tool: str
    task: str
    reads: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    write: Callable[[Any | None], dict[str, Any]]
    tool_budget: int | None = None
    max_steps: int | None = None

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Run the agent; retry once with the submit tool forced; else write the unavailable marker plus an error."""
        self.slot.arm(state)
        reason = self._run(self.task, state, force_tool=None)
        if self.slot.value is None and not reason.startswith("loop:"):
            logger.log_warning(f"[{self.name}] no valid {self.submit_tool}: retrying with it forced")
            retry_task = f"{self.task}\n\nYour previous attempt ended without a valid {self.submit_tool} call. Call {self.submit_tool} now."
            reason = self._run(retry_task, state, force_tool=self.submit_tool)
        if self.slot.value is not None:
            return self.write(self.slot.value)
        logger.log_warning(f"[{self.name}] unavailable: {reason}")
        return {**self.write(None), "errors": [{"node": self.name, "error": reason}]}

    def _run(self, task: str, state: Mapping[str, Any], *, force_tool: str | None) -> str:
        """One agent run; returns why it left no submission (an empty submission is `"no valid submission"`)."""
        try:
            self.agent.run(
                task,
                context=self.reads(state),
                force_tool=force_tool,
                tool_budget=self.tool_budget,
                max_steps=self.max_steps,
            )
        except AgentLoopError as exc:
            return f"loop: {exc}"
        except AgentError as exc:
            return f"agent failed: {exc}"
        return "no valid submission"
