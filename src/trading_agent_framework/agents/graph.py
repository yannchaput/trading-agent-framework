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
    from trading_agent_framework.agents.submit import NodeRecorder

logger = ColorLogger(logging.getLogger(__name__), "AgentNode")


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for a state slot written by parallel nodes: right-hand keys win, `None` counts as empty."""
    return {**(left or {}), **(right or {})}


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
        recorder: The recorder the agent's submit tool records into.
        submit_tool: Name of that submit tool (forced on the retry).
        task: The task prompt.
        reads: Picks the state slots the agent sees, passed as `context=`.
        write: Turns the submission (`None` when unavailable) into the state update.
        tool_budget: Per-run tool-call cap, passed to `agent.run`.
        max_steps: Per-run superstep cap, passed to `agent.run`.
    """

    name: str
    agent: AgentLike
    recorder: NodeRecorder
    submit_tool: str
    task: str
    reads: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    write: Callable[[Any | None], dict[str, Any]]
    tool_budget: int | None = None
    max_steps: int | None = None

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Run the agent; retry once with the submit tool forced; else write the unavailable marker plus an error."""
        self.recorder.arm(state)
        task, force_tool = self.task, None
        reason = "no valid submission"
        for attempt in range(2):
            try:
                result = self._run(task, state, force_tool=force_tool)
            except AgentLoopError as exc:
                reason = f"loop: {exc}"
                break  # a loop is not retried
            except AgentError as exc:
                reason = f"agent failed: {exc}"
            else:
                reason = "no valid submission"
                if result.output and not self.recorder.submitted:  # the model's text is the only clue to why it did not submit
                    logger.log_info(f"[{self.name}] {result.output}")
            if self.recorder.submitted:
                break
            if attempt == 0:
                logger.log_warning(f"[{self.name}] no valid {self.submit_tool}: retrying with it forced")
                task = f"{self.task}\n\nYour previous attempt ended without a valid {self.submit_tool} call. Call {self.submit_tool} now."
                force_tool = self.submit_tool
        if self.recorder.submitted:
            logger.log_info(f"[{self.name}] submitted: {self.recorder.submission}")
            return self.write(self.recorder.submission)
        logger.log_warning(f"[{self.name}] unavailable: {reason}")
        return {**self.write(None), "errors": [{"node": self.name, "error": reason}]}

    def _run(self, task: str, state: Mapping[str, Any], *, force_tool: str | None) -> AgentRunResult:
        """One agent run; raises `AgentLoopError` or `AgentError` when the agent fails."""
        return self.agent.run(
            task,
            context=self.reads(state),
            force_tool=force_tool,
            tool_budget=self.tool_budget,
            max_steps=self.max_steps,
        )
