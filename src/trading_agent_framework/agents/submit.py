"""The recorder behind an agent's submit tools: it arms a stage, guards the tools and records the first valid submission.

A submit tool validates an agent's submission in code and returns `{"error": ...}` so the model can correct itself in the
same run. `SubmitRecorder` is the base of every strategy's recorder; `NodeRecorder` is the ready-made one for a graph
node whose stage is a single tool. A stage may carry a `LengthGate` (`agents/tools/text.py`): `submit_texts` runs the
caller's length review before the submission is validated, and arming the stage gives its gate's attempts back.

Unlike `memory/tools.py`, this module may use `from __future__ import annotations`: it builds no tool schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from trading_agent_framework.agents.tools.text import LengthGate


class HandoffError(ValueError):
    """A submission that breaks a rule; the message tells the model what is wrong and what is allowed."""


class SubmitRecorder:
    """Holds what the armed stage must validate against, and the first valid submission for it.

    A subclass sets `TOOL_NAMES` (stage -> tool name), arms a stage with `_arm(stage, context)` and implements `_validate`,
    which returns the submission or raises `HandoffError`. `submit` is what a submit tool returns: `{"status": "recorded"}`
    or `{"error": ...}` (also kept in `last_error`). Some recorders keep their own final-submission attributes, so
    `submission` stays None there.
    """

    TOOL_NAMES: ClassVar[Mapping[str, str]] = {}
    NOT_EXPECTED_SUFFIX: ClassVar[str] = ""
    ALREADY_RECORDED: ClassVar[str] = "already recorded for this run"

    def __init__(self, gates: Mapping[str, LengthGate] | None = None) -> None:
        """Create a recorder that accepts no submission until armed.

        Args:
            gates: Optional `LengthGate` per stage; arming a stage resets its gate.
        """
        self._gates = gates or {}
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        """Arm the recorder to accept the submission for `stage` with its validation context; reset the stage's gate."""
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None
        if gate := self._gates.get(stage):
            gate.reset()

    @property
    def gate(self) -> LengthGate | None:
        """The armed stage's length gate, if it has one."""
        return self._gates.get(self._stage or "")

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission, else None."""
        return self._submission

    @property
    def missing_tool(self) -> str:
        """The tool the armed stage still waits for: what a forced retry forces."""
        return self._tool_name(self._stage or "")

    def _tool_name(self, stage: str) -> str:
        """The submit tool's name for `stage` (empty for an unknown one when reading `missing_tool`)."""
        return self.TOOL_NAMES.get(stage, "")

    def submit(self, stage: str, *args: Any) -> dict[str, Any]:
        """Validate and record the agent's submission for `stage`; return a result or error payload for the agent."""
        if self._stage != stage:
            return self._fail(f"{self._tool_name(stage)} is not expected at this point{self.NOT_EXPECTED_SUFFIX}")
        if self._submitted:
            return self._fail(self.ALREADY_RECORDED)
        try:
            value = self._validate(stage, *args)
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def submit_texts(
        self,
        stage: str,
        review: Callable[[LengthGate], dict[str, str] | tuple[Any, ...]],
        *args: Any,
    ) -> dict[str, Any]:
        """Like `submit`, after the stage's length gate reviewed the texts.

        The guards run first, so a call the recorder would refuse anyway spends no gate attempt.

        Args:
            stage: The stage being submitted.
            review: Called with the stage's gate when it has one. Returns `{"error": ...}` to refuse the submission, or
                the tuple of values to validate and record in place of `args`.
            *args: The submission as the agent sent it.

        Returns:
            What `submit` returns, or the gate's refusal as an error payload.
        """
        if self._stage != stage:
            return self._fail(f"{self._tool_name(stage)} is not expected at this point{self.NOT_EXPECTED_SUFFIX}")
        if self._submitted:
            return self._fail(self.ALREADY_RECORDED)
        if gate := self._gates.get(stage):
            out = review(gate)
            if isinstance(out, dict):
                return self._fail(out["error"])
            args = out
        return self.submit(stage, *args)

    def _validate(self, stage: str, *args: Any) -> Any:
        """The submission for `stage`, or a `HandoffError` that tells the model what to fix."""
        raise NotImplementedError

    def _fail(self, message: str) -> dict[str, Any]:
        """Remember `message` as the last error and return it as the error payload."""
        self.last_error = message
        return {"error": message}


class NodeRecorder(SubmitRecorder):
    """A recorder for one tool whose submission is validated against the state its node was armed with."""

    def __init__(
        self,
        tool: str,
        validate: Callable[[Any, Mapping[str, Any]], None],
        gate: LengthGate | None = None,
    ) -> None:
        """Create the recorder.

        Args:
            tool: The submit tool's name; it is also the stage.
            validate: Called with the payload and the armed state; raises `HandoffError` to refuse the payload.
            gate: Optional length gate, reset each time the node is armed.
        """
        super().__init__({tool: gate} if gate else None)
        self.tool = tool
        self._check = validate

    def arm(self, state: Mapping[str, Any]) -> None:
        """Arm the tool's stage with `state` as the validation context."""
        self._arm(self.tool, dict(state))

    def _tool_name(self, stage: str) -> str:
        """The one tool's name."""
        return self.tool

    def _validate(self, stage: str, *args: Any) -> Any:
        """Check the payload against the armed state and return it."""
        self._check(args[0], self._context)
        return args[0]
