"""Text-length tools for agents that submit free text: a character counter and the gate a submit tool puts its texts through.

Models cannot count characters, so an agent counts with `count_characters` before it submits. A submit tool then hands its
labelled texts to a `LengthGate`, which refuses an over-long text a bounded number of times (naming each count, so the
model can correct itself) and on the last attempt accepts the submission, cutting any text above its hard cap at a
sentence. The module knows nothing of any strategy: labels are plain strings the caller chooses.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from the
function's real annotations.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.utils.log import ColorLogger


def trim_text(text: str, *, max_chars: int, min_chars: int) -> str:
    """`text` cut to at most `max_chars`, never reworded.

    The longest run of whole sentences that fits, if it has at least `min_chars`; otherwise a cut at a word with a trailing
    ellipsis (or hard, when there is no space).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    end = 0  # length of the longest prefix that ends a sentence (. ! or ? before whitespace or at the end) and fits
    for index, char in enumerate(text[:max_chars]):
        if char in ".!?" and (index + 1 == len(text) or text[index + 1].isspace()):
            end = index + 1
    whole = text[:end].rstrip()
    if len(whole) >= min_chars:
        return whole
    head = text[: max_chars - 1]
    cut = max((index for index, char in enumerate(head) if char.isspace()), default=-1)
    if cut <= 0:
        return head + "…"
    return head[:cut].rstrip() + "…"


@dataclass(frozen=True)
class TextLimit:
    """The two lengths of one kind of text.

    Attributes:
        stated: What the prompt quotes to the model; a longer text is refused while attempts remain.
        hard: Where the last attempt cuts a longer text. Never quoted to the model.

    Raises:
        ValueError: Unless `1 <= stated <= hard`.
    """

    stated: int
    hard: int

    def __post_init__(self) -> None:
        """Validate the two lengths."""
        if not 1 <= self.stated <= self.hard:
            raise ValueError(f"a text limit needs 1 <= stated <= hard, got stated={self.stated}, hard={self.hard}")


class LengthGate:
    """Refuses over-long texts up to `max_attempts - 1` times, then accepts them cut at their hard cap.

    One gate per agent run: `reset` starts a new run. Only a refusal for length uses an attempt, so the caller's own
    validation errors never spend one. Not thread-safe (one agent's submit tool calls it).
    """

    def __init__(self, max_attempts: int, *, name: str, log_tag: str) -> None:
        """Create a gate.

        Args:
            max_attempts: Submissions allowed per run before the texts are cut instead of refused; 1 never refuses.
            name: The submit tool's name, written into messages.
            log_tag: Prefix of the gate's log lines (the strategy's tag).

        Raises:
            ValueError: `max_attempts` is below 1.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._max_attempts = max_attempts
        self._name = name
        self._refusals = 0
        self._logger = ColorLogger(logging.getLogger(__name__), log_tag)

    def reset(self) -> None:
        """Give the attempts back: a new agent run starts."""
        self._refusals = 0

    def review(self, texts: Sequence[tuple[str, Any, TextLimit]]) -> dict[str, str] | list[Any]:
        """Check labelled texts against their limits.

        Args:
            texts: `(label, value, limit)` per text, in the caller's order. A label may repeat. A value that is not a
                string is passed through for the caller's own validation to refuse.

        Returns:
            `{"error": ...}` listing every text over its stated limit with its count, while refusals remain; otherwise
            the values in the same order, stripped and, above their hard cap, cut at a sentence (a WARNING per cut).
        """
        over = [(label, len(value.strip()), limit.stated) for label, value, limit in texts if isinstance(value, str) and len(value.strip()) > limit.stated]
        if over and self._refusals < self._max_attempts - 1:
            self._refusals += 1
            listed = "; ".join(f"{label} has {count} characters (limit {stated})" for label, count, stated in over)
            self._logger.log_info(f"{self._name}: refused for length: {listed}")
            return {"error": f"too long: {listed}. Count them with count_characters, shorten those texts and call {self._name} again with the whole submission."}
        return [self._kept(label, value, limit) for label, value, limit in texts]

    def _kept(self, label: str, value: Any, limit: TextLimit) -> Any:
        """`value` stripped and cut at `limit.hard` (with a warning) when it is a longer string; anything else unchanged."""
        if not isinstance(value, str):
            return value
        kept = trim_text(value, max_chars=limit.hard, min_chars=limit.stated // 2)
        if len(value.strip()) > limit.hard:
            self._logger.log_warning(f"{self._name}: {label} trimmed from {len(value.strip())} to {len(kept)} characters (limit {limit.hard})")
        return kept


def count_step(submit_tool: str, texts: str) -> str:
    """The prompt paragraph that makes an agent count its texts before it submits.

    Args:
        submit_tool: Name of the submit tool the agent ends with.
        texts: The texts to count, in the order of the count list, e.g. ``every `title`, then `body```.
    """
    return f"""
Length check: before you call `{submit_tool}`, call `count_characters` once with all your texts: {texts}, in that order.
It returns the length of each text in characters. `{submit_tool}` refuses a text over its limit.
If a count is over its limit, shorten that text and count it again. Submit only texts whose last count is within the limit, unchanged.
"""


def text_tools() -> list[Callable[..., dict[str, Any]]]:
    """The text tools: `count_characters`."""

    def count_characters(texts: list[str]) -> dict[str, list[int]]:
        """Count the characters of each text. Returns `counts`: one number per text, in the order given."""
        return {"counts": [len(str(text).strip()) for text in texts]}

    return [count_characters]
