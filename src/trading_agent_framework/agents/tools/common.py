"""Common tools to reuse with other tools accross agents."""

from collections.abc import Callable
from typing import Any

Tool = Callable[..., dict[str, Any]]


def only(tools: list[Tool], names: set[str]) -> list[Tool]:
    """The tools named in `names`: each agent gets exactly the tools its prompt describes, and no more schema than that."""
    return [tool for tool in tools if tool.__name__ in names]
