from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


class _Overlap:
    """Counts how many tool calls run at the same moment."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def tool(self, name: str, *, trades: bool = False) -> Callable[..., dict[str, Any]]:
        def call(symbol: str) -> dict[str, Any]:
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            time.sleep(0.05)
            with self._lock:
                self.running -= 1
            return {"symbol": symbol}

        call.__name__ = name
        call.__doc__ = f"{name} a symbol."
        if trades:
            vars(call)["mutates_trading"] = True
        return call


def _run(tools: list[Callable[..., dict[str, Any]]]) -> None:
    """One model turn calling every tool at once (LangGraph runs such calls on worker threads), then a closing turn."""
    turn = AIMessage(content="", tool_calls=[ToolCall(name=t.__name__, args={"symbol": "SPY"}, id=f"call_{i}") for i, t in enumerate(tools)])
    model = FakeToolCallingChatModel(messages=iter([turn, AIMessage(content="done")]))
    _manager().create(name="agent", system_prompt="x", model=model, tools=tools).run("go")


def test_an_agent_that_can_trade_runs_its_tool_calls_one_at_a_time() -> None:
    # Regression: parallel calls of one turn raced on the backtest broker ("dictionary changed size during iteration")
    # and on the order-readiness checks; the same race exists in live and paper.
    overlap = _Overlap()

    _run([overlap.tool("get_account_balance"), overlap.tool("get_positions"), overlap.tool("get_last_price"), overlap.tool("submit_order", trades=True)])

    assert overlap.peak == 1


def test_a_read_only_agent_keeps_its_parallel_tool_calls() -> None:
    # All four calls must be inside at once to pass the barrier: run one at a time, it breaks and the run raises.
    barrier = threading.Barrier(4, timeout=5)

    def make(name: str) -> Callable[..., dict[str, Any]]:
        def call(symbol: str) -> dict[str, Any]:
            barrier.wait()
            return {"symbol": symbol}

        call.__name__ = name
        call.__doc__ = f"{name} a symbol."
        return call

    _run([make("search_news"), make("get_fred_latest"), make("get_bars"), make("get_quote")])

    assert not barrier.broken
