"""LangChain agent creation/execution for strategies (`strategy.agents`).

Plain eager imports here are fine (unlike `brokers/__init__.py`'s lazy `__getattr__`): nothing in
`config.py`, `results.py` or `manager.py` imports `langchain` at module level, so importing this
package -- or `trading_agent_framework.core.strategy`, which imports it -- never pulls LangChain
into a strategy that doesn't create an agent.
"""

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord

__all__ = [
    "AgentHandle",
    "AgentManager",
    "AgentRunResult",
    "LLMCredentials",
    "ToolCallRecord",
]
