from trading_agent_framework.utils import get_version

from .agent_news_binary import NewsBinaryStrategy

__version__ = get_version("trading_agent_framework")

__all__ = [
    "NewsBinaryStrategy",
]
