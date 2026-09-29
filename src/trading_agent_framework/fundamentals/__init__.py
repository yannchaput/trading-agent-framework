"""SEC EDGAR fundamentals: `sec.py` (pure translation) and `edgar_client.py` (cached I/O)."""

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["SecEdgarClient"]
