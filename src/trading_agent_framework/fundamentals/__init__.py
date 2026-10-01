"""SEC EDGAR fundamentals: `sec.py` (pure translation), `edgar_client.py` (cached I/O), and the quality
screen (`quality.py` pure gates and score, `annual_store.py` and `splits.py` I/O, `screen.py` wiring)."""

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.quality import Candidate, ScreenParams, ScreenResult
from trading_agent_framework.fundamentals.screen import QualityScreen, build_quality_screen
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["Candidate", "QualityScreen", "ScreenParams", "ScreenResult", "SecEdgarClient", "build_quality_screen"]
