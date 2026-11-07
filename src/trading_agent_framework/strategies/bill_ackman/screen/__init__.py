"""The fundamentals quality screen: which companies were simple, predictable, cash-generative, lightly indebted
and reasonably priced on a date (`quality.py` pure gates and score, `annual_figures.py` pure SEC reduction,
`annual_store.py` and `splits.py` I/O, `screen.py` wiring).
"""

from trading_agent_framework.strategies.bill_ackman.screen.quality import Candidate, ScreenParams, ScreenResult
from trading_agent_framework.strategies.bill_ackman.screen.screen import QualityScreen, build_quality_screen
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["Candidate", "QualityScreen", "ScreenParams", "ScreenResult", "build_quality_screen"]
