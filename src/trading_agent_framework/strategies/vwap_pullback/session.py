"""Per-session state of the vwap_pullback strategy: rebuilt at every session, kept in `strategy.vars.session`.

Nothing here survives a restart or carries over to the next day, by design: the strategy is strictly
intraday, so a fresh session starts from an empty book and fresh candidates. `Scanner.prepare_session`
builds it, `Scanner.scan` and `Desk` mutate it, `tools.py` renders it for the agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent_framework.strategies.vwap_pullback.features import BarContext, BarStamp
from trading_agent_framework.strategies.vwap_pullback.setups import Setup
from trading_agent_framework.strategies.vwap_pullback.trades import TradeBook
from trading_agent_framework.utils.clock import MarketSession


@dataclass
class CandidateInfo:
    """A stage-1 survivor: its daily measures, plus its latest stage-2 scores.

    The daily fields are fixed for the session; the z-scores are overwritten by every scan that ranks
    the symbol (they stay at their last value when it drops out of the ranking).
    """

    symbol: str
    daily_atr: float  # daily ATR(14), the unit of every ATR-based threshold for this symbol
    beta: float  # 60-day beta to the benchmark, used for beta-adjusted relative strength
    composite: float = 0.0  # mean of z(return), z(RS), z(RVOL) at the last scan
    z_rs: float = 0.0  # cross-sectional z-score of relative strength (the "catalyst: none" gate reads it)
    z_rvol: float = 0.0  # cross-sectional z-score of relative volume (same)


@dataclass
class SessionState:
    """Everything the strategy knows about the current trading session."""

    day: date  # the session's calendar date, market time; a different date means a new session
    session: MarketSession  # open/close timestamps (early closes included)
    bar_stamp: BarStamp  # "close" in backtests, "open" live: how minute bars are indexed
    session_open_equity: Decimal  # equity at preparation, the base of the daily loss circuit breaker
    candidates: dict[str, CandidateInfo] = field(default_factory=dict)  # stage-1 survivors, by symbol
    baselines: dict[str, pd.Series] = field(default_factory=dict)  # per-minute cumulative-volume baselines (RVOL)
    setups: dict[str, Setup] = field(default_factory=dict)  # tracked setups, advanced every tick
    contexts: dict[str, list[BarContext]] = field(default_factory=dict)  # today's completed 5-minute bars per candidate
    book: TradeBook = field(default_factory=TradeBook)  # this session's trades (pending, open, closed)
    headlines: dict[str, list[dict[str, str]]] = field(default_factory=dict)  # lean headlines per symbol
    headlines_fetched_at: dict[str, datetime] = field(default_factory=dict)  # throttles news fetches per symbol
    new_headline: set[str] = field(default_factory=set)  # symbols with a headline the exit agent has not reviewed
    decided: set[str] = field(default_factory=set)  # triggered symbols the entry agent entered or passed this tick
    unknown_positions_checked: bool = False  # the restart check (orphan orders/positions) ran for this session
    flattened: bool = False  # the 15:50 flatten ran: no more entries, late fills are sold instead of protected
