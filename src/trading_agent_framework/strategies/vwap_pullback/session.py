"""Per-session state of the vwap_pullback strategy: rebuilt at every session, kept in `strategy.vars.session`."""

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
    """A stage-1 survivor: its daily measures, plus its latest stage-2 scores."""

    symbol: str
    daily_atr: float
    beta: float
    composite: float = 0.0
    z_rs: float = 0.0
    z_rvol: float = 0.0


@dataclass
class SessionState:
    day: date
    session: MarketSession
    bar_stamp: BarStamp
    session_open_equity: Decimal
    candidates: dict[str, CandidateInfo] = field(default_factory=dict)
    baselines: dict[str, pd.Series] = field(default_factory=dict)
    setups: dict[str, Setup] = field(default_factory=dict)
    contexts: dict[str, list[BarContext]] = field(default_factory=dict)
    book: TradeBook = field(default_factory=TradeBook)
    headlines: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    headlines_fetched_at: dict[str, datetime] = field(default_factory=dict)
    new_headline: set[str] = field(default_factory=set)  # symbols with a headline the exit agent has not reviewed
    decided: set[str] = field(default_factory=set)  # triggered symbols the entry agent entered or passed this tick
    unknown_positions_checked: bool = False
    flattened: bool = False
