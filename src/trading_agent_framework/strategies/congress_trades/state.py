"""What the strategy remembers between daily checks.

`StateStore` keeps, in one small JSON file per mode written atomically: the DocIDs of the filings the last completed run was
built from (`processed`, the "nothing new" test), that run's holdings and target, the tickers this strategy has ever ordered
(`traded`: the only positions it may sell), a trade left unfilled (`pending_trade`) and the abandoned-run streak. A missing or
corrupt file is an empty state: the next check then treats every known filing as new and builds the portfolio again.
`StateStore` never raises on an I/O problem: bookkeeping must not stop a run (the per-run log, `RunLog`, is in `strategies/common/run_log.py`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.strategies.common.json_state import InvalidState, JsonStateStore, is_int, valid_strings, valid_weights
from trading_agent_framework.strategies.common.pipeline import PendingTrade, decode_pending, valid_pending
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "CongressTradesState")

STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class CongressState:
    """What congress_trades carries from one run to the next.

    Attributes:
        processed: DocIDs of the yearly report and PTRs the last completed run was built from.
        holdings: Ticker -> `{tier, value_low, value_high}` after that run.
        target: Ticker -> weight of that run's portfolio.
        traded: Tickers ever ordered by this strategy.
        pending_trade: Orders left unfilled that a later tick retries, if any.
        abandoned_streak: Abandoned runs in a row.
        last_run: ISO date of the last completed run.
    """

    processed: list[str] = field(default_factory=list)  # DocIDs of the yearly report and PTRs the last completed run was built from
    holdings: dict[str, dict[str, int]] = field(default_factory=dict)  # ticker -> {tier, value_low, value_high} after that run
    target: dict[str, float] = field(default_factory=dict)  # ticker -> weight of that run's portfolio
    traded: list[str] = field(default_factory=list)  # tickers ever ordered by this strategy
    pending_trade: PendingTrade | None = None
    abandoned_streak: int = 0  # abandoned runs in a row
    last_run: str | None = None  # ISO date of the last completed run


def _valid_holdings(value: Any) -> bool:
    keys = {"tier", "value_low", "value_high"}
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, dict) and set(v) == keys and all(is_int(n) for n in v.values()) for k, v in value.items())


def _valid(raw: dict[str, Any]) -> bool:
    last_run = raw.get("last_run")
    streak = raw.get("abandoned_streak", 0)
    return (
        valid_strings(raw.get("processed", []))
        and _valid_holdings(raw.get("holdings", {}))
        and valid_weights(raw.get("target", {}))
        and valid_strings(raw.get("traded", []))
        and valid_pending(raw.get("pending_trade"))
        and is_int(streak)
        and streak >= 0
        and (last_run is None or isinstance(last_run, str))
    )


def _decode(raw: dict[str, Any]) -> CongressState:
    if not _valid(raw):
        raise InvalidState
    return CongressState(
        processed=list(raw.get("processed", [])),
        holdings={ticker: dict(record) for ticker, record in raw.get("holdings", {}).items()},
        target=dict(raw.get("target", {})),
        traded=list(raw.get("traded", [])),
        pending_trade=decode_pending(raw.get("pending_trade")),
        abandoned_streak=raw.get("abandoned_streak", 0),
        last_run=raw.get("last_run"),
    )


class StateStore(JsonStateStore[CongressState]):
    """JSON store of the congress_trades state."""

    def __init__(self, path: Path, *, fresh: bool = False) -> None:
        """`fresh` wipes the file at construction; used in backtests so one run never inherits the last."""
        super().__init__(path, version=STATE_VERSION, empty=CongressState, decode=_decode, encode=asdict, log=logger, fresh=fresh)
