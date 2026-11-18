"""Pure research loop for the bear put spread: a day-by-day replay over aligned SPY, VIX and regime series.

Every decision comes from `rules.py` and every price from `pricing.py`; this module only holds the state (cash and
the one open spread) and the bookkeeping. Positions are valued at model mid minus the cost of closing them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from trading_agent_framework.strategies.bear_put_spread.pricing import CostModel, VolModel, spread_mid
from trading_agent_framework.strategies.bear_put_spread.rules import SpreadParameters, contracts_for, exit_reason, monthly_expiry, should_open, strikes


@dataclass(frozen=True, slots=True)
class Trade:
    """One spread from entry to exit (or to the end of the window).

    Attributes:
        entry: Entry session.
        exit: Exit session; None while still open at the window's end.
        reason: `"time"`, `"target"`, `"regime"` or `"open"` (still open).
        spy_entry: SPY at entry.
        spy_exit: SPY at exit, or at the last session while open.
        long_strike: Strike of the bought put.
        short_strike: Strike of the sold put.
        expiry: Expiry date.
        contracts: Number of spreads.
        debit: Entry value per share (mid).
        exit_value: Exit value per share (mid), or the last value while open.
        costs: Entry plus exit costs in dollars (the exit costs are assumed while open).
        pnl: `(exit_value - debit) * 100 * contracts - costs`, dollars.
        pnl_pct: `pnl` over the equity at entry.
    """

    entry: date
    exit: date | None
    reason: str
    spy_entry: float
    spy_exit: float
    long_strike: float
    short_strike: float
    expiry: date
    contracts: int
    debit: float
    exit_value: float
    costs: float
    pnl: float
    pnl_pct: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """What a replay produced.

    Attributes:
        trades: Every spread, oldest first; the last may be open.
        equity: `(session, equity)` per session, positions valued net of exit costs.
        skipped: Bearish sessions with no spread open where not even one contract fit.
    """

    trades: list[Trade]
    equity: list[tuple[date, float]]
    skipped: int


@dataclass(slots=True)
class _Open:
    entry: date
    spy: float
    long_strike: float
    short_strike: float
    expiry: date
    contracts: int
    debit: float
    equity: float


def _trade(pos: _Open, exit_day: date | None, reason: str, spy: float, value: float, costs: CostModel) -> Trade:
    total_costs = costs.open_cost(pos.contracts) + costs.close_cost(pos.contracts)
    pnl = (value - pos.debit) * 100 * pos.contracts - total_costs
    return Trade(pos.entry, exit_day, reason, pos.spy, spy, pos.long_strike, pos.short_strike, pos.expiry, pos.contracts, pos.debit, value, total_costs, pnl, pnl / pos.equity)


def simulate(
    sessions: Sequence[date],
    spy: Sequence[float],
    vix: Sequence[float],
    regimes: Mapping[date, int],
    params: SpreadParameters,
    model: VolModel,
    costs: CostModel,
    start_equity: float,
) -> SimulationResult:
    """Replay the strategy over `sessions`, deciding and trading at each session's close.

    Args:
        sessions: Session dates, oldest first; each must have a regime reading.
        spy: SPY closes aligned with `sessions`.
        vix: VIX closes aligned with `sessions`.
        regimes: The regime (1, 0, -1) each session would have had, from the closes before it.
        params: Strategy rules.
        model: Volatility model.
        costs: Cost model.
        start_equity: Starting cash, dollars; idle cash earns nothing.

    Returns:
        The trades, the equity per session and the count of skipped entries.
    """
    cash = start_equity
    pos: _Open | None = None
    trades: list[Trade] = []
    equity: list[tuple[date, float]] = []
    skipped = 0
    value = 0.0
    for day, price, vol_index in zip(sessions, spy, vix, strict=True):
        if pos is not None:
            value = spread_mid(price, pos.long_strike, pos.short_strike, vol_index, (pos.expiry - day).days / 365, model)
            reason = exit_reason(day, pos.expiry, regimes[day], value, pos.long_strike - pos.short_strike, params)
            if reason is not None:
                cash += value * 100 * pos.contracts - costs.close_cost(pos.contracts)
                trades.append(_trade(pos, day, reason, price, value, costs))
                pos = None
                equity.append((day, cash))
                continue  # a roll waits for the next session
        elif should_open(regimes[day], False):
            long_strike, short_strike = strikes(price, params)
            expiry = monthly_expiry(day, params)
            debit = spread_mid(price, long_strike, short_strike, vol_index, (expiry - day).days / 365, model)
            count = contracts_for(cash, debit, costs.open_cost(1), params)
            if count == 0:
                skipped += 1
            else:
                pos = _Open(day, price, long_strike, short_strike, expiry, count, debit, cash)
                cash -= debit * 100 * count + costs.open_cost(count)
                value = debit
        mark = 0.0 if pos is None else value * 100 * pos.contracts - costs.close_cost(pos.contracts)
        equity.append((day, cash + mark))
    if pos is not None:
        trades.append(_trade(pos, None, "open", spy[-1], value, costs))
    return SimulationResult(trades, equity, skipped)
