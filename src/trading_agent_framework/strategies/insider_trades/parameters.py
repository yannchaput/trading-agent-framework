"""`InsiderParams`: every threshold of the insider_trades strategy in one frozen dataclass."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InsiderParams:
    watchlist: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "BAC", "XOM", "CVX", "PFE", "INTC", "F", "KO")  # the stocks the strategy may hold: insiders' Form 4 trades are read for these only
    signal_window_days: int = 30  # how far back, in days, an insider trade still counts as a signal
    history_years: int = 3  # how many years of Form 4 history the research tools may read
    max_total_weight: float = 0.95  # largest sum of stock weights, as a fraction of portfolio value; the rest stays cash
    min_weight: float = 0.03  # smallest weight of one stock; must be at least `rebalance_band`
    max_position_weight: float = 0.20  # largest weight of one stock
    min_buy_holding_fraction: float = 0.01  # an open-market buy counts as a signal from this fraction of the insider's holding up
    min_sell_holding_fraction: float = 0.10  # an open-market sale counts as a signal from this fraction of the insider's holding up
    weight_tolerance: float = 0.001  # slack, as a fraction of portfolio value, when a submitted weight is checked against a limit
    rebalance_band: float = 0.01  # drift, as a fraction of portfolio value, below which a position is left alone
    min_trade_pct: float = 0.005  # smallest order, as a fraction of portfolio value
    max_trade_retries: int = 3  # later ticks that re-run the trading stage after orders were left unfilled (0 disables)
    max_consecutive_abandoned: int = 3  # abandoned runs in a row that abort a backtest
    reason_max_chars: int = 300  # longest `reason` a submit tool accepts
    agent_temperature: float | None = 0.3  # sampling temperature of the agents; None leaves the LLM server's default
    order_wait_seconds: float = 60.0  # how long `check_orders` waits for the orders to reach a final state

    @property
    def even_weight(self) -> float:
        """The weight each watchlist stock gets when the book is split evenly."""
        return self.max_total_weight / len(self.watchlist)

    def __post_init__(self) -> None:
        floats = (
            self.max_total_weight,
            self.min_weight,
            self.max_position_weight,
            self.min_buy_holding_fraction,
            self.min_sell_holding_fraction,
            self.weight_tolerance,
            self.rebalance_band,
            self.min_trade_pct,
            self.order_wait_seconds,
        )
        finite = all(math.isfinite(value) for value in floats)
        tickers = [ticker.strip().upper() for ticker in self.watchlist]
        # every check that divides by, or counts, the watchlist is guarded: the empty case must report, not raise
        has_watchlist = bool(tickers)
        problems = {
            "watchlist must not be empty": not has_watchlist,
            "watchlist tickers must not be blank": not all(tickers),
            "watchlist must not repeat a ticker": len(set(tickers)) != len(tickers),
            "weights, holding fractions, weight_tolerance, rebalance_band, min_trade_pct and order_wait_seconds must be finite": not finite,
            "max_total_weight must be in (0, 1]": finite and not 0 < self.max_total_weight <= 1,
            "min_weight must be above 0 and at most max_position_weight, which is at most max_total_weight": finite and not 0 < self.min_weight <= self.max_position_weight <= self.max_total_weight,
            "min_weight must be at least rebalance_band (a chosen stock below the band would never be bought)": finite and self.min_weight < self.rebalance_band,
            "watchlist x min_weight must fit in max_total_weight": finite and has_watchlist and len(tickers) * self.min_weight > self.max_total_weight + 1e-9,
            "the even weight must lie between min_weight and max_position_weight": finite and has_watchlist and not self.min_weight - 1e-9 <= self.even_weight <= self.max_position_weight + 1e-9,
            "holding fractions must be in (0, 1]": finite and not (0 < self.min_buy_holding_fraction <= 1 and 0 < self.min_sell_holding_fraction <= 1),
            "weight_tolerance must be in [0, 1)": finite and not 0 <= self.weight_tolerance < 1,
            "rebalance_band must be in (0, 1)": finite and not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": finite and not 0 <= self.min_trade_pct < 1,
            "order_wait_seconds must be above 0": finite and self.order_wait_seconds <= 0,
            "signal_window_days must be at least 1": self.signal_window_days < 1,
            "history_years must be at least 1": self.history_years < 1,
            "max_trade_retries must be at least 0": self.max_trade_retries < 0,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "reason_max_chars must be at least 1": self.reason_max_chars < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("InsiderParams: " + "; ".join(failed))
