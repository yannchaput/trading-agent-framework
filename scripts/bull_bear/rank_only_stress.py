"""TODO item 5: stress test of bull_bear's rank-only book, 2005-2026.

Spec: docs/superpowers/specs/2026-11-27-bull-bear-rank-only-stress-test-design.md

Run from the repo root: uv run python scripts/bull_bear/rank_only_stress.py [--self-check]
"""

import sys
from collections.abc import Callable
from datetime import date

import pandas as pd


def rebalance_days(sessions: list[date]) -> list[date]:
    """Per ISO week, the first session on a Tuesday or later."""
    weeks: dict[tuple[int, int], date] = {}
    for day in sorted(sessions):
        if day.weekday() >= 1:
            weeks.setdefault(day.isocalendar()[:2], day)
    return list(weeks.values())


def simulate(
    prices: dict[str, pd.DataFrame],
    sessions: list[date],
    picks: Callable[[date, set[str]], list[str]],
    weight: float,
    cost: float = 0.0005,
    sleeve: str = "SHV",
) -> pd.Series:
    """Equity at each session close, from 1.0.

    On a rebalance day every holding is valued at that open (else its last close) and reset to `weight` per pick; the
    rest goes to `sleeve` when it has an open, else stays cash at 0%. `cost` is charged on every dollar traded. A pick
    with no open that day is left out.

    shortcut: a holding whose prices stop keeps its last close until the next rebalance, where it is sold at it.
    """
    opens = {symbol: frame["open"].to_dict() for symbol, frame in prices.items()}
    closes = {symbol: frame["close"].to_dict() for symbol, frame in prices.items()}
    rebalances = set(rebalance_days(sessions))
    cash, shares, last = 1.0, {}, {}
    equity = []
    for day in sessions:
        if day in rebalances:
            held = {symbol for symbol in shares if symbol != sleeve}
            wanted = [s for s in picks(day, held) if day in opens.get(s, {})]
            at_open = {s: opens[s][day] if day in opens[s] else last[s] for s in {*shares, *wanted}}
            if day in opens.get(sleeve, {}):
                at_open[sleeve] = opens[sleeve][day]
            old = {s: n * at_open[s] for s, n in shares.items()}
            value = cash + sum(old.values())
            new = {s: weight * value for s in wanted}
            rest = value - sum(new.values())
            if sleeve in at_open:
                new[sleeve] = rest
            fees = cost * sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in {*new, *old})
            if sleeve in new:
                new[sleeve] -= fees
                cash = 0.0
            else:
                cash = rest - fees
            shares = {s: dollars / at_open[s] for s, dollars in new.items()}
            last.update({s: at_open[s] for s in shares})
        for symbol in shares:
            if day in closes[symbol]:
                last[symbol] = closes[symbol][day]
        equity.append(cash + sum(n * last[s] for s, n in shares.items()))
    return pd.Series(equity, index=sessions)


def max_drawdown(equity: pd.Series) -> tuple[float, date, date]:
    """(worst fall as a negative fraction, its peak date, its trough date)."""
    falls = equity / equity.cummax() - 1
    trough = falls.idxmin()
    return float(falls.min()), equity.loc[:trough].idxmax(), trough


def _frame(rows: list[tuple[date, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["day", "open", "close"]).set_index("day")


def self_check() -> None:
    """Assert the weekly book and drawdown maths on hand-worked cases."""
    days = [date(2024, 1, d) for d in (2, 3, 4, 8, 10, 11)]  # Tue, Wed, Thu, Mon, Wed (holiday Tue), Thu
    assert rebalance_days(days) == [date(2024, 1, 2), date(2024, 1, 10)]
    dead = {"A": _frame([(days[0], 10.0, 10.0), (days[1], 10.0, 8.0)])}  # prices stop after day 2
    equity = simulate(dead, days, lambda day, held: ["A"] if day == days[0] else [], weight=0.5, cost=0.0)
    assert [round(v, 6) for v in equity] == [1.0, 0.9, 0.9, 0.9, 0.9, 0.9]  # sold at its last close
    costly = simulate(dead, days[:2], lambda day, held: ["A"], weight=0.5, cost=0.001)
    assert round(costly.iloc[0], 6) == 0.9995
    # With a sleeve: the 50% not in A goes to SHV and earns its return; both trades pay the cost.
    with_sleeve = {"A": _frame([(d, 10.0, 10.0) for d in days[:2]]), "SHV": _frame([(days[0], 100.0, 100.0), (days[1], 100.0, 102.0)])}
    parked = simulate(with_sleeve, days[:2], lambda day, held: ["A"], weight=0.5, cost=0.001)
    assert [round(v, 6) for v in parked] == [0.999, 1.00898]  # 0.499 in SHV after the cost, up 2%
    seen: list[tuple[date, set[str]]] = []
    simulate(dead, days, lambda day, held: seen.append((day, held)) or ["A"], weight=0.5)
    assert seen[1] == (days[4], {"A"})
    dd, peak, trough = max_drawdown(pd.Series([1.0, 2.0, 1.0, 1.5], index=days[:4]))
    assert (dd, peak, trough) == (-0.5, days[1], days[2])
    print("self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
