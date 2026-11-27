"""TODO item 6: momentum score variants against today's score on the rank-only book.

Spec: docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md

Run from the repo root: uv run python scripts/bull_bear/score_variants.py [--self-check]
"""

import math
import statistics
import sys
from bisect import bisect_left
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import cast

import pandas as pd
from rank_only_stress import CLOSES, SLEEVE, TOP_N, WEIGHTS, WINDOWS, load_data, max_drawdown, simulate

from trading_agent_framework.strategies.common.scoring import (
    MomentumRow,
    annualized_volatility,
    compute_return_from_prices,
    momentum_score,
    score_stock,
)
from trading_agent_framework.strategies.common.universe_history import month_universe
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

VARIANTS = ("today", "vol_scaled", "residual", "high_52w", "r3_skip")
RESIDUAL_SESSIONS = 36 * 21 + 1
LONG_HISTORY = 800  # sessions read per stock: residual momentum needs RESIDUAL_SESSIONS
DEV = (date(2005, 1, 1), date(2015, 12, 31))
HOLDOUT = (date(2016, 1, 1), date(2026, 11, 24))
# The cached bars start in 2003-06, so no stock has RESIDUAL_SESSIONS before 2006-06-02: residual and its own
# `today` book start here in development.
RESIDUAL_DEV_START = date(2006, 7, 1)
OUT = Path("cache/stress_2005/score_variants_equity.csv")


def residual_score(closes: list[float], spy: list[float]) -> float | None:
    """12-1 month momentum of the returns SPY does not explain, over their own deviation (market-only Blitz et al.).

    36 returns of 21 sessions each are regressed on SPY's; the score covers months 2 to 12 back (the latest is skipped).
    """
    if len(closes) < RESIDUAL_SESSIONS:
        return None
    stock, market = closes[-RESIDUAL_SESSIONS::21], spy[-RESIDUAL_SESSIONS::21]
    stock = [after / before - 1 for before, after in pairwise(stock)]
    market = [after / before - 1 for before, after in pairwise(market)]
    slope, intercept = statistics.linear_regression(market, stock)
    window = [s - intercept - slope * m for s, m in zip(stock, market, strict=True)][-12:-1]
    spread = statistics.stdev(window)
    # An exact fit leaves float noise, not a zero.
    return sum(window) / spread if spread > 1e-12 else None


def variant_score(name: str, row: MomentumRow, closes: list[float], spy: list[float]) -> float | None:
    """The variant's score of a stock that passed today's filters, or None when it has none.

    `closes` and `spy` are the same sessions, oldest first, all before the trade day; `row` is today's score of them.
    """
    value: float | None = None
    if name == "today":
        value = row.score
    elif name == "vol_scaled":
        volatility = annualized_volatility(closes, 126)
        value = row.score / volatility if volatility else None
    elif name == "residual":
        value = residual_score(closes, spy)
    elif name == "high_52w":
        value = closes[-1] / max(closes[-252:])
    elif name == "r3_skip":
        recent = compute_return_from_prices(closes, 63, CONFIG["skip_days"])
        if recent is not None:
            value = momentum_score(row.inputs.ret_12_1m, row.inputs.ret_6_1m, recent, CONFIG["w_12m"], CONFIG["w_6m"], CONFIG["w_3m"])
    else:
        raise ValueError(f"unknown variant {name}")
    return value if value is not None and math.isfinite(value) else None


def sort_key(variant: float, today: float) -> tuple[float, float]:
    """Best first on the variant's score, ties by today's (many stocks sit exactly at their 52-week high)."""
    return -variant, -today


def book_picks(ranked: list[str], held: set[str], cooldown: dict[str, int], top_n: int = 10, keep_rank: int = 35) -> list[str]:
    """bull_bear's book rule without the judge: keep holdings ranked within `keep_rank`, fill to `top_n` from the top."""
    kept = [s for s in ranked[:keep_rank] if s in held]
    new = [s for s in ranked if s not in held and s not in cooldown]
    return sorted(kept + new[: top_n - len(kept)], key=ranked.index)


def advance_cooldown(cooldown: dict[str, int], sold: set[str], reviews: int = 4) -> dict[str, int]:
    """One rebalance later: old blocks run down, and the names just sold are blocked for the next `reviews` ones."""
    left = {s: n - 1 for s, n in cooldown.items() if n > 1}
    return left | dict.fromkeys(sold, reviews)


def ranked(day: date, held: set[str], listed: list[str], frames: dict[str, pd.DataFrame], sessions: list[date], variant: str) -> list[str]:
    """The listed symbols today's filters keep for a trade on `day`, best first on the variant's score.

    The filters, the held volatility exemption and the no-look-ahead slicing are those of
    `rank_only_stress.ranked_symbols`; only the sort key changes. Only sessions strictly before `day` are read.
    """
    previous = sessions[bisect_left(sessions, day) - 1]
    exempt = {**CONFIG, "max_volatility": math.inf}
    spy = frames["SPY"]["close"].to_dict()
    keys = {}
    for symbol in listed:
        frame = frames.get(symbol)
        if frame is None:
            continue
        end = bisect_left(frame.index, day)
        if end == 0 or frame.index[end - 1] != previous:
            continue
        past = frame.iloc[max(0, end - LONG_HISTORY) : end]
        recent = past.iloc[-CLOSES:]
        row = score_stock(
            symbol,
            recent["close"].tolist(),
            recent["volume"].tolist(),
            exempt if symbol in held else CONFIG,
            traded_closes=recent["traded_close"].tolist(),
            traded_volumes=recent["traded_volume"].tolist(),
        )
        if row is None:
            continue
        both = [(close, spy[d]) for d, close in zip(past.index, past["close"].tolist(), strict=True) if d in spy]
        value = variant_score(variant, row, [c for c, _ in both], [m for _, m in both])
        if value is not None:
            keys[symbol] = sort_key(value, row.score)
    return sorted(keys, key=keys.__getitem__)


def window_stats(equity: pd.Series) -> dict[str, float]:
    """CAGR, daily Sharpe (no risk-free rate), max drawdown, and the worst fall inside each stress window covered."""
    first, last = cast("date", equity.index[0]), cast("date", equity.index[-1])
    years = (last - first).days / 365.25
    daily = equity.pct_change().dropna()
    stats = {
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)),
        "max_dd": max_drawdown(equity)[0],
    }
    for label, (start, stop) in WINDOWS.items():
        part = cast("pd.Series", equity[(equity.index >= start) & (equity.index <= stop)])
        if len(part):
            stats[label] = max_drawdown(part)[0]
    return stats


def passes(stats: dict[str, float], base: dict[str, float]) -> bool:
    """The user's rule: a higher CAGR than today's and a Sharpe no lower."""
    return stats["cagr"] > base["cagr"] and stats["sharpe"] >= base["sharpe"]


def run(variant: str, first: date, last: date, data: tuple) -> tuple[pd.Series, dict[str, float]]:
    """A fresh book from `first` to `last` ranked on `variant`: its equity and its stats."""
    history, frames, sessions = data
    window = [day for day in sessions if first <= day <= last]
    books: list[list[str]] = []
    counts: list[int] = []

    def picks(day: date, held: set[str]) -> list[str]:
        order = ranked(day, held, month_universe(history, day, exclude={SLEEVE}), frames, sessions, variant)
        counts.append(len(order))
        books.append(order[:TOP_N])
        return books[-1]

    equity = simulate(frames, window, picks, WEIGHTS[0])
    stats = window_stats(equity)
    stats["ranked"] = sum(counts) / len(counts)
    stats["turnover"] = statistics.mean(len(set(now) - set(before)) / TOP_N for before, now in pairwise(books))
    return equity, stats


def table(title: str, results: dict[str, dict[str, float]]) -> None:
    """Print one markdown row of stats per book."""
    columns = list(dict.fromkeys(key for stats in results.values() for key in stats))
    print(f"\n### {title}\n\n| Book | {' | '.join(columns)} |")
    print("|" + "---|" * (len(columns) + 1))
    for name, stats in results.items():
        cells = []
        for key in columns:
            value = stats.get(key)
            if value is None:
                cells.append("-")
            else:
                cells.append(f"{value:.2f}" if key == "sharpe" else f"{value:.0f}" if key == "ranked" else f"{value:.1%}")
        print(f"| {name} | {' | '.join(cells)} |", flush=True)


def _from_months(returns: list[float]) -> list[float]:
    """Daily closes whose every 21-session step compounds to the given monthly return."""
    closes = [100.0]
    for monthly in returns:
        start = closes[-1]
        closes += [start * (1 + monthly) ** (day / 21) for day in range(1, 22)]
    return closes


def self_check() -> None:
    """Assert each score and the book rule on hand-worked cases."""
    closes = [20.0 * 1.002**i * (1.01 if i % 2 else 1.0) for i in range(300)]
    row = score_stock("A", closes, [5e6] * 300, CONFIG)
    assert row is not None
    assert variant_score("today", row, closes, closes) == row.score

    vol = annualized_volatility(closes, 126)
    assert vol is not None and variant_score("vol_scaled", row, closes, closes) == row.score / vol
    assert variant_score("vol_scaled", row, [10.0] * 300, closes) is None  # zero volatility

    peak = [5.0] * 300 + [10.0, 12.0, 9.0, 11.0]
    assert variant_score("high_52w", row, peak, peak) == 11 / 12
    assert variant_score("high_52w", row, [50.0] + [10.0] * 252, closes) == 1.0  # the old high is outside 252

    r3 = closes[-22] / closes[-85] - 1
    expected = momentum_score(row.inputs.ret_12_1m, row.inputs.ret_6_1m, r3, CONFIG["w_12m"], CONFIG["w_6m"], CONFIG["w_3m"])
    assert variant_score("r3_skip", row, closes, closes) == expected

    market = [0.03 if i % 3 else -0.02 for i in range(36)]
    spy = _from_months(market)
    assert len(spy) == 757
    assert variant_score("residual", row, _from_months([2 * m for m in market]), spy) is None  # exact fit
    noise = [0.005 if i % 2 else -0.005 for i in range(24)] + [0.01 + (0.001 if i % 2 else -0.001) for i in range(11)] + [0.0]
    winner = _from_months([m + e for m, e in zip(market, noise, strict=True)])
    loser = _from_months([m - e for m, e in zip(market, noise, strict=True)])
    up, down = variant_score("residual", row, winner, spy), variant_score("residual", row, loser, spy)
    assert up is not None and up > 0 and down is not None and down < 0
    assert variant_score("residual", row, winner[1:], spy[1:]) is None  # 756 closes

    tied = [("low", 1.0, 0.1), ("far", 0.9, 5.0), ("high", 1.0, 0.9)]
    assert [name for name, v, t in sorted(tied, key=lambda x: sort_key(x[1], x[2]))] == ["high", "low", "far"]

    names = [f"N{i}" for i in range(1, 50)]
    names[29], names[39], names[0] = "X", "Y", "Z"  # ranks 30, 40, 1
    picks = book_picks(names, {"X", "Y"}, {"Z": 1})
    assert picks == [*names[1:10], "X"], picks
    assert advance_cooldown({"Z": 1, "W": 3}, {"Y"}) == {"W": 2, "Y": 4}
    assert passes({"cagr": 0.2, "sharpe": 1.0}, {"cagr": 0.1, "sharpe": 1.0})
    assert not passes({"cagr": 0.2, "sharpe": 0.9}, {"cagr": 0.1, "sharpe": 1.0})
    assert not passes({"cagr": 0.1, "sharpe": 1.5}, {"cagr": 0.1, "sharpe": 1.0})

    days = [date(2023, 1, 1) + timedelta(days=i) for i in range(301)]

    def stock(drift: float, on: list[date]) -> pd.DataFrame:
        close = [20.0 * (1 + drift) ** i * (1.001 if i % 2 else 1.0) for i in range(len(on))]
        return pd.DataFrame({"close": close, "volume": 5e6, "traded_close": close, "traded_volume": 5e6}, index=on)

    frames = {"UP": stock(0.002, days), "FLAT": stock(0.0005, days), "HALT": stock(0.003, days[:-2]), "SPY": stock(0.001, days)}
    frames["UP"].loc[days[-1], ["close", "traded_close"]] = 0.01  # the trade day's bar must not be read
    listed = ["FLAT", "UP", "HALT"]
    assert ranked(days[-1], set(), listed, frames, days, "today") == ["UP", "FLAT"]  # HALT: no previous-session bar
    assert ranked(days[-1], set(), listed, frames, days, "high_52w") == ["UP", "FLAT"]
    assert ranked(days[-1], set(), listed, frames, days, "residual") == []  # under 757 sessions
    print("self-check ok")


def main() -> None:
    """Run development for every variant, then the holdout once for those that passed, and print the verdicts."""
    self_check()
    data = load_data()
    curves: dict[str, pd.Series] = {}
    dev: dict[str, dict[str, float]] = {}
    passed = []
    for variant in VARIANTS:
        first = RESIDUAL_DEV_START if variant == "residual" else DEV[0]
        curves[f"dev {variant}"], dev[variant] = run(variant, first, DEV[1], data)
        base = dev["today"]
        if variant == "residual":
            curves["dev today from 2006-07"], base = run("today", first, DEV[1], data)
            dev["today from 2006-07"] = base
        if variant != "today" and passes(dev[variant], base):
            passed.append(variant)
        table(f"Development, after {variant}", dev)
    print(f"\nPassed development: {passed or 'none'}")
    stress = pd.read_csv("cache/stress_2005/equity.csv", index_col=0)["0.098"]
    stress.index = pd.Index([date.fromisoformat(str(day)) for day in stress.index])
    stress = cast("pd.Series", stress[(stress.index >= DEV[0]) & (stress.index <= DEV[1])])
    expected = window_stats(stress)["cagr"]
    print(f"Sanity: today dev CAGR {dev['today']['cagr']:.2%}, stress test curve {expected:.2%}")
    assert abs(dev["today"]["cagr"] - expected) < 0.01, "today does not reproduce the stress test's book"
    holdout: dict[str, dict[str, float]] = {}
    if passed:
        for variant in ["today", *passed]:
            curves[f"holdout {variant}"], holdout[variant] = run(variant, *HOLDOUT, data)
        table("Holdout 2016-2026", holdout)
    for variant in VARIANTS[1:]:
        if variant not in passed:
            verdict = "FAILED development"
        elif passes(holdout[variant], holdout["today"]):
            verdict = "KEPT"
        else:
            verdict = "FAILED holdout"
        print(f"{variant}: {verdict}")
    pd.DataFrame(curves).sort_index().to_csv(OUT)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
