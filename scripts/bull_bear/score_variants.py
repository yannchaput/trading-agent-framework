"""TODO item 6: momentum score variants against today's score on the rank-only book.

Spec: docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md

Run from the repo root: uv run python scripts/bull_bear/score_variants.py [--self-check | --book-rules VARIANT...]
"""

import math
import statistics
import sys
from datetime import date, timedelta
from itertools import pairwise
from typing import cast

import pandas as pd
from rank_only_stress import CACHE, SLEEVE, TOP_N, WEIGHTS, WINDOWS, eligible, load_data, max_drawdown, simulate

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
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
OUT = CACHE / "score_variants_equity.csv"
BOOK = BullBearParams()


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


def book_picks(ranked: list[str], held: set[str], top_n: int = BOOK.max_book_size, keep_rank: int = BOOK.retention_rank) -> list[str]:
    """bull_bear's book rule without the judge: keep holdings ranked within `keep_rank`, fill to `top_n` from the top.

    No re-entry cooldown: in bull_bear only a judge sale starts one, and every sale here is a forced exit.
    """
    kept = [s for s in ranked[:keep_rank] if s in held]
    new = [s for s in ranked if s not in held]
    return sorted(kept + new[: top_n - len(kept)], key=ranked.index)


def ranked(day: date, held: set[str], listed: list[str], frames: dict[str, pd.DataFrame], sessions: list[date], variant: str) -> list[str]:
    """The listed symbols today's filters keep for a trade on `day`, best first on the variant's score.

    Filters and no-look-ahead slicing are `rank_only_stress.eligible`'s; only the sort key differs from today's.
    """
    spy = frames["SPY"]["close"].to_dict()
    keys = {}
    for symbol, row, past in eligible(day, held, listed, frames, sessions, LONG_HISTORY):
        both = [(close, spy[d]) for d, close in zip(past.index, past["close"].tolist(), strict=True) if d in spy]
        value = variant_score(variant, row, [c for c, _ in both], [m for _, m in both])
        if value is not None:
            keys[symbol] = (value, row.score)  # ties by today's score: many stocks sit exactly at their 52-week high
    return sorted(keys, key=keys.__getitem__, reverse=True)


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


def run(variant: str, first: date, last: date, data: tuple, real_rules: bool = False) -> tuple[pd.Series, dict[str, float]]:
    """A fresh book from `first` to `last` ranked on `variant`: its equity and its stats.

    `real_rules` holds with bull_bear's book rules (`book_picks`) instead of resetting to the top `TOP_N`.
    """
    history, frames, sessions = data
    window = [day for day in sessions if first <= day <= last]
    books: list[list[str]] = []
    counts: list[int] = []

    def picks(day: date, held: set[str]) -> list[str]:
        order = ranked(day, held, month_universe(history, day, exclude={SLEEVE}), frames, sessions, variant)
        counts.append(len(order))
        books.append(book_picks(order, held) if real_rules else order[:TOP_N])
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
    formats = {"sharpe": "{:.2f}", "ranked": "{:.0f}"}
    for name, stats in results.items():
        cells = [formats.get(key, "{:.1%}").format(stats[key]) if key in stats else "-" for key in columns]
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

    names = [f"N{i}" for i in range(1, 50)]
    names[29], names[39] = "X", "Y"  # ranks 30 and 40
    picks = book_picks(names, {"X", "Y"})
    assert picks == [*names[:9], "X"], picks

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
    # A forced exit starts no cooldown (only a judge sale does, in bull_bear/graph.py): A is bought back at once.
    assert book_picks(["B"], {"A"}, top_n=1, keep_rank=1) == ["B"]
    assert book_picks(["A", "B"], {"B"}, top_n=1, keep_rank=1) == ["A"]
    print("self-check ok")


def book_rules(kept: list[str]) -> None:
    """Step 2: `today` and the kept variants under bull_bear's real book rules, same pass rule on both windows."""
    data = load_data()
    won = {variant: True for variant in kept}
    curves: dict[str, pd.Series] = {}
    for title, window in (("Real book rules, 2005-2015", DEV), ("Real book rules, 2016-2026", HOLDOUT)):
        results: dict[str, dict[str, float]] = {}
        for variant in ["today", *kept]:
            curves[f"{title} {variant}"], results[variant] = run(variant, *window, data, real_rules=True)
        table(title, results)
        for variant in kept:
            won[variant] &= passes(results[variant], results["today"])
    for variant, passed in won.items():
        print(f"{variant} under the real book rules: {'KEPT' if passed else 'FAILED'}")
    pd.DataFrame(curves).sort_index().to_csv(OUT.with_name("score_variants_book_rules_equity.csv"))


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
    stress = pd.read_csv(CACHE / "equity.csv", index_col=0)[f"{WEIGHTS[0]:.3f}"]
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
    elif "--book-rules" in sys.argv:
        self_check()
        book_rules(sys.argv[sys.argv.index("--book-rules") + 1 :])
    else:
        main()
