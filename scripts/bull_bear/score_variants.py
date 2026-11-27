"""TODO item 6: momentum score variants against today's score on the rank-only book.

Spec: docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md

Run from the repo root: uv run python scripts/bull_bear/score_variants.py [--self-check]
"""

import math
import statistics
import sys
from itertools import pairwise

from trading_agent_framework.strategies.common.scoring import (
    MomentumRow,
    annualized_volatility,
    compute_return_from_prices,
    momentum_score,
    score_stock,
)
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

VARIANTS = ("today", "vol_scaled", "residual", "high_52w", "r3_skip")
RESIDUAL_SESSIONS = 36 * 21 + 1


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
    print("self-check ok")


def main() -> None:
    """Run both windows and print the verdicts."""
    raise NotImplementedError


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
