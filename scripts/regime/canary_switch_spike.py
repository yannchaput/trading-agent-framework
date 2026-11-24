"""Spike: does Keller's canary switch (BAA) cut bull_bear's falls without giving up its gains?

Design (rules fixed before running): docs/superpowers/specs/2026-11-24-canary-switch-spike-design.md

Run:
  uv run python scripts/regime/canary_switch_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/canary_switch_spike.py run     # replay on bull_bear's two saved runs and on SPY 2008-2026
"""

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd


def _main_checkout() -> Path:
    """Root of the main checkout: the git-ignored inputs (logs/, cache/) live there, not in a worktree."""
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=True)
    return Path(common.stdout.strip()).parent


MAIN = _main_checkout()
CACHE = MAIN / "cache" / "canary_switch"
CANARIES = ("SPY", "VWO", "VEA", "BND")
DEFENSIVE = ("TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND")
BULL = "BULL"  # the bull leg's key in a target: bull_bear (or SPY in the sanity check)
STOCK_COST = 0.001  # 10 bps on the bull_bear side of a trade
ETF_COST = 0.0005  # 5 bps on the ETF side
SHARE = 0.85  # the bar: at least this share of bull_bear's gain
MONTHS = 12  # month-end closes before a decision that the momentum scores need


def month_ends(days: list[date]) -> list[date]:
    """The last session of each calendar month in `days` (the last, possibly partial, month included)."""
    last: dict[tuple[int, int], date] = {}
    for day in sorted(days):
        last[(day.year, day.month)] = day
    return list(last.values())


def fast_momentum(p0: float, past: list[float]) -> float:
    """13612W: `(12 r1 + 4 r3 + 2 r6 + r12) / 19`, `rN` the return over the last N month-ends (`past`, oldest first)."""
    r = {n: p0 / past[-n] - 1 for n in (1, 3, 6, 12)}
    return (12 * r[1] + 4 * r[3] + 2 * r[6] + r[12]) / 19


def slow_momentum(p0: float, past: list[float]) -> float:
    """`p0` against the mean of itself and the last 12 month-end closes."""
    return p0 / (sum([p0, *past[-MONTHS:]]) / (len(past[-MONTHS:]) + 1)) - 1


def is_bull(fast: dict[str, float]) -> bool:
    """Bull only when every canary's fast momentum is strictly above 0 (a zero counts as bear)."""
    return all(value > 0 for value in fast.values())


def defensive_weights(slow: dict[str, float]) -> dict[str, float]:
    """The top 3 of `DEFENSIVE` by slow momentum (ties by symbol) at 1/3 each; a pick scoring below BIL becomes BIL."""
    picks = sorted(DEFENSIVE, key=lambda s: (-slow[s], s))[:3]
    weights: dict[str, float] = {}
    for symbol in picks:
        held = "BIL" if slow[symbol] < slow["BIL"] else symbol
        weights[held] = weights.get(held, 0.0) + 1 / 3
    return weights


def decide(px: pd.DataFrame, day: date, ends: list[date]) -> dict[str, float]:
    """The target on `day`: `{BULL: 1.0}` in bull mode, else the defensive weights.

    `px` holds daily closes (date index, one column per ETF); the scores use `day`'s close and the closes of the last
    12 month-ends of `ends` strictly before it. Raises `ValueError` with fewer than 12 of them or a NaN close.
    """
    earlier = [e for e in ends if e < day][-MONTHS:]
    if len(earlier) < MONTHS:
        raise ValueError(f"{day}: only {len(earlier)} earlier month-ends, {MONTHS} needed")

    def closes(symbol: str) -> tuple[float, list[float]]:
        p0, past = float(px.loc[day, symbol]), [float(px.loc[e, symbol]) for e in earlier]
        if pd.isna([p0, *past]).any():
            raise ValueError(f"{day}: a close of {symbol} is missing")
        return p0, past

    if is_bull({s: fast_momentum(*closes(s)) for s in CANARIES}):
        return {BULL: 1.0}
    return defensive_weights({s: slow_momentum(*closes(s)) for s in DEFENSIVE})


def prices(columns: dict[str, list[float]], months: int) -> pd.DataFrame:
    """Month-end closes (one row per month from 2024-01-31) for the given columns."""
    index = [(pd.Timestamp("2024-01-31") + pd.offsets.MonthEnd(i)).date() for i in range(months)]
    return pd.DataFrame(columns, index=index)


def check() -> None:
    """Self-check on synthetic data: no network, no files."""
    # 1. month_ends: the last session of each calendar month in the data (the last, partial month included)
    days = [d.date() for d in pd.bdate_range("2026-01-26", "2026-03-02")]
    assert month_ends(days) == [date(2026, 1, 30), date(2026, 2, 27), date(2026, 3, 2)]
    # 2. fast_momentum
    assert abs(fast_momentum(110.0, [100.0] * 12) - 0.10) < 1e-12 and fast_momentum(100.0, [100.0] * 12) == 0.0
    past = [100.0 + i for i in range(12)]
    hand = (12 * (112 / 111 - 1) + 4 * (112 / 109 - 1) + 2 * (112 / 106 - 1) + (112 / 100 - 1)) / 19
    assert abs(fast_momentum(112.0, past) - hand) < 1e-12
    # 3. is_bull: every canary strictly above 0 (a zero is bear)
    assert is_bull({"SPY": 0.01, "VWO": 0.01, "VEA": 0.01, "BND": 0.01})
    assert not is_bull({"SPY": 0.01, "VWO": 0.0, "VEA": 0.01, "BND": 0.01})
    assert not is_bull({"SPY": 0.01, "VWO": 0.01, "VEA": -0.01, "BND": 0.01})
    # 4. defensive_weights: top 3, a pick below BIL becomes BIL, BIL's weights add up, the total is 1
    rest = {"TIP": -1.0, "DBC": -1.0, "BIL": -1.0, "IEF": -1.0, "TLT": -1.0, "LQD": -1.0, "BND": -1.0}
    top = defensive_weights({**rest, "TLT": 0.05, "IEF": 0.04, "TIP": 0.03, "BIL": 0.01})
    assert {s: round(w, 9) for s, w in top.items()} == {"TLT": round(1 / 3, 9), "IEF": round(1 / 3, 9), "TIP": round(1 / 3, 9)}
    swap = defensive_weights({**rest, "TLT": 0.05, "BIL": 0.01, "IEF": 0.0})
    assert abs(swap["TLT"] - 1 / 3) < 1e-12 and abs(swap["BIL"] - 2 / 3) < 1e-12 and set(swap) == {"TLT", "BIL"}
    assert abs(sum(swap.values()) - 1) < 1e-12 and abs(sum(top.values()) - 1) < 1e-12
    # 5. slow_momentum: the last close against the mean of itself and the last 12 month-end closes
    assert abs(slow_momentum(113.0, [100.0] * 12) - (113 / ((113 + 1200) / 13) - 1)) < 1e-12
    # 6. decide: too little history or a NaN close raises; every canary rising is bull; a falling canary is bear
    symbols = [*CANARIES, "TIP", "DBC", "BIL", "IEF", "TLT", "LQD"]
    rising = {s: [100.0 + i for i in range(14)] for s in dict.fromkeys(symbols)}
    px = prices(rising, 14)
    ends = month_ends(list(px.index))
    assert decide(px, px.index[13], ends) == {BULL: 1.0}
    for short in (px.iloc[:12], px.iloc[:11]):
        try:
            decide(short, short.index[-1], month_ends(list(short.index)))
        except ValueError:
            continue
        raise AssertionError("decide must raise with fewer than 12 earlier month-ends")
    nan = px.copy()
    nan.loc[nan.index[13], "VEA"] = float("nan")
    try:
        decide(nan, nan.index[13], ends)
        raise AssertionError("decide must raise on a NaN close")
    except ValueError:
        pass
    falling = {**rising, "VWO": [200.0 - 5 * i for i in range(14)]}
    bear = decide(prices(falling, 14), px.index[13], ends)
    assert BULL not in bear and abs(sum(bear.values()) - 1) < 1e-12
    print("self-check OK")


def main() -> None:
    """Entry point: `check` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
