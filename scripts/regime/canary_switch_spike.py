"""Spike: does Keller's canary switch (BAA) cut bull_bear's falls without giving up its gains?

Design (rules fixed before running): docs/superpowers/specs/2026-11-24-canary-switch-spike-design.md

Run:
  uv run python scripts/regime/canary_switch_spike.py check   # self-check on synthetic data
  uv run python scripts/regime/canary_switch_spike.py run     # replay on bull_bear's two saved runs and on SPY 2008-2026
"""

import pickle
import subprocess
import sys
from bisect import bisect_right
from datetime import date, timedelta
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
RUNS = {"5Y": "2026-11-05_012455_backtesting", "COVID": "2026-11-09_224910_backtesting"}  # bull_bear's two point-in-time runs
SYMBOLS = sorted({*CANARIES, *DEFENSIVE, "SHV"})
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


def valid_days(px: pd.DataFrame, days: list[date]) -> list[date]:
    """The decision days among `days` that `decide` can score (enough history, no missing close)."""
    ends = month_ends(list(px.index))
    out = []
    for day in days:
        try:
            decide(px, day, ends)
        except ValueError:
            continue
        out.append(day)
    return out


def effective_targets(px: pd.DataFrame, decision_days: list[date], sessions: list[date]) -> dict[date, dict[str, float]]:
    """Each decision keyed by the first session after it (a later decision for the same session wins).

    A decision with no later session is dropped. Decisions before the window all key its first session: the latest
    sets the opening holdings.
    """
    ends = month_ends(list(px.index))
    targets: dict[date, dict[str, float]] = {}
    for day in sorted(decision_days):
        i = bisect_right(sessions, day)
        if i < len(sessions):
            targets[sessions[i]] = decide(px, day, ends)
    return targets


def fixed_targets(weights: dict[str, float], px: pd.DataFrame, sessions: list[date]) -> dict[date, dict[str, float]]:
    """The same weights from the first session and again after every month-end inside the window (monthly rebalance)."""
    targets = {sessions[0]: weights}
    for day in month_ends(list(px.index)):
        i = bisect_right(sessions, day)
        if i < len(sessions):
            targets[sessions[i]] = weights
    return targets


def replay(bull: pd.Series, px: pd.DataFrame, sessions: list[date], targets: dict[date, dict[str, float]], bull_cost: float = STOCK_COST) -> pd.DataFrame:
    """Daily `equity` of an account that holds `targets` (weights by asset, `BULL` = the `bull` curve, else ETFs of `px`).

    A target keyed by a session rebalances at that session's close, after the holdings earned that day's returns, and
    pays `bull_cost` on the change in the BULL leg plus `ETF_COST` on the change in each ETF. The opening target pays
    `ETF_COST` on its ETF part only. `bear` is True on the sessions held in bear mode (before that day's rebalance).
    """
    returns = pd.concat([bull.reindex(sessions).pct_change().rename(BULL), px.reindex(sessions).ffill().pct_change()], axis=1).fillna(0.0)
    target = targets[sessions[0]]
    start = float(bull.reindex(sessions).iloc[0])
    opening_cost = ETF_COST * start * sum(w for a, w in target.items() if a != BULL)
    holdings = {a: w * (start - opening_cost) for a, w in target.items()}
    equity, bear = [], []
    for i, day in enumerate(sessions):
        bear.append(BULL not in target)
        if i > 0:
            holdings = {a: v * (1 + returns.loc[day, a]) for a, v in holdings.items()}
            if day in targets:
                total = sum(holdings.values())
                new = {a: w * total for a, w in targets[day].items()}
                cost = bull_cost * abs(new.get(BULL, 0.0) - holdings.get(BULL, 0.0))
                cost += ETF_COST * sum(abs(new.get(a, 0.0) - holdings.get(a, 0.0)) for a in {*new, *holdings} if a != BULL)
                holdings = {a: w * (total - cost) for a, w in targets[day].items()}
                target = targets[day]
        equity.append(sum(holdings.values()))
    return pd.DataFrame({"equity": equity, "bear": bear}, index=sessions)


def max_drawdown(equity: pd.Series) -> float:
    """The worst fall from a running peak (negative)."""
    return float((equity / equity.cummax() - 1).min())


def gain(equity: pd.Series) -> float:
    """Final value over the first, minus 1."""
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def bear_return(rows: pd.DataFrame) -> float:
    """The compounded daily return of `equity` over the sessions held in bear mode."""
    daily = rows["equity"].pct_change()[rows["bear"]].fillna(0.0)
    return float((1 + daily).prod() - 1)


def passes(gated: dict[str, tuple[float, float]], alone: dict[str, tuple[float, float]], bear_ret: float) -> bool:
    """The bar: per window `(gain, max drawdown)`; a smaller drawdown and a gain of at least `SHARE` x bull_bear's, and a bear return >= 0."""
    return bear_ret >= 0 and all(gated[w][1] > alone[w][1] and gated[w][0] >= SHARE * alone[w][0] for w in alone)


def etf_closes() -> pd.DataFrame:
    """Yahoo total-return daily closes of every ETF from 2006 (date index), cached."""
    path = CACHE / "etfs.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import yfinance as yf

    raw = yf.download(SYMBOLS, start="2006-01-01", progress=False, auto_adjust=True)["Close"][SYMBOLS]
    raw.index = list(raw.index.date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(raw))
    return raw


def bull_curve(run: str) -> pd.Series:
    """bull_bear's `portfolio_value` of a saved run, indexed by session date."""
    curve = pd.read_parquet(MAIN / "logs" / "bull_bear" / "backtesting" / run / "equity.parquet")["portfolio_value"]
    curve.index = [stamp.date() for stamp in curve.index]
    return curve


def line(name: str, equity: pd.Series, rows: pd.DataFrame | None = None) -> str:
    """One report row: gain, max drawdown and, for a switched account, the months in bear mode."""
    bear = f" bear sessions {int(rows['bear'].sum())}/{len(rows)}" if rows is not None else ""
    return f"  {name:34} gain {gain(equity):+8.1%} dd {max_drawdown(equity):6.1%}{bear}"


def stretches(rows: pd.DataFrame, bull: pd.Series) -> list[str]:
    """Each run of bear-mode sessions: its dates, the account's return and bull_bear's over the same sessions."""
    out, start = [], None
    flags = list(rows["bear"]) + [False]
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            first, last = rows.index[start], rows.index[i - 1]
            mine = rows["equity"].iloc[i - 1] / rows["equity"].iloc[start - 1] - 1 if start > 0 else 0.0
            theirs = bull.loc[last] / bull.loc[rows.index[start - 1]] - 1 if start > 0 else 0.0
            out.append(f"    {first} -> {last}: account {mine:+.1%}, bull_bear {theirs:+.1%}")
            start = None
    return out


def report() -> bool:
    """Run both windows and the SPY check, print every table and return the verdict."""
    px = etf_closes()
    ends = month_ends(list(px.index))
    gated: dict[str, tuple[float, float]] = {}
    alone: dict[str, tuple[float, float]] = {}
    for name, run in RUNS.items():
        curve = bull_curve(run)
        sessions = list(curve.index)
        monthly = effective_targets(px, valid_days(px, [e for e in ends if e <= sessions[-1]]), sessions)
        tuesdays = [d for d in px.index if d.weekday() == 1 and sessions[0] - timedelta(days=14) <= d <= sessions[-1]]
        weekly = effective_targets(px, valid_days(px, tuesdays), sessions)
        rows = replay(curve, px, sessions, monthly)
        mix = replay(curve, px, sessions, fixed_targets({BULL: 0.7, "SHV": 0.3}, px, sessions))
        spy = px["SPY"].reindex(sessions).ffill()
        gated[name], alone[name] = (gain(rows.equity), max_drawdown(rows.equity)), (gain(curve), max_drawdown(curve))
        print(f"\n{name}: {sessions[0]} -> {sessions[-1]}, {len(sessions)} sessions")
        print(line("bull_bear alone", curve))
        print(line("canary switch, monthly (the bar)", rows.equity, rows))
        tuesday_rows = replay(curve, px, sessions, weekly)
        print(line("canary switch, Tuesdays (info)", tuesday_rows.equity, tuesday_rows))
        print(line("70% bull_bear / 30% SHV", mix.equity))
        print(line("SPY buy and hold", spy))
        print("  bear-mode stretches (monthly):\n" + ("\n".join(stretches(rows, curve)) or "    none"))
    decisions = valid_days(px, ends)
    sessions = [d for d in px.index if d > decisions[0]]
    spy_curve = px["SPY"].reindex(sessions) / px["SPY"].loc[sessions[0]] * 10_000
    rows = replay(spy_curve, px, sessions, effective_targets(px, [d for d in decisions if d <= sessions[-1]], sessions), ETF_COST)
    split = rows[[d >= date(2022, 8, 1) for d in rows.index]]
    sixty = replay(spy_curve, px, sessions, fixed_targets({BULL: 0.6, "IEF": 0.4}, px, sessions), ETF_COST)
    print(f"\nSPY check: {sessions[0]} -> {sessions[-1]}")
    print(line("canary on SPY", rows.equity, rows))
    print(line("SPY buy and hold", spy_curve))
    print(line("60% SPY / 40% IEF", sixty.equity))
    overall = bear_return(rows)
    print(f"  bear-mode return, all: {overall:+.1%}; since 2022-08 (the paper's out-of-sample): {bear_return(split):+.1%}")
    ok = passes(gated, alone, overall)
    for name in RUNS:
        floor = SHARE * alone[name][0]
        print(f"  {name}: smaller drawdown {gated[name][1] > alone[name][1]}, gain {gated[name][0]:+.1%} vs {SHARE:.0%} x {alone[name][0]:+.1%} = {floor:+.1%}: {gated[name][0] >= floor}")
    print(f"  bear-mode return >= 0: {overall >= 0}\n{'GO' if ok else 'NO'}")
    return ok


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
    # 7. replay, bull mode only: equity follows the bull curve, never bear, no cost
    sess = [date(2024, 1, 2) + timedelta(days=i) for i in range(5)]
    flat_px = pd.DataFrame({"TLT": [50.0] * 5}, index=sess)
    rows = replay(pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=sess), flat_px, sess, {sess[0]: {BULL: 1.0}})
    assert list(rows.equity) == [100.0, 105.0, 110.0, 115.0, 120.0] and not rows.bear.any()
    # 8. replay, a switch to TLT decided for session 2: cost 10 bps on the bull side + 5 bps on the ETF side at that
    #    close, then flat while the bull curve keeps rising; "bear" is True on the sessions held in bear mode (3 and 4)
    rows = replay(pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=sess), flat_px, sess, {sess[0]: {BULL: 1.0}, sess[2]: {"TLT": 1.0}})
    assert abs(rows.equity.iloc[2] - (110 - 0.001 * 110 - 0.0005 * 110)) < 1e-9 and abs(rows.equity.iloc[4] - rows.equity.iloc[2]) < 1e-9
    assert list(rows.bear) == [False, False, False, True, True]
    # 9. replay on a flat bull curve with an opening target that holds no ETF: equity stays at the start, no NaN
    rows = replay(pd.Series([100.0] * 5, index=sess), flat_px, sess, {sess[0]: {BULL: 1.0}})
    assert (rows.equity == 100.0).all() and not rows.equity.isna().any()
    # 10. effective_targets: a decision keys the first session after it; earlier ones key the first session (the latest
    #     wins); a decision with no later session is dropped
    px20 = prices({s: [100.0 + i for i in range(20)] for s in dict.fromkeys(symbols)}, 20)
    window = [px20.index[i] + timedelta(days=1) for i in range(14, 19)]
    targets = effective_targets(px20, [px20.index[i] for i in range(13, 20)], window)
    assert list(targets) == window and all(target == {BULL: 1.0} for target in targets.values())
    assert effective_targets(px20, [px20.index[19]], window) == {}
    # 11. max_drawdown, gain and bear_return
    curve = pd.Series([100.0, 120.0, 90.0, 130.0])
    assert abs(max_drawdown(curve) + 0.25) < 1e-12 and abs(gain(curve) - 0.30) < 1e-12
    two = pd.DataFrame({"equity": [100.0, 110.0, 99.0], "bear": [False, True, True]})
    assert abs(bear_return(two) - (1.1 * 0.9 - 1)) < 1e-12
    # 12. passes: smaller drawdown and gain >= 0.85 x bull_bear's on both windows, and a bear return >= 0
    alone = {"5Y": (4.68, -0.252), "COVID": (1.37, -0.377)}
    good = {"5Y": (4.0, -0.20), "COVID": (1.2, -0.30)}
    assert passes(good, alone, 0.0)
    assert not passes({**good, "COVID": (1.37 * 0.84, -0.30)}, alone, 0.0)
    assert not passes({**good, "5Y": (4.0, -0.252)}, alone, 0.0)
    assert not passes(good, alone, -0.001)
    print("self-check OK")


def main() -> None:
    """Entry point: `check` or `run` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "run":
        report()
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
