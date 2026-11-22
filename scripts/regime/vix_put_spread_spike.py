"""Spike: bear put spreads as insurance while VIX alone is defensive.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-put-spread-spike-design.md

Run (jumpmodels is not a project dependency; the sibling spike modules import it):
  uv run --with jumpmodels python scripts/regime/vix_put_spread_spike.py check   # self-check
  uv run --with jumpmodels python scripts/regime/vix_put_spread_spike.py dev     # 2016 -> 2024: run, read

Decisions before the open of session t from closes up to t-1, executed at the open of t.
"""

import importlib.util
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_staged = _load("vix_staged_spike", REPO / "scripts" / "regime" / "vix_staged_spike.py")
market, entry_trigger, run_switch, falls, metrics = _staged.market, _staged.entry_trigger, _staged.run_switch, _staged.falls, _staged.metrics
portfolio_exposure = _staged.portfolio_exposure

# Rules (design doc, fixed before running)
LONG_OTM, SHORT_OTM, MIN_DTE, EXIT_DTE, TARGET, SKEW = 0.98, 0.90, 30, 7, 0.80, 0.8
HALF_SPREAD, COMMISSION, REGULATORY = 0.05, 1.00, 0.10  # Tastytrade, per contract per leg
MIXES, BUDGETS, START, SWITCH_COST = (1.0, 0.75, 0.5), (0.0, 0.005, 0.01, 0.02), 100_000.0, 0.0010


def _cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_price(spot: float, strike: float, vol: float, years: float) -> float:
    """Black-Scholes European put per share, rate 0, no dividend."""
    root = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * years) / root
    return strike * _cdf(-(d1 - root)) - spot * _cdf(-d1)


def spread_mid(spot: float, long_k: float, short_k: float, vix: float, years: float, vol_scale: float) -> float:
    """Bear put spread value per share, each leg at its own skewed volatility, clamped to [0, width]."""

    def vol(k: float) -> float:
        return vix / 100 * vol_scale + SKEW * max(0.0, math.log(spot / k))

    value = put_price(spot, long_k, vol(long_k), years) - put_price(spot, short_k, vol(short_k), years)
    return min(max(value, 0.0), long_k - short_k)


def monthly_expiry(today: date) -> date:
    """The nearest third Friday at least MIN_DTE calendar days away."""
    y, mo = today.year, today.month
    while True:
        first = date(y, mo, 1)
        expiry = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
        if (expiry - today).days >= MIN_DTE:
            return expiry
        y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)


def open_cost(n: int) -> float:
    """Dollars to open n spreads (two legs each)."""
    return n * 2 * (HALF_SPREAD * 100 + COMMISSION + REGULATORY)


def close_cost(n: int) -> float:
    """Dollars to close n spreads (two legs each)."""
    return n * 2 * (HALF_SPREAD * 100 + REGULATORY)


def simulate(px: pd.DataFrame, spy_prev: pd.Series, vix: pd.Series, defensive: pd.Series, w_def: float, budget: float, vol_scale: float = 1.0) -> tuple[pd.Series, list[dict]]:
    """Daily close equity ($100,000 start) and the closed spread trades of one variant.

    `px` has Open/Close of SPY and SHV per session; `spy_prev[t]` is the SPY close of t-1; `vix` holds VIX closes
    (`vix.shift(1)` gives the value known at the open). `defensive[t]` is VIX alone's flag for session t.
    """
    days = list(defensive.index)
    vix_prev = vix.shift(1)
    spy = shv = 0.0
    equity, trades, spread, done, w_prev, last_eq = {}, [], None, False, None, START
    for i, d in enumerate(days):
        if i:
            prev = days[i - 1]
            spy *= px.at[d, "Open_SPY"] / px.at[prev, "Close_SPY"]
            shv *= px.at[d, "Open_SHV"] / px.at[prev, "Close_SHV"]
        cash = START if i == 0 else 0.0
        s_open, vix_open = px.at[d, "Open_SPY"], vix_prev[d]
        if spread:
            width = spread["long"] - spread["short"]
            reason = None
            if not defensive[d]:
                reason = "episode end"
            elif spread["mark"] >= TARGET * width:
                reason = "target"
            elif (spread["expiry"] - d).days <= EXIT_DTE:
                reason = "roll"
            if reason:
                value = spread_mid(s_open, spread["long"], spread["short"], vix_open, (spread["expiry"] - d).days / 365, vol_scale)
                proceeds = value * 100 * spread["n"] - close_cost(spread["n"])
                cash += proceeds
                trades.append({"open": spread["opened"], "close": d, "reason": reason, "n": spread["n"], "paid": spread["paid"], "paid_pct": spread["paid_pct"], "pnl": proceeds - spread["paid"]})
                spread, done = None, done or reason == "target"
        if defensive[d] and (i == 0 or not defensive[days[i - 1]]):
            done = False  # a new episode
        if defensive[d] and spread is None and not done and budget > 0:
            long_k, short_k, expiry = round(LONG_OTM * spy_prev[d]), round(SHORT_OTM * spy_prev[d]), monthly_expiry(d)
            debit = spread_mid(s_open, long_k, short_k, vix_open, (expiry - d).days / 365, vol_scale)
            n = math.floor(budget * last_eq / (debit * 100 + open_cost(1)))
            if n > 0:
                paid = debit * 100 * n + open_cost(n)
                cash -= paid
                spread = {"long": long_k, "short": short_k, "expiry": expiry, "n": n, "paid": paid, "paid_pct": paid / last_eq, "opened": d, "mark": debit}
        w = w_def if defensive[d] else 1.0
        sleeve = (spy + shv + cash) * (1 - (SWITCH_COST * abs(w - w_prev) if w_prev is not None else 0.0))
        closed = w * sleeve * px.at[d, "Close_SPY"] / s_open + (1 - w) * sleeve * px.at[d, "Close_SHV"] / px.at[d, "Open_SHV"]
        spy, shv = w * closed, (1 - w) * closed  # weights back to target at the close: drift ignored, as portfolio_exposure
        mark = 0.0
        if spread:
            spread["mark"] = spread_mid(px.at[d, "Close_SPY"], spread["long"], spread["short"], vix[d], (spread["expiry"] - d).days / 365, vol_scale)
            mark = spread["mark"] * 100 * spread["n"]
        last_eq = equity[d] = spy + shv + mark
        w_prev = w
    return pd.Series(equity), trades


def report(mode: str) -> dict:
    """Run every mix x budget on a window; print the money table, the spread statistics and the pass bar."""
    m, px = market()
    start, end = _staged._exit.WINDOWS[mode]
    days = [d for d in m.index if start <= d <= end and d in px.index]
    years = (days[-1] - days[0]).days / 365.25
    calm = ((m.vix / m.vix3m).shift(1) < _staged._exit.RATIO_EXIT).loc[days]
    defensive = run_switch(None, entry_trigger(m).loc[days], calm)
    spy_prev, px = m.close.shift(1).loc[days], px.loc[days]
    fl = [f for f in falls(m.close) if days[0] <= f[0] and f[1] <= days[-1]]
    pos = {d: i for i, d in enumerate(days)}
    in_fall = {d for hi, lo, _ in fl for d in days[max(0, pos[hi] - 20) : pos[lo] + 1]}
    print(f"{mode}: {days[0]} -> {days[-1]}, {len(days)} sessions, {len(fl)} SPY falls >= 8%, defensive {defensive.mean():.0%} of sessions")
    head = f"{'defensive mix':16s} {'budget':>6s} {'total':>8s} {'CAGR':>6s} {'max DD':>7s} {'Sharpe':>6s} {'spreads/yr':>10s} {'premium/yr':>10s}"
    head += f" {'spread P&L: in falls / outside':>32s} {'total @0.8 VIX':>14s}"
    print("\n" + head)
    res = {}
    for w in MIXES:
        for b in BUDGETS:
            eq, trades = simulate(px, spy_prev, m.vix, defensive, w, b)
            r = metrics(eq.pct_change().fillna(eq.iloc[0] / START - 1), years)
            res[(w, b)] = r
            mix = f"{int(w * 100)}% SPY + {int(round((1 - w) * 100))}% SHV"
            if not trades:
                print(f"{mix:16s} {b:6.1%} {r['total']:+8.1%} {r['cagr']:+6.1%} {r['mdd']:+7.1%} {r['sharpe']:6.2f}")
                continue
            inside = sum(t["pnl"] for t in trades if t["open"] in in_fall)
            outside = sum(t["pnl"] for t in trades if t["open"] not in in_fall)
            stress, _ = simulate(px, spy_prev, m.vix, defensive, w, b, vol_scale=0.8)
            pnl, premium = f"{inside:+,.0f} / {outside:+,.0f}", sum(t["paid_pct"] for t in trades) / years
            row = f"{mix:16s} {b:6.1%} {r['total']:+8.1%} {r['cagr']:+6.1%} {r['mdd']:+7.1%} {r['sharpe']:6.2f} {len(trades) / years:10.1f} {premium:10.1%}"
            print(f"{row} {pnl:>32s} {stress.iloc[-1] / START - 1:+14.1%}")
    bh_mdd = res[(1.0, 0.0)]["mdd"]
    print(f"\n=== pass bar: total above the same mix without spreads, max DD no worse than buy and hold ({bh_mdd:+.1%})")
    passing = {}
    for (w, b), r in res.items():
        if b > 0:
            ok = r["total"] > res[(w, 0.0)]["total"] and r["mdd"] >= bh_mdd
            print(f"  {int(w * 100)}% SPY, budget {b:.1%}: {r['total']:+.1%} vs {res[(w, 0.0)]['total']:+.1%} without, max DD {r['mdd']:+.1%} -> {'PASS' if ok else 'FAIL'}")
            if ok:
                passing[(w, b)] = r["total"]
    chosen = max(passing, key=passing.get, default=None)
    print(f"chosen: {chosen or 'none -> verdict NO, the test window stays unused'}")
    return {"results": res, "chosen": chosen}


def check() -> None:
    """Assert pricing, the no-spread ledger against portfolio_exposure, and the spread mechanics on synthetic data."""
    # 1. pricing: a deep in-the-money spread is near its width, a far out-of-the-money one near 0, always in [0, width]
    assert abs(spread_mid(80.0, 98, 90, 20, 0.05, 1.0) - 8) < 0.05 and spread_mid(200.0, 98, 90, 20, 0.05, 1.0) < 1e-3
    assert monthly_expiry(date(2026, 12, 17)) == date(2027, 2, 19) and monthly_expiry(date(2026, 12, 16)) == date(2027, 1, 15)
    idx = [date(2021, 1, 4) + timedelta(days=i) for i in range(90)]
    rng = np.random.default_rng(2)
    cols = {}
    for s in ("SPY", "SHV"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 90)))
        cols[f"Close_{s}"], cols[f"Open_{s}"] = close, close * np.exp(rng.normal(0, 0.003, 90))
    px = pd.DataFrame(cols, index=idx)
    spy_prev, vix = px.Close_SPY.shift(1).fillna(100.0), pd.Series(20.0, index=idx)
    defensive = pd.Series([10 <= i < 70 for i in range(90)], index=idx)
    # 2. no spreads: the ledger equals portfolio_exposure at every mix
    for w in MIXES:
        eq, trades = simulate(px, spy_prev, vix, defensive, w, 0.0)
        r = eq.pct_change().fillna(eq.iloc[0] / START - 1)
        assert not trades and np.allclose(r, portfolio_exposure(1 - (1 - w) * defensive.astype(float), px), atol=1e-12)
    # 3. flat market: spreads open on the switch, roll at 7 days to expiry, close when the episode ends
    flat = pd.DataFrame({c: 100.0 for c in px.columns}, index=idx)
    _, trades = simulate(flat, pd.Series(100.0, index=idx), vix, defensive, 0.5, 0.01)
    assert trades[0]["open"] == idx[10] and trades[-1]["reason"] == "episode end" and trades[-1]["close"] == idx[70]
    assert {t["reason"] for t in trades[:-1]} == {"roll"} and all(t["pnl"] < 0 for t in trades)
    # 4. crash: the target closes the spread and no new one opens in the same episode
    crash = flat.copy()
    crash.loc[idx[15] :, :] = 80.0
    crash[["Open_SHV", "Close_SHV"]] = 100.0
    _, trades = simulate(crash, crash.Close_SPY.shift(1).fillna(100.0), vix, defensive, 0.5, 0.01)
    assert trades[0]["reason"] == "target" and len(trades) == 1 and trades[0]["pnl"] > 0
    # 5. a new episode after a target buys insurance again
    second = pd.Series([10 <= i < 40 or 50 <= i < 70 for i in range(90)], index=idx)
    _, trades = simulate(crash, crash.Close_SPY.shift(1).fillna(100.0), vix, second, 0.5, 0.01)
    assert trades[0]["reason"] == "target" and any(t["open"] == idx[50] for t in trades)
    print("self-check OK")


def main() -> None:
    """Entry point: `check` or `dev` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "dev":
        report(mode)
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
