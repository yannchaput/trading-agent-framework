"""Spike: a fast exit after a VIX switch -- get back into SPY early in the rebound.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-exit-rule-spike-design.md

Run (jumpmodels is not a project dependency; the sibling spike module imports it):
  uv run --with jumpmodels python scripts/regime/vix_exit_spike.py check   # self-check of the state machine
  uv run --with jumpmodels python scripts/regime/vix_exit_spike.py dev     # 2016 -> 2024: run, read
  (test, 2008 -> 2015: not built; no variant passed dev)

A daily verdict before the open of session t, from closes up to t-1.
"""

import importlib.util
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


_leaders = _load("leaders_crack_spike", REPO / "scripts" / "regime" / "leaders_crack_spike.py")
falls, run_switch, evaluate = _leaders.falls, _leaders.run_switch, _leaders.evaluate
portfolio = _load("vix_jump_spike", REPO / "scripts" / "regime" / "vix_jump_spike.py").portfolio  # SPY / SHV, switch at the open, 10 bp

# Rules (design doc, fixed before running)
RATIO_CONFIRM, VIXY_CONFIRM, RATIO_EXIT = 0.95, 0.10, 0.90  # VIX-alone entry trigger and calm exit
CALM_SESSIONS, REBOUND, DMI_N = 3, 0.04, 14
WHIPSAW_SESSIONS, MAX_DD_BAR = 20, -0.20
WINDOWS = {"dev": (date(2016, 1, 4), date(2024, 12, 31)), "test": (date(2008, 1, 2), date(2015, 12, 31))}


def entry_trigger(m: pd.DataFrame) -> pd.Series:
    """VIX alone's entry trigger on each morning, from closes up to the day before; VIXY NaN (before 2011) counts as false."""
    ratio = (m.vix / m.vix3m).shift(1)
    vixy_5d = (m.VIXY / m.VIXY.shift(5) - 1).shift(1)
    return (ratio >= RATIO_CONFIRM) | (vixy_5d >= VIXY_CONFIRM)


def run_exit(trigger: pd.Series, calm: pd.Series, spy_prev: pd.Series, exit_sig: pd.Series | None = None, rebound: float | None = None) -> tuple[pd.Series, list[date]]:
    """Defensive flag per session (True = defensive) and the sessions where a price exit fired.

    All inputs are indexed by session t and known before its open (`spy_prev[t]` = SPY close of t-1). Give exactly one of
    `exit_sig` (price exit when True) or `rebound` (price exit when SPY is that far above the episode's lowest close).
    Each morning: normal -> defensive on the trigger when armed, or on a failed rally (disarmed and SPY below the
    previous episode's low); defensive since t-1 -> normal on a price exit or after CALM_SESSIONS calm sessions
    (counted like `run_switch`: not on the switch session); `armed` is updated last (False after a price exit,
    True when the trigger is off).
    """
    assert (exit_sig is None) != (rebound is None), "give exactly one of exit_sig / rebound"
    state, armed, quiet, episode_low, prev_low = False, True, 0, 0.0, None
    out, price_exits = {}, []
    for d in trigger.index:
        price_exit = False
        if not state:
            if (armed and trigger[d]) or (not armed and prev_low is not None and spy_prev[d] < prev_low):
                state, quiet, episode_low = True, 0, spy_prev[d]
        else:
            episode_low = min(episode_low, spy_prev[d])
            quiet = quiet + 1 if calm[d] else 0
            price_exit = bool(exit_sig[d]) if exit_sig is not None else spy_prev[d] >= (1 + rebound) * episode_low
            if price_exit or quiet >= CALM_SESSIONS:
                state, prev_low = False, episode_low
        if price_exit:
            price_exits.append(d)
            armed = False
        elif not trigger[d]:
            armed = True
        out[d] = state
    return pd.Series(out), price_exits


def market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yahoo daily data from 2007-06 (adjusted): SPY open/high/low/close + VIX inputs, and SPY / SHV opens and closes."""
    import yfinance as yf

    raw = yf.download(["SPY", "SHV", "^VIX", "^VIX3M", "VIXY"], start="2007-06-01", end="2026-11-21", progress=False, auto_adjust=True)
    m = pd.DataFrame({k.lower(): raw[k]["SPY"] for k in ("Open", "High", "Low", "Close")})
    m["vix"], m["vix3m"], m["VIXY"] = raw["Close"]["^VIX"], raw["Close"]["^VIX3M"], raw["Close"]["VIXY"]
    px = pd.concat({f"{f}_{s}": raw[f][s] for f in ("Open", "Close") for s in ("SPY", "SHV")}, axis=1)
    m, px = m.dropna(subset=["close"]), px.dropna()
    m.index, px.index = m.index.date, px.index.date
    return m, px


def wilder(x: pd.Series, n: int = DMI_N) -> pd.Series:
    """Wilder smoothing: the first value (n-th non-NaN) is the simple mean of the first n, then (previous * (n-1) + x) / n."""
    v, out = x.to_numpy(float), np.full(len(x), np.nan)
    seed = int(np.argmax(~np.isnan(v))) + n - 1
    if seed < len(v):
        out[seed] = v[seed - n + 1 : seed + 1].mean()
        for i in range(seed + 1, len(v)):
            out[i] = (out[i - 1] * (n - 1) + v[i]) / n
    return pd.Series(out, index=x.index)


def exit_signals(m: pd.DataFrame) -> dict[str, pd.Series | float]:
    """The four price exits. Bool Series (row t from closes up to t-1) for E1, E3, E4; the rebound size for E2."""
    c, r = m.close, np.log(m.close).diff()
    up, down = m.high.diff(), -m.low.diff()
    plus_dm, minus_dm = up.where((up > down) & (up > 0), 0.0), down.where((down > up) & (down > 0), 0.0)
    plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan
    # +DI > -DI is wilder(+DM) > wilder(-DM): the true range divides both sides, so it drops out

    def lag(signal: pd.Series) -> pd.Series:
        return signal.shift(1, fill_value=False)  # row t from closes up to t-1

    return {
        "E1 price reclaim": lag(c > c.shift(1).rolling(5).max()),
        "E2 rebound off the low": REBOUND,
        "E3 volatility compression": lag((r.rolling(5).std() < r.rolling(20).std()) & (c > c.rolling(5).mean())),
        "E4 DMI": lag(wilder(plus_dm) > wilder(minus_dm)),
    }


def metrics(r: pd.Series, years: float) -> dict[str, float]:
    """Total return, CAGR, max drawdown, Sharpe (no risk-free) and Calmar of a daily return series."""
    level = (1 + r).cumprod()
    cagr, mdd = level.iloc[-1] ** (1 / years) - 1, (level / level.cummax() - 1).min()
    return {"total": level.iloc[-1] - 1, "cagr": cagr, "mdd": mdd, "sharpe": r.mean() / r.std() * np.sqrt(252), "calmar": cagr / -mdd}


def run_window(mode: str, m: pd.DataFrame, px: pd.DataFrame) -> dict[str, dict]:
    """VIX alone and the four exit variants over a window: defensive flags, price exits, SPY / SHV portfolio metrics."""
    start, end = WINDOWS[mode]
    days = [d for d in m.index if start <= d <= end and d in px.index]
    years = (days[-1] - days[0]).days / 365.25
    trigger, spy_prev = entry_trigger(m).loc[days], m.close.shift(1).loc[days]
    calm = ((m.vix / m.vix3m).shift(1) < RATIO_EXIT).loc[days]
    runs = {"VIX alone": (run_switch(None, trigger, calm), [])}
    for name, sig in exit_signals(m).items():
        runs[name] = run_exit(trigger, calm, spy_prev, rebound=sig) if isinstance(sig, float) else run_exit(trigger, calm, spy_prev, sig.loc[days])
    results = {}
    for name, (state, exits) in runs.items():
        switches = int((state != state.shift(1)).iloc[1:].sum())
        results[name] = {"state": state, "exits": exits, **metrics(portfolio(state, px), years), "switches_yr": switches / years, "defensive": state.mean()}
    hold = px.Close_SPY.loc[days].pct_change().fillna(px.at[days[0], "Close_SPY"] / px.at[days[0], "Open_SPY"] - 1)
    results["SPY buy and hold"] = {"state": None, "exits": [], **metrics(hold, years)}
    results["days"], results["years"] = days, years
    return results


def passes(res: dict[str, dict], name: str) -> bool:
    """Pass bar against VIX alone: higher Sharpe, higher total return, max drawdown no worse than -20%."""
    r, base = res[name], res["VIX alone"]
    return r["sharpe"] > base["sharpe"] and r["total"] > base["total"] and r["mdd"] >= MAX_DD_BAR


def choose(results: dict[str, dict]) -> str | None:
    """The passing exit variant with the best Sharpe (tie: Calmar), or None."""
    passing = [n for n in results if n.startswith("E") and passes(results, n)]
    return max(passing, key=lambda n: (results[n]["sharpe"], results[n]["calmar"]), default=None)


def whipsaws(state: pd.Series, exits: list[date], spy: pd.Series) -> list[date]:
    """Price exits followed within WHIPSAW_SESSIONS by a new switch at a lower SPY close (closes before each session)."""
    sessions = list(state.index)
    pos = {d: i for i, d in enumerate(sessions)}
    starts = [d for i, d in enumerate(sessions) if i and state.iloc[i] and not state.iloc[i - 1]]
    bad = []
    for e in exits:
        later = [s for s in starts if 0 < pos[s] - pos[e] <= WHIPSAW_SESSIONS]
        if later and spy[sessions[pos[later[0]] - 1]] < spy[sessions[pos[e] - 1]]:
            bad.append(e)
    return bad


def report(mode: str) -> dict[str, dict]:
    """Print the money table, the entry table, the trough-to-return diagnostics, whipsaws and the pass bar; return the results."""
    m, px = market()
    res = run_window(mode, m, px)
    days, years = res.pop("days"), res.pop("years")
    fl = [f for f in falls(m.close) if days[0] <= f[0] and f[1] <= days[-1]]
    print(f"{mode}: {days[0]} -> {days[-1]}, {len(days)} sessions, {len(fl)} SPY falls >= 8% fully inside")
    print(f"\n{'variant':28s} {'total':>8s} {'CAGR':>6s} {'max DD':>7s} {'Sharpe':>6s} {'Calmar':>6s} {'sw/yr':>6s} {'defensive':>9s}")
    for name, r in res.items():
        tail = f" {r['switches_yr']:6.1f} {r['defensive']:9.0%}" if r["state"] is not None else ""
        print(f"{name:28s} {r['total']:+8.1%} {r['cagr']:+6.1%} {r['mdd']:+7.1%} {r['sharpe']:6.2f} {r['calmar']:6.2f}{tail}")
    for name, r in res.items():
        if r["state"] is None:
            continue
        evaluate(name, r["state"], m.close, fl, years)
        sessions = list(r["state"].index)
        pos = {d: i for i, d in enumerate(sessions)}
        print("  trough -> back to SPY:")
        for hi, lo, depth in fl:
            back = next((s for s in sessions[pos[lo] + 1 :] if not r["state"][s]), None) if r["state"][lo] else None
            if back is None:
                print(f"    {hi} -> {lo} ({depth:.1%}): {'still defensive at the window end' if r['state'][lo] else 'not defensive at the trough'}")
            else:
                print(f"    {hi} -> {lo} ({depth:.1%}): back {back}, {pos[back] - pos[lo]} sessions after the trough, SPY open {px.at[back, 'Open_SPY'] / m.close[lo] - 1:+.1%} from the trough")
        bad = whipsaws(r["state"], r["exits"], m.close)
        print(f"  price exits {len(r['exits'])}, whipsaws {len(bad)}: {', '.join(map(str, bad))}")
    chosen = choose(res)
    print("\n=== pass bar vs VIX alone: Sharpe higher, total higher, max DD no worse than -20%")
    for name in res:
        if name.startswith("E"):
            print(f"  {name}: Sharpe {res[name]['sharpe']:.2f}, total {res[name]['total']:+.1%}, max DD {res[name]['mdd']:+.1%} -> {'PASS' if passes(res, name) else 'FAIL'}")
    print(f"chosen: {chosen or 'none -> verdict NO, the test window stays unused'}")
    return res


def check() -> None:
    """Assert the state machine and the entry trigger on synthetic series (sessions are numbered s1..s12)."""
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(12)]

    def ser(vals: list) -> pd.Series:
        return pd.Series(vals, index=idx)

    def on(*sessions: int) -> pd.Series:
        return ser([k in sessions for k in range(1, 13)])

    def run(trigger, spy=None, calm=None, exit_sig=None, rebound=None):
        """Defensive session numbers and price-exit session numbers of one synthetic run."""
        if rebound is None and exit_sig is None:
            exit_sig = on()
        state, exits = run_exit(trigger, on() if calm is None else calm, ser([100.0] * 12) if spy is None else spy, exit_sig, rebound)
        return [k + 1 for k, v in enumerate(state) if v], [idx.index(d) + 1 for d in exits]

    # 1. a price exit while the trigger is still on: out, and not re-entered (disarmed)
    assert run(on(1, 2, 3, 4, 5, 6), exit_sig=on(3)) == ([1, 2], [3])
    # 2. one session with the trigger off re-arms it
    assert run(on(1, 2, 3, 4, 5, 6, 8), exit_sig=on(3)) == ([1, 2, 8, 9, 10, 11, 12], [3])
    # 3. failed rally: disarmed, SPY closes below the previous episode's low
    spy = ser([100.0] * 4 + [99.0] + [100.0] * 7)
    assert run(on(1, 2, 3, 4, 5, 6), spy=spy, exit_sig=on(3)) == ([1, 2] + list(range(5, 13)), [3])
    # 4. calm exit counted like run_switch: the switch session does not count
    trigger, calm = on(1), on(1, 2, 3, 4)
    assert run(trigger, calm=calm) == ([1, 2, 3], [])
    assert list(run_switch(None, trigger, calm)) == [k in (1, 2, 3) for k in range(1, 13)]
    # 5. `armed` is updated after the failed-rally check: trigger off and a new low on the same morning still switches
    assert run(on(1, 2, 3, 4, 6), spy=spy, exit_sig=on(3)) == ([1, 2] + list(range(5, 13)), [3])
    # 6. rebound exit measured from the episode low (97), not from the switch close (100)
    assert run(on(*range(1, 13)), spy=ser([100.0, 97.0] + [100.9] * 10), rebound=0.04) == ([1, 2], [3])
    assert run(on(*range(1, 13)), spy=ser([100.0, 97.0] + [100.8] * 10), rebound=0.04) == (list(range(1, 13)), [])
    # 7. VIXY missing (before 2011): the ratio alone drives the trigger, no error
    m = pd.DataFrame({"vix": [20, 20, 21, 21], "vix3m": [22, 22, 21, 21], "VIXY": [float("nan")] * 4}, index=idx[:4])
    assert list(entry_trigger(m)) == [False, False, False, True]
    # 8. no look-ahead: perturbing session k's own prices (up or down) must not change session k's signal (row k uses closes up to k-1)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    px = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=[date(2020, 1, 1) + timedelta(days=i) for i in range(80)])
    base = exit_signals(px)
    for k in range(40, 80):
        for factor in (0.5, 1.5):
            moved = px.copy()
            moved.iloc[k] = px.iloc[k] * factor
            for name, sig in exit_signals(moved).items():
                if isinstance(sig, pd.Series):
                    assert sig.iloc[k] == base[name].iloc[k], (name, k, factor)
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "dev":
        report(mode)
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
