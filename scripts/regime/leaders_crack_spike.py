"""Spike: leaders crack first -- early detection of a SPY fall.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-leaders-crack-first-spike-design.md

Run:
  uv run python scripts/regime/leaders_crack_spike.py dev        # 2016 -> 2024: run, read
  (holdout: never earned by this spike; the 2025-2026 window was spent by vix_jump_spike.py final, so it refuses to run)

A daily verdict before the open of session t, from closes up to t-1.
"""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("early_regime_spike", REPO / "scripts" / "news" / "early_regime_spike.py")
_news = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_news)  # reuse the SPY fall zigzag and the switch state machine of the news spike
falls, run_switch = _news.falls, _news.run_switch

HISTORY = REPO / "data" / "universe" / "us_stock_universe_history.json"
BARS = REPO / "cache" / "universe_history" / "bars"
ETFS = ["QQQ", "SMH", "XLK"]
SAME_COMPANY = {"GOOG": "GOOGL", "BRK-B": "BRK-A"}  # one ticker per company

# Rules (design doc, fixed before running)
N_LEADERS, HIGH_WINDOW = 10, 60
CRACK_BASKET, CRACK_SPY, CALM_BASKET, EXIT_SESSIONS = -0.07, -0.03, -0.03, 3
RATIO_CONFIRM, VIXY_CONFIRM, RATIO_EXIT = 0.95, 0.10, 0.90
EARLY_SESSIONS, TARGET_DD = 20, -0.03
WINDOWS = {"dev": (date(2016, 1, 4), date(2024, 12, 31)), "holdout": (date(2025, 1, 2), date(2026, 11, 20))}


def market() -> pd.DataFrame:
    """Daily closes (Yahoo, adjusted) of SPY, the leader ETFs and the volatility inputs."""
    import yfinance as yf

    raw = yf.download(["SPY", *ETFS, "^VIX", "^VIX3M", "VIXY"], start="2015-06-01", end="2026-11-21", progress=False, auto_adjust=True)["Close"]
    m = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m"}).dropna(subset=["SPY"])
    m.index = m.index.date
    return m


def monthly_leaders() -> dict[str, list[str]]:
    """First N_LEADERS symbols of each point-in-time monthly list, one per company, with cached bars."""
    months = json.loads(HISTORY.read_text())["months"]
    out = {}
    for month, symbols in months.items():
        picked, companies = [], set()
        for s in symbols:
            company = SAME_COMPANY.get(s, s)
            if company in companies or not (BARS / f"{s}.parquet").exists():
                continue
            picked.append(s)
            companies.add(company)
            if len(picked) == N_LEADERS:
                break
        out[month] = picked
    return out


def stock_basket_returns(sessions: list[date], leaders: dict[str, list[str]]) -> pd.Series:
    """Equal-weight daily return of the month's leaders (members without a bar that day are left out)."""
    symbols = sorted({s for members in leaders.values() for s in members})
    closes = {s: pd.read_parquet(BARS / f"{s}.parquet")["Adj Close"] for s in symbols}
    rets = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    rets.index = rets.index.date
    rets = rets.reindex(sessions)
    out = {}
    for d in sessions:
        members = leaders.get(f"{d:%Y-%m}")
        if members:
            out[d] = rets.loc[d, members].mean()  # skips NaN
    return pd.Series(out, dtype=float)


def drawdown(level: pd.Series) -> pd.Series:
    """Level against its max over the last HIGH_WINDOW sessions, minus 1."""
    return level / level.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW).max() - 1


def evaluate(name: str, state: pd.Series, spy: pd.Series, fall_list: list, years: float) -> tuple[float, float]:
    """Print the early / on time / late / missed table of one variant; return (median drawdown, false per year)."""
    sessions = list(state.index)
    pos = {d: i for i, d in enumerate(sessions)}
    starts = [d for i, d in enumerate(sessions) if state.iloc[i] and (i == 0 or not state.iloc[i - 1])]
    print(f"\n=== {name}: {len(starts)} defensive switches, {state.mean():.0%} of sessions defensive")
    dds, lags, credited = [], [], set()
    for hi, lo, depth in fall_list:
        i_hi = pos[hi]
        win0 = sessions[max(0, i_hi - EARLY_SESSIONS)]
        before = [s for s in starts if win0 <= s <= hi]
        after = [s for s in starts if hi < s <= lo]
        credited.update(before + after)
        if before and state[hi]:
            lead = i_hi - pos[before[0]]
            dds.append(0.0)
            lags.append(-lead)
            print(f"  fall {hi} -> {lo} ({depth:.1%}): EARLY, defensive {before[0]}, {lead} sessions before the high")
        elif state[hi]:  # defensive at the high since a switch older than the early window: protected all along
            since = max(s for s in starts if s <= hi)
            credited.add(since)
            dds.append(0.0)
            lags.append(-(i_hi - pos[since]))
            print(f"  fall {hi} -> {lo} ({depth:.1%}): EARLY, already defensive since {since}, {i_hi - pos[since]} sessions before the high")
        elif after:
            s = after[0]
            dd = spy[sessions[pos[s] - 1]] / spy[hi] - 1  # previous session's close, known before the open
            n = pos[s] - i_hi
            dds.append(dd)
            lags.append(n)
            print(f"  fall {hi} -> {lo} ({depth:.1%}): {'ON TIME' if dd >= TARGET_DD else 'LATE'}, defensive {s}, {dd:+.1%} from the high, {n} sessions in")
        else:
            dds.append(-1.0)  # missed counts as the worst outcome
            print(f"  fall {hi} -> {lo} ({depth:.1%}): MISSED")
    false = [s for s in starts if s not in credited]
    med_dd = float(np.median(dds)) if dds else float("nan")
    print(f"  median over {len(dds)} falls: drawdown at switch {med_dd:+.1%} (missed = -100%), sessions from the high {np.median(lags):+.0f}")
    print(f"  false switches: {len(false)} ({len(false) / years:.1f}/year): {', '.join(map(str, false))}")
    print("  defensive share by year: " + ", ".join(f"{y}: {s.mean():.0%}" for y, s in state.groupby([d.year for d in state.index])))
    return med_dd, len(false) / years


def main() -> None:
    """Entry point: `dev` or `holdout` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "holdout":
        sys.exit("The 2025-2026 holdout was spent on 2026-11-22 (vix_jump_spike.py final); see the specs. Not re-run.")
    start, end = WINDOWS[mode]
    m = market()
    all_sessions = list(m.index)
    leaders = monthly_leaders()
    print("L1 leaders, samples: " + " | ".join(f"{k}: {','.join(leaders[k])}" for k in ("2016-01", "2020-02", "2022-01", "2024-07") if k in leaders))

    spy = m["SPY"]
    spy_dd = drawdown(spy).shift(1)
    baskets = {
        "L1 stocks (point-in-time top 10)": stock_basket_returns(all_sessions, leaders),
        "L2 ETFs (QQQ, SMH, XLK)": m[ETFS].pct_change().mean(axis=1),
    }
    vix_on = ((m.vix / m.vix3m).shift(1) >= RATIO_CONFIRM) | ((m.VIXY / m.VIXY.shift(5) - 1).shift(1) >= VIXY_CONFIRM)
    vix_calm = (m.vix / m.vix3m).shift(1) < RATIO_EXIT

    dd_of = {name: drawdown((1 + r.dropna()).cumprod()).reindex(all_sessions).shift(1) for name, r in baskets.items()}
    valid = [d for d in all_sessions if start <= d <= end and all(not np.isnan(dd[d]) for dd in dd_of.values()) and not np.isnan(spy_dd[d])]
    years = (valid[-1] - valid[0]).days / 365.25
    fl = [f for f in falls(spy) if valid[0] <= f[0] and f[1] <= valid[-1]]  # falls whose high and trough lie in the window
    print(f"\n{mode}: {valid[0]} -> {valid[-1]}, {len(valid)} sessions, {len(fl)} SPY falls >= 8% fully inside")

    results = {}
    for name, dd in dd_of.items():
        crack = (dd <= CRACK_BASKET) & (spy_dd >= CRACK_SPY)
        calm = dd > CALM_BASKET
        state = run_switch(None, crack.loc[valid], calm.loc[valid])
        results[name] = evaluate(name, state, spy, fl, years)
    base_dd, base_false = evaluate("baseline: VIX alone", run_switch(None, vix_on.loc[valid], vix_calm.loc[valid]), spy, fl, years)

    print("\n=== development pass bar: median drawdown at switch better than -3%, false switches/year <= VIX alone")
    for name, (med, fpy) in results.items():
        ok = med > TARGET_DD and fpy <= base_false
        print(f"  {name}: median {med:+.1%}, {fpy:.1f} false/year (VIX alone {base_dd:+.1%}, {base_false:.1f}/year) -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
