"""Spike: filters to cut VIX alone's false alarms.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-false-alarm-filter-spike-design.md

Run (jumpmodels is not a project dependency; the sibling spike modules import it):
  uv run --with jumpmodels python scripts/regime/vix_filter_spike.py check   # self-check
  uv run --with jumpmodels python scripts/regime/vix_filter_spike.py dev     # 2016 -> 2024: run, read
  uv run --with jumpmodels python scripts/regime/vix_filter_spike.py test    # 2008 -> 2015, ONCE, chosen variant only

A daily verdict before the open of session t, from closes up to t-1.
"""

import contextlib
import importlib.util
import io
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
_exit = _staged._exit
market, entry_trigger, run_switch, falls, metrics, portfolio_exposure = _staged.market, _staged.entry_trigger, _staged.run_switch, _staged.falls, _staged.metrics, _staged.portfolio_exposure
evaluate = _exit.evaluate

# Variants (design doc, fixed before running)
RATIO_A, SPY_DROP, SPY_HIGH_N, CREDIT_N, BREADTH_N, BREADTH_MAX = 1.0, 0.98, 20, 20, 50, 4
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
TEST_GUARD = REPO / "cache" / "regime" / "vix_filter_test.done"
MIXES = (0.0, 0.5)  # SPY weight while defensive: 100% SHV, 50/50


def extra() -> pd.DataFrame:
    """Yahoo adjusted closes of HYG, IEF and the 9 sector ETFs from 2007-06, date index."""
    import yfinance as yf

    c = yf.download(["HYG", "IEF", *SECTORS], start="2007-06-01", end="2026-11-21", progress=False, auto_adjust=True)["Close"]
    c.index = c.index.date
    return c


def triggers(m: pd.DataFrame, x: pd.DataFrame) -> dict[str, pd.Series]:
    """Entry trigger per variant, row t from closes up to t-1 (each filter lagged one session)."""
    base = entry_trigger(m)
    x = x.reindex(m.index)
    vixy_5d = (m.VIXY / m.VIXY.shift(5) - 1).shift(1)
    spy_weak = (m.close <= SPY_DROP * m.close.rolling(SPY_HIGH_N).max()).shift(1, fill_value=False)
    credit = x.HYG / x.IEF
    credit_weak = (credit < credit.rolling(CREDIT_N).mean()).shift(1, fill_value=False)
    above = (x[SECTORS] > x[SECTORS].rolling(BREADTH_N).mean()).sum(axis=1)
    breadth_weak = (above <= BREADTH_MAX).shift(1, fill_value=False)
    return {
        "VIX alone": base,
        "a. ratio threshold 1.0": ((m.vix / m.vix3m).shift(1) >= RATIO_A) | (vixy_5d >= _exit.VIXY_CONFIRM),
        "b. SPY 2% under 20-day high": base & spy_weak,
        "c. credit HYG/IEF weak": base & credit_weak,
        "d. sector breadth <= 4 of 9": base & breadth_weak,
    }


def entry_stats(name: str, state: pd.Series, spy: pd.Series, fall_list: list, years: float) -> tuple[float, float, int]:
    """evaluate() run quietly: (median drawdown at the switch, false alarms per year, missed falls)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        med, fpy = evaluate(name, state, spy, fall_list, years)
    return med, fpy, buf.getvalue().count("MISSED")


def report(mode: str, only: set[str] | None = None) -> dict:
    """Print the money table per variant and mix, the entry statistics and the pass bar; return the results.

    `only` limits the variants run (the test window shows VIX alone and the chosen variant, nothing else).
    """
    m, px = market()
    x = extra()
    start, end = _exit.WINDOWS[mode]
    days = [d for d in m.index if start <= d <= end and d in px.index]
    years = (days[-1] - days[0]).days / 365.25
    calm = ((m.vix / m.vix3m).shift(1) < _exit.RATIO_EXIT).loc[days]
    fl = [f for f in falls(m.close) if days[0] <= f[0] and f[1] <= days[-1]]
    print(f"{mode}: {days[0]} -> {days[-1]}, {len(days)} sessions, {len(fl)} SPY falls >= 8% fully inside")
    hold = metrics(portfolio_exposure(pd.Series(1.0, index=days), px.loc[days]), years)
    print(f"SPY buy and hold: {hold['total']:+.1%}, max DD {hold['mdd']:+.1%}")
    print(f"\n{'variant':30s} {'mix':>8s} {'total':>8s} {'CAGR':>6s} {'max DD':>7s} {'Sharpe':>6s} {'sw/yr':>6s} {'false/yr':>8s} {'med DD at switch':>16s} {'missed':>6s}")
    res = {}
    for name, trig in triggers(m, x).items():
        if only is not None and name not in only:
            continue
        state = run_switch(None, trig.loc[days], calm)
        med, fpy, missed = entry_stats(name, state, m.close, fl, years)
        sw = int((state != state.shift(1)).iloc[1:].sum()) / years
        for w in MIXES:
            r = metrics(portfolio_exposure(1 - (1 - w) * state.astype(float), px.loc[days]), years)
            res[(name, w)] = r
            mix = "100% SHV" if w == 0 else "50/50"
            print(f"{name:30s} {mix:>8s} {r['total']:+8.1%} {r['cagr']:+6.1%} {r['mdd']:+7.1%} {r['sharpe']:6.2f} {sw:6.1f} {fpy:8.1f} {med:+16.1%} {missed:6d}")
    print(f"\n=== pass bar: total above VIX alone at both mixes, max DD no worse than buy and hold ({hold['mdd']:+.1%})")
    passing = {}
    for name in dict.fromkeys(n for n, _ in res if n != "VIX alone"):
        ok = all(res[(name, w)]["total"] > res[("VIX alone", w)]["total"] and res[(name, w)]["mdd"] >= hold["mdd"] for w in MIXES)
        print(
            f"  {name}: " + ", ".join(f"{'100% SHV' if w == 0 else '50/50'} {res[(name, w)]['total']:+.1%} vs {res[('VIX alone', w)]['total']:+.1%}" for w in MIXES) + f" -> {'PASS' if ok else 'FAIL'}"
        )
        if ok:
            passing[name] = res[(name, 0.5)]["total"]
    chosen = max(passing, key=lambda n: passing[n], default=None)
    print(f"chosen: {chosen or 'none -> verdict NO, the test window stays unused'}")
    return {"results": res, "chosen": chosen}


def check() -> None:
    """Assert that every filtered trigger implies VIX alone's (or the stricter ratio) and that no filter reads session t."""
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(120)]
    rng = np.random.default_rng(3)

    def walk(n: int = 120, scale: float = 0.01) -> np.ndarray:
        return 100 * np.exp(np.cumsum(rng.normal(0, scale, n)))

    m = pd.DataFrame({"close": walk(), "vix": 20 + 5 * rng.random(120), "VIXY": walk(scale=0.05)}, index=idx)
    m["vix3m"] = m.vix * (0.9 + 0.2 * rng.random(120))
    x = pd.DataFrame({t: walk() for t in ["HYG", "IEF", *SECTORS]}, index=idx)
    base = triggers(m, x)
    # 1. filters only remove switches: each filtered trigger implies VIX alone's; (a) implies it too (stricter ratio)
    for name, trig in base.items():
        assert not (trig & ~base["VIX alone"]).any(), name
    # 2. no look-ahead: perturbing session k's closes never changes session k's trigger
    for k in range(60, 120, 7):
        for factor in (0.7, 1.4):
            m2, x2 = m.copy(), x.copy()
            m2.iloc[k], x2.iloc[k] = m.iloc[k] * factor, x.iloc[k] * np.where(x.columns == "IEF", 1.0, factor)  # IEF kept: a ratio must move
            for name, trig in triggers(m2, x2).items():
                assert trig.iloc[k] == base[name].iloc[k], (name, k, factor)
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    if mode == "dev":
        report(mode)
        return
    if mode == "test":
        if TEST_GUARD.exists():
            sys.exit(f"The 2008-2015 test was already run ({TEST_GUARD} exists); see the spec. Not re-run.")
        chosen = report("dev")["chosen"]
        if chosen is None:
            sys.exit("No variant passed dev: the test window stays unused.")
        print("\n" + "=" * 100 + f"\nTEST (one shot): {chosen} against VIX alone\n")
        TEST_GUARD.parent.mkdir(parents=True, exist_ok=True)
        TEST_GUARD.write_text(f"{chosen}\n")
        report("test", only={"VIX alone", chosen})
        return
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
