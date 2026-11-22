"""Spike: a jump model on VIX inputs -- keep VIX's speed, cut its false switches.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-jump-model-spike-design.md

Run (jumpmodels is not a project dependency):
  uv run --with jumpmodels python scripts/regime/vix_jump_spike.py dev        # 2016 -> 2024
  (holdout: spent on 2026-11-22 by `final`; the `holdout` mode now refuses to run)
  uv run --with jumpmodels python scripts/regime/vix_jump_spike.py curve      # dev exploration: speed vs false-alarm trade-off
  uv run --with jumpmodels python scripts/regime/vix_jump_spike.py costs      # dev: SPY when normal, SHV when defensive, per variant
  uv run --with jumpmodels python scripts/regime/vix_jump_spike.py final      # the single holdout run (2025-01-02 -> 2026-11-20):
                                                                              # VIX alone (1 session) and B penalty 15, chosen on dev

A daily verdict before the open of session t, from closes up to t-1.
"""

import contextlib
import importlib.util
import io
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_leaders = _load("leaders_crack_spike", REPO / "scripts" / "regime" / "leaders_crack_spike.py")
falls, run_switch, evaluate = _leaders.falls, _leaders.run_switch, _leaders.evaluate

# Model (design doc, fixed before running)
PENALTY, CLIP_SD = 50.0, 3.0
RATIO_CONFIRM, VIXY_CONFIRM, RATIO_EXIT = 0.95, 0.10, 0.90  # VIX-alone baseline
TARGET_DD = -0.03
SWITCH_COST = 0.0010  # 10 bp per switch (sell one ETF, buy the other), at the open
WINDOWS = {"dev": (date(2016, 1, 4), date(2024, 12, 31)), "holdout": (date(2025, 1, 2), date(2026, 11, 20))}


def market() -> pd.DataFrame:
    """Daily closes (Yahoo, adjusted) of SPY and the volatility inputs, from December 2007 (VIX3M start)."""
    import yfinance as yf

    raw = yf.download(["SPY", "^VIX", "^VIX3M", "VIXY"], start="2007-06-01", end="2026-11-21", progress=False, auto_adjust=True)["Close"]
    m = raw.rename(columns={"^VIX": "vix", "^VIX3M": "vix3m"}).dropna(subset=["SPY"])
    m.index = m.index.date
    return m


def ew(x: pd.Series, halflife: int) -> pd.Series:
    """Exponentially weighted mean with the given half-life in sessions."""
    return x.ewm(halflife=halflife, min_periods=halflife).mean()


def features(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Feature sets B (volatility inputs) and C (the paper's), shifted one session: row t uses closes up to t-1."""
    r = m.SPY.pct_change()
    dd10 = np.sqrt(ew(r.pow(2) * (r < 0), 10))
    b = pd.DataFrame({"downside_dev_10": dd10, "vix_ratio": m.vix / m.vix3m, "vix_chg_5": np.log(m.vix / m.vix.shift(5))})
    c = pd.DataFrame({"downside_dev_10": dd10})
    for hl in (20, 60):
        c[f"sortino_{hl}"] = ew(r, hl) / np.sqrt(ew(r.pow(2) * (r < 0), hl))
    return {"B: jump model on VIX inputs": b.shift(1).dropna(), "C: paper's model (SPY features)": c.shift(1).dropna()}


def online_states(x: pd.DataFrame, sessions: list[date], penalty: float = PENALTY) -> pd.Series:
    """Defensive flag per session: refit every 6 months on an expanding window, online prediction in between."""
    out = []
    refits = sorted({date(d.year, 1 if d.month <= 6 else 7, 1) for d in sessions})
    for k, refit in enumerate(refits):
        block_end = refits[k + 1] if k + 1 < len(refits) else date.max
        train = x[x.index < refit]
        clipper = DataClipperStd(mul=CLIP_SD).fit(train)
        scaler = StandardScalerPD().fit(clipper.transform(train))
        model = JumpModel(n_components=2, jump_penalty=penalty).fit(scaler.transform(clipper.transform(train)))
        bear = int(np.argmax(model.centers_[:, list(x.columns).index("downside_dev_10")]))  # higher downside deviation
        hist = x[x.index < block_end]
        states = model.predict_online(scaler.transform(clipper.transform(hist)))  # causal: run from the training start
        block = states[[d for d in hist.index if refit <= d < block_end]]
        out.append(block == bear)
    flag = pd.concat(out)
    return flag.loc[[d for d in sessions if d in flag.index]]


def summary(name: str, state: pd.Series, spy: pd.Series, fall_list: list, years: float) -> str:
    """One table row: evaluate() quietly, then count its fall classes."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        med, fpy = evaluate(name, state, spy, fall_list, years)
    text = buf.getvalue()
    dds = [float(x) / 100 for x in re.findall(r"(?:ON TIME|LATE), defensive \S+, ([+-][\d.]+)% from the high", text)]
    caught = f"{np.median(dds):+.1%}" if dds else "-"
    counts = {k: text.count(f"): {k}") for k in ("EARLY", "ON TIME", "LATE", "MISSED")}
    return f"{name:34s} {counts['EARLY']:>5d} {counts['ON TIME']:>7d} {counts['LATE']:>4d} {counts['MISSED']:>6d}   {med:+7.1%}   {caught:>8s}   {fpy:5.1f}   {state.mean():5.0%}"


def curve_states(m: pd.DataFrame, valid: list[date]) -> dict[str, pd.Series]:
    """Defensive flags of every curve point: B at several penalties, VIX alone with an N-session confirmation."""
    out = {}
    ratio = (m.vix / m.vix3m).shift(1)
    vix_on = ((ratio >= RATIO_CONFIRM) | ((m.VIXY / m.VIXY.shift(5) - 1).shift(1) >= VIXY_CONFIRM)).loc[valid]
    calm = (ratio < RATIO_EXIT).loc[valid]
    for n in (1, 2, 3):
        confirmed = vix_on.astype(int).rolling(n).sum() >= n  # stressed n sessions in a row
        out[f"VIX alone, {n} session(s) in a row"] = run_switch(None, confirmed, calm)
    x = features(m)["B: jump model on VIX inputs"]
    for penalty in (0.0, 5.0, 15.0, 35.0, 50.0):
        out[f"B, jump penalty {penalty:g}"] = online_states(x, valid, penalty)
    return out


def curve(m: pd.DataFrame, valid: list[date], fl: list, years: float) -> None:
    """Dev exploration: every curve point through the same fall evaluation."""
    print(f"{'variant':34s} {'early':>5s} {'on time':>7s} {'late':>4s} {'missed':>6s}   {'median*':>7s}   {'med.caught':>8s}   {'false/yr':>5s}   {'defensive':>5s}")
    for name, state in curve_states(m, valid).items():
        print(summary(name, state, m.SPY, fl, years), flush=True)
    print("* median drawdown at the switch over all falls, early = 0%, missed = -100%; med.caught = on time and late only")


def open_close() -> pd.DataFrame:
    """Adjusted daily opens and closes of SPY and SHV (Yahoo)."""
    import yfinance as yf

    raw = yf.download(["SPY", "SHV"], start="2015-06-01", end="2026-11-21", progress=False, auto_adjust=True)
    px = pd.concat({f"{f}_{s}": raw[f][s] for f in ("Open", "Close") for s in ("SPY", "SHV")}, axis=1).dropna()
    px.index = px.index.date
    return px


def portfolio(state: pd.Series, px: pd.DataFrame) -> pd.Series:
    """Daily returns of SPY when normal, SHV when defensive; the switch trades at the open (overnight move = old position)."""
    rets, prev = [], None
    days = list(state.index)
    for i, d in enumerate(days):
        now = "SHV" if state[d] else "SPY"
        held = prev or now
        if i == 0:
            r = px.at[d, f"Close_{now}"] / px.at[d, f"Open_{now}"] - 1
        else:
            overnight = px.at[d, f"Open_{held}"] / px.at[days[i - 1], f"Close_{held}"] - 1
            intraday = px.at[d, f"Close_{now}"] / px.at[d, f"Open_{now}"] - 1
            cost = SWITCH_COST if now != held else 0.0
            r = (1 + overnight) * (1 + intraday) * (1 - cost) - 1
        rets.append(r)
        prev = now
    return pd.Series(rets, index=days)


def stats(r: pd.Series, years: float) -> str:
    """CAGR, max drawdown, Sharpe (no risk-free) and Calmar of a daily return series."""
    level = (1 + r).cumprod()
    cagr = level.iloc[-1] ** (1 / years) - 1
    mdd = (level / level.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252)
    return f"{level.iloc[-1] - 1:+8.1%}  {cagr:+6.1%}  {mdd:+7.1%}  {sharpe:5.2f}  {cagr / -mdd:5.2f}"


def final(m: pd.DataFrame, valid: list[date], fl: list, years: float) -> None:
    """The single holdout run: the two variants chosen on dev, fall table and SPY / SHV portfolio, nothing tuned."""
    states = curve_states(m, valid)
    chosen = {k: states[k] for k in ("VIX alone, 1 session(s) in a row", "B, jump penalty 15")}
    for name, state in chosen.items():
        evaluate(name, state, m.SPY, fl, years)
    px = open_close()
    days = [d for d in valid if d in px.index]
    spy = px.Close_SPY.loc[days].pct_change().fillna(px.at[days[0], "Close_SPY"] / px.at[days[0], "Open_SPY"] - 1)
    print(f"\n{'variant':34s} {'total':>8s}  {'CAGR':>6s}  {'max DD':>7s}  {'Sharpe':>5s}  {'Calmar':>5s}  {'switches/yr':>11s}  {'defensive':>9s}")
    print(f"{'SPY buy and hold':34s} {stats(spy, years)}")
    for name, state in chosen.items():
        state = state.loc[days]
        switches = int((state != state.shift(1)).iloc[1:].sum())
        print(f"{name:34s} {stats(portfolio(state, px), years)}  {switches / years:11.1f}  {state.mean():9.0%}")


def costs(m: pd.DataFrame, valid: list[date], years: float) -> None:
    """Dev: price the speed / false-alarm trade-off with a SPY / SHV portfolio per curve point."""
    px = open_close()
    days = [d for d in valid if d in px.index]
    spy = px.Close_SPY.loc[days].pct_change().fillna(px.at[days[0], "Close_SPY"] / px.at[days[0], "Open_SPY"] - 1)
    print(f"{'variant':34s} {'total':>8s}  {'CAGR':>6s}  {'max DD':>7s}  {'Sharpe':>5s}  {'Calmar':>5s}  {'switches/yr':>11s}  {'defensive':>9s}")
    print(f"{'SPY buy and hold':34s} {stats(spy, years)}")
    for name, state in curve_states(m, days).items():
        state = state.loc[days]
        switches = int((state != state.shift(1)).iloc[1:].sum())
        print(f"{name:34s} {stats(portfolio(state, px), years)}  {switches / years:11.1f}  {state.mean():9.0%}", flush=True)
    print(f"switch cost {SWITCH_COST:.2%} at the open; SHV = short T-bills ETF")


def main() -> None:
    """Entry point: `dev`, `holdout` or `curve` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "holdout":
        sys.exit("The 2025-2026 holdout was spent on 2026-11-22 by `final`; see the spec. Not re-run.")
    start, end = WINDOWS["holdout" if mode == "final" else "dev" if mode in ("curve", "costs") else mode]
    m = market()
    feats = features(m)
    valid = [d for d in m.index if start <= d <= end and all(d in f.index for f in feats.values())]
    years = (valid[-1] - valid[0]).days / 365.25
    fl = [f for f in falls(m.SPY) if valid[0] <= f[0] and f[1] <= valid[-1]]
    print(f"{mode}: {valid[0]} -> {valid[-1]}, {len(valid)} sessions, {len(fl)} SPY falls >= 8% fully inside")
    if mode == "curve":
        return curve(m, valid, fl, years)
    if mode == "costs":
        return costs(m, valid, years)
    if mode == "final":
        return final(m, valid, fl, years)

    results = {}
    for name, x in feats.items():
        state = online_states(x, valid)
        results[name] = evaluate(name, state, m.SPY, fl, years)

    ratio = (m.vix / m.vix3m).shift(1)
    vix_on = (ratio >= RATIO_CONFIRM) | ((m.VIXY / m.VIXY.shift(5) - 1).shift(1) >= VIXY_CONFIRM)
    base_dd, base_false = evaluate("baseline: VIX alone", run_switch(None, vix_on.loc[valid], (ratio < RATIO_EXIT).loc[valid]), m.SPY, fl, years)

    print("\n=== development pass bar: median drawdown at switch better than -3%, false switches/year below VIX alone")
    for name, (med, fpy) in results.items():
        ok = med > TARGET_DD and fpy < base_false
        print(f"  {name}: median {med:+.1%}, {fpy:.1f} false/year (VIX alone {base_dd:+.1%}, {base_false:.1f}/year) -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
