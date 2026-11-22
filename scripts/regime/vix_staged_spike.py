"""Spike: staged exposure after a VIX switch -- cut the return cost of being defensive.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-staged-exposure-spike-design.md

Run (jumpmodels is not a project dependency; the sibling spike modules import it):
  uv run --with jumpmodels python scripts/regime/vix_staged_spike.py check   # self-check of the state machine
  uv run --with jumpmodels python scripts/regime/vix_staged_spike.py dev     # 2016 -> 2024: run, read

A daily SPY exposure before the open of session t, from closes up to t-1.
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


_exit = _load("vix_exit_spike", REPO / "scripts" / "regime" / "vix_exit_spike.py")
market, entry_trigger, exit_signals, metrics = _exit.market, _exit.entry_trigger, _exit.exit_signals, _exit.metrics
run_switch, falls, portfolio = _exit.run_switch, _exit.falls, _exit.portfolio


# Rules (design doc, fixed before running)
CALM_SESSIONS, CONFIRM_DROP, FLOOR, SWITCH_COST = 3, 0.97, 0.5, 0.0010


def run_staged(trigger: pd.Series, calm: pd.Series, spy_prev: pd.Series, e1: pd.Series, e3: pd.Series, high60: pd.Series | None = None) -> pd.Series:
    """SPY exposure per session in [0, 1]: B (back in by thirds) or, with `high60`, C (half until the fall is confirmed).

    Inputs are indexed by session t and known before its open (`spy_prev[t]` = SPY close of t-1, `high60[t]` = highest
    close of the 60 sessions up to t-1). Each morning of an episode: a new low sends the returned tranches out, else E1
    / E3 (levels) bring tranche 1 / 2 back; the calm exit (3 calm sessions, counted like `run_switch`) is checked every
    morning and returns everything. Tranche 3 returns only through the calm exit.
    """
    in_episode, back, low, quiet, confirmed, prev_trigger, out = False, [False, False], 0.0, 0, False, False, {}
    for d in trigger.index:
        e = 1.0
        if not in_episode:
            if trigger[d]:
                in_episode, back, low, quiet, confirmed = True, [False, False], spy_prev[d], 0, False
        else:
            if spy_prev[d] < low:
                back, low = [False, False], spy_prev[d]
            else:
                back = [back[0] or bool(e1[d]), back[1] or bool(e3[d])]
            quiet = quiet + 1 if calm[d] else 0
            if quiet >= CALM_SESSIONS:
                in_episode = False
        if in_episode:
            if high60 is not None and not confirmed:
                confirmed = (trigger[d] and prev_trigger) or spy_prev[d] <= CONFIRM_DROP * high60[d]
            e = sum(back) / 3
            if high60 is not None and not confirmed:
                e = max(e, FLOOR)
        out[d] = e
        prev_trigger = bool(trigger[d])
    return pd.Series(out)


def portfolio_exposure(e: pd.Series, px: pd.DataFrame) -> pd.Series:
    """Daily returns of `e` SPY + (1 - e) SHV, rebalanced at the open; cost SWITCH_COST x |change in e|.

    The overnight move is earned by the old weights, the open-to-close move by the new ones; the first session has
    no overnight and no cost. At e in {0, 1} this is `portfolio()` of `vix_jump_spike.py`.
    """
    old = e.shift(1)
    overnight = old * (px.Open_SPY / px.Close_SPY.shift(1) - 1) + (1 - old) * (px.Open_SHV / px.Close_SHV.shift(1) - 1)
    intraday = e * (px.Close_SPY / px.Open_SPY - 1) + (1 - e) * (px.Close_SHV / px.Open_SHV - 1)
    r = (1 + overnight) * (1 + intraday) * (1 - SWITCH_COST * (e - old).abs()) - 1
    r.iloc[0] = intraday.iloc[0]
    return r


def check() -> None:
    """Assert the state machines and the exposure portfolio on synthetic series (sessions are numbered s1..s12)."""
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(12)]

    def ser(vals: list) -> pd.Series:
        return pd.Series(vals, index=idx)

    def on(*sessions: int) -> pd.Series:
        return ser([k in sessions for k in range(1, 13)])

    def run(trigger, e1=None, e3=None, calm=None, spy=None, high60=None) -> list[float]:
        """Exposure per session s1..s12, rounded."""
        e = run_staged(trigger, on() if calm is None else calm, ser([100.0] * 12) if spy is None else spy, on() if e1 is None else e1, on() if e3 is None else e3, high60)
        return [round(float(v), 4) for v in e]

    third, two_thirds = round(1 / 3, 4), round(2 / 3, 4)
    calm_789 = on(7, 8, 9)
    # 1. E1 and E3 never true: B is exactly VIX alone (also over several episodes and calm exits)
    trigger, calm = on(1, 2, 7, 8), on(3, 4, 5, 6, 9, 10, 11)
    assert run(trigger, calm=calm) == [1.0 - float(s) for s in run_switch(None, trigger, calm)]
    # 2. tranches come back by thirds, the calm exit returns the rest
    assert run(on(1), e1=on(3, 4, 5, 6, 7, 8, 9), e3=on(5, 6, 7, 8, 9), calm=calm_789) == [0, 0, third, third] + [two_thirds] * 4 + [1.0] * 4
    # 3. new low sends the returned tranches out; a tranche whose signal is still true returns the next morning (level rule)
    spy = ser([100.0] * 5 + [99.0] + [100.0] * 6)
    assert run(on(1), e1=on(3, 4, 5, 6, 7, 8, 9), e3=on(5, 6, 7, 8, 9), calm=calm_789, spy=spy) == [0, 0, third, third, two_thirds, 0] + [two_thirds] * 2 + [1.0] * 4
    # 4. calm exit and new low on the same morning: the calm exit wins
    assert run(on(1), calm=calm_789, spy=ser([100.0] * 8 + [99.0] + [100.0] * 3))[8] == 1.0
    # 5. C: half until confirmed (two stressed sessions in a row, or SPY 3% below its 60-session high)
    high60 = ser([100.0] * 12)
    assert run(on(1), high60=high60)[:3] == [0.5, 0.5, 0.5]
    assert run(on(1, 2), high60=high60)[:3] == [0.5, 0.0, 0.0]
    assert run(on(1), spy=ser([96.9] + [100.0] * 11), high60=high60)[:2] == [0.0, 0.0]
    assert run(on(1), spy=ser([97.1] + [100.0] * 11), high60=high60)[0] == 0.5
    # 6. exposure portfolio: equals portfolio() at 0 / 1, and half a switch costs 5 bp
    rng = np.random.default_rng(1)
    cols = {}
    for s in ("SPY", "SHV"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 12)))
        cols[f"Close_{s}"], cols[f"Open_{s}"] = close, close * np.exp(rng.normal(0, 0.003, 12))
    px = pd.DataFrame(cols, index=idx)
    defensive = on(2, 3, 4, 8, 9)
    assert np.allclose(portfolio_exposure(1.0 - defensive.astype(float), px), portfolio(defensive, px), atol=1e-12)
    flat = pd.DataFrame({c: [100.0] * 12 for c in px.columns}, index=idx)
    assert abs(portfolio_exposure(ser([1.0, 0.5] + [0.5] * 10), flat).iloc[1] + 0.0005) < 1e-12
    print("self-check OK")


def main() -> None:
    """Entry point: `check` or `dev` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
