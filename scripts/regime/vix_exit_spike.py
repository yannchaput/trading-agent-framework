"""Spike: a fast exit after a VIX switch -- get back into SPY early in the rebound.

Design (rules fixed before running): docs/superpowers/specs/2026-11-22-vix-exit-rule-spike-design.md

Run (jumpmodels is not a project dependency; the sibling spike module imports it):
  uv run --with jumpmodels python scripts/regime/vix_exit_spike.py check   # self-check of the state machine
  uv run --with jumpmodels python scripts/regime/vix_exit_spike.py dev     # 2016 -> 2024: run, read
  uv run --with jumpmodels python scripts/regime/vix_exit_spike.py test    # 2008 -> 2015, once, chosen variant only

A daily verdict before the open of session t, from closes up to t-1.
"""

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_leaders = _load("leaders_crack_spike", REPO / "scripts" / "regime" / "leaders_crack_spike.py")
run_switch = _leaders.run_switch

# Rules (design doc, fixed before running)
RATIO_CONFIRM, VIXY_CONFIRM, RATIO_EXIT = 0.95, 0.10, 0.90  # VIX-alone entry trigger and calm exit
CALM_SESSIONS = 3


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
    print("self-check OK")


def main() -> None:
    """Entry point: `check`, `dev` or `test` (see the module docstring)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        return check()
    sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
