"""Pure alternative triggers for the bear put spread research: earlier than the regime, with cheap puts.

Each trigger is an on/off state machine over SPY and VIX closes. A session's state uses only the closes BEFORE it
(like `regime_by_session`), so the spread opens at that session's close with no look-ahead. The states use the
regime's own convention (-1 while a spread should be held, 0 otherwise), so `research.simulate` takes them
unchanged in place of the regime readings.

- `t1`: open when SPY closes below its 50-day average and VIX < 25; hold until SPY closes back above it.
- `t2`: open when SPY is below its 50-day average and the 50-day is below the 200-day; hold until the 50-day
  crosses back above the 200-day.
- `t3`: open on a new 20-session closing low and VIX < 25; hold until SPY closes above its 10-day average.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

TRIGGERS = ("t1", "t2", "t3")
VIX_CEILING = 25.0  # a spike above this means puts are already dear: the failure of the regime trigger


def _sma(values: Sequence[float], k: int, window: int) -> float | None:
    return sum(values[k - window + 1 : k + 1]) / window if k >= window - 1 else None


def _t1(spy: Sequence[float], vix: Sequence[float], k: int, on: bool) -> bool:
    avg = _sma(spy, k, 50)
    if avg is None:
        return False
    return spy[k] < avg and (on or vix[k] < VIX_CEILING)


def _t2(spy: Sequence[float], vix: Sequence[float], k: int, on: bool) -> bool:
    fast, slow = _sma(spy, k, 50), _sma(spy, k, 200)
    if fast is None or slow is None:
        return False
    return fast < slow and (on or spy[k] < fast)


def _t3(spy: Sequence[float], vix: Sequence[float], k: int, on: bool) -> bool:
    if on:
        avg = _sma(spy, k, 10)
        return avg is not None and spy[k] <= avg
    return k >= 19 and spy[k] < min(spy[k - 19 : k]) and vix[k] < VIX_CEILING


_RULES: dict[str, Callable[[Sequence[float], Sequence[float], int, bool], bool]] = {"t1": _t1, "t2": _t2, "t3": _t3}


def trigger_states(name: str, spy: Sequence[float], vix: Sequence[float]) -> list[int]:
    """The -1 (hold a spread) / 0 (flat) state each session would have had.

    Args:
        name: One of `TRIGGERS`.
        spy: SPY closes, oldest first.
        vix: VIX closes aligned with `spy`.

    Returns:
        One state per session; the state at index `i` uses the closes up to `i - 1`, and the first is 0.

    Raises:
        ValueError: If `name` is not a known trigger.
    """
    if name not in _RULES:
        raise ValueError(f"unknown trigger {name!r}; choose from {', '.join(TRIGGERS)}")
    rule = _RULES[name]
    states = [0]
    on = False
    for k in range(len(spy) - 1):
        on = rule(spy, vix, k, on)
        states.append(-1 if on else 0)
    return states
