"""Market facts for the pods' ETFs and the score of last week's views against what happened."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from trading_agent_framework.core.regime import RegimeReading
from trading_agent_framework.strategies.common.scoring import annualized_volatility, compute_return_from_prices
from trading_agent_framework.strategies.sector_pods.pods import DEFENSIVE_ETFS, PODS

MIN_BARS = 201  # a 200-day average plus the day before it

_RETURN_SESSIONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}


def _pct(value: float) -> float:
    return round(value * 100, 1)


def week_return_pct(closes: Sequence[float]) -> float | None:
    """One-week return of a close series, in percent, with the same window and rounding as `etf_facts`' `return_1w_pct`.

    `outcome` needs SPY's last-week return to put the portfolio's own return in context. `etf_facts` also needs 201
    closes (a 200-day average) and computes volatility, trend and the other returns, none of which SPY needs here, so
    this reads only the 5 sessions that matter: 6 closes are enough.

    Args:
        closes: Daily closes, oldest first.

    Returns:
        The return over the last `_RETURN_SESSIONS["1w"]` (5) sessions in percent, rounded to 0.1; None when there are
        fewer than 6 closes or the starting close is not positive.
    """
    value = compute_return_from_prices(closes, _RETURN_SESSIONS["1w"])
    return None if value is None else _pct(value)


def etf_facts(closes: Sequence[float], spy: Sequence[float]) -> dict[str, float] | None:
    """Return, relative-strength, volatility and trend facts for one ETF, in percent.

    Args:
        closes: Daily closes of the ETF, oldest first.
        spy: Daily closes of SPY, oldest first; a relative key is left out when it lacks the bars.

    Returns:
        None when `closes` has fewer than `MIN_BARS`. Otherwise `return_{1w,1m,3m,6m}_pct`, `relative_{1m,3m}_pct`
        (the ETF's return minus SPY's, in points), `volatility_20d_pct` (annualized) and `vs_200d_avg_pct` (the last
        close against the simple mean of the last 200 closes).
    """
    if len(closes) < MIN_BARS:
        return None
    facts: dict[str, float] = {}
    returns: dict[str, float] = {}
    for name, sessions in _RETURN_SESSIONS.items():
        value = compute_return_from_prices(closes, sessions)
        if value is None:
            continue
        returns[name] = value
        facts[f"return_{name}_pct"] = _pct(value)
    for name in ("1m", "3m"):
        spy_return = compute_return_from_prices(spy, _RETURN_SESSIONS[name])
        if spy_return is not None and name in returns:
            facts[f"relative_{name}_pct"] = _pct(returns[name] - spy_return)
    volatility = annualized_volatility(closes, 20)
    if volatility is not None:
        facts["volatility_20d_pct"] = _pct(volatility)
    facts["vs_200d_avg_pct"] = _pct(closes[-1] / fmean(closes[-200:]) - 1)
    return facts


def outcome(
    weights: Mapping[str, float],
    pods: Mapping[str, Mapping[str, Any] | None],
    facts: Mapping[str, Mapping[str, Mapping[str, float]]],
    spy_1w_pct: float | None,
) -> dict[str, Any]:
    """Score last week's book and pod views against the week that followed.

    Args:
        weights: Last week's target weights by symbol (SHV or symbols without facts add nothing).
        pods: Last week's pod views by pod key; each has `scores`, a list of
            `{"symbol", "score" (-2..+2), "reason"}`. A missing pod, a None view or the unavailable marker scores None.
        facts: This week's `etf_facts` by pod key then symbol.
        spy_1w_pct: SPY's return over the same week, or None.

    Returns:
        The window, `positions_1w_pct` (each weighted symbol's 1-week return), `invested_weight` (the weight of the
        symbols that have facts), `portfolio_1w_pct` (the return on the whole portfolio, the sum of weight times
        return, so SHV, cash and symbols without facts contribute 0; None when no weighted symbol has facts),
        `spy_1w_pct` and, per pod, the mean 1-week return of its positively scored symbols minus its negatively scored ones (None when
        either side is empty or the pod is missing).
    """
    # shortcut: the outcome uses the last 5 sessions, not the sessions since the previous review; exact after a holiday week or an abandoned week only if DailySeries gains dates
    returns = {symbol: ret for pod_facts in facts.values() for symbol, symbol_facts in pod_facts.items() if (ret := symbol_facts.get("return_1w_pct")) is not None}
    positions = {symbol: returns[symbol] for symbol in weights if symbol in returns}
    invested = sum(weights[symbol] for symbol in positions)
    portfolio = sum(weights[symbol] * ret for symbol, ret in positions.items()) if positions else None
    pod_scores: dict[str, dict[str, float | None]] = {}
    for key in (pod.key for pod in PODS):
        view = pods.get(key) or {}
        scores = {item["symbol"]: item["score"] for item in view.get("scores") or []}
        positive = [returns[s] for s, score in scores.items() if score > 0 and s in returns]
        negative = [returns[s] for s, score in scores.items() if score < 0 and s in returns]
        spread = fmean(positive) - fmean(negative) if positive and negative else None
        pod_scores[key] = {"positive_minus_negative_1w_pct": None if spread is None else round(spread, 1)}
    return {
        "window": "last 5 sessions",
        "positions_1w_pct": positions,
        "invested_weight": round(invested, 4),
        "portfolio_1w_pct": None if portfolio is None else round(portfolio, 1),
        "spy_1w_pct": spy_1w_pct,
        "pods": pod_scores,
    }


def stress_view(reading: RegimeReading | None, market_fall: bool | None) -> dict[str, Any] | None:
    """The market-stress picture the risk manager and the portfolio manager read.

    The condition names use the default regime parameters (50-day average, 5% off the high), the only ones sector_pods
    runs with.

    Args:
        reading: The strategy's full regime reading, None until there is enough history.
        market_fall: The "VIX alone" early warning, None when `reading` is None.

    Returns:
        None without a reading; otherwise the early warning, the regime score (0 to 4), the four conditions and the
        defensive ETFs.
    """
    if reading is None:
        return None
    return {
        "early_warning": market_fall,
        "regime_score": reading.score,
        "conditions": {
            "vix_above_vix3m": reading.inverted,
            "vix_spike": reading.vix_spike,
            "spy_below_sma50": reading.below_sma,
            "spy_5pct_off_high": reading.off_high,
        },
        "defensive_etfs": list(DEFENSIVE_ETFS),
    }
