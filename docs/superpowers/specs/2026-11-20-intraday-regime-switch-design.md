# Intraday regime switch (UPRO / cash / SPXU): research design (short)

Fresh start (branch `feat/intraday-regime-switch` from `main`). Experimentation mode, decided by the user: a research spike, no reviewers and no test-first process; code lives in `scripts/research/intraday_regime/`, outside `src/`, and nothing touches the production regime or any strategy. Earlier work (bear call spread, fast-regime screens, the daily and 20-minute spikes) is recorded in `docs/superpowers/replays/2026-11-19-*` and on the unmerged branch `feat/fast-regime-research`.

## Intent

A strategy that reads the market regime very fast from intraday data and adapts: **bullish -> UPRO** (3x S&P), **bearish -> SPXU** (3x short S&P), **neutral -> cash**; each position has a **trailing stop** as a safety net. The detector is the key part; the user expects volatility (VIX) to be the fastest input, with noise handled by confirmation thresholds and the trailing-stop distance. News-headline sentiment comes second, as a filter on this detector (a later design).

## Success bar (the user's, 2026-11-20)

- **Positive every calendar year**, and **beats SPY buy-and-hold** over the same window.
- **P&L per regime change** reported as a diagnostic.
- Judged on the development window 2016-01-04 to 2024-06-28; the held-out window 2024-07-01 to the last session is scored once, at the end, on the single chosen setting.

## Approaches (decided with the user)

- **C, volatility-led hybrid (chosen):** a VIXY move is the fast trigger; a trend band and thresholds confirm it and calm it; the trailing stop catches what slips through.
- A, trend band alone: kept as the baseline row (to see what volatility adds).
- B, volatility alone: inside C (C with the confirmation off is close to B).
- D, a machine-learning classifier: rejected (about ten falls to learn from, one clean held-out test).

## The detector (fixed rules; parameters on a small grid)

Inputs on **5-minute regular-session bars** (09:30-16:00 New York), Alpaca consolidated feed (SIP), split/dividend-adjusted, from 2016: SPY, VIXY, UPRO, SPXU. No intraday VIX index history exists (Yahoo keeps 60 days, Alpaca none), so **VIXY is the VIX proxy** (a futures fund with daily decay: only its move versus its previous close is used, never its level).

At each bar close:
- `v` = VIXY / VIXY's previous session close - 1 (the volatility move today)
- `s` = SPY / SPY's previous session close - 1 (the price move today)
- `d` = SPY / mean of SPY's 5-minute closes over the last 5 sessions - 1 (the trend band)

States and transitions (an exit threshold is looser than its entry threshold, so the regime does not flip on every wiggle):
- **-> bearish:** `v >= V_UP` and (`s <= -S_CONF` or `d <= -BAND`)
- **-> bullish:** `d >= +BAND` and `v <= 0`
- **bullish -> neutral:** `v >= V_UP / 2` (the volatility override: leave UPRO before the trend turns) or `d < 0`
- **bearish -> neutral:** `v <= 0` and `d > -BAND / 2`
- A trailing stop that fires sends the position to cash; the strategy re-enters only after the detector passes through neutral (no re-entry into the same regime it was stopped out of).

**Baseline row (A):** the same states with every `v` condition removed.

**Grid (fixed now; 16 cells for C):** `V_UP` in {5%, 8%}, `S_CONF` in {0.5%, 1.0%}, `BAND` in {1%, 2%}, trailing stop in {4%, 8%} of the ETF's highest price since entry (UPRO) or lowest-inverse equivalent (SPXU: its own highest price since entry). Baseline A: `BAND` x stop = 4 cells.

**Choosing one cell (fixed now):** among the cells that pass the bar on the development window, take the one whose neighbours (each parameter moved one step) pass the most often; ties by the higher development return. If none passes: the result is NO and nothing is run on the held-out window.

## Execution model

- A signal at a bar's close trades at the **next bar's open**; cost 3 bps per side (spread and slippage on UPRO/SPXU; commission 0 at Alpaca).
- Trailing stops are checked on each bar's low (UPRO) / low of SPXU, fill at the stop price, or at the bar's open when it gaps through.
- Positions are held overnight and across days (the user reconsidered a flat-at-night rule on 2026-11-20 and kept overnight holding); 100% of equity per position (so 3x S&P exposure); no position on neutral. Overnight gaps hit the trailing stop at the next open (filled at the open).
- **Cash account (the user's live account, no margin):** UPRO and SPXU are bought outright (SPXU is long an inverse fund, never a short sale), so no margin is needed and the pattern-day-trader rule does not apply. Settlement is T+1: a buy paid with the proceeds of a same-day sale must not be sold before those proceeds settle (the next session), or it is a good-faith violation. The replay enforces it: a position opened with unsettled proceeds cannot be sold the same session, neither by a regime exit nor by its trailing stop (the stop is armed from the next session's open); a position opened with settled cash can be sold any time.
- Benchmarks on the same bars: SPY buy-and-hold, UPRO buy-and-hold.

## Report

Per cell: total return, per calendar year return (each must be > 0), max drawdown, number of regime changes, time in each state, and **P&L per regime change** split into bullish, bearish and stopped-out trades; the chosen cell's trade list as CSV under `logs/` (git-ignored). A short record in `docs/superpowers/replays/` at the end.

## Left to the builder

Script layout under `scripts/research/intraday_regime/`, the bar cache format under `cache/` (git-ignored), the 5-day mean's exact window (the last 5 x 78 bars or session-based), and how the grid is printed.
