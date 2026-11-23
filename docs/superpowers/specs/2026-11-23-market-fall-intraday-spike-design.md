# Market-fall intraday bet: research spike design (short)

## Intent and why

bull_bear loses money while the market falls (as in Oct-Nov 2026). The user wants a **second strategy on a separate
account that makes money in those falls**: it reads `in_market_fall` (`utils/helpers.py`, the VIX-alone signal) each
morning and, on a fall day, holds a bearish position for the day only. The strategy must be **profitable on its own**;
offsetting bull_bear's losses is the reason, but a smaller combined drawdown alone is not success.

Path, decided by the user: research -> framework backtest -> Alpaca paper -> Tastytrade (margin or cash) live.

This spike answers the first question only: **on `in_market_fall` days, does a long inverse-ETF position held from the
open to the close make money?** The framework strategy is designed after a GO, in its own brainstorm.

## Approaches (decided with the user, 2026-11-23)

- **A. Measure first, then an intraday inverse ETF (chosen).** Uses only what the framework supports today (stocks and
  ETFs, minute bars, trailing stops in the backtester). Stops after a day or two of work if there is no edge.
- B. Build options support (asset type, multi-leg orders on Alpaca and Tastytrade, an option backtester), then bear
  call / put spreads. Rejected for now: weeks of work before knowing there is an edge; same-day option prices before
  February 2024 can only be modelled; the two earlier spread studies pointed at the signal, not the structure.
- C. Short SPY or short MES futures. Rejected for now: needs shorting or futures support, as large as B.

## Goals

- One script, `scripts/regime/fall_intraday_spike.py`, run with `uv run python`. It prints one row per cell and per
  calendar year, and writes the chosen cell's trades as CSV under `logs/` (git-ignored).
- A short record in `docs/superpowers/replays/` at the end.

## Non-goals

- No production code, no `Strategy`, no change under `src/`, no tests (experimentation mode, as in the earlier
  `scripts/regime/` spikes).
- No options, no shorting, no futures, no other signal than `in_market_fall` as it is today.

## Anti-goals

- **A flattering result.** Costs on every trade, stops filled against the trader, no look-ahead.
- **Searching until something passes.** The grid, the bar and the choice rule below are fixed before the first run.
- **Touching 2025-2026 before the cell is chosen.**

## Data and signal

- Signal: `in_market_fall` called for each session `d` with Yahoo daily closes of `^VIX`, `^VIX3M`, `VIXY` **up to
  `d - 1`** (250+ sessions of history), so day `d` is traded on what was known before its open.
- Prices: Alpaca 1-minute bars (SIP, split/dividend-adjusted, `ALPACA_DATA_*` credentials), regular session only, for
  SPY, SH (-1x), SDS (-2x), SPXU (-3x); cached under `cache/` (git-ignored).
- Development window: 2016-01-04 to 2024-12-31. Held-out window: 2025-01-02 to the last complete session, run **once**
  on the chosen cell, nothing changed after it.

## Trade model

- On a fall day: buy at the open of the 09:30 bar, sell at the close of the 15:59 bar (flat every night).
- Fake account of **5,000** (the user's EUR 5,000, modelled in USD at 5,000; no FX), 100% of the account per trade;
  costs 3 bps per side (spread and slippage; commission 0). Fractional shares (6 decimals, as `fractional_qty`):
  whole shares on split-adjusted prices would be wrong, since SPXU's reverse splits inflate its adjusted 2016 price far
  above what traded then.
- Trailing stop: % below the ETF's highest price since entry, checked on each bar's low against the high reached on
  earlier bars (a bar's own high cannot raise the stop it is tested against); fills at the stop price, or at the bar's
  open if it gaps through. No re-entry the same day.
- Two account cases, both reported:
  - **unconstrained** (margin over $25k: every fall day);
  - **cash account, conservative**: 50% of the account on every fall day. Under T+1, a sale at day d's close is
    settled on d+1, so a broker may well let a cash account trade full size every day; this case is a lower bound
    until the broker confirms.

## Grid (fixed now: 12 cells)

Instrument {SH, SDS, SPXU} plus **-1 x SPY** (a short-SPY reference, sign check only) x trailing stop {none, 2%, 4%}.

Reference rows (not candidates): the same trade on **every** session and on **non-fall** sessions only (does the
signal add anything?), SPY buy-and-hold, and the combined account "SPY + chosen cell, 50/50" (hedge diagnostic).

## Bar (fixed now)

1. **Sign gate:** SPY's mean open-to-close return on dev fall days, after costs on the short side, must be negative
   (i.e. -1 x SPY makes money). If it fails: **NO**, stop, nothing else is judged.
2. A cell passes on dev (unconstrained case) if: total return after costs > 0, at most 2 losing calendar years out of
   9, and it beats the same cell on every session (the signal adds value).
3. Choice: among passing ETF cells (not -1 x SPY), the one with the highest dev total return; ties by smaller max
   drawdown. If none passes: **NO**, the held-out window is not run.
4. **Held-out (run once):** GO if the chosen cell's total return on 2025-01-02 to the last session is > 0 in the
   unconstrained case. The cash-account result is reported next to it.

Reported per cell: total, CAGR, per calendar year, max drawdown, number of trades, win rate, mean trade, trades ended
by stop vs close.

## Left to the builder

Script layout, cache format, how the 250-session history is fed to `in_market_fall` (one replay over the whole series
is fine if it reproduces the per-day call), table printing.
