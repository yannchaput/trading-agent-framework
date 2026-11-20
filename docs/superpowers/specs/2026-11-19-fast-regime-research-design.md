# Faster bearish regime: research design

## Intent and why

The bear call spread (`docs/superpowers/replays/2026-11-19-bear-call-spread-research.md`) and the same signal on a SPY short or inverse ETFs all lost money, while the same trades timed in hindsight made money. The signal is late: today's regime (`core/regime.py`) turns bearish on average when SPY is already **-5.8%** below the swing high (`uv run regime scorecard`, 13 falls of 8%+ since 2016), and it raises 86 bearish days outside any fall. Two of its four conditions are lagging by construction (SPY below its 50-day average; SPY 5% under its 252-session high), and it reads closes strictly before the session (one more day).

The user wants a signal that turns bearish **in the first days of a fall** (fast detection, not predicting the peak), before investigating inverse ETFs, because any instrument needs that timing.

## Goals, non-goals, anti-goals

- **Goal:** a research-only faster regime that switches on at a mean of **-3% or better** into a fall (scorecard `mean_on_fall >= -0.03`), **with no more false-alarm bearish days than today's regime** on the same window.
- **Non-goals:** changing the production regime (`strategy.regime`, `regime_by_session`, `uv run regime scorecard|recompute` keep their exact output); trading tests (spread, short, inverse ETF) — those come in a later design, only if this one succeeds; news sentiment and the put/call ratio (phase 2, only if the cheap inputs fail); predicting the peak.
- **Anti-goals:** a "fast" signal that buys speed with more false alarms (that is what lost money); thresholds tuned on the data; looking at the held-out period before the final pick.

## Constraints

- Every input must have daily history long enough for the full replay that the code can download. Checked on Yahoo on 2026-11-19, first dates with no gaps before 2016: `^VIX9D` 2011-01-03, `^VVIX` 2007-01-03, `HYG` 2007-04-11, `IEF` 2002-07-30, `RSP` 2003-05-01 (`^VIX3M` 2006-07-17). So every input supports the same 503-session window (`RegimeParameters.min_bars`) as today's regime from 2016-01-04: the variant is replayed exactly like `regime_by_session` (stateless per session, fresh state machine over the last 252 sessions), and the download starts early enough to cover it. `^CPC` / `^PCALL` are not on Yahoo, so the put/call ratio is out.
- **Development window: sessions 2016-01-04 to 2024-06-28** (10 of the 13 falls). **Held-out window: 2024-07-01 to the last download** (3 falls: Jul-Aug 2024, Feb-Apr 2025, Jan-Mar 2026). The held-out window is scored once, at the end, for the final variant only.
- The suite never touches the network (fakes, as in `tests/core/`). Pure functions where the existing regime code is pure.
- Thresholds below are fixed by this document, from common practice, before any run. None is tuned.

## Approaches (decided with the user, 2026-11-19)

- **A. Screen each input alone, then B. a faster regime from the winners (chosen).** A is cheap and says which inputs carry any speed; B turns the winners into one regime every instrument can use.
- C. A rebuilt CNN Fear & Greed (5 of its 7 parts from Yahoo): rejected for now; it averages slow and fast parts and is built to read mood, not to catch a fall early.

## Phase A: the screen

Each input becomes a raw daily label: -1 (bearish) on a session where its condition holds, else 0. Like today's regime, the label of session `d` uses closes **strictly before** `d` (the one exception is input 1). Each label is scored with `core/regime_scorecard.score` on the development window. The screen also scores, as reference rows, today's regime and each of its four conditions as a raw label (to see which existing conditions are slow).

| # | Input | Bearish when (fixed) |
|---|---|---|
| 1 | Same-day closes | today's regime, computed from closes **up to and including** `d` (one day sooner; a live version would read a quote near the close) |
| 2 | VIX9D over VIX | `VIX9D / VIX >= 1.0` |
| 3 | VIX jump | VIX up `>= 20%` over 3 sessions (`VIX[t] / VIX[t-3] - 1 >= 0.20`) |
| 4 | VVIX level | `VVIX >= 110` |
| 5 | Credit stress | `HYG / IEF` down `>= 1.5%` over 10 sessions |
| 6 | Short drawdown | SPY `>= 3%` under its 20-session high close |
| 7 | Breadth proxy | `RSP / SPY` down `>= 1.0%` over 10 sessions |

SKEW is not screened (known weak).

**Selection rule (fixed):** inputs 2-7 are compared against the two conditions B replaces (below the 50-day average, 5% under the high, scored as raw labels on the same window). An input **qualifies** if its `false_bear_days` is at most the larger of those two conditions' `false_bear_days`. The **two qualifying inputs with the best `mean_on_fall`** (closest to zero) win; ties go to fewer false days. Input 1 is judged separately: B uses same-day closes if row 1 beats today's regime on `mean_on_fall` with no more `false_bear_days`.

If no input 2-7 qualifies **and** input 1 does not win either, the work stops there with a NO result (recorded), and the next step would be phase 2 (news), a new design. *(My call, from the builder check:)* if input 1 wins and no input 2-7 qualifies, phase B still runs, as today's four conditions on same-day closes (that is itself a faster regime).

## Phase B: the faster regime

The same state machine as `regime_path` (score from four conditions, bearish at score >= 2, ends at an 8% bounce off the episode low or when the score drops under 2, the re-entry block, 3-session hold for neutral/bullish), with the two lagging conditions (`below_sma`, `off_high`) **replaced by the two winners** of phase A (one winner: replace `off_high` only). `inverted` and `vix_spike` stay. *(My call, from the builder check:)* if input 1 won, **every** condition of the variant reads same-day closes (one timing, not a mix); otherwise every condition reads closes strictly before the session. The winners were screened on prior-day closes; reading them one day sooner can only make them earlier, and the development-window scorecard of the variant is the check that their false alarms did not grow. A live version of a same-day variant needs a quote near the close for every input it uses (VIX, VIX3M, SPY and the winners).

- **Success on the development window:** `mean_on_fall >= -0.03` and `false_bear_days <=` today's regime's.
- If it succeeds, score **once** on the held-out window, next to today's regime on the same window. The held-out result is reported as it comes (3 falls: weak evidence); passing it there (better `mean_on_fall` and no more `false_bear_days` than today's regime on the held-out window) is the gate for a later trading design.
- If it fails on the development window: report it; no threshold changes, no second combination.

## The how

- **Data:** a research download of `^VIX9D`, `^VVIX`, `HYG`, `IEF`, `RSP` aligned on SPY's sessions (forward fill up to 5 sessions, the existing rule), **separate from** the production `regime_data.SYMBOLS` download, so the production regime's inputs and cache do not change.
- **Code:** a pure module for the input labels and the variant regime; the variant state machine must not change the production readings: either a shared core that `regime_path` also uses (existing `regime_path` tests and the scorecard's `replay disagreements: 0` must stay green and identical) or a separate function. Builder's call; a shared core is preferred over a copy if it keeps `regime_path`'s behaviour byte-for-byte.
- **Command:** two new subcommands of `uv run regime` (read-only, Yahoo): `screen` (phase A table on the development window, reference rows first, winners marked by the rule above) and `fast` (phase B on the development window; `--held-out` scores the held-out window, prints a warning that it is the one-time look). `scorecard` and `recompute` are untouched.
- **Output:** printed tables (per row: `mean_on_fall`, `false_bear_days`, `recall`, `changes`, episodes caught); the result is recorded in `docs/superpowers/replays/` and in `core/CLAUDE.md` (one line pointing to it).

## Left to the builder

Module and function names, the shared-core versus separate-function choice (within the constraint above), table layout, and how the CLI passes the winners from `screen` to `fast` (flags naming the conditions are enough; no file handoff needed).
