# Spike: bear put spreads as insurance while VIX alone is defensive

Date: 2026-11-22. Status: **design, not run**. Type: research spike (one throwaway script, no production code).
Branch: `research/vix-exit-rule`.

## Intent

The user's goal: run bull_bear normally, and when a fall is under way switch to a defensive posture that keeps part of
the book invested (to catch the rebound) and adds SPY bear put spreads as coverage. VIX alone is the entry and exit
(fast: median -1.6% from the high on dev; 3 of 3 holdout falls at -2.0% to -2.6%). Defensive mixes so far lose return
on false switches and after troughs; the best is 50% SPY + 50% SHV (+180.4% on dev against buy and hold +240.9%).
Inverse ETFs made it far worse. A put spread's loss is fixed on the day it is bought, so it should not fight the
rebound. SPY stands in for bull_bear (as in every earlier spike).

Prior art: `strategies/bear_put_spread` (removed, `c932e39`): $5-wide SPY spreads on slower triggers (old regime,
t1-t3) were all NO-GO; the call-spread record says the structure made money with hindsight entries, the signal lost it.
VIX alone was never tried as the trigger.

## Goals, non-goals, anti-goals

- Goal: for each defensive mix, does adding spreads raise total return over the same mix without them?
- Non-goals: no strategy integration, no broker or options entities, no real option data, no production code.
- Anti-goals: changing a strike, expiry, exit or budget after seeing `dev`; running 2008-2015 more than once or for a
  variant that failed `dev`; any use of 2025-2026 (spent).

## Approaches considered

- **Spreads on the VIX-alone switch, on top of a partial mix (user's choice, option b).**
- Exit to SHV + spreads only (option a): drops the rebound the user wants to keep through bull_bear.
- Puts bought only after confirmation: later and dearer. A rolling put calendar held all the time: pays premium in
  calm markets too.

## Rules (fixed before running)

Decisions are made before the open of session t from closes up to t-1 and executed at the open of t.

- **Defensive flag**: VIX alone, unchanged (`run_switch(None, entry_trigger, calm)` as in the earlier spikes).
- **Mixes** (SPY weight while defensive): w = 1.0, 0.75, 0.5 (rest SHV). Normal: 100% SPY. User's list; "75% SPY + 25%
  SPY" read as 75% SPY + 25% SHV (my call).
- **Spread**: long put at `round(0.98 x SPY close (t-1))`, short put at `round(0.90 x SPY close (t-1))` ($1 strikes);
  expiry the nearest third Friday at least 30 calendar days away.
- **Budget per purchase** b = 0 (no spreads), 0.5%, 1%, 2% of equity at the close of t-1: contracts =
  floor(b x equity / (debit x 100 + open cost per contract)); 0 contracts = no spread.
- **Each morning, in this order**:
  1. A spread is open: close it if the flag is off (episode over), or if its value at the close of t-1 is at least 80%
     of its width (**target**: no new spread until the next episode), or if 7 calendar days or fewer remain to expiry
     (**roll**).
  2. Defensive, no spread, and no target hit in this episode: open one (a roll reopens the same morning).
  3. The rest of the equity (SPY + SHV + option cash) is set to w / 1 - w; 10 bp per full change of w, charged as in
     `portfolio_exposure` (`vix_staged_spike.py`); weights reset to target each morning (drift ignored).
- **Pricing** (the removed research's model): Black-Scholes put, rate 0, no dividend, volatility per strike
  `VIX / 100 x vol_scale + 0.8 x max(0, ln(spot / strike))`. At the open: SPY open and the VIX close of t-1; marks at
  the close: SPY and VIX closes of t. Spread value clamped to [0, width]. vol_scale 1.0 for the verdict; 0.8 (the
  level the call-spread research measured against real Alpaca quotes, cheaper puts) shown as information.
- **Costs** (Tastytrade, per contract per leg): $0.05 half spread per share on every open and close, $1 commission on
  opens, $0.10 regulatory on opens and closes.
- **Account**: $100,000 start; SPY and SHV adjusted prices (as before).

## Evaluation

- Per variant (3 mixes x 4 budgets): total return, CAGR, max DD, Sharpe; spreads opened per year, premium paid per
  year (% of equity), spread P&L split into trades opened inside a fall window (20 sessions before a high to its
  trough, falls >= 8%) and outside; total at vol_scale 0.8.
- `check`: b = 0 reproduces `portfolio_exposure` exactly on synthetic data; synthetic spread mechanics (open, roll,
  target, close on episode end). `dev` must show w = 1.0, b = 0 at +240.9% and w = 0.5, b = 0 at +180.4%.
- **Development pass bar** (dev 2016-01-04 to 2024-12-31, vol_scale 1.0): total return above the same mix with b = 0,
  and max DD no worse than buy and hold's (-33.7%). Chosen: the highest total among passing variants. None pass:
  verdict NO, the test window stays unused.
- **Test**: 2008-01-02 to 2015-12-31, once, the chosen variant and its mix at b = 0, nothing changed; same bar.

## Constraints

- Code: `scripts/regime/vix_put_spread_spike.py`, modes `check`, `dev` (and `test` only if a variant passes); it
  imports from `vix_staged_spike.py`. No new dependency. Run with `uv run --with jumpmodels`.

## Left to the builder

- Output layout; how the ledger is coded.
