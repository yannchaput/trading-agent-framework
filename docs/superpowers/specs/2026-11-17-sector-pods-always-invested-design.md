# sector_pods: always invested (PM weights set the mix, code sets the exposure)

## Intent and why

The 2-year backtest `logs/sector_pods/backtesting/2026-11-16_181406_backtesting` made 10.6% CAGR against SPY's 19.1%. Its book was
53% invested on average (56% in bullish weeks), the rest parked in SHV: the PM sizes each ETF on its own (median 3.5%, about 14
holdings) and the total is whatever that adds up to. Replayed on weekly total-return closes, the PM's own picks scaled to 98%
invested would have made 15.6% CAGR (max drawdown -16.6% weekly) instead of 10.2% (-6.7%). The agents' selection is worth
keeping (it matched equal-weight ETFs); the cash is the loss. The user chose: **always invested**.

## Goals

- Every completed week trades a book invested at `1 - cash_buffer` (98%), whatever total the PM submits.
- The PM's relative weights are kept: an ETF it gave twice the weight of another keeps twice the weight, unless a cap cuts it.
- The caps still hold on the traded book: 25% per ETF, 40% per pod, and keep-or-trim for unresearched ETFs.

## Non-goals

- No regime-based exposure, no defensive mode, no switch to turn this off (YAGNI; the user chose always invested).
- No change to the pods, the risk manager, the caps' values or the cash buffer.
- No stock-picking variant (a separate question, assessed in the same session and parked).

## Anti-goals

- A week abandoned or left half in SHV because of the scaling. If the caps make 98% unreachable (say the PM held only two
  pods: 2 x 40% = 80%), the rest stays in SHV with a log line; the week still trades.
- The PM's chosen mix silently replaced: the scaled book is logged and recorded next to what the PM submitted.

## Approaches

- **A. Code scales the book (chosen).** `Execute` scales the PM's weights to 98% and spreads what a cap cuts over the others.
  Deterministic, cannot undershoot, and is what the 15.6% replay measured.
- **B. The submit tool refuses a book under 95%.** Keeps the agent in control but needs it to do arithmetic it gets wrong
  (the note-length episode); extra turns, possible refusal on the last attempt, and it would need A as a fallback anyway.
- **C. Prompt only.** Cheapest, nothing enforces it; this model treats stated numbers loosely.

## Design

- **`fill_book(weights, state, params) -> dict[str, float]`** in `handoff.py`, pure, next to `check_weights`:
  - Unresearched ETFs (`_frozen`) keep their submitted weight. ETFs the PM gave 0 (or left out) stay out.
  - The other ETFs are scaled in proportion so the book totals `1 - cash_buffer`.
  - Water-filling: an ETF above `max_etf_weight` is clipped; a pod above `max_pod_weight` is cut back to the cap in proportion;
    what was cut is spread in proportion over the ETFs not yet at a cap; repeat until nothing is left over or no ETF can take more.
  - If no ETF can absorb the rest (every one at a cap, or the PM listed nothing scalable), the rest stays in SHV.
  - The result passes `check_weights` (asserted by the tests, and re-checked by `Execute`).
- **`Execute._trade`**: after `check_weights` accepts the PM's book, it trades `fill_book(...)` instead of the raw weights; logs
  one INFO line `book filled from 53.0% to 98.0% invested` (plus a note when a cap held it below 98%); the `reviews.jsonl`
  record gains `book` (the traded weights), next to `pm` (what the PM submitted). `_trades_due` compares the traded book.
- **PM prompt** (`prompts.py`): weights now set the mix, the book is always invested at 98% and scaled; drop "Build a small book
  only with clear risk-off evidence"; the caps are stated as applying to the scaled book. The validator keeps checking the
  submitted weights against the same caps (unchanged).
- **Docs**: `sector_pods/CLAUDE.md` (Architecture: caps, keep-or-trim, the record's `book`).

## Left to the builder

- Tolerance and iteration bound of the water-filling (the caps already use `_TOLERANCE`).
- Exact log wording and whether `outcome.invested_weight` needs a note in the prompt (it will read about 0.98 from now on).

## Evidence to check after the change

A sector_pods backtest on the same window: invested weight about 0.98 every week, no week left below it except when the
log says a cap held it, CAGR against the 15.6% replay (agents are not deterministic: one run is not proof).
