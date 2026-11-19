# Bear call spread: hold through bounces -- design (short)

Follows `2026-11-18-bear-call-spread-research-design.md` (the real-regime run was NO-GO, the hindsight run GO).

## Intent

Get the real strategy closer to the hindsight result without predicting falls better. Reading the default run's
`trades.csv` (a diagnosis, not a tested cause): trades opened inside a 10%+ fall made +$346; trades opened outside one
lost -$2,392, and the loss sits in regime exits (50 of 77 exits, -$3,935 in total; 14 of them mid-fall at -$1,374,
36 on false alarms at -$2,562). The regime flips off bearish on a bounce and the spread is bought back at a loss.

## Goals, non-goals

- **Goal:** test, once, whether removing the regime exit turns the verdict to GO.
- **Non-goals:** a new entry signal, any change to strikes, expiry, sizing, the target, the stop, the time exit, the
  pricing or the costs. No parameter search.
- **Anti-goal:** a GO found by trying variants until one passes.

## Approaches (decided with the user, 2026-11-19)

- **A. No regime exit** (a spread closes only on time, stop or target). **The decision test.** One change, nothing
  to tune. Risk: a false alarm followed by a rally now runs to the stop (2x the credit) instead of a small regime exit.
- **B. Exit only when the regime turns bullish (+1)**, ignoring neutral (0). **Run once alongside A, for information
  only**: it does not decide GO.
- C. Exit after N non-bearish sessions, or only past a loss of X: rejected, it adds a number we would end up tuning.

No other variant is run.

## The how

- `SpreadParameters` gets `regime_exit`: `"not_bearish"` (today's rule, the default, so existing runs and tests are
  unchanged), `"bullish"` (B: exit when regime == 1) or `"never"` (A). `exit_reason` keeps its order (time, stop,
  target, then the regime check per `regime_exit`).
- `bear-call-research` gets `--regime-exit {not_bearish,bullish,never}` (default `not_bearish`); the summary line
  names the rule used.
- Entries are unchanged: `should_open` still opens on -1 when no spread is open, so with A a spread stays open
  through a bounce and the next entry waits for it to close.

## Verdict and reporting

- **GO bar unchanged** (fixed before running): closed P&L > 0, at most 2 losing calendar years, drawdown <= 35%,
  $5,000 start, 2016-2026, default pricing and costs. Also run A at `--call-vol-ratio 0.70` as the stress run, as before.
- **2023-2026 reported separately** from `years.csv` (the years we never designed against; only one 10%+ fall, 2025,
  so it is weak evidence and does not decide GO on its own).
- Results go in `bear_call_spread/CLAUDE.md` next to the earlier runs, including the exit-reason breakdown and the
  inside-a-fall / outside-a-fall split of A.

## Left to the builder

Test layout, the exact report wording, and the stored value names if `not_bearish` reads badly in the code.
