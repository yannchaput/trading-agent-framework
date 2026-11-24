# Canary switch on bull_bear (Keller BAA): research spike design

## Intent and why

bull_bear loses money when the market falls. Every attempt to bet on the falls themselves failed (inverse ETFs, the
intraday VIXY/SPY signal, put spreads on the VIX switch, the earnings short side). The user now wants **one account
that switches**: it runs bull_bear while the regime is bullish, and in a bearish regime it holds something that makes
money, even a little, until the regime turns bullish again.

The switch is Keller's "canary" from **Bold Asset Allocation** (BAA, Keller 2022, SSRN 4166845). A small set of
ETFs that are not traded, the canaries, decides the mode once a month by their fast momentum. In bear mode the
account holds the paper's defensive basket, the strongest of bonds, inflation-linked bonds, commodities and T-bills.
In bull mode it holds **bull_bear** (the user's choice; the paper's offensive basket is not used). The canary is a
new signal for this project: every earlier switch used SPY and VIX.

Path if it passes: the gate built into bull_bear and LLM backtests on both windows (approach B below), then paper.

## Goals

- One script, `scripts/regime/canary_switch_spike.py` (`uv run python`), modes `check` (assert self-check on
  synthetic data, no network) and `run`, in the style of `scripts/regime/vix_staged_spike.py`.
- Replay the canary switch on bull_bear's saved daily equity on its two point-in-time windows, and the same switch
  on SPY over 2008-2026 as a check of the rule itself.
- A pre-registered GO / NO-GO and a record in `docs/superpowers/replays/` at the end.

## Non-goals

- No production code, nothing under `src/` or `tests/`; no LLM run; no change to bull_bear.
- No offensive basket from the paper (bull mode is bull_bear, or SPY in the sanity check).
- No parameter search: the canary, the momentum, the defensive basket and its ranking are the paper's, as written.

## Anti-goals

- **Tuning the canary to bull_bear's two windows.** These curves are already seen and cover only two index falls; any
  rule picked on them would be fitted. The paper's rule is scored as is, once.
- **A flattering switch:** the decision uses only month-end closes known at the time, it acts one session later,
  and every switch pays costs.
- Network access in the self-check.

## Data

- **bull_bear daily equity** (`portfolio_value`, index = session close, tz America/New_York) of the two point-in-time
  runs, read from the main checkout's git-ignored `logs/bull_bear/backtesting/`:
  - `2026-11-05_012455_backtesting`: 2021-09-20 to 2026-10-05, +468% (SPY +91%), max drawdown -25.2%.
  - `2026-11-09_224910_backtesting` (COVID): 2020-01-02 to 2021-05-26, +137% (SPY +32%), max drawdown -37.7%.
- **ETFs** from Yahoo, daily, total return (`auto_adjust=True`) and opens not needed: canaries **SPY, VWO, VEA,
  BND**; defensive **TIP, DBC, BIL, IEF, TLT, LQD, BND**; reference **SHV**. History from 2007 (VEA from July 2007,
  BND April 2007, BIL May 2007); the first month with every score defined is the 13th month-end after the latest
  start, so the SPY check runs from about mid-2008.
- Known biases, reported in the record: bull_bear's curves are flattered by the universe and the window end (see the
  2Y flaw check of 2026-11-01); they hold only two index falls (COVID 2020 and 2022, plus the April 2025 dip).

## The canary rule (the paper's, fixed)

- **Fast momentum (13612W)** of a canary at a month-end close: `(12 r1 + 4 r3 + 2 r6 + 1 r12) / 19`, `rN` the total
  return over the last N month-ends.
- **Mode:** bull when all four canaries (SPY, VWO, VEA, BND) have a fast momentum above 0; bear when any is at or
  below 0. Decided at each month's last close.
- **Defensive basket (bear mode):** rank TIP, DBC, BIL, IEF, TLT, LQD, BND by slow relative momentum
  `p0 / mean(p0..p12) - 1` (the last close against the average of the last 13 month-end closes, itself included);
  hold the top 3 at 1/3 each; a pick whose score is below BIL's is replaced by BIL (BIL may then be held more than
  once, summing its weights). Recomputed every month-end while in bear mode.

## Replay

- **Timing:** the mode and the basket decided at month-end close `t` take effect from the close of the next session
  `t+1` (a one-session delay standing in for "trade at the next open"; bull_bear's curve has closes only). The
  account earns the old mode's return from `t` to `t+1`.
- **Bull mode:** the account's daily return is bull_bear's (`portfolio_value` close to close). **Bear mode:** the
  basket's daily return (weights fixed at the switch or month-end, drifting with prices until the next month-end).
- **Costs:** each switch pays **10 bps** on the bull_bear side (selling or buying a stock book) plus **5 bps** on the
  ETF side; a month-end change inside the defensive basket pays 5 bps on the turnover.
- **Account:** starts at each window's first session with bull_bear's own starting value; the canary's first
  decision is the latest month-end before the window start (enough ETF history exists). Money never sits idle: bull
  mode is fully bull_bear, bear mode fully the basket.

## Grid

One cell for the bar: **the paper's monthly check**. One information row, not judged: the same rule checked **every
Tuesday close** (bull_bear's rebalance day), to show what a faster check would have changed. On a Tuesday, `p0` is
that Tuesday's close and the lookbacks are the last 1, 3, 6, 12 (and 13 for the slow score) month-end closes before
it.

## Reference rows (not candidates)

- bull_bear alone; **70% bull_bear / 30% SHV** rebalanced monthly (the one lever of the 2026-11-12 study that cut
  drawdown); SPY buy and hold.
- **Canary on SPY, 2008-2026:** bull mode SPY, bear mode the basket, against SPY and against 60% SPY / 40% IEF.
- Per window: months in bear mode, the switch dates around each fall (2020-02/03, 2022, 2025-04, 2026-06/07), and
  the account's and bull_bear's return over each bear-mode stretch.

## Bar (fixed now, the user's choice (b) with 85%)

On **each** of the two bull_bear windows, the monthly cell must have:

1. a **max drawdown smaller** (less negative) than bull_bear alone, **and**
2. a **total gain at least 85% of bull_bear's** (gain = final / start - 1: 5Y at least +397.8%, COVID at least
   +116.5%).

And, on the SPY check 2008-2026:

3. the **defensive basket's compounded return over all bear-mode months is at least 0** (the user's "make money,
   even a little, in a bear regime").

GO needs 1 and 2 on both windows and 3. Anything else is NO. There is no held-out window: the bull_bear curves are
seen and the rule is not fitted to them; the paper (July 2022) leaves 2022-08 to 2026-11 out of its own sample,
reported separately on the SPY check.

## Constraints

- Python 3.14, `uv`, no new dependency; read-only (Yahoo); writes only caches under the main checkout's
  `cache/canary_switch/` and the record.
- Built in the worktree `.worktrees/bearish-regime` (branch `research/bearish-regime`). The git-ignored inputs
  (`logs/`, `cache/`) exist only in the main checkout: the script resolves them from the main checkout's root
  (`git rev-parse --path-format=absolute --git-common-dir` -> its parent), as `scripts/earnings/short_side_spike.py`
  does.
- Lint: `ruff check` and `ruff format --check` clean on the script and the Markdown.
- The self-check covers at least: the 13612W formula, the mode rule (a zero counts as bear), the slow momentum and
  the top-3 with BIL replacement (including BIL held twice), the one-session delay, the switch costs, the bar's
  three tests.

## Approaches (decided with the user, 2026-11-24)

- **A. Overlay on bull_bear's saved daily equity (chosen).** Cheap (no LLM), the method of the 2026-11-12 lever
  study. Limit: on switching back in, it assumes bull_bear's current book can be bought at once.
- B. The gate built into bull_bear and LLM backtests: real switch-in trades, but days of GPU and a rerun noise of
  about 25 points between identical 2Y runs, so one run per window cannot judge it. The step after a GO.
- C. Like A with the books rebuilt from `reviews.jsonl` (switch-in waits for the next Tuesday): more faithful, but
  the rebuilt books already drift from the real runs (+438% vs +468%).

## Decided with the user

The account switches between bull_bear (bull mode) and the BAA defensive basket (bear mode) on Keller's canary;
approach A; the bar is (b): smaller max drawdown on both windows and at least 85% of bull_bear's gain.

## My calls (correctable)

The paper's monthly check as the only judged cell, Tuesday checks as information; condition 3 (defensive basket at
least 0 over the bear months of 2008-2026) as the test of "make money in a bear regime"; one-session delay; 10 bps
per switch on the stock side and 5 bps on the ETF side; Yahoo total-return closes everywhere, SPY included (bull_bear's
curve already includes its dividends); the Tuesday row's lookbacks counted in month-ends before the Tuesday; the
canary list SPY, VWO, VEA, BND (the paper's, not the EFA/EEM/AGG variant); the 70/30 SHV mix as the reference to beat
on drawdown.

## Left to the builder

Script layout, the month-end calendar (last session of each month in SPY's dates), the table printing, the self-check's
synthetic data.
