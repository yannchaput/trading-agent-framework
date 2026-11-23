# Market-fall intraday bet (inverse ETF open to close): record and result

Spec: `docs/superpowers/specs/2026-11-23-market-fall-intraday-spike-design.md`; plan: `docs/superpowers/plans/2026-11-23-market-fall-intraday-spike.md`. Script: `scripts/regime/fall_intraday_spike.py` (modes `check`, `dev`, `test`), kept on the branch `research/market-fall-intraday`.

## Question and fixed bar

On days when `in_market_fall` is true (decided from closes up to the day before), does buying an inverse S&P ETF (SH, SDS, SPXU) at the open and selling at the close make money, with an optional trailing stop (none, 2%, 4%)? Fake account 5,000, fractional shares, 3 bps per side, Alpaca SIP minute bars, dev window 2016-01-04 to 2024-12-31. Bar, fixed before any run: **(1) sign gate**: a -1 x SPY open-to-close trade on fall days must make money after costs, else NO and nothing else is judged; (2) a passing cell has total > 0, at most 2 losing calendar years and beats the same cell on every session; (3) the held-out window 2025-01-02 to the last session is scored once on the chosen cell.

## Result (dev, run once 2026-11-23): NO at the sign gate

2,264 sessions, 898 fall days (40%). SPY buy and hold +236.8% (max drawdown -33.8%).

- **Sign gate FAILS:** -1 x SPY held open to close on fall days makes **-0.066% a day after costs**. Nothing else was judged and **the held-out window (2025-2026) was not run and stays unused**.
- Why, in plain numbers (descriptive, SPY open to close, no costs): fall days mean **+0.007%**, median +0.064%, up 54.2% of the time; non-fall days mean +0.031%, up 54.8%. The 6 bps round trip is the whole loss.
- The signal marks **volatile** days, not **down** days: fall days move twice as much (standard deviation 1.11% against 0.57%) with the same small upward drift. A short has no direction to ride. By year the fall-day mean was negative only in 2018 (-0.17%) and about zero in 2022-2024.
- SH has no Alpaca minute bars before 2016-06-27; its earlier fall days would have been counted as missing (not reached).

## Reading

Same lesson as the bear call spread record: the structure (here, the instrument and the intraday hold) is not the problem, the signal is. `in_market_fall` was built to move bull_bear out early, which it does at a median 1.6% below the SPY high; it says nothing about the *next day's* direction, and a day-trade bet needs exactly that. Leverage (SDS, SPXU) would multiply the same zero edge and the decay and costs.

## Not tested (open, no held-out window needed to ask them)

- The overnight leg (close to next open) on fall days: gaps are where fall moves and rebounds happen, and open-to-close excludes them. Cheap to measure, but a bet held overnight is a different strategy (and the earlier tests of holding inverse funds through falls were NO).
- Fall *onset* days only (the first 1-3 sessions of a fall) instead of every fall day: 40% of sessions are signalled and the signal stays on a median 32 sessions after the trough, so most signalled days are not onsets. This splits the sample, so it must be declared as exploration on dev.
