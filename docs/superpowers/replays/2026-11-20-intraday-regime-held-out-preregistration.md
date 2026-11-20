# Intraday regime switch: held-out pre-registration (written before the run)

> Code not kept: the research branch, its scripts and its logs/caches were dropped on 2026-11-20; this record is what remains.

Decided with the user on 2026-11-20, after about 70 tries on the development window 2016-01-04 to 2024-06-28
(spike scripts on `feat/intraday-regime-switch`, `scripts/research/intraday_regime/`). This is the one look at the
held-out window; nothing is changed after it, whatever it shows.

- **Rule:** UPRO only (bullish -> UPRO, bearish or neutral -> cash). Detector `K v=5% k=1 trail=4%` of
  `scripts/research/intraday_regime/vol_band.py`: VIXY trigger 5% (exit UPRO when VIXY is up 2.5% on the day), band =
  1 x SPY's daily volatility over the 20 sessions before the day, 4% trailing stop on UPRO, signal at a 5-minute bar's
  close and trade at the next bar's open, 3 bps per side, cash-account T+1 settlement lock, positions held overnight.
- **Window:** 2024-07-01 to the last complete session downloaded. Bars from 2024-05-01 only warm up the indicators;
  trading starts flat on the first bar of 2024-07-01.
- **Bar (relaxed by the user from "positive every year"):** total return above SPY buy-and-hold over the same window,
  and at most one losing calendar year, no worse than -5.0% (2024 counts from July, the current year to date).
- **Reported:** total, per calendar year, max drawdown, trades by exit cause, SPY and UPRO buy-and-hold.

## Result (run once, 2026-11-20, after the section above was committed in c11a4f6): NO-GO

Window 2024-07-01 09:30 to 2026-11-19 15:55 (601 sessions), `scripts/research/intraday_regime/held_out.py`.

| | Total | 2024 (from July) | 2025 | 2026 (to Nov 19) | Max drawdown |
|---|---|---|---|---|---|
| Strategy (UPRO only) | **+22.0%** | +24.1% | +0.0% | -1.7% | 20.2% |
| SPY buy-and-hold | **+34.7%** | +8.1% | +17.7% | +5.8% | 20.8% |
| UPRO buy-and-hold | +55.6% | | | | |

- Beats SPY: **FAIL** (+22.0% against +34.7%). At most one losing year no worse than -5%: PASS (2026 -1.7%).
- 75 trades, 150 regime changes. Regime exits: 62 trades, mean +0.70%, compounded +47.8%. Trailing stops:
  13 trades, mean -1.44%, compounded -17.4% (2025: regime +6.3%, stops -5.9%; 2026: regime +12.0%, stops -12.3%).
- The development window's edge (+300% against SPY's +227%, drawdown 22% against about 34%) did not carry over:
  the strategy matched SPY's drawdown but earned about two-thirds of its return. The trailing stop, which helped in
  2016-2024, cost the most here. Nothing is changed after this look.
