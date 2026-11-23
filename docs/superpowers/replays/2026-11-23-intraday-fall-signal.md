# Intraday fall signal (VIXY / SPY, 10-minute loop): record and result

Spec: `docs/superpowers/specs/2026-11-23-intraday-fall-signal-spike-design.md`; plan: `docs/superpowers/plans/2026-11-23-intraday-fall-signal-spike.md`. Script: `scripts/regime/intraday_fall_signal_spike.py` (modes `check`, `dev`, `test`).

## Question and fixed bar

Does a signal read every 10 minutes (VIXY against D-1's close at 09:40 and against an hour ago later in the day, above D-1's VWAP and its SMA50, optional VIXY volume 1.5x the same-clock-time D-1..D-5 average, SPY down 0.5% on the day and down over the hour) pick moments when an inverse S&P ETF makes money, with a volatility-sized trailing stop, a fade exit, a 3-in-5 day-trade cap, a daily loss limit and SHV overnight? Grid fixed before any run: 16 cells (strength loose/strict x volume off/on x SDS/SPXU x stop 3/5 ATR). Dev window 2016-01-04 to 2024-06-28; held-out 2024-07-01 onward, scored once. Bar, first condition: **signal gate**, per signal setting, a -1 x SPY trade from the session's first signal to 15:55 (no stop, no limits, 3 bps per side) must have a positive mean; if every setting fails: NO and nothing else is judged.

## Result (dev, run once 2026-11-23): NO at the signal gate

2,136 sessions, 2,131 with signal values; SPY buy and hold +210.7% (max drawdown -33.8%).

| Signal setting | Gate mean per trade | |
|---|---|---|
| loose, no volume | -0.112% | fails |
| loose, volume | -0.090% | fails |
| strict, no volume | -0.107% | fails |
| strict, volume | -0.089% | fails |

All four fail, so no cell was simulated and **the held-out window (2024-07 onward) was not run and stays unused**.

Descriptive numbers behind it (SPY, entry at the first signal to 15:55, no costs; scratch script, not in the repo):
- The signal is rare: **240 sessions (loose) and 167 (strict) of 2,131**, about 20-28 a year, mostly later in the day (202 and 146 of them are hour-trigger signals; only 38 and 21 are 09:40 gap signals).
- After the first signal SPY goes **up** a little: mean **+0.052%** (loose) and **+0.047%** (strict), up on 52% and 49% of the days; the next 60 minutes average +0.09% to +0.12%. A short loses even before costs.
- By year the mean is positive in 2016 (+0.33%), 2020-2024 and negative only in 2017-2019 (2018: -0.16% on 53 days). No year has enough signals to trust.
- Volume (the new input) did not help: the volume settings are slightly better than the no-volume ones (-0.090% against -0.112%) but still negative. VIXY volume is split-adjusted by Alpaca, so before about 2019 it is a few thousand shares a day and quantised (5% of minutes at zero in 2016); the ratio is noisier there.

## A modelling artifact to know about (does not affect the verdict)

The report line "SHV only (no trades, SHV overnight): **-26.2%**" is not a finding. As specified, the account buys SHV at every close and sells it at the next open at 1 bp each side: about 2 bps a night, **-35% over 2,136 nights**, against roughly +13% of T-bill carry. Holding SHV continuously, and selling it only when a trade starts, would cost almost nothing. The gate does not involve SHV, so the NO stands; any future version of this account should model SHV as held until needed.

## Reading

Same family of result as the two earlier records: the daily `in_market_fall` signal marked volatile days without direction (`2026-11-23-market-fall-intraday.md`), and this intraday version of the stress signal also does not point down. After VIXY jumps and SPY is weak, SPY has on average bounced slightly into the close: the intraday move looks more like short-term mean reversion than continuation. That is a description of 2016-2024, from 240 signal days, not a tested long strategy; the user's goal here is a bearish bet, and nothing here supports one.

## Not tested (open)

- The signal with the opposite side (buy the dip after the stress signal): different goal from the fall account; would need its own pre-registered bar and the 2024-07 window is not clean for it either way.
- Other signal families the spec did not include (SPY's own breadth or leaders-crack inputs were tested daily before, see the earlier records); a 5-minute loop or an outer `in_market_fall` gate.
