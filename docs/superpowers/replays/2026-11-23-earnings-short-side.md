# Earnings short side (put spreads after a bad reaction): replay record

Design: `docs/superpowers/specs/2026-11-23-earnings-short-side-spike-design.md`. Script:
`scripts/earnings/short_side_spike.py` (`dev`, 2026-11-23). Raw output: `logs/earnings_short_side_dev.txt`
(git-ignored).

## Question

After a stock drops hard on its earnings release (abnormal return <= -D against SPY, closed in the lower half of its
range, volume >= 2x its 20-session average, price >= $10, dollar volume >= $20M), does it keep falling enough, after
costs, for a debit put spread to make money? All year, no market gate.

## The bar (fixed before the run)

Gate: the mean -1x trade (gain capped at 10%, 80 bps per side) must be positive in a cell; all 6 cells out is a NO and
nothing else is judged. Then the account bar, the held-out window and the real-spread step.

## Dev result, 2016-01-04 to 2023-12-29

Data: 1,983 universe symbols with bars, 4 without a SEC CIK, 32,833 item 2.02 events, 32,598 kept after the universe
and feature filters; 2,012 sessions. Buy and hold over the window: SHV +12.4%, SPY +172.3%.

| Cell | Qualifying trades | Mean trade, 80 bps (the bar) | Mean trade, 10 bps (info) |
|---|---|---|---|
| D5% H5 | 3,389 | -1.88% | -0.49% |
| D5% H10 | 3,389 | -1.98% | -0.60% |
| D5% H20 | 3,389 | -3.12% | -1.75% |
| D8% H5 | 1,908 | -1.94% | -0.55% |
| D8% H10 | 1,908 | -2.02% | -0.63% |
| D8% H20 | 1,908 | -3.32% | -1.95% |

The mean is negative in every calendar year of every cell (best year: -0.9%, D8% H10 in 2021; worst: -6.5%, D8%
H20 in 2016). The longer the hold, the worse the mean: the stock does not keep falling after the reaction day, it
drifts up with the market.

## Verdict: NO (sign gate)

Every cell fails the gate at the put spread's cost, and every cell also fails at the 10 bps rows, which are still
capped at a 10% gain. The account replay, the held-out window (2024 onward) and the real option prices were not run:
the held-out window stays unused.

## Reviewer's read-only measurement (information only, taken after the bar was fixed)

The final reviewer recomputed the dev means at 0 bps on the same events. They do not change the verdict.

| Cell | Uncapped short | Capped short | Market-adjusted (SPY minus stock) |
|---|---|---|---|
| D5% H5 | -0.12% | -0.29% | +0.07% |
| D5% H10 | +0.12% | -0.40% | +0.44% |
| D5% H20 | -0.41% | -1.55% | +0.28% |
| D8% H5 | -0.12% | -0.35% | +0.11% |
| D8% H10 | +0.21% | -0.43% | +0.53% |
| D8% H20 | -0.46% | -1.75% | +0.37% |

- The gross edge of a plain short is about zero; uncapped at 10 bps per side D8% H10 is about break-even (+0.01%).
- The 10% cap takes 0.2 to 1.3 points. The spread's round trip (about 1.6% of the stock's price) then decides the
  result: no cell comes close to covering it.
- After a bad reaction the stocks do lag SPY by about 0.4-0.5% over 10 sessions: a small post-earnings drift exists
  market-adjusted. A market-neutral (beta-hedged) short is a different bet that this spike did not test.

## Known gaps

- Delisted stocks are absent from Yahoo and so from the universe lists: bankruptcies after a drop would have been
  winning shorts, takeovers losing ones. The direction of the bias is unknown; it is unlikely to turn -1.9% into a
  positive mean.
- The ticker -> CIK map is today's, so renamed or delisted tickers may lose their events (4 of 1,987 symbols had no
  CIK; a renamed company's old ticker may map to another company).
- The gain cap at 10% (the spread's short leg) removes crash gains from the gate, which makes it harder than a plain
  short (see the reviewer's table above for the uncapped means).
- Two item 2.02 filings of one company on one reaction day (an original and an amendment) are not merged: 18 of 3,389
  (D5%) and 10 of 1,908 (D8%) gate trades are doubled. The account replay skips them ("already held"). It cannot
  flip the sign.
- Step 2 (option prices) was not run. A data probe showed Alpaca does serve expired put contracts and their daily
  bars (AAPL, MSFT, NVDA, February-March 2024), so a later spike on single stocks could use it.
