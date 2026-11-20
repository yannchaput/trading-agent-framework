# Leveraged ETF switch rules (UPRO, TQQQ), 1993-2026: record and result

> Code not kept: the research script (`scripts/research/leveraged_switch.py`) and its output
> (`logs/research/leveraged_switch/report.txt`, `wealth.csv`) were not committed; this record is what remains.

Spec: `docs/superpowers/specs/2026-11-20-leveraged-etf-switch-design.md`. Plan:
`docs/superpowers/plans/2026-11-20-leveraged-etf-switch.md`. Run once on 2026-11-20; nothing was changed after it.

## Question

Which rule for how much 3x to hold (UPRO on the S&P 500, TQQQ on the Nasdaq-100) hurts least, with T-bills instead
of an inverse ETF when it steps out: R2 graded regime (bullish 3x, neutral 1x, bearish T-bills), R3 volatility-scaled
(3x weight `min(1, 0.30 / (3 x 20-session volatility))`), R4 200-day trend (3x above the 200-session average, else
T-bills)? Against B1 3x buy-and-hold, B2 1x buy-and-hold, B3 1/3 3x + 2/3 1x rebalanced monthly. **Bar:** Calmar
above B3's on the full window and on both sub-windows, and a worst calendar year better than B1's.

## Method (as in the spec)

Synthetic 3x from `^SP500TR` and `^NDX` daily returns, `3 r - 2 rf - ER / 252` (ER 0.91% and 0.84%, `rf` from
`^IRX`); one full session of lag (a signal from the close of t trades at the close of t+1); 10 bps per unit traded;
R3 trades only when a weight is 10 points or more off. R2 runs `core/regime.py` (`regime_by_session`, default
parameters) on `^GSPC`, `^VIX`, `^VIX3M`; before VIX3M exists (2006-07-17) its term-structure condition is always
false. Window 1993-01-04 to 2026-11-20; sub-windows 1993-2015 and 2016-2026, each started fresh. Rebound: median
return over the 63 sessions after each 15%+ fall's trough in the underlying, next to B1's.

Validation of the synthetic 3x against the real ETFs (reported, not adjusted):

```
UPRO: 2009-06-26 to 2026-11-20, CAGR real +29.8% vs synthetic +31.6%, daily correlation 0.9981
TQQQ: 2010-02-12 to 2026-11-20, CAGR real +37.2% vs synthetic +35.2%, daily correlation 0.9986
```

## Result (as printed)

```
UPRO full: 1993-01-04 to 2026-11-20
rule          multiple    CAGR  max DD Calmar worst yr lost yrs 3x share changes    rebound rule/B1 (n)
B1 3x hold       116.9  +15.1%   97.9%   0.15   -85.7%       11     100%       0   +51.3% / +51.3% (15)
B2 1x hold        30.1  +10.6%   55.3%   0.19   -37.0%        6       0%       0   +16.9% / +51.3% (15)
B3 1/3 mix        72.0  +13.5%   79.1%   0.17   -59.2%        9     100%     407   +27.7% / +51.3% (15)
R2 regime         20.6   +9.3%   79.3%   0.12   -49.1%       13      57%     550    +5.6% / +51.3% (15)
R3 vol 30%       135.4  +15.6%   76.3%   0.20   -45.5%       12     100%     566   +17.9% / +51.3% (15)
R4 200-day       279.8  +18.1%   61.5%   0.29   -49.3%        9      76%     227    +0.4% / +51.3% (15)

UPRO 1993-2015
B1 3x hold         9.2  +10.1%   97.9%   0.10   -85.7%        8     100%       0   +45.2% / +45.2% (10)
B2 1x hold         7.4   +9.1%   55.3%   0.16   -37.0%        4       0%       0   +15.1% / +45.2% (10)
B3 1/3 mix        10.5  +10.8%   79.1%   0.14   -59.2%        6     100%     276   +24.6% / +45.2% (10)
R2 regime          2.2   +3.4%   79.3%   0.04   -49.1%       10      55%     382    +2.2% / +45.2% (10)
R3 vol 30%        15.5  +12.7%   76.3%   0.17   -45.5%        9     100%     381   +17.0% / +45.2% (10)
R4 200-day        30.5  +16.0%   61.5%   0.26   -49.3%        6      73%     164    +0.2% / +45.2% (10)

UPRO 2016-end
B1 3x hold        13.3  +26.9%   76.6%   0.35   -56.1%        3     100%       0    +66.8% / +66.8% (5)
B2 1x hold         4.1  +13.9%   33.8%   0.41   -18.1%        2       0%       0    +20.0% / +66.8% (5)
B3 1/3 mix         7.0  +19.6%   50.1%   0.39   -32.2%        3     100%     131    +34.0% / +66.8% (5)
R2 regime          9.6  +23.2%   42.0%   0.55   -26.6%        3      62%     168     +6.6% / +66.8% (5)
R3 vol 30%         9.1  +22.6%   37.2%   0.61   -30.2%        3     100%     184    +20.7% / +66.8% (5)
R4 200-day         9.6  +23.2%   55.8%   0.42   -43.0%        3      82%      62     +1.2% / +66.8% (5)

TQQQ full: 1993-01-04 to 2026-11-20
B1 3x hold        31.4  +10.7%  100.0%   0.11   -91.6%       10     100%       0   +55.3% / +55.3% (31)
B2 1x hold        70.2  +13.4%   82.9%   0.16   -41.9%        7       0%       0   +18.9% / +55.3% (31)
B3 1/3 mix       144.0  +15.8%   96.6%   0.16   -64.4%       10     100%     407   +29.9% / +55.3% (31)
R2 regime         31.7  +10.7%   93.5%   0.11   -76.4%       14      57%     550   +11.0% / +55.3% (31)
R3 vol 30%       349.7  +18.9%   74.7%   0.25   -40.5%       11     100%     566   +19.7% / +55.3% (31)
R4 200-day       975.7  +22.5%   88.2%   0.26   -66.1%       12      77%     235    +1.2% / +55.3% (31)

TQQQ 1993-2015
B1 3x hold         1.9   +2.8%  100.0%   0.03   -91.6%        7     100%       0   +53.8% / +53.8% (24)
B2 1x hold        12.9  +11.8%   82.9%   0.14   -41.9%        4       0%       0   +18.6% / +53.8% (24)
B3 1/3 mix        14.5  +12.3%   96.6%   0.13   -64.4%        7     100%     276   +29.5% / +53.8% (24)
R2 regime          2.9   +4.8%   93.5%   0.05   -76.4%       11      55%     382    +6.7% / +53.8% (24)
R3 vol 30%        29.5  +15.9%   74.7%   0.21   -40.5%        8     100%     355   +19.6% / +53.8% (24)
R4 200-day        54.9  +19.0%   88.2%   0.22   -66.1%        9      75%     180    +1.3% / +53.8% (24)

TQQQ 2016-end
B1 3x hold        17.6  +30.2%   81.8%   0.37   -79.2%        3     100%       0    +74.4% / +74.4% (6)
B2 1x hold         5.6  +17.1%   35.6%   0.48   -33.0%        3       0%       0    +22.6% / +74.4% (6)
B3 1/3 mix        10.3  +23.9%   55.9%   0.43   -52.7%        3     100%     131    +38.9% / +74.4% (6)
R2 regime         11.0  +24.7%   55.4%   0.45   -42.4%        3      62%     168    +16.6% / +74.4% (6)
R3 vol 30%        12.5  +26.2%   42.1%   0.62   -37.3%        2     100%     211    +24.5% / +74.4% (6)
R4 200-day        19.0  +31.1%   56.8%   0.55   -41.9%        3      82%      55     +1.1% / +74.4% (6)

UPRO R2 regime: FAIL (Calmar full 0.12 <= B3 0.17; Calmar 1993-2015 0.04 <= B3 0.14)
UPRO R3 vol 30%: PASS
UPRO R4 200-day: PASS
TQQQ R2 regime: FAIL (Calmar full 0.11 <= B3 0.16; Calmar 1993-2015 0.05 <= B3 0.13)
TQQQ R3 vol 30%: PASS
TQQQ R4 200-day: PASS
```

Selected calendar years (full table in `report.txt`):

| Year | UPRO B1 | UPRO R3 | UPRO R4 | TQQQ B1 | TQQQ R3 | TQQQ R4 |
|---|---|---|---|---|---|---|
| 2000 | -43% | -25% | -49% | -92% | -37% | -66% |
| 2001 | -45% | -40% | +3% | -89% | -25% | -29% |
| 2002 | -63% | -38% | -7% | -87% | -31% | +1% |
| 2008 | -86% | -46% | +1% | -89% | -41% | -31% |
| 2011 | -11% | -7% | -14% | -10% | -5% | -42% |
| 2015 | -4% | -13% | -33% | +15% | -3% | -11% |
| 2020 | +12% | +19% | +36% | +108% | +48% | +93% |
| 2022 | -56% | -30% | -43% | -79% | -37% | -42% |

## Conclusion

- **R3 (volatility-scaled) and R4 (200-day trend) pass on both UPRO and TQQQ; R2 (graded regime) fails on both**, on
  the full window and the 1993-2015 half (Calmar 0.04 and 0.05 there), where it runs without its VIX/VIX3M condition
  before 2006 and loses 49% (UPRO) and 76% (TQQQ) in 2000. It passes on 2016-2026 alone, the window it was built on.
- **R4 has the highest return** (UPRO +18.1%/y, TQQQ +22.5%/y, against 3x buy-and-hold +15.1% and +10.7%) and is the
  best in long bear markets (UPRO 2008 +1%, TQQQ 2002 +1%), but it **misses the rebound by design**: about +1% over
  the 63 sessions after a trough, against +51% to +55% for 3x buy-and-hold, and it loses in choppy years (UPRO 2015
  -33%, TQQQ 2011 -42%).
- **R3 has the best worst year on both** (UPRO -45.5%, TQQQ -40.5%), the smallest drawdown on TQQQ (74.7%, against
  R4's 88.2%; on UPRO R4's 61.5% beats R3's 76.3%) and the smallest on both over 2016-2026 (UPRO 37.2%, TQQQ 42.1%, the
  best Calmar there, 0.61 and 0.62). It **keeps part of the rebound**: +17% to +25% over the 63 sessions, about a third
  of 3x buy-and-hold's, as its weight rises when volatility fades.
- **No rule makes 3x safe.** Full-window drawdowns stay 61-88% (UPRO R4 61.5%, TQQQ R4 88.2%), against 55.3% and
  82.9% for 1x buy-and-hold. The rules make leverage pay for itself; they do not make it gentle.

## Limits

- One run, one parameter per rule; the whole 1993-2026 history is now seen, so no held-out period is left for a
  combination of R3 and R4 (a new idea, not tested here).
- Synthetic UPRO is about 1.8 points a year optimistic (no swap spread) and synthetic TQQQ about 2 points pessimistic
  (Nasdaq-100 price index, no dividends) against the real ETFs since launch.
- Before 2006-07-17 the regime runs without its VIX/VIX3M condition (R2 only).
- Close-to-close with one session of lag; no taxes; T-bills at the 13-week rate with no fee.
