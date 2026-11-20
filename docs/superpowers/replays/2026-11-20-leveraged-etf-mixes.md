# SPY / UPRO / TQQQ fixed-weight mixes (2016-2026): record

> Code not kept: the spike script and its log were dropped on 2026-11-20; this record is what remains.

Asked by the user on 2026-11-20 after the intraday regime switch was NO-GO on the held-out window
(`2026-11-20-intraday-regime-held-out-preregistration.md`): what does buying and holding the leveraged funds give, alone
and mixed with SPY? Yahoo daily adjusted closes, 2016-01-08 to 2026-11-19; target weights restored on the first session
of each month at 0.03% on the traded amount. The user's fourth mix ("50% SPY, 25% UPRO, 25% SPY") was run as written
(75/25) and as the likely intent (SPY 50%, UPRO 25%, TQQQ 25%). No look-ahead issue (fixed weights), but the window is
hindsight: no 2008-style bear market (both funds started after it) and TQQQ rides the Nasdaq's best decade.

```
2016-01-08 .. 2026-11-19, monthly rebalance, 0.03% on the traded amount
mix                                  total per year worst DD       (when) worst yr losing yrs
SPY 100%                             +344%   +14.7%    33.7%   2020-03-23   -18.2%          2
UPRO 100%                           +1338%   +27.8%    76.8%   2020-03-23   -56.8%          2
TQQQ 100%                           +2612%   +35.5%    81.7%   2022-12-28   -79.1%          3
UPRO 1/3, SPY 2/3                    +663%   +20.6%    49.8%   2020-03-23   -32.5%          2
UPRO 1/3, TQQQ 1/3, SPY 1/3         +1459%   +28.8%    61.7%   2020-03-23   -56.2%          2
UPRO 40%, TQQQ 40%, SPY 20%         +1724%   +30.6%    66.5%   2020-03-23   -62.0%          2
SPY 75%, UPRO 25% (as written)       +575%   +19.2%    46.0%   2020-03-23   -29.0%          2
SPY 50%, UPRO 25%, TQQQ 25%         +1123%   +25.9%    55.3%   2020-03-23   -48.1%          2

by year:
Date                            2016   2017  2018   2019   2020  2021  2022   2023  2024  2025 2026
SPY 100%                        +19%   +22%   -5%   +31%   +18%  +29%  -18%   +26%  +25%  +18%  +6%
UPRO 100%                       +58%   +71%  -25%  +102%   +10%  +99%  -57%   +69%  +64%  +32%  +2%
TQQQ 100%                       +39%  +118%  -20%  +134%  +110%  +83%  -79%  +198%  +58%  +34%  -6%
UPRO 1/3, SPY 2/3               +31%   +37%  -11%   +53%   +24%  +50%  -32%   +40%  +37%  +23%  +6%
UPRO 1/3, TQQQ 1/3, SPY 1/3     +39%   +67%  -15%   +87%   +51%  +69%  -56%   +90%  +50%  +29%  +3%
UPRO 40%, TQQQ 40%, SPY 20%     +43%   +78%  -18%   +99%   +54%  +78%  -62%  +104%  +54%  +31%  +2%
SPY 75%, UPRO 25% (as written)  +28%   +33%   -9%   +47%   +23%  +44%  -29%   +37%  +34%  +22%  +6%
SPY 50%, UPRO 25%, TQQQ 25%     +34%   +55%  -12%   +72%   +46%  +58%  -48%   +73%  +43%  +26%  +5%
```

## Result

- Every mix beats SPY over the window, and return and drawdown rise together: no weighting gets UPRO-like returns at
  SPY-like risk.
- None is positive every year: all lose in 2018 and 2022, in proportion to the leverage (the user's bar for the
  intraday strategy is not met by buy-and-hold either).
- Monthly rebalancing helped: UPRO/TQQQ/SPY by thirds beat UPRO alone (+1459% vs +1338%) with a smaller worst
  drawdown (61.7% vs 76.8%).
- Closest to the user's goals: SPY 75% / UPRO 25% or SPY 2/3 / UPRO 1/3 (19-21% a year, worst years -29% and -32%).
