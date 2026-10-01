Yes. For your use case, I would make this a strict intraday reversal strategy, not a generic “price far from VWAP → short” strategy.

The key is to separate extension from exhaustion:

Extension tells you where to look. Exhaustion tells you when to enter.

And because you want this as a complement to your momentum strategy, I'd explicitly design it to perform in choppy/mean-reverting conditions, rather than trying to predict the whole market.

1. Strategy architecture

I'd structure it as a state machine:

NORMAL
  │
  │ strong directional move
  ▼
EXTENDED
  │
  │ extreme VWAP deviation
  ▼
EXHAUSTION WATCH
  │
  │ momentum deterioration + failed continuation
  ▼
REVERSAL CONFIRMED
  │
  │ enter fade
  ▼
MEAN REVERSION
  │
  ├── VWAP reached → EXIT
  ├── stop hit     → EXIT
  ├── time limit   → EXIT
  └── EOD          → EXIT

This is much more robust than trying to encode everything into one giant entry condition.

2. Start with a liquid universe

This strategy is particularly sensitive to spreads and slippage.

I'd initially restrict it to something like:

Price > $10
ADV20 > $20M
Average 1-min dollar volume > $1M
Spread < 10–15 bps

For your first backtest, I'd actually make it even simpler:

SP500 + Nasdaq100 stocks.

You want to determine whether the signal itself works before introducing universe-selection effects.

3. Calculate session VWAP

Use regular-session VWAP, resetting every trading day.

For each bar:

typical_price = (high + low + close) / 3

cum_pv += typical_price * volume
cum_volume += volume

vwap = cum_pv / cum_volume

Then calculate normalized VWAP distance:

vwap_distance = (close - vwap) / vwap

But I wouldn't use a fixed percentage as the primary measure.

A $500 stock and a $20 stock have very different intraday volatility.

Instead calculate something like:

atr_5m = ATR(5m, 14)

distance_atr = (close - vwap) / atr_5m

This becomes much more useful.

For example:

distance_atr > +2.0 → significantly extended upward
distance_atr < -2.0 → significantly extended downward
4. Detect the initial strong move

You don't want:

price 2 ATR from VWAP
→ immediately short

because the stock may simply be trending strongly.

You want evidence that the extension came from a real directional impulse.

For example, on 5-minute bars:

return_30m = close / close.shift(6) - 1

Then require something like:

30-min return > +1.5%

for a potential short.

But I'd make this volatility-adjusted:

move_atr = (close - close_30m_ago) / atr_5m

Potential initial condition:

move_atr > +2.5

for an upside exhaustion setup.

Symmetrically:

move_atr < -2.5

for downside exhaustion.

5. Detect the extreme extension

Now combine the move with VWAP distance.

For a short setup:

strong upward move
        AND
price significantly above VWAP

For example:

distance_atr > 2.0

So your first candidate becomes:

candidate_short = (
    move_atr > 2.5
    and distance_atr > 2.0
)

This is not an entry.

It merely means:

"Something interesting is happening."

6. Measure momentum deterioration

This is the most important part.

You need to detect:

price is still going up, but its ability to continue upward is deteriorating.

There are several ways to measure this.

A. Short-term returns decreasing

For example:

bar -3 → +0.45%
bar -2 → +0.32%
bar -1 → +0.11%
current → +0.03%

Momentum is weakening.

You could calculate:

r1 = close.pct_change(1)
r3 = close.pct_change(3)

momentum_decay = r1 < r3 / 3
B. RSI exhaustion

Use RSI as a confirmation, not the primary signal.

For example:

RSI(5) > 80

followed by:

RSI starts falling

is more interesting than simply:

RSI > 80
C. Range expansion → compression

This is particularly interesting.

During the impulse:

large candles
large volume

Then suddenly:

smaller candles

while price remains near the extreme.

For example:

range_now < 0.6 * range_max_last_6_bars

This gives you a crude measure of loss of directional energy.

7. Detect failed continuation

This is where I'd make the strategy considerably more selective.

Suppose the stock does:

100
101
102
103
104
105
106

You don't want to short at 104.

Instead wait for:

106
106.20
106.05
105.80

The important event is:

new high → failure to continue → break of short-term structure

For example:

recent_high = high.rolling(6).max()

failed_breakout = (
    high >= recent_high.shift(1)
    and close < high.shift(1)
)

Even better:

new intraday high
        ↓
cannot hold high
        ↓
close below previous 1–3 bar low

That's a much more meaningful reversal trigger.

8. Add volume exhaustion

Volume can help distinguish:

Healthy trend
price ↑
volume ↑
new highs ↑

from:

Exhaustion
price ↑
volume ↓
new highs barely advancing

A simple metric:

volume_ratio = volume / volume.rolling(20).mean()

Then look for:

initial impulse:
volume_ratio > 1.5

exhaustion:
volume_ratio falling

An interesting pattern is:

huge volume
     ↓
price makes extreme
     ↓
volume remains elevated
     ↓
price stops advancing

That can represent absorption/exhaustion.

Don't require volume to simply become low; that could exclude useful reversals.

9. Put the pieces together

Your short candidate could initially be:

extension = (
    distance_atr > 2.0
    and move_atr > 2.5
)

exhaustion = (
    momentum_decay
    and range_compression
    and failed_continuation
)

short_signal = extension and exhaustion

And the long side is simply inverted.

But I would actually use a scoring model rather than requiring every condition.

For example:

Extension
+2   VWAP distance > 2 ATR
+1   30m move > 2 ATR

Exhaustion
+2   failed breakout
+1   momentum decay
+1   range compression
+1   RSI reversal
+1   volume exhaustion/absorption

Entry threshold = 5

This gives you much better control during backtesting.

10. Entry should be after confirmation

This is crucial.

Don't enter:

price reaches +2 ATR VWAP
→ SHORT

Instead:

              EXTREME
                 ●
              ●     ●
           ●          ●
        ●               ●
     ●                   ●
─────── VWAP ───────────────
                       ↑
                exhaustion
                       ↑
                 ENTER SHORT

For example:

trigger = (
    close < low.shift(1)
    and close < ema(close, 3)
)

Or:

new high
→ rejection
→ break of 3-bar low
→ short

This sacrifices some entry price in exchange for substantially better confirmation.

11. Stop loss

Don't use a fixed percentage initially.

Use the exhaustion extreme.

For a short:

stop = exhaustion_high + 0.1 * ATR

For a long:

stop = exhaustion_low - 0.1 * ATR

This makes the trade thesis very clear:

"If price breaks the exhaustion extreme, my exhaustion hypothesis was wrong."

12. Profit target

The natural target is VWAP.

So:

target = VWAP

But I'd actually implement two exits.

Target A — partial

For example:

50% position → 1R
Target B — VWAP
remaining 50% → VWAP

Or simply test:

Exit at VWAP

versus:

Exit at 0.5 × distance-to-VWAP

versus:

Exit at 1R

This will be interesting in your backtests.

13. Add a time stop

This is particularly important for mean reversion.

Your thesis is:

extreme price should revert relatively quickly.

If it doesn't, the thesis is deteriorating.

For example:

Maximum holding time = 30–90 minutes

So:

if elapsed_minutes > 60:
    exit()

I would test this systematically.

Potential values:

15m
30m
45m
60m
90m
120m
14. Force EOD liquidation

For this strategy, I would strictly prohibit overnight positions.

Your original concept is:

intraday exhaustion
→ intraday mean reversion

If you hold overnight, you're changing the strategy into something else.

So:

15:45–15:55 ET
→ close everything

depending on your execution constraints.

15. Avoid the biggest trap: strong trends

This is probably going to be your biggest source of losses.

Example:

VWAP
  │
  │
  │      ↑
  │     ↑
  │    ↑
  │   ↑
  │  ↑
  │ ↑
  └──────────────

Price can remain:

+2 ATR
+2.5 ATR
+3 ATR
+3.5 ATR

for a long time.

A naïve mean-reversion strategy will repeatedly short this.

Therefore add a trend-strength filter.

For example:

ADX > 30
AND
price > VWAP
AND
VWAP slope strongly positive

→ don't short.

This is especially important for your strategy because you already have a momentum strategy.

You want:

Momentum strategy:
    strong trend

Mean reversion strategy:
    exhausted move / weakening trend

rather than having both fight over the same market regime.

16. VWAP slope is a very useful filter

Calculate:

vwap_slope = vwap - vwap.shift(12)

Normalize it:

vwap_slope_atr = vwap_slope / atr_5m

Then:

Short exhaustion

Prefer:

VWAP above price? No
VWAP slope positive but flattening

rather than:

VWAP slope strongly positive

The latter means the market may still be aggressively trending.

You can therefore explicitly detect:

VWAP slope positive
+
VWAP slope decreasing

which is another exhaustion clue.

17. Market regime filter

Given your experience with your momentum strategy being highly correlated to SP500, I'd make the strategy market-aware but not market-dependent.

For each stock:

Stock extension
+
Stock exhaustion

Then compare against:

SPY
QQQ
sector ETF

For example, if a stock is:

+3 ATR above VWAP

while SPY is:

+0.3 ATR above VWAP

the stock-specific extension is much more interesting.

Conversely:

stock +3 ATR
SPY +3 ATR
sector +2.5 ATR

is much more likely to be a broad market momentum event.

That's exactly the situation where I'd be more cautious about fading it.

18. An interesting enhancement: relative VWAP extension

This could become one of the strongest features in your implementation.

Calculate:

stock VWAP deviation
minus
sector VWAP deviation

Example:

Stock: +3.0 ATR
Sector: +1.0 ATR

Relative extension = +2.0 ATR

That's potentially much more interesting than:

Stock: +3.0 ATR
Sector: +2.9 ATR

because in the second case the entire sector is moving.

You could therefore create:

relative_extension =
    stock_vwap_zscore
    - sector_vwap_zscore

and use it as one of your strongest features.

19. I'd implement the first version like this

Don't over-engineer V1.

Candidate
Universe:
    SP500 + Nasdaq100

Timeframe:
    5-minute

Session:
    regular US session

VWAP:
    session VWAP

Candidate:
    distance from VWAP > 2 ATR
    AND
    30-minute move > 2.5 ATR
Exhaustion

Require 2 of:

1. failed new high/low
2. momentum decay
3. range compression
4. RSI reversal
5. volume exhaustion
Entry
break of short-term reversal structure
Stop
beyond exhaustion extreme + volatility buffer
Target
VWAP
Time stop
60 minutes
EOD
flat
20. The strategy should therefore look like this

For a short:

                EXTREME
                   ▲
                   │
             ┌─────┴─────┐
             │ exhaustion│
             └─────┬─────┘
                   │
             failed high
                   │
                   ▼
                ENTRY
                   │
                   │
                   ▼
                 VWAP
───────────────────┼────────────────
                   │
                   │

The important distinction is:

                    WRONG
                      ↓
VWAP ────────────────●────────
                    SHORT
                      ↑
              merely extended


                    RIGHT
                      ↓
VWAP ────────────────●───────
                     ↑
                extension
                     ↑
                exhaustion
                     ↑
               failed breakout
                     ↑
                   SHORT
21. What I would measure in your backtest

Don't just look at CAGR.

For this particular strategy I'd collect:

Metric	Why
Win rate	Mean reversion should generally have reasonable hit rate
Avg winner	Determines whether winners compensate for failures
Avg loser	Critical because trends can run away
Profit factor	Core robustness metric
MAE	Helps optimize stop placement
MFE	Helps optimize VWAP/partial exits
Time-to-MFE	Determines time stop
Time-to-VWAP	Directly tests thesis
VWAP distance at entry	Determines optimal extension
Market beta	Check diversification
SPY correlation	Important for your portfolio
Sector correlation	Detect broad-sector events
Performance by VIX regime	Very useful
Performance by market trend	Critical
Performance by time of day	Potentially very significant

And especially create a distribution of:

Entry VWAP deviation
       ↓
      P&L

You may discover something like:

1.5–2.0 ATR → poor
2.0–2.5 ATR → mediocre
2.5–3.0 ATR → good
3.0–4.0 ATR → excellent
>4.0 ATR   → dangerous

That would tell you where the actual exhaustion regime exists rather than assuming the threshold.

22. One important design decision for your project

Given the strategies you've already tested, I'd make this the opposite regime of your ORB strategy.

Your portfolio could eventually look something like:

                    MARKET REGIME
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       TRENDING       NEUTRAL        REVERTING
          │              │              │
          ▼              ▼              ▼
    Cross-sectional    VWAP          Exhaustion
      momentum       pullback        mean reversion
          │              │              │
          └──────────────┼──────────────┘
                         │
                    Portfolio

And your Intraday Exhaustion Mean Reversion should ideally make money precisely in periods where your momentum strategy struggles—not merely generate another version of the same equity curve.

My recommended V1

I'd start with 5-minute bars + VWAP + ATR normalization + failed breakout + momentum decay + VWAP target + 60-minute time stop + mandatory EOD close.

Then run a feature-ablation backtest:

A: VWAP extension only
B: + strong move
C: + failed breakout
D: + momentum decay
E: + volume
F: + market/sector relative extension

That will tell you which part is actually generating the edge, rather than optimizing a black-box collection of indicators.