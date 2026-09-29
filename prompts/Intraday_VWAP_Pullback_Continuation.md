# Intraday VWAP Pullback Continuation

The basic idea is:

Find stocks already demonstrating abnormal strength, wait for the first pullback toward VWAP/short EMA, then enter when the stock resumes upward.

This is materially different from ORB:

ORB: "price broke the range → buy."
VWAP pullback: "price demonstrated strength → pulled back without losing structure → buyers returned → buy."

That distinction is important because your ORB problem—many false breakouts—is largely avoided by requiring the market to prove continuation after the initial move.

There is also empirical motivation for concentrating on intraday continuation. Research has found intraday return continuation at specific half-hour intervals, while short-lived reversals appear associated with liquidity imbalances and bid-ask effects.

The strategy I would test first
1. Universe

Use the same cross momentum universe file

Then dynamically select perhaps 20–50 candidates per day.

The important variable isn't simply liquidity.

You want:

unusual intraday activity.

For example:

relative_volume = current_volume / expected_volume_at_this_time

and look for:

RVOL > 1.5–2.0
2. Don't enter at the open

This is one of the biggest changes I'd make relative to ORB.

I'd initially ignore:

09:30–09:40

and potentially even:

09:30–09:45

The opening period has particularly complicated price discovery and relatively high price impact.

Instead, let the first move establish itself.

For example:

09:35 ───── initial move
       \
        \
         \ pullback
          \
           → continuation entry
3. Detect an "impulse"

You want a stock that is clearly behaving differently from the rest of the universe.

For example, over the first 20–40 minutes:

stock_return       > +1.0%
relative_strength  > +0.5%
RVOL               > 1.5
price              > VWAP

But I wouldn't hard-code those values initially.

Create normalized features:

intraday_return_z
relative_volume_z
relative_strength_z
distance_from_vwap
atr_normalized_move

Then let the strategy rank candidates.

4. Wait for the pullback

This is the key.

After the impulse, don't chase.

Wait for:

price ↓
volume ↓
volatility ↓
but
price remains above VWAP

Ideally:

             impulse
                /\
               /  \
              /    \
             /      \
            /        \____
           /               \
          /                 \  ← pullback
         /                   \
--------VWAP------------------\------
                               \
                                ↑
                             entry

The pullback should ideally have:

Healthy
price > VWAP
pullback_volume < impulse_volume
pullback_duration < impulse_duration
RS remains positive
Bad
price < VWAP
huge selling volume
large bearish candle
relative strength collapses
5. Entry = resumption, not touching VWAP

This is where I think you can eliminate a lot of false positives.

Don't enter because:

price touched VWAP

Instead:

pullback
   ↓
compression
   ↓
bullish reversal candle
   ↓
break of micro-structure
   ↓
BUY

For example:

pullback_low = L

trigger =
    close > previous_5m_high
    AND close > VWAP
    AND volume > pullback_volume_average

This gives you a confirmation entry.

6. The really interesting feature: relative strength

This is where I would make your strategy more sophisticated than a standard VWAP strategy.

Calculate:

stock_return - SPY_return

or better:

stock_return - beta * SPY_return

during the entire setup.

You don't want simply:

"AAPL went +1.5%."

You want:

"AAPL went +1.5% while SPY went +0.1%."

And during the pullback:

SPY     continues upward
AAPL    pulls back slightly
AAPL/SPY relative strength remains strong

or:

SPY     -0.2%
AAPL    +0.3%

That is potentially much more interesting.

It gives you an intraday idiosyncratic momentum strategy rather than simply buying leveraged SPY.

7. Exit architecture

I wouldn't use a single fixed take-profit.

I'd test:

Initial stop

Below:

pullback_low - 0.1 ATR

or:

pullback_low

Then:

TP1

Something like:

+1R

Take 25–50% off.

Remaining position

Trail using:

VWAP

or:

9 EMA

or a volatility-based trailing stop.

The interesting question for your CAGR isn't necessarily:

"How do I maximize win rate?"

It's:

"How long can I keep the rare large intraday winners?"

That's much more important for a high-CAGR strategy.