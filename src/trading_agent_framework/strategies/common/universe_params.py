"""The liquidity thresholds of the US stock universe: the one home of the numbers.

The universe build (`batch_us_stock_universe`, live batch and point-in-time history) filters on these, and
cross_momentum's `CONFIG` reads the same constants, so the strategy's filters and the universe's can never drift apart.
"""

MIN_PRICE = 10.0
MIN_DOLLAR_VOLUME = 20_000_000
MIN_MARKET_CAP = 2_000_000_000
