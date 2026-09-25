"""Pure Alpaca symbol translation.

`batch_us_stock_universe.normalize_symbol` folds share-class separators
(space/dot/slash) onto yfinance's dash form (``"BRK-B"``), since the universe
file is also read by the yfinance-backed backtest data source. Alpaca's own
API uses dot notation instead (``"BRK.B"``) and rejects the dash form as an
unknown symbol. These two functions translate between the two conventions at
the Alpaca boundary; no I/O, no state -- same rules as `orders.py`.
"""

from __future__ import annotations

import re

_DASH_SHARE_CLASS = re.compile(r"^([A-Z]+)-([A-Z])$")
_DOT_SHARE_CLASS = re.compile(r"^([A-Z]+)\.([A-Z])$")


def to_alpaca_symbol(symbol: str) -> str:
    """Our dash share-class suffix to Alpaca's dot form (``"BRK-B"`` -> ``"BRK.B"``)."""
    match = _DASH_SHARE_CLASS.match(symbol)
    return f"{match.group(1)}.{match.group(2)}" if match else symbol


def from_alpaca_symbol(symbol: str) -> str:
    """Alpaca's dot share-class suffix back to our dash form (``"BRK.B"`` -> ``"BRK-B"``)."""
    match = _DOT_SHARE_CLASS.match(symbol)
    return f"{match.group(1)}-{match.group(2)}" if match else symbol
