"""Sector metadata provider using yfinance. Cached per instance."""

import logging

logger = logging.getLogger(__name__)


class SectorProvider:
    """Provide sector classifications via yfinance, cached per instance.

    Known limitation: In backtesting mode, yfinance returns current (live)
    sector classification, not point-in-time data. Acceptable for a diagnostic.
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def get_sector(self, symbol: str) -> str:
        """Return sector string or 'UNKNOWN'. Caches per instance."""
        if symbol in self._cache:
            return self._cache[symbol]
        sector = self._fetch_sector(symbol)
        self._cache[symbol] = sector
        return sector

    def get_sector_map(self, symbols: list[str]) -> dict[str, str]:
        """Bulk lookup returning {symbol: sector}."""
        return {sym: self.get_sector(sym) for sym in symbols}

    def _fetch_sector(self, symbol: str) -> str:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info is None:
                return "UNKNOWN"
            sector = info.get("sector")
            if sector is None or not isinstance(sector, str) or not sector.strip():
                return "UNKNOWN"
            return sector.strip()
        except Exception:
            logger.debug("Failed to fetch sector for %s", symbol, exc_info=True)
            return "UNKNOWN"
