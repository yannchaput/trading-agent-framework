"""`AlpacaMarketData`: Alpaca IEX prices, quotes and bars, plus the calendar they are windowed on.

Extracted from `AlpacaBroker` so a broker that trades elsewhere (`IbkrBroker`) reuses the exact
same data path. I/O only: every request and parse lives in the pure `market_data`/`account` modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from trading_agent_framework.brokers.alpaca import account, market_data, orders
from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials


class AlpacaMarketData:
    def __init__(
        self,
        data_client: market_data.AlpacaStockDataClient | None,
        calendar_client: orders.AlpacaTradingClient,
    ) -> None:
        self._data_client = data_client
        self._calendar_client = calendar_client

    @classmethod
    def from_credentials(cls, creds: AlpacaCredentials) -> AlpacaMarketData:
        """Both clients from the `ALPACA_DATA_*` credentials (the calendar lives on the trading API)."""
        data_client = cast("market_data.AlpacaStockDataClient", build_stock_data_client(creds))
        calendar_client = cast("orders.AlpacaTradingClient", build_trading_client(creds))
        return cls(data_client, calendar_client)

    @property
    def calendar_client(self) -> orders.AlpacaTradingClient:
        return self._calendar_client

    def _require_data_client(self) -> market_data.AlpacaStockDataClient:
        if self._data_client is None:
            raise BrokerError(
                "no market data client configured; construct the broker with data_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return self._data_client

    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self.get_last_prices([asset])[asset]

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        client = self._require_data_client()
        prices: dict[Asset, Decimal | None] = {}
        for chunk in market_data.chunk_assets(assets):
            request = market_data.build_latest_trade_request(chunk)
            try:
                response = client.get_stock_latest_trade(request)
            except Exception as exc:
                raise BrokerError(
                    f"Failed to fetch latest trades ({len(chunk)} symbols): {exc}"
                ) from exc
            prices.update(market_data.parse_latest_trades(response, chunk))
        return prices

    def get_quote(self, asset: Asset) -> Quote | None:
        client = self._require_data_client()
        request = market_data.build_latest_quote_request([asset])
        try:
            response = client.get_stock_latest_quote(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the latest quote for {asset.symbol}: {exc}") from exc
        return market_data.parse_quote(response, asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        end: datetime,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        if not assets:
            return {}
        client = self._require_data_client()
        sessions = self._sessions_before(end, length, timestep)
        start = market_data.bars_start(end, length, timestep, sessions)
        in_session = None if include_after_hours else sessions
        bars: dict[Asset, Bars] = {}
        for chunk in market_data.chunk_assets(assets):
            request = market_data.build_bars_request(chunk, timestep, start, end)
            try:
                barset = client.get_stock_bars(request)
            except Exception as exc:
                raise BrokerError(
                    f"Failed to fetch {timestep} bars ({len(chunk)} symbols): {exc}"
                ) from exc
            bars.update(market_data.parse_bars(barset, chunk, timestep, length, sessions=in_session))
        return bars

    def _sessions_before(self, end: datetime, length: int, timestep: str) -> list[MarketSession]:
        first_day = market_data.calendar_lookback_start(end, length, timestep)
        last_day = end.astimezone(market_data.MARKET_TZ).date()
        request = account.build_calendar_request(first_day, last_day)
        try:
            days = self._calendar_client.get_calendar(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca calendar: {exc}") from exc
        return account.parse_calendar(days, market_data.MARKET_TZ)
