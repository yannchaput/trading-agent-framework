"""A strategy's Alpaca account for the Live and Paper tabs: env-file discovery, credentials and read-only fetches.

Streamlit-free. Keys are read with `dotenv_values` -- never `load_dotenv` -- so one strategy's keys never reach
`os.environ` and switching strategies cannot mix accounts.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import dotenv_values

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.config.env import AlpacaCredentials, find_project_root
from trading_agent_framework.dashboard.pnl import AccountPnl, BrokerPosition, Fee, Fill, daily_account_pnl, market_day
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.trading.models import Position


def default_env_dir() -> Path:
    return find_project_root() / "env"


def _env_values(path: Path) -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(path).items()}


def strategies_for_mode(mode: str, env_dir: Path) -> list[str]:
    """The strategies with an `env_dir/.env.<strategy>.<mode>` file trading on Alpaca (no `BROKER` means alpaca)."""
    if not env_dir.is_dir():
        return []
    prefix, suffix = ".env.", f".{mode}"
    names = []
    for path in env_dir.glob(f"{prefix}*{suffix}"):
        broker = (_env_values(path).get("BROKER") or "alpaca").strip().lower()
        if broker == "alpaca":
            names.append(path.name[len(prefix) : -len(suffix)])
    return sorted(names)


def load_account_credentials(strategy: str, mode: str, env_dir: Path) -> tuple[AlpacaCredentials, AlpacaCredentials]:
    """The trading and the market-data keys of `env_dir/.env.<strategy>.<mode>`."""
    path = env_dir / f".env.{strategy}.{mode}"
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    values = _env_values(path)
    return AlpacaCredentials.for_trading(env=values), AlpacaCredentials.for_data(env=values)


def account_is_paper(strategy: str, mode: str, env_dir: Path) -> bool:
    """Whether the strategy's trading keys hit Alpaca's paper endpoint (`BROKER_API_IS_PAPER`), whatever its mode."""
    return load_account_credentials(strategy, mode, env_dir)[0].is_paper


PAGE_SIZE = 100  # Alpaca's maximum for /account/activities


class AccountTradingClient(Protocol):
    """The only `TradingClient` calls the dashboard makes: GETs. No order method, so the reader cannot trade."""

    def get(self, path: str, data: dict[str, Any] | str | None = None, **kwargs: Any) -> Any: ...
    def get_all_positions(self) -> list[Position]: ...


class AccountDataClient(Protocol):
    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...


def _parse_activity(item: dict[str, Any]) -> Fill | Fee | None:
    # Parsed by hand: the Trading API's activities have no `account_id`, which alpaca-py's activity models require.
    kind = item.get("activity_type")
    if kind == "FILL":
        return Fill(
            time=datetime.fromisoformat(item["transaction_time"]),
            symbol=item["symbol"],
            side=item["side"],
            qty=float(item["qty"]),
            price=float(item["price"]),
        )
    if kind == "FEE" and item.get("status") != "canceled":
        return Fee(day=date.fromisoformat(str(item["date"])[:10]), amount=float(item["net_amount"]))
    return None


def fetch_activities(client: AccountTradingClient) -> tuple[list[Fill], list[Fee]]:
    """Every fill and fee of the account, oldest first, following `page_token` until a short page."""
    fills: list[Fill] = []
    fees: list[Fee] = []
    params: dict[str, Any] = {"activity_types": "FILL,FEE", "direction": "asc", "page_size": PAGE_SIZE}
    while True:
        page = cast(list[dict[str, Any]], client.get("/account/activities", params))
        for item in page:
            parsed = _parse_activity(item)
            if isinstance(parsed, Fill):
                fills.append(parsed)
            elif isinstance(parsed, Fee):
                fees.append(parsed)
        if len(page) < PAGE_SIZE:
            return fills, fees
        params = {**params, "page_token": page[-1]["id"]}


def fetch_daily_closes(client: AccountDataClient, symbols: Sequence[str], first_day: date, today: date) -> dict[str, dict[date, float]]:
    """Raw (unadjusted, like the fills) IEX daily closes from `first_day` up to, not including, today."""
    if not symbols or first_day >= today:
        return {}
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=cast(TimeFrame, TimeFrame.Day),
        start=datetime.combine(first_day, time.min, MARKET_TZ),
        end=datetime.combine(today, time.min, MARKET_TZ),
        feed=DataFeed.IEX,
        adjustment=Adjustment.RAW,
    )
    barset = client.get_stock_bars(request)
    return {symbol: {market_day(bar.timestamp): bar.close for bar in bars} for symbol, bars in barset.data.items() if bars}


def _float(value: str | None) -> float | None:
    return None if value is None else float(value)


def to_broker_position(position: Position) -> BrokerPosition:
    return BrokerPosition(
        symbol=position.symbol,
        qty=float(position.qty),
        avg_entry_price=float(position.avg_entry_price),
        current_price=_float(position.current_price),
        market_value=_float(position.market_value),
        unrealized_pl=_float(position.unrealized_pl),
        unrealized_plpc=_float(position.unrealized_plpc),
    )


def load_account_pnl(trading: AccountTradingClient, data: AccountDataClient, today: date) -> AccountPnl:
    """The account's daily P&L: its fills and fees, the daily closes of every symbol it traded, its positions now."""
    fills, fees = fetch_activities(trading)
    positions = [to_broker_position(p) for p in trading.get_all_positions()]
    closes: dict[str, dict[date, float]] = {}
    if fills:
        first_day = min(market_day(f.time) for f in fills)
        closes = fetch_daily_closes(data, sorted({f.symbol for f in fills}), first_day, today)
    return daily_account_pnl(fills, fees, closes, positions, today)


def load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
    """The daily P&L of the strategy's Alpaca account, fetched now."""
    trading_creds, data_creds = load_account_credentials(strategy, mode, env_dir)
    trading = cast(AccountTradingClient, build_trading_client(trading_creds))
    data = cast(AccountDataClient, build_stock_data_client(data_creds))
    return load_account_pnl(trading, data, today)
