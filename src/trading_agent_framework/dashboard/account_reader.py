"""A strategy's Alpaca account for the Live and Paper tabs: env-file discovery, credentials and read-only fetches.

Streamlit-free. Keys are read with `dotenv_values` -- never `load_dotenv` -- so one strategy's keys never reach
`os.environ` and switching strategies cannot mix accounts.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import dotenv_values

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.config.env import AlpacaCredentials, find_project_root
from trading_agent_framework.dashboard.pnl import AccountPnl, BrokerPosition, Dividend, Fee, Fill, ReturnBases, daily_account_pnl, market_day
from trading_agent_framework.strategies.common.yahoo_daily_bars import HISTORY_BARS, YahooDailyBars
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.trading.models import Position

    from trading_agent_framework.entities.bars import Bars


def default_env_dir() -> Path:
    return find_project_root() / "env"


def _env_values(path: Path) -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(path).items()}


_DISPLAY_VALUES = frozenset({"true", "1", "yes"})


def strategies_for_mode(mode: str, env_dir: Path) -> list[str]:
    """The strategies with an `env_dir/.env.<strategy>.<mode>` file trading on Alpaca (no `BROKER` means alpaca) and
    opted in with `DASHBOARD_DISPLAY=true` (missing means hidden: a retired strategy keeps its file off the picker)."""
    if not env_dir.is_dir():
        return []
    prefix, suffix = ".env.", f".{mode}"
    names = []
    for path in env_dir.glob(f"{prefix}*{suffix}"):
        values = _env_values(path)
        broker = (values.get("BROKER") or "alpaca").strip().lower()
        displayed = values.get("DASHBOARD_DISPLAY", "").strip().lower() in _DISPLAY_VALUES
        if broker == "alpaca" and displayed:
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
BENCHMARK_SYMBOL = "SPY"  # what the daily % is compared with, as in the backtests
_BASE_DAYS = 7  # calendar days read before the first day, so that day has a base even over a long weekend


class AccountTradingClient(Protocol):
    """The only `TradingClient` calls the dashboard makes: GETs. No order method, so the reader cannot trade."""

    def get(self, path: str, data: dict[str, Any] | str | None = None, **kwargs: Any) -> Any: ...
    def get_all_positions(self) -> list[Position]: ...


class BenchmarkBars(Protocol):
    """`strategies.common.yahoo_daily_bars.YahooDailyBars`: the backtests' own daily bars, each stamped at its session's close."""

    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]: ...


class AccountDataClient(Protocol):
    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...


def _parse_activity(item: dict[str, Any]) -> Fill | Fee | Dividend | None:
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
    if kind in ("DIV", "DIVNRA"):  # DIVNRA: the non-resident tax withheld on a dividend, negative
        return Dividend(day=date.fromisoformat(str(item["date"])[:10]), amount=float(item["net_amount"]))
    return None


def fetch_activities(client: AccountTradingClient) -> tuple[list[Fill], list[Fee], list[Dividend]]:
    """Every fill, fee and dividend (credits and their withholding) of the account, oldest first, following `page_token`
    until a short page."""
    fills: list[Fill] = []
    fees: list[Fee] = []
    dividends: list[Dividend] = []
    params: dict[str, Any] = {"activity_types": "FILL,FEE,DIV,DIVNRA", "direction": "asc", "page_size": PAGE_SIZE}
    while True:
        page = cast(list[dict[str, Any]], client.get("/account/activities", params))
        for item in page:
            parsed = _parse_activity(item)
            if isinstance(parsed, Fill):
                fills.append(parsed)
            elif isinstance(parsed, Fee):
                fees.append(parsed)
            elif isinstance(parsed, Dividend):
                dividends.append(parsed)
        if len(page) < PAGE_SIZE:
            return fills, fees, dividends
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


def fetch_benchmark_closes(source: BenchmarkBars, first_day: date, today: date) -> dict[date, float]:
    """The benchmark's closes from a week before `first_day` to today, whose bar is still forming (its close is the
    latest price). Yahoo's, like the backtests': consolidated, official closes, split- and dividend-adjusted."""
    bars = source.bars([BENCHMARK_SYMBOL], today).get(BENCHMARK_SYMBOL)
    if bars is None:
        return {}
    closes = {market_day(stamp): float(close) for stamp, close in bars.df["close"].items()}
    return {day: close for day, close in closes.items() if day >= first_day - timedelta(days=_BASE_DAYS)}


def fetch_equity_closes(client: AccountTradingClient, first_day: date, today: date) -> dict[date, float]:
    """The account's equity at each close from a week before `first_day` (so the first day has a base) to today.
    Alpaca's portfolio history labels a close with the *next* calendar day (Monday's close under Tuesday, Friday's
    under Saturday), so the date is moved back by one. Equity is 0 before the account is funded: kept, and read as
    "no base" downstream."""
    history = cast(
        dict[str, Any],
        client.get("/account/portfolio/history", {"timeframe": "1D", "start": (first_day - timedelta(days=_BASE_DAYS)).isoformat(), "end": (today + timedelta(days=1)).isoformat()}),
    )
    return {
        datetime.fromtimestamp(stamp, UTC).date() - timedelta(days=1): float(equity)
        for stamp, equity in zip(history.get("timestamp") or [], history.get("equity") or [], strict=True)
        if equity is not None
    }


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
    fills, fees, dividends = fetch_activities(trading)
    positions = [to_broker_position(p) for p in trading.get_all_positions()]
    closes: dict[str, dict[date, float]] = {}
    if fills:
        first_day = min(market_day(f.time) for f in fills)
        closes = fetch_daily_closes(data, sorted({f.symbol for f in fills}), first_day, today)
    return daily_account_pnl(fills, fees, closes, positions, today, dividends)


def load_return_bases(trading: AccountTradingClient, benchmark: BenchmarkBars, first_day: date, today: date) -> ReturnBases:
    """What the daily % are measured against: the account's equity at each close and the benchmark's closes, both from
    a week before `first_day` (the first fill's day)."""
    return ReturnBases(equity=fetch_equity_closes(trading, first_day, today), benchmark=fetch_benchmark_closes(benchmark, first_day, today))


def load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
    """The daily P&L of the strategy's Alpaca account, fetched now."""
    trading_creds, data_creds = load_account_credentials(strategy, mode, env_dir)
    trading = cast(AccountTradingClient, build_trading_client(trading_creds))
    data = cast(AccountDataClient, build_stock_data_client(data_creds))
    return load_account_pnl(trading, data, today)


def load_strategy_return_bases(strategy: str, mode: str, env_dir: Path, first_day: date, today: date) -> ReturnBases:
    """The equity and benchmark closes the strategy's daily % are measured against, fetched now."""
    trading_creds, _ = load_account_credentials(strategy, mode, env_dir)
    trading = cast(AccountTradingClient, build_trading_client(trading_creds))
    # One Yahoo bar per session: enough of them to reach back a week before the first day.
    history = max(HISTORY_BARS, (today - first_day).days + _BASE_DAYS)
    return load_return_bases(trading, YahooDailyBars(history=history), first_day, today)
