"""`InsiderFacts`: the known Form 4 data of the watchlist for the strategy's current market date, labelled and windowed.

One `source.known(strategy.clock.now())` call per market date serves every tool and stage. The clock gate is applied here on top of
the source's own: a filing whose `filed` is not strictly before the market date of `strategy.clock.now()` is invisible, whatever the
source returned (the same rule as every data tool: the cutoff comes from `strategy.clock`, never the wall clock).

Labels come from the code (`label_trades`): every attributed trade of every known filing of a ticker is both the trades to label and
the history they are labelled against, so a trade is labelled per its own year. A ticker whose last load failed raises
`InsiderDataError` here, naming the ticker; a ticker the source knows nothing about simply has no filings.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from trading_agent_framework.strategies.insider_trades.form4.classify import Labelled, label_trades
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing
from trading_agent_framework.strategies.insider_trades.form4.signal import StockFacts, summarise, window_trades
from trading_agent_framework.strategies.insider_trades.form4.source import KnownInsiderData
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import InsiderDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class KnownSource(Protocol):
    def known(self, as_of: datetime) -> KnownInsiderData: ...


class InsiderFacts:
    def __init__(self, strategy: Strategy, source: KnownSource, params: InsiderParams) -> None:
        self.strategy = strategy
        self.source = source
        self.params = params
        self._cached: tuple[date, KnownInsiderData] | None = None

    def today(self) -> date:
        return self.strategy.clock.now().astimezone(MARKET_TZ).date()

    def known(self) -> KnownInsiderData:
        """The filings filed strictly before today, as the source knows them: one source call per market date."""
        now = self.strategy.clock.now()
        today = now.astimezone(MARKET_TZ).date()
        if self._cached is None or self._cached[0] != today:
            raw = self.source.known(now)
            filings = {ticker: [f for f in rows if f.filed < today] for ticker, rows in raw.filings.items()}
            self._cached = (today, KnownInsiderData(filings, dict(raw.failed)))
        return self._cached[1]

    def _filings(self, ticker: str) -> list[Form4Filing]:
        known = self.known()
        if ticker in known.failed:
            raise InsiderDataError(f"{ticker}: Form 4 data could not be loaded: {known.failed[ticker]}")
        return known.filings.get(ticker, [])

    def labelled(self, ticker: str) -> list[Labelled]:
        """Every known trade of the ticker, labelled against all the known trades of the ticker."""
        trades = [trade for filing in self._filings(ticker) for trade in filing.trades]
        return label_trades(trades, trades, years=self.params.history_years)

    def window(self, ticker: str) -> list[Labelled]:
        return window_trades(self.labelled(ticker), today=self.today(), window_days=self.params.signal_window_days)

    def facts(self, ticker: str) -> StockFacts:
        today = self.today()
        window_start = today - timedelta(days=self.params.signal_window_days)
        return summarise(ticker, self.window(ticker), today=today, window_start=window_start)

    def filing(self, accession: str) -> Form4Filing | None:
        """A known filing by accession number, from any watchlist ticker; None when it is unknown or not yet filed."""
        for filings in self.known().filings.values():
            for filing in filings:
                if filing.accession == accession:
                    return filing
        return None
