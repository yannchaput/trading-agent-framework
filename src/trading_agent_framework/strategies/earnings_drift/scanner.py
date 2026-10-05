# src/trading_agent_framework/strategies/earnings_drift/scanner.py
"""Today's candidates: earnings events → surprise → reaction → gates (spec §3.6).

Runs once per cycle, right after the close. Reads SEC through an `EventProvider`, news through the broker's
`NewsProvider` (cut at `strategy.clock.now()`), daily bars through `strategy.get_historical_prices_for_assets`
(in a backtest, the no-look-ahead gate). Never orders anything.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.earnings_drift.event_source import EventProvider
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent, events_reacting_on
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.reaction import bar_dates, reaction_features
from trading_agent_framework.strategies.earnings_drift.screening import Candidate, gate
from trading_agent_framework.strategies.earnings_drift.surprise import article_time, articles_for, pick_surprise
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

_MAX_HEADLINES = 5


@dataclass
class ScanResult:
    today: date
    trading_dates: list[date]
    candidates: list[Candidate] = field(default_factory=list)  # best abnormal return first
    rejections: dict[str, str] = field(default_factory=dict)  # symbol -> reason
    hollow: bool = False


class Scanner:
    def __init__(self, strategy: Strategy, params: DriftParams, universe: Sequence[str], source: EventProvider, *, benchmark: str = "SPY") -> None:
        self._strategy = strategy
        self._params = params
        self._universe = [symbol.upper() for symbol in universe]
        self._source = source
        self._benchmark = benchmark

    def prepare(self, held: Collection[str]) -> ScanResult:
        strategy, params = self._strategy, self._params
        now = strategy.clock.now()
        today = now.astimezone(MARKET_TZ).date()
        benchmark = strategy.get_historical_prices(self._benchmark, params.bars_lookback_sessions, "day")
        frame = benchmark.df if benchmark is not None else pd.DataFrame()
        trading_dates = sorted({d for d in (bar_dates(frame) if not frame.empty else []) if d <= today})
        result = ScanResult(today=today, trading_dates=trading_dates)
        if today not in trading_dates:
            strategy.log_warning(f"[earnings_drift] {self._benchmark} has no bar for {today}: no candidates this cycle")
            result.trading_dates = [*trading_dates, today]  # today still counts as a session for the holdings
            return result
        if self._source.needs_load():
            report = self._source.load(self._universe, as_of=now, since=today - timedelta(days=params.event_lookback_days))
            if report.failed:
                sample = ", ".join(sorted(report.failed)[:5])
                strategy.log_warning(f"[earnings_drift] SEC submissions failed for {len(report.failed)} of {len(self._universe)} symbols (e.g. {sample})")
            if report.is_hollow(params.sec_hollow_fraction, params.sec_hollow_min_failures):
                self._source.discard()
                strategy.log_error(f"[earnings_drift] hollow scan: SEC failed for {len(report.failed)} symbols; no candidates this cycle")
                result.hollow = True
                return result
        events = events_reacting_on(self._source.all_events(), today, trading_dates, now)
        if not events:
            strategy.log_info(f"[earnings_drift] no earnings reactions on {today}")
            return result
        news = self._news(events, now)
        bars = self._bars([event.symbol for event in events])
        held_symbols = {symbol.upper() for symbol in held}
        for event in events:
            reason, candidate = self._judge(event, today, now, news, bars, frame, held_symbols)
            if candidate is not None:
                result.candidates.append(candidate)
            elif reason is not None:
                result.rejections[event.symbol] = reason
        result.candidates.sort(key=lambda c: c.reaction.abnormal_pct, reverse=True)
        counts: dict[str, int] = {}
        for reason in result.rejections.values():
            counts[reason] = counts.get(reason, 0) + 1
        strategy.log_info(f"[earnings_drift] {today}: {len(events)} earnings reactions, {len(result.candidates)} candidates, rejected {counts}")
        return result

    def _judge(
        self,
        event: EarningsEvent,
        today: date,
        now: datetime,
        news: Mapping[str, list[Mapping[str, Any]]] | None,
        bars: Mapping[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        held: set[str],
    ) -> tuple[str | None, Candidate | None]:
        if event.symbol in held:
            return "already_held", None
        if news is None or event.symbol not in news:
            return "no_news", None
        lookback = timedelta(hours=self._params.surprise_lookback_hours)
        window = articles_for(news[event.symbol], event.symbol, start=event.accepted_at - lookback, end=now)
        picked = pick_surprise(window)
        frame = bars.get(event.symbol)
        reaction = reaction_features(frame, benchmark, today, baseline_sessions=self._params.volume_baseline_sessions) if frame is not None else None
        reason = gate(picked.surprise if picked else None, reaction, self._params, held=False)
        if reason is not None or picked is None or reaction is None:
            return reason, None
        headlines = []
        for article in window[:_MAX_HEADLINES]:
            created = article_time(article)
            if created is not None:
                headlines.append((created.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M"), str(article.get("headline") or "")))
        return None, Candidate(event=event, reaction_day=today, surprise=picked, reaction=reaction, headlines=tuple(headlines))

    def _news(self, events: Sequence[EarningsEvent], now: datetime) -> dict[str, list[Mapping[str, Any]]] | None:
        """Articles per event symbol, read in chunks; a failed chunk's symbols are missing (`no_news`); None without a provider."""
        try:
            provider = self._strategy.broker.news_provider()
        except BrokerError as exc:
            self._strategy.log_warning(f"[earnings_drift] no news provider: {exc}")
            return None
        if provider is None:
            self._strategy.log_warning("[earnings_drift] this broker has no news provider: every event is rejected as no_news")
            return None
        start = min(event.accepted_at for event in events) - timedelta(hours=self._params.surprise_lookback_hours)
        symbols = sorted({event.symbol for event in events})
        found: dict[str, list[Mapping[str, Any]]] = {}
        size = self._params.news_symbols_per_call
        for first in range(0, len(symbols), size):
            chunk = symbols[first : first + size]
            try:
                articles = provider.get_news(chunk, start=start, end=now, limit=self._params.news_limit)
            except BrokerError as exc:
                self._strategy.log_warning(f"[earnings_drift] news failed for {', '.join(chunk)}: {exc}")
                continue
            for symbol in chunk:
                found[symbol] = list(articles)
        return found

    def _bars(self, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
        try:
            fetched = self._strategy.get_historical_prices_for_assets([Asset(symbol) for symbol in symbols], self._params.bars_lookback_sessions, "day")
        except (BrokerError, BacktestError) as exc:
            self._strategy.log_warning(f"[earnings_drift] daily bars failed for {len(symbols)} symbols: {exc}")
            return {}
        return {asset.symbol: bars.df for asset, bars in fetched.items() if bars is not None and not bars.df.empty}
