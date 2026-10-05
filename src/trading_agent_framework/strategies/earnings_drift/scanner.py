# src/trading_agent_framework/strategies/earnings_drift/scanner.py
"""Today's candidates: earnings events → surprise → reaction → gates (spec §3.6).

Runs once per cycle, right after the close. Reads SEC through an `EventProvider`, news through the broker's
`NewsProvider` (cut at `strategy.clock.now()`), daily bars through `strategy.get_historical_prices_for_assets`
(in a backtest, the no-look-ahead gate). Never orders anything.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    def __init__(
        self,
        strategy: Strategy,
        params: DriftParams,
        universe: Sequence[str],
        source: EventProvider,
        *,
        benchmark: str = "SPY",
        load_as_of: Callable[[], datetime | None] | None = None,
        volume_share: float = 1.0,
    ) -> None:
        """`load_as_of`: the SEC cache's freshness time (a backtest's end); None, or an answer of None, means `now`.

        `volume_share`: the bar feed's share of the consolidated volume (1.0 for SIP backtests, `live_volume_share`
        for IEX in paper/live); `min_dollar_volume` is scaled by it. `rel_volume` compares same-feed volumes: unscaled.
        """
        if not 0 < volume_share <= 1:
            raise ValueError(f"volume_share must be in (0, 1], got {volume_share}")
        self._strategy = strategy
        self._params = params
        self._gate_params = replace(params, min_dollar_volume=params.min_dollar_volume * volume_share)
        self._universe = [symbol.upper() for symbol in universe]
        self._source = source
        self._benchmark = benchmark
        self._load_as_of: Callable[[], datetime | None] = load_as_of or (lambda: None)

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
            # The freshness time only: a backtest loads once, so a cached file must be newer than the run's end.
            as_of = self._load_as_of() or now
            report = self._source.load(self._universe, as_of=as_of, since=today - timedelta(days=params.event_lookback_days))
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
        start, end = self._news_window(event, now)
        window = articles_for(news[event.symbol], event.symbol, start=start, end=end)
        picked = pick_surprise(window)
        frame = bars.get(event.symbol)
        reaction = reaction_features(frame, benchmark, today, baseline_sessions=self._params.volume_baseline_sessions) if frame is not None else None
        reason = gate(picked.surprise if picked else None, reaction, self._gate_params, held=False)
        if reason is not None or picked is None or reaction is None:
            return reason, None
        headlines = []
        for article in window[:_MAX_HEADLINES]:
            created = article_time(article)
            if created is not None:
                headlines.append((created.astimezone(MARKET_TZ).strftime("%Y-%m-%d %H:%M"), str(article.get("headline") or "")))
        return None, Candidate(event=event, reaction_day=today, surprise=picked, reaction=reaction, headlines=tuple(headlines))

    def _news_window(self, event: EarningsEvent, now: datetime) -> tuple[datetime, datetime]:
        """From `surprise_lookback_hours` before the release to `surprise_window_hours` after it, never beyond `now`."""
        start = event.accepted_at - timedelta(hours=self._params.surprise_lookback_hours)
        return start, min(now, event.accepted_at + timedelta(hours=self._params.surprise_window_hours))

    def _news(self, events: Sequence[EarningsEvent], now: datetime) -> dict[str, list[Mapping[str, Any]]] | None:
        """Articles per event symbol, one query each; a failed query's symbol is missing (`no_news`); None without a provider.

        One symbol and a narrow window per query: the provider answers the newest `news_limit` articles first, so a
        query shared by several symbols, or reaching far past the release, drops the oldest ones, and the earnings
        headline is among the first articles after the release.
        """
        try:
            provider = self._strategy.broker.news_provider()
        except BrokerError as exc:
            self._strategy.log_warning(f"[earnings_drift] no news provider: {exc}")
            return None
        if provider is None:
            self._strategy.log_warning("[earnings_drift] this broker has no news provider: every event is rejected as no_news")
            return None
        found: dict[str, list[Mapping[str, Any]]] = {}
        for event in events:
            start, end = self._news_window(event, now)
            try:
                found[event.symbol] = list(provider.get_news([event.symbol], start=start, end=end, limit=self._params.news_limit))
            except BrokerError as exc:
                self._strategy.log_warning(f"[earnings_drift] news failed for {event.symbol}: {exc}")
        return found

    def _bars(self, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
        try:
            fetched = self._strategy.get_historical_prices_for_assets([Asset(symbol) for symbol in symbols], self._params.bars_lookback_sessions, "day")
        except (BrokerError, BacktestError) as exc:
            self._strategy.log_warning(f"[earnings_drift] daily bars failed for {len(symbols)} symbols: {exc}")
            return {}
        return {asset.symbol: bars.df for asset, bars in fetched.items() if bars is not None and not bars.df.empty}
