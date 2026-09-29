"""Session preparation (stage 1) and the per-tick scan (stage 2, setups, headlines), spec §2-§3.

The I/O side of candidate selection: bars and news come in through the strategy and broker, go through the
pure `features`/`screening`/`setups` modules, and land in `SessionState`. Price reads go through the strategy
(and so, in a backtest, through `BacktestBroker._source_bars`, the no-look-ahead gate); news is cut at the
strategy clock.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.vwap_pullback.features import (
    BarStamp,
    cumulative_volume_by_minute,
    intraday_contexts,
    minute_starts,
    rvol_baseline,
    session_slice,
)
from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.screening import daily_profile, rank_stage2, select_stage1, snapshot_from
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState, advance
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

Preload = Callable[[Sequence[Asset], str], None]

_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_SESSION_MINUTES = 390
_NEWS_LOOKBACK = timedelta(hours=18)  # from the open back to roughly the previous close
_EMPTY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz=MARKET_TZ))


class Scanner:
    def __init__(self, strategy: Strategy, params: VwapPullbackParameters, universe: Sequence[str], *, benchmark: str = "SPY", preload: Preload | None = None) -> None:
        self._strategy = strategy
        self._params = params
        self._universe = list(universe)
        self._benchmark = benchmark
        self._preload = preload

    @property
    def bar_stamp(self) -> BarStamp:
        return "close" if self._strategy.is_backtesting else "open"

    # --- stage 1 -------------------------------------------------------------------

    def prepare_session(self) -> SessionState:
        """Stage 1 for the next (or current) session: candidates, their beta and ATR, and their RVOL baselines."""
        strategy = self._strategy
        session = strategy.clock.next_session()
        if session is None:
            raise BrokerError("no upcoming market session to prepare")
        day = session.open.astimezone(MARKET_TZ).date()
        bench = Asset(self._benchmark)
        universe = [Asset(symbol) for symbol in self._universe]
        if self._preload is not None:
            self._preload([*universe, bench], "day")
        daily = strategy.get_historical_prices_for_assets([*universe, bench], self._params.stage1_lookback_sessions + 1, "day")
        bench_daily = _before(daily.get(bench), day)
        profiles = []
        for asset in universe:
            frame = _before(daily.get(asset), day)
            profile = daily_profile(asset.symbol, frame, bench_daily, self._params) if frame is not None else None
            if profile is not None:
                profiles.append(profile)
        chosen = select_stage1(profiles, self._params)
        state = SessionState(day=day, session=session, bar_stamp=self.bar_stamp, session_open_equity=strategy.get_portfolio_value())
        state.candidates = {p.symbol: CandidateInfo(symbol=p.symbol, daily_atr=p.daily_atr, beta=p.beta) for p in chosen}
        state.baselines = self._baselines([Asset(p.symbol) for p in chosen], day)
        strategy.log_info(f"stage 1 for {day}: {len(chosen)} candidates out of {len(profiles)} profiled symbols")
        return state

    def _baselines(self, assets: Sequence[Asset], day: date) -> dict[str, pd.Series]:
        if not assets:
            return {}
        if self._preload is not None:
            self._preload([*assets, Asset(self._benchmark)], "minute")
        length = (self._params.rvol_baseline_sessions + 1) * _SESSION_MINUTES
        bars = self._strategy.get_historical_prices_for_assets(assets, length, "minute", include_after_hours=False)
        baselines: dict[str, pd.Series] = {}
        for asset in assets:
            found = bars.get(asset)
            if found is not None and not found.df.empty:
                baseline = self._baseline(found.df, day)
                if not baseline.empty:
                    baselines[asset.symbol] = baseline
        return baselines

    def _baseline(self, df: pd.DataFrame, day: date) -> pd.Series:
        starts = minute_starts(df.index, self.bar_stamp).tz_convert(MARKET_TZ)
        dates = sorted({d for d in starts.date if d < day})[-self._params.rvol_baseline_sessions :]
        per_session = []
        for session_day in dates:
            open_at = datetime.combine(session_day, _REGULAR_OPEN, tzinfo=MARKET_TZ)
            rows = session_slice(df, open_at, datetime.combine(session_day, _REGULAR_CLOSE, tzinfo=MARKET_TZ), self.bar_stamp)
            if not rows.empty:
                per_session.append(cumulative_volume_by_minute(rows, open_at, self.bar_stamp))
        return rvol_baseline(per_session)

    # --- stage 2 and setups -----------------------------------------------------------

    def scan(self, state: SessionState) -> None:
        """One tick: contexts for every candidate, stage-2 ranking, setups advanced, headlines refreshed."""
        now = self._strategy.get_datetime()
        bench = Asset(self._benchmark)
        minutes_open = int((now - state.session.open).total_seconds() // 60)
        length = max(10, min(minutes_open + 5, _SESSION_MINUTES + 10))
        bars = self._strategy.get_historical_prices_for_assets([*(Asset(s) for s in state.candidates), bench], length, "minute", include_after_hours=False)
        bench_df = self._session_frame(bars.get(bench), state)
        snapshots = []
        for symbol, info in state.candidates.items():
            contexts = intraday_contexts(
                self._session_frame(bars.get(Asset(symbol)), state), bench_df,
                session_open=state.session.open, now=now, bar_stamp=state.bar_stamp, beta=info.beta,
                baseline=state.baselines.get(symbol, pd.Series(dtype=float)), minutes=self._params.bar_minutes,
            )
            state.contexts[symbol] = contexts
            snapshot = snapshot_from(symbol, contexts)
            if snapshot is not None:
                snapshots.append(snapshot)
        sticky = {symbol for symbol, setup in state.setups.items() if setup.state is not SetupState.WATCH}
        ranked = rank_stage2(snapshots, self._params, sticky)
        for candidate in ranked:
            info = state.candidates[candidate.symbol]
            info.composite, info.z_rs, info.z_rvol = candidate.composite, candidate.z_rs, candidate.z_rvol
        tracked = {candidate.symbol for candidate in ranked}
        for symbol in [s for s, setup in state.setups.items() if s not in tracked and setup.state is SetupState.WATCH]:
            del state.setups[symbol]
        for symbol in sorted(tracked):
            setup = state.setups.get(symbol, Setup(symbol=symbol))
            state.setups[symbol] = advance(setup, state.contexts.get(symbol, []), state.candidates[symbol].daily_atr, self._params)
        self.refresh_headlines(state, now)

    @staticmethod
    def _session_frame(bars: Bars | None, state: SessionState) -> pd.DataFrame:
        if bars is None or bars.df.empty:
            return _EMPTY
        return session_slice(bars.df, state.session.open, state.session.close, state.bar_stamp)

    def refresh_headlines(self, state: SessionState, now: datetime) -> None:
        """Headlines for pullback/triggered setups and open trades, at most once per `exit_review_minutes` per symbol.

        A headline that was not in the previous fetch marks the symbol in `state.new_headline` (an exit-review
        event); a symbol's first fetch is its baseline, not news.
        """
        wanted = {s for s, setup in state.setups.items() if setup.state in (SetupState.PULLBACK, SetupState.TRIGGERED)}
        wanted |= {trade.symbol for trade in state.book.open_trades()}
        if not wanted:
            return
        try:
            provider = self._strategy.broker.news_provider()
        except BrokerError as exc:
            self._strategy.log_warning(f"no news provider, setups go without headlines: {exc}")
            return
        if provider is None:
            return
        refresh = timedelta(minutes=self._params.exit_review_minutes)
        since = state.session.open - _NEWS_LOOKBACK
        for symbol in sorted(wanted):
            fetched = state.headlines_fetched_at.get(symbol)
            if fetched is not None and now - fetched < refresh:
                continue
            try:
                articles = provider.get_news([symbol], start=since, end=now, limit=self._params.headlines_per_symbol * 3)
            except BrokerError as exc:
                self._strategy.log_warning(f"news for {symbol} unavailable: {exc}")
                continue
            rows = lean_headlines(articles, self._params.headlines_per_symbol)
            previous = state.headlines.get(symbol)
            if previous is not None and {r["headline"] for r in rows} - {r["headline"] for r in previous}:
                state.new_headline.add(symbol)
            state.headlines[symbol] = rows
            state.headlines_fetched_at[symbol] = now


def _before(bars: Bars | None, day: date) -> pd.DataFrame | None:
    """Only the daily rows dated before `day`: a restart mid-session must not read today's forming daily bar."""
    if bars is None or bars.df.empty:
        return None
    df = bars.df
    dates = df.index.tz_convert(MARKET_TZ).date
    return df[dates < day]
