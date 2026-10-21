"""One daily check: did a watchlist stock's insider filings change? If so research those stocks, then trade; if a trade is left over, re-check only the trading.

The agents decide; this module validates (through the `InsiderRecorder`), remembers (`StateStore`) and calls the `TradeDesk`.
The daily rule (see the design spec): the strategy ticks every day but acts only on a change, and code decides what changed, not an
agent. A stock changed when a P/S filing it did not know at the last COMPLETED run is now known, when a filing that was inside its
signal window then has left it, or on the first run. With no change and no pending trade no agent runs and no order is sent.
A changed stock with nothing in its window gets a note written by code; any other one is researched (one agent run per stock),
and the stocks that did not change reuse their stored note. The trader (one agent, two submits: the target, then the trade report)
then sees every stock's class. A trade the trading stage could not finish is kept in the state and re-checked on the following
days by running only the trading stage against the stored target, at most `max_trade_retries` times; a change replaces it. A stage
that never yields a valid submission abandons the run: nothing is traded, the filings stay "new" and the streak goes up.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.trade_desk import Audit, TradeDesk
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.form4.classify import Label, Labelled
from trading_agent_framework.strategies.insider_trades.form4.signal import InsiderSummary, Override, SignalClass, StockFacts, apply_overrides, signal_class, summarise
from trading_agent_framework.strategies.insider_trades.form4.source import KnownInsiderData, LoadReport
from trading_agent_framework.strategies.insider_trades.handoff import InsiderRecorder
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.strategies.insider_trades.state import InsiderState, PendingTrade, StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, InsiderDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

AGENT_RESEARCHER = "researcher"
AGENT_TRADER = "trader"
AGENT_TRADER_RETRY = "trader_retry"


class SourceLike(Protocol):
    """What the pipeline needs from the Form 4 source (`Form4Source`, or a fake in tests)."""

    @property
    def loaded_on(self) -> date | None: ...

    def load(self, *, as_of: datetime, since: date) -> LoadReport: ...

    def known(self, as_of: datetime) -> KnownInsiderData: ...


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...


class AgentLookup(Protocol):
    """What the pipeline needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike: ...


class RunAbandoned(Exception):
    """A stage could not produce what the run needs; the run ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class RunOutcome:
    completed: bool
    abandoned_streak: int  # abandoned runs in a row after this one (unchanged by a quiet day, 0 after a completed run)


@dataclass(frozen=True, slots=True)
class _Stock:
    """One watchlist stock as the trader sees it: its note (researched, stored or written by code) and the class that follows from it."""

    ticker: str
    signal: SignalClass
    summary: str
    overrides: tuple[Override, ...]
    as_of: str  # ISO date the note was written
    changed: bool
    stored: bool = True  # False for the placeholder of a stock that failed to load and has no note yet: it is not kept in the state


class InsiderPipeline:
    def __init__(
        self,
        *,
        strategy: Strategy,
        params: InsiderParams,
        source: SourceLike,
        facts: InsiderFacts,
        agents: AgentLookup,
        recorder: InsiderRecorder,
        state: StateStore,
        run_log: RunLog,
        desk: TradeDesk,
        load_window: Callable[[], tuple[datetime, date]],
        researcher_task: str,
        trader_task: str,
        trader_retry_task: str,
        retry_prompt: Callable[[str, str], str],
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._source = source
        self._facts = facts
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = run_log
        self._desk = desk
        self._load_window = load_window
        self._researcher_task = researcher_task
        self._trader_task = trader_task
        self._trader_retry_task = trader_retry_task
        self._retry_prompt = retry_prompt
        self._watchlist = [ticker.strip().upper() for ticker in params.watchlist]
        self._phase = "filings"  # where a failure is attributed: filings, researcher, trading
        self._pending_retry = False

    # --- one daily check ---------------------------------------------------------------------------

    def run(self) -> RunOutcome:
        state = self._state.load()
        now = self._strategy.clock.now().astimezone(MARKET_TZ)
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "outcome": None}
        self._phase, self._pending_retry = "filings", False
        try:
            return self._run(now, state, record)
        except RunAbandoned as exc:
            stage, error = exc.stage, exc
        except InsiderDataError as exc:
            stage, error = "filings", exc
        except (BrokerError, BacktestError) as exc:
            stage, error = ("execution" if self._phase == "trading" else "broker"), exc
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"[insider_trades] run abandoned at the {stage} stage (streak {streak}): {error}")
        pending = state.pending_trade
        if self._pending_retry and pending is not None:
            pending = replace(pending, days=pending.days + 1)  # an attempt that fails still uses up one of the retries
        traded = sorted({*state.traded, *self._desk.traded})
        self._state.save(replace(state, abandoned_streak=streak, pending_trade=pending, traded=traded))
        orders = self._orders_record() if self._phase == "trading" else []
        self._log.append({**record, "outcome": "abandoned", "stage": stage, "error": str(error), "abandoned_streak": streak, "orders": orders})
        return RunOutcome(completed=False, abandoned_streak=streak)

    def _run(self, now: datetime, state: InsiderState, record: dict[str, Any]) -> RunOutcome:
        strategy = self._strategy
        today = now.date()
        failed = self._refresh(today)
        known = self._facts.known()
        current = {ticker: {f.accession for f in known.filings.get(ticker, []) if f.trades} for ticker in self._watchlist if ticker not in failed}
        changed = [ticker for ticker in self._watchlist if ticker in current and self._changed(ticker, current[ticker], state)]
        record["failed"] = failed
        record["changed"] = changed
        if changed:
            return self._full_run(now, state, current, changed, failed, record)
        if state.pending_trade is None:
            strategy.log_info("[insider_trades] nothing new: no change in the insider filings since the last run, so nothing to do")
            self._log.append({**record, "outcome": "nothing_new"})
            return RunOutcome(completed=True, abandoned_streak=state.abandoned_streak)
        return self._retry_pending(now, state, state.pending_trade, record)

    # --- the filings --------------------------------------------------------------------------------

    def _refresh(self, today: date) -> dict[str, str]:
        """Load the Form 4s when this tick needs it; the tickers that failed to load, with why.

        A backtest loads once for the whole run and again only while a ticker is missing; one missing ticker abandons the run (a
        half-read history never labels a trade), and the next tick tries again. Paper and live load once per market date and ride
        out a failed ticker (it is skipped); only every ticker failing abandons the run.
        """
        strategy = self._strategy
        # Backtest: load again only while a ticker is missing. Paper/live: load once per market date.
        if strategy.is_backtesting:
            # Read the source itself: `InsiderFacts` caches the first read of a market date, which must not happen before the load.
            needed = self._source.loaded_on is None or bool(self._source.known(strategy.clock.now()).failed)
        else:
            needed = self._source.loaded_on != today
        # Nothing to load: the failed tickers of the last read stand for this tick too.
        if not needed:
            return dict(self._source.known(strategy.clock.now()).failed)
        # Load the window this run needs (see `load_window()` in agent_insider_trades.py).
        as_of, since = self._load_window()
        report = self._source.load(as_of=as_of, since=since)
        for ticker, error in report.failed.items():
            strategy.log_warning(f"[insider_trades] {ticker}: Form 4 data could not be loaded: {error}")
        # A backtest needs every ticker: a history with a gap would mislabel trades.
        if report.failed and strategy.is_backtesting:
            raise RunAbandoned("filings", f"Form 4 data could not be loaded for {', '.join(report.failed)}: {next(iter(report.failed.values()))}")
        # Paper/live skip the failed tickers, but with none loaded there is nothing to trade on.
        if not report.loaded:
            raise RunAbandoned("filings", f"Form 4 data could not be loaded for any ticker: {next(iter(report.failed.values()), 'the watchlist is empty')}")
        # Log what was kept and what the load set aside, so a run shows why a filing did not count.
        filings = [f for read in self._source.known(strategy.clock.now()).filings.values() for f in read]
        strategy.log_info(
            f"[insider_trades] loaded {len(report.loaded)} tickers: {len(filings)} Form 4 filings, {sum(len(f.trades) for f in filings)} trades kept; "
            f"set aside: {report.ignored_forms} 4/A or Form 5, {report.other_issuer} filings of another issuer, "
            f"{sum(f.non_insider for f in filings)} rows of non-officer/director owners, {sum(f.other_codes for f in filings)} other transaction codes, "
            f"{sum(f.unpriced for f in filings)} unpriced rows"
        )
        # The failed tickers are skipped by the caller (`_run` leaves them out of the change check).
        return dict(report.failed)

    def _changed(self, ticker: str, current: set[str], state: InsiderState) -> bool:
        """A stock changed when a P/S filing is new to the last completed run, a filing it then had in its window has left it, or there is no note to reuse."""
        if state.last_run is None or self._stored(ticker, state) is None:
            return True
        window = self._facts.facts(ticker).accessions
        return bool(current - set(state.processed)) or bool(set(state.in_window.get(ticker, [])) - window)

    # --- a pending trade: only the trading stage ---------------------------------------------------------

    def _retry_pending(self, now: datetime, state: InsiderState, pending: PendingTrade, record: dict[str, Any]) -> RunOutcome:
        strategy, params = self._strategy, self._params
        if pending.days >= params.max_trade_retries:
            strategy.log_warning(f"[insider_trades] giving up on the unfinished trade after {pending.days} later checks: the target was not reached")
            self._state.save(replace(state, pending_trade=None))
            self._log.append({**record, "outcome": "pending_gave_up", "target": pending.target})
            return RunOutcome(completed=True, abandoned_streak=state.abandoned_streak)
        strategy.log_info(f"[trader] checking the unfinished trade again (check {pending.days + 1} of {params.max_trade_retries})")
        self._phase, self._pending_retry = "trading", True
        audit = self._retry_trade(pending.target, state, record)
        remaining = None if audit.complete else PendingTrade(target=pending.target, days=pending.days + 1)
        self._state.save(replace(state, traded=self._desk.traded, pending_trade=remaining, abandoned_streak=0, last_run=now.date().isoformat()))
        self._log.append({**record, "outcome": "pending_complete" if audit.complete else "pending_retry", "target": pending.target})
        return RunOutcome(completed=True, abandoned_streak=0)

    def _retry_trade(self, target: Mapping[str, float], state: InsiderState, record: dict[str, Any]) -> Audit:
        """Run the retry agent against the stored `target`; returns the audit afterwards (complete = nothing unfilled, nothing off target)."""
        strategy, desk = self._strategy, self._desk
        desk.begin_run(target, state.traded)
        before = desk.audit()
        record["shortfalls_before"] = [asdict(s) for s in before.shortfalls]
        if not before.shortfalls:
            if before.in_flight:
                strategy.log_info(f"[trader] the account is at the target counting orders still working ({', '.join(before.in_flight)}): no new order, checked again later")
            else:
                strategy.log_info("[trader] the account is already at the target: no order needed")
            return before
        self._recorder.expect_report(desk.orders_view)
        context = {
            "current_datetime": strategy.clock.now().astimezone(MARKET_TZ).isoformat(),
            "target": [{"ticker": ticker, "weight": weight} for ticker, weight in sorted(target.items())],
            "shortfalls": _shortfalls(before),
        }
        try:
            self._run_stage(AGENT_TRADER_RETRY, self._trader_retry_task, context, stage=AGENT_TRADER)
        except RunAbandoned as exc:
            if not desk.orders:
                raise  # nothing was sent: abandon the whole run
            # Orders went out but were never reported: they are checked again on later days.
            strategy.log_error(f"[trader] no valid trade report, but {len(desk.orders)} order(s) were sent and will be checked again later: {exc}")
        return self._audited(record)

    # --- a change: research the changed stocks, then trade ---------------------------------------------

    def _full_run(self, now: datetime, state: InsiderState, current: dict[str, set[str]], changed: list[str], failed: dict[str, str], record: dict[str, Any]) -> RunOutcome:
        strategy = self._strategy
        today = now.date()
        strategy.log_info(f"[insider_trades] {len(changed)} stock(s) changed: {', '.join(changed)}")

        # 1. Researcher: one run per changed stock that has insider trades in its window.
        self._phase = "researcher"
        stocks = [self._stock(ticker, now, state, changed, failed) for ticker in self._watchlist]
        record["stocks"] = [{"ticker": s.ticker, "class": s.signal.value, "changed": s.changed, "summary": s.summary} for s in stocks]

        # 2. Trader: the target, then the orders and the check that they filled.
        self._phase, self._pending_retry = "trading", False
        target = self._trade(now, state, stocks, failed, record)
        audit = self._audited(record)

        kept = {ticker: state.in_window[ticker] for ticker in failed if ticker in state.in_window}
        self._state.save(
            InsiderState(
                processed=sorted(set(state.processed).union(*current.values())),
                in_window={**kept, **{t: sorted(self._facts.facts(t).accessions) for t in current}},
                notes={s.ticker: _note_record(s) for s in stocks if s.stored},
                target=target,
                traded=self._desk.traded,
                pending_trade=None if audit.complete else PendingTrade(target=target, days=0),
                abandoned_streak=0,
                last_run=today.isoformat(),
            )
        )
        self._log.append({**record, "outcome": "completed" if audit.complete else "completed_pending"})
        return RunOutcome(completed=True, abandoned_streak=0)

    def _stock(self, ticker: str, now: datetime, state: InsiderState, changed: list[str], failed: dict[str, str]) -> _Stock:
        """The stock's note: researched when it changed, written by code when nothing is in its window, else the stored one."""
        today = now.date()
        stored = self._stored(ticker, state)
        if ticker not in changed:
            if stored is not None:
                return stored
            # a stock that failed to load with no note yet: placeholder for the trader, never kept in the state
            return _Stock(ticker, SignalClass.NONE, f"Form 4 data could not be loaded ({failed.get(ticker, 'unknown error')}): no signal is known", (), today.isoformat(), changed=False, stored=False)
        window = self._facts.window(ticker)
        if not window:
            n = self._params.signal_window_days
            return _Stock(ticker, SignalClass.NONE, f"no open-market insider trade in the last {n} days", (), today.isoformat(), changed=True)
        facts = self._facts.facts(ticker)
        self._recorder.expect_note(facts)
        previous = state.notes.get(ticker)
        context = {"current_datetime": now.isoformat(), "ticker": ticker, "window_days": self._params.signal_window_days, "previous_note": previous}
        self._run_stage(AGENT_RESEARCHER, self._researcher_task, context)
        note = self._recorder.note
        assert note is not None  # `submitted` is true only with a recorded note
        signal = self._signal(ticker, window, note.overrides, today)
        self._strategy.log_info(f"[researcher] {ticker}: {signal.value}: {note.summary}")
        for override in note.overrides:
            self._strategy.log_info(f"[researcher]   override {override.insider_id}: {override.from_label.value} -> {override.to_label.value}: {override.reason}")
        return _Stock(ticker, signal, note.summary, note.overrides, today.isoformat(), changed=True)

    def _stored(self, ticker: str, state: InsiderState) -> _Stock | None:
        """The stock's stored note as a `_Stock`; None when there is none or it cannot be read back (the stock is then researched again)."""
        note = state.notes.get(ticker)
        if note is None:
            return None
        try:
            overrides = tuple(Override(o["insider_id"], Label(o["from_label"]), Label(o["to_label"]), o["reason"]) for o in note["overrides"])
            return _Stock(ticker, SignalClass(note["class"]), note["summary"], overrides, note["as_of"], changed=False)
        except ValueError, KeyError:
            return None

    def _summary_of(self, ticker: str, window: list[Labelled], overrides: tuple[Override, ...], today: date) -> StockFacts:
        start = today - timedelta(days=self._params.signal_window_days)
        return summarise(ticker, apply_overrides(window, overrides), today=today, window_start=start)

    def _signal(self, ticker: str, window: list[Labelled], overrides: tuple[Override, ...], today: date) -> SignalClass:
        params = self._params
        return signal_class(
            self._summary_of(ticker, window, overrides, today),
            min_buy_fraction=Decimal(str(params.min_buy_holding_fraction)),
            min_sell_fraction=Decimal(str(params.min_sell_holding_fraction)),
        )

    # --- the trading stage ---------------------------------------------------------------------------------

    def _trade(self, now: datetime, state: InsiderState, stocks: list[_Stock], failed: dict[str, str], record: dict[str, Any]) -> dict[str, float]:
        """Run the trader on every stock's class: it submits the target, trades, and reports; returns the target it recorded."""
        strategy, desk, recorder, params = self._strategy, self._desk, self._recorder, self._params
        desk.begin_run(None, state.traded)
        recorder.expect_trade({s.ticker: s.signal for s in stocks}, self._on_target, desk.orders_view)
        weights = self._current_weights([s.ticker for s in stocks])
        context = {
            "current_datetime": now.isoformat(),
            "stocks": [self._stock_context(s, state, weights[s.ticker], failed) for s in stocks],
            "constraints": {
                "min_weight": params.min_weight,
                "max_weight": params.max_position_weight,
                "max_total_weight": params.max_total_weight,
                "even_weight": round(params.even_weight, 4),
            },
        }
        try:
            self._run_stage(AGENT_TRADER, self._trader_task, context)
        except RunAbandoned as exc:
            if recorder.target is None:
                raise  # no valid target: nothing was traded, abandon the whole run
            # The target is in (so orders may be too) but the report never was: the audit decides what is left to check.
            strategy.log_error(f"[trader] no valid trade report: {exc}")
        assert recorder.target is not None  # a report is only armed after a valid target
        target = {t.ticker: t.weight for t in recorder.target if t.weight > 0}
        record["target"] = [asdict(t) for t in recorder.target]
        strategy.log_info(f"[trader] target: {len(target)} positions, {sum(target.values()):.1%} invested")
        for position in recorder.target:
            strategy.log_info(f"[trader]   {position.ticker} {position.weight:.1%}" + (f": {position.reason}" if position.reason else ""))
        return target

    def _on_target(self, weights: dict[str, float]) -> dict[str, Any]:
        """What `submit_target` answers once the target is valid: the desk now knows it, and the trader gets what is off it."""
        self._desk.set_target(weights)
        return {"shortfalls": _shortfalls(self._desk.audit())}

    def _audited(self, record: dict[str, Any]) -> Audit:
        """The desk's audit after the trading stage, with the orders and what is still off target logged."""
        strategy = self._strategy
        after = self._desk.audit()
        record["orders"] = self._orders_record()
        record["shortfalls_after"] = [asdict(s) for s in after.shortfalls]
        for view in after.unfilled:
            strategy.log_warning(f"[trader] {view.side} {view.quantity:g} {view.symbol} is {view.status}: it will be checked again")
        for shortfall in after.shortfalls:
            strategy.log_warning(f"[trader] off target: {shortfall.detail}")
        return after

    def _stock_context(self, stock: _Stock, state: InsiderState, current_weight: float | None, failed: dict[str, str]) -> dict[str, Any]:
        insiders: list[InsiderSummary] = []
        if stock.ticker not in failed:
            today = self._facts.today()
            insiders = list(self._summary_of(stock.ticker, self._facts.window(stock.ticker), stock.overrides, today).insiders)
        return {
            "ticker": stock.ticker,
            "class": stock.signal.value,
            "summary": stock.summary,
            "insiders": [
                {
                    "name": i.name,
                    "role": i.role,
                    "opportunistic_bought": _dollars(i.bought.get(Label.OPPORTUNISTIC, Decimal(0))),
                    "opportunistic_sold": _dollars(i.sold.get(Label.OPPORTUNISTIC, Decimal(0))),
                    "max_buy_fraction": round(float(i.max_buy_fraction), 4),
                    "sell_fraction": round(float(i.sell_fraction), 4),
                }
                for i in insiders
            ],
            "current_weight": current_weight,
            "previous_weight": state.target.get(stock.ticker),
        }

    def _current_weights(self, tickers: list[str]) -> dict[str, float | None]:
        """Each stock's share of the portfolio value now (long positions only); None when the account or the price cannot be read."""
        strategy = self._strategy
        try:
            value = strategy.broker.get_account().portfolio_value
            held = {p.asset.symbol: p.quantity for p in strategy.get_positions() if p.side is PositionSide.LONG and p.quantity > 0}
        except (BrokerError, BacktestError) as exc:
            strategy.log_warning(f"[trader] current weights are unavailable: {exc}")
            return dict.fromkeys(tickers)
        if value <= 0:
            return dict.fromkeys(tickers)
        weights: dict[str, float | None] = {}
        for ticker in tickers:
            quantity = held.get(ticker)
            if quantity is None:
                weights[ticker] = 0.0
                continue
            try:
                price = strategy.get_last_price(ticker)
            except (BrokerError, BacktestError) as exc:
                strategy.log_warning(f"[trader] no price for {ticker}: {exc}")
                weights[ticker] = None
                continue
            weights[ticker] = round(float(quantity * price / value), 4) if price is not None else None
        return weights

    # --- helpers -------------------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, task: str, context: dict[str, Any], *, stage: str | None = None) -> None:
        """Run an agent until the armed submission is complete: once, then once more forcing the tool still missing; else abandon."""
        recorder = self._recorder
        agent = self._agents[agent_name]
        run_id = uuid.uuid4().hex
        prompt, force_tool, error = task, None, ""
        for _ in range(2):
            recorder.last_error, error = None, ""  # an error from the first attempt must not be reported for the second
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool)
                self._strategy.log_info(f"[{agent_name}] {result.output}")
                for i, tool_call in enumerate(result.tool_calls):
                    self._strategy.log_debug(f"[{agent_name}] tool_call_{i}: {tool_call}")
            except AgentError as exc:
                error = str(exc)
                self._strategy.log_error(f"[{agent_name}] run failed: {exc}")
            if recorder.submitted:
                return
            tool = recorder.missing_tool
            error = recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = self._retry_prompt(tool, error), tool
        raise RunAbandoned(stage or agent_name, error)

    def _orders_record(self) -> list[dict[str, Any]]:
        return [asdict(view) for view in self._desk.orders_view().values()]


def _shortfalls(audit: Audit) -> list[dict[str, Any]]:
    return [{"symbol": s.symbol, "kind": s.kind, "dollars": round(s.value), "detail": s.detail} for s in audit.shortfalls]


def _dollars(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _note_record(stock: _Stock) -> dict[str, Any]:
    return {
        "summary": stock.summary,
        "class": stock.signal.value,
        "overrides": [{"insider_id": o.insider_id, "from_label": o.from_label.value, "to_label": o.to_label.value, "reason": o.reason} for o in stock.overrides],
        "as_of": stock.as_of,
    }
