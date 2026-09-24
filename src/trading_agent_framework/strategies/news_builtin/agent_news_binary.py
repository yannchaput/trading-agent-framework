"""News-builtin strategy: an LLM agent reads broad-market news and holds SPY/QQQ (bullish) or a defensive ETF.

Port of the lumibot `agent_alpaca_news_builtin` strategy onto this framework: `PrebuiltTools` (trading,
account, market data, indicators, memory) plus the `search_news` tool. News is gated on the strategy clock,
so it works in backtests as well as paper/live.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.agents.tools import PrebuiltTools
from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.news_builtin.grounding import require_search_news_before
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, FatalStrategyError

# A persistent LLM misconfiguration (wrong URL/model, dead server) would otherwise yield a flat "successful" backtest,
# so a backtest aborts (FatalStrategyError, which the executor propagates) after this many failed runs in a row.
MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS = 3


def _decision_was_recorded(tool_calls: Sequence[ToolCallRecord]) -> bool:
    """Whether a `remember_decision` call in `tool_calls` actually succeeded (no `"error"` in its result).

    Guards against a run ending with no decision recorded and no exception raised. Two distinct causes
    observed: (1) the model writes what should have been the call as JSON text in its final message
    instead of a real tool call -- `AgentManager`'s tool-call-repair middleware now recovers that case
    generically, before this ever sees it; (2) the model places a real trade via `submit_order` and then
    just writes a prose summary, skipping `remember_decision` entirely -- nothing to repair there (no
    call was attempted, real or malformed), so `on_trading_iteration` retries with a corrective follow-up
    turn instead. This check, and the warning it drives, stays as the backstop for whatever gets past
    both: a retry that itself fails to record anything.
    """
    return any(call.name == "remember_decision" and '"error"' not in call.result for call in tool_calls)


def _build_system_prompt(*, symbols: Sequence[str], defensive_symbol: str, news_symbols: str) -> str:
    allowed = ", ".join([*symbols, defensive_symbol])
    risky = " or ".join(symbols)
    return (
        f"You are a news-driven allocator. Your only allowed instruments are {allowed} "
        f"({defensive_symbol} is the defensive ETF). Never short, never use margin, never trade anything else, "
        "never trade USD or FOREX.\n\n"
        "On every run, follow this workflow:\n"
        "1. Call search_memory to recall recent decisions and the current regime thesis.\n"
        f"2. Scan broad-market news: call search_news with symbols='{news_symbols}', include_content=False and limit=30. "
        "This is required, not optional: remember_decision and submit_order are refused with an error until "
        "search_news has returned a result (even an empty one) in this run -- if either refuses that way, call "
        "search_news now, then retry.\n"
        "3. Pick the single most relevant article: among the broad scan's headlines, prefer the most recent one "
        "that is actually about the regime call (broad market, not a single-company story) -- even if an older "
        "headline reads as more dramatic. Only reach further back if nothing recent is on-topic. If the article "
        "you're about to pick already appears in a decision from search_memory, it is not new evidence: look for "
        "a fresher on-topic article instead, or treat this run as having no new signal. Call search_news again "
        "with a narrow start/end window around the chosen article's created_at (ISO 8601 with timezone), "
        "include_content=True and limit=3, to try to read it in full. This call is required, not optional: "
        "remember_decision and submit_order are refused with an error until search_news has been called with "
        "include_content=True in this run -- if either refuses that way, make that call now, then retry. But what "
        "matters is making the call, not what it returns: some articles (terse data-print/quote wires) never "
        "carry a body no matter which one you pick or how the window is narrowed, and that is a fine outcome -- "
        "if this call comes back with no content, do not keep searching for a different article chasing it; "
        "proceed to decide using the headline alone.\n"
        "4. Compare article timestamps with the current datetime given in the task and ignore stale news.\n"
        "5. Decide the regime. The portfolio snapshot in the context gives current_regime, computed from what is actually "
        f"held ('risk_on' = {risky}, 'defensive' = {defensive_symbol}, 'mixed' = both, 'none' = nothing), and "
        "sessions_in_regime, the trading days it has lasted. Unclear, mixed, stale or no relevant news means KEEP the "
        "current holding: trade nothing. That includes an empty book (current_regime 'none'): there is no "
        "default position, so stay in cash until the news gives a clear signal. With an empty book there is no regime to "
        f"flip, so ONE clear signal is enough to enter: bullish -> buy {risky}, bearish -> buy {defensive_symbol}. "
        "A single article or data print must not flip an existing regime. "
        f"Leaving {risky} for {defensive_symbol} needs two INDEPENDENT bearish signals in this run (a jobs, CPI, PPI, "
        "Redbook, ADP or claims release is ONE signal however many headlines cover it), or the same bearish call already "
        "recorded on the previous run (see search_memory). With only one signal, do not trade: call remember_decision "
        "with text starting 'PENDING bearish flip:' and the reason. "
        f"Going from {defensive_symbol} back to {risky} needs one clear bullish signal. If current_regime is 'mixed', "
        "finish the rotation your last recorded decision started.\n"
        "6. Every run, call get_positions and get_account_balance before trading, and never rely on memory for what you "
        "hold: memory records what you intended, not what filled. The context below also carries a portfolio snapshot; "
        "use it to cross-check (each position shows its pct_of_portfolio). Then check get_orders, count an open buy order "
        "from get_orders as already held, and trade only the difference. Sell the current position before buying the "
        "other: submit the sell first, then the buy in the same run. A sell you submitted is credited toward the buy. "
        "Size a buy at no more than 95% of the SMALLER of 'buying_power' and ('cash' + the proceeds of the sells you "
        "submitted in this run that were accepted (no 'error')), where proceeds = sold quantity x last price "
        "(quantity = floor(0.95 * that amount / last price)), leaving room for fees. Never size a buy against "
        "'buying_power' alone: on a margin account it exceeds your cash, and buying on margin is forbidden. "
        "If you decided on a regime in step 5 and the combined value of the instruments for the regime is worth less "
        "than 90% of portfolio_value (a leftover from a partial rotation, or cash you decided to invest), you are not "
        "aligned with the regime: buy the shortfall, spending up to the sizing amount above. An empty book with no "
        "clear signal stays in cash. "
        "Never sell more than get_positions says you hold. "
        "If submit_order comes back with an 'error', the order was refused and nothing was placed -- read the reason, "
        "correct the quantity and try once more, or skip the trade this run. "
        "Keep position sizing reasonable and never place a duplicate order. An order you submit is final for the run: "
        "settle the decision before the first order, and never cancel an order you submitted in this run "
        "(cancel_order refuses it). Cancel only orders from earlier runs, as listed by get_orders.\n"
        "7. Record the outcome: call remember_decision after every decision, and open_thesis or close_thesis when the "
        "regime call changes. Call remember_decision exactly once per run: once it returns status 'recorded', the run "
        "is over, so reply with a one-line summary and make no further tool call.\n"
        "**Do not forget the gating process:** search_memory -> search_news(include_content=False and limit=30) "
        "-> search_news(include_content=True and limit=3) -> (get_positions & get_account_balance) -> trade if this is the decision "
        "-> remember_decision."
    )


class NewsBinaryStrategy(Strategy):
    AGENT_NAME = "news_binary_trader"
    TASK_PROMPT = "Research current broad-market news and rebalance if needed. The current datetime is in the context below."

    parameters = {
        "backtesting_start": datetime(2026, 1, 1, tzinfo=MARKET_TZ),
        "backtesting_end": datetime(2026, 9, 18, tzinfo=MARKET_TZ),
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 10,  # No warmup needed (just in case)
        "budget": 10000,
        "commission": 0.0,  # Commission is 0 on US ETF (Alpaca)
    }

    strategy_parameters = {
        "symbols": ("SPY", "QQQ"),
        "defensive_symbol": "SHV",
        "news_symbols": "SPY,QQQ,DIA,IWM",
        "backtest_every_n_iterations": 1,  # Number of days between 2 effective iterations
    }

    def initialize(self) -> None:
        self.sleeptime = "3H" if self.is_backtesting else "1H"
        # Enable agent telemetry in live/trading
        # self.agent_telemetry = True
        self.vars.iteration_count = 0
        self.vars.consecutive_agent_errors = 0
        self.vars.strategy_parameters = self.strategy_parameters
        self.vars.regime = None
        self.vars.sessions_in_regime = 0
        self.vars.regime_last_date = None
        system_prompt = _build_system_prompt(
            symbols=self.vars.strategy_parameters["symbols"],
            defensive_symbol=self.vars.strategy_parameters["defensive_symbol"],
            news_symbols=self.vars.strategy_parameters["news_symbols"],
        )
        self.agents.create(
            name=self.AGENT_NAME,
            system_prompt=system_prompt,
            tools=require_search_news_before([*PrebuiltTools.all(self), *news_tools(self)]),
        )
        self.log_info(f"NewsBuiltinStrategy initialized (sleeptime={self.sleeptime})")
        self.log_info(f"Agent run with system prompt: \n{system_prompt}")

    def on_trading_iteration(self) -> None:
        self.vars.iteration_count += 1
        if self.is_backtesting and not self._backtest_iteration_is_due():
            return
        context = {"current_datetime": self.get_datetime().isoformat(), "portfolio": self._portfolio_snapshot()}
        run_id = uuid.uuid4().hex
        result = self._run_agent(self.TASK_PROMPT, context=context, run_id=run_id)
        if result is None:
            return
        self._log_agent_result(result)
        if _decision_was_recorded(result.tool_calls):
            return
        # Observed: the model places a real trade via submit_order, then just writes a prose summary
        # and stops -- no remember_decision call at all, real or malformed. Reusing `run_id` keeps the
        # follow-up turn part of the same logical run, so the news-grounding gate (already satisfied by
        # this run's own search_news call) doesn't ask it to re-ground itself for one more tool call.
        self.log_warning(f"[{self.AGENT_NAME}] run ended without a successful remember_decision call -- retrying once so a decision is still recorded.")
        retry_prompt = (
            f"Your last reply this run was:\n{result.output}\n\n"
            "You did not call remember_decision before answering, so nothing was recorded. Call "
            "remember_decision now, with the `text` argument summarizing that decision (including any "
            "trade you placed). Call no other tool first."
        )
        retry_result = self._run_agent(retry_prompt, context=context, run_id=run_id, force_tool="remember_decision")
        if retry_result is None:
            return
        self._log_agent_result(retry_result, prefix="retry_")
        if not _decision_was_recorded(retry_result.tool_calls):
            self.log_warning(f"[{self.AGENT_NAME}] retry also ended without a successful remember_decision call -- no decision was recorded this run.")

    def _run_agent(self, task_prompt: str, *, context: dict[str, object], run_id: str, force_tool: str | None = None) -> AgentRunResult | None:
        """Run `self.AGENT_NAME` once; on `AgentError`, log/count it (matching the old inline handling) and return `None`."""
        try:
            result = self.agents[self.AGENT_NAME].run(task_prompt, context=context, run_id=run_id, force_tool=force_tool)
        except AgentError as exc:
            # A failed LLM call must not kill a live loop, and one flaky call must not kill a backtest.
            self.vars.consecutive_agent_errors += 1
            self.log_error(f"[{self.AGENT_NAME}] run failed: {exc}")
            if self.is_backtesting and self.vars.consecutive_agent_errors >= MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS:
                raise FatalStrategyError(f"[{self.AGENT_NAME}] failed {self.vars.consecutive_agent_errors} runs in a row, aborting the backtest; last error: {exc}") from exc
            return None
        self.vars.consecutive_agent_errors = 0
        return result

    def _log_agent_result(self, result: AgentRunResult, *, prefix: str = "") -> None:
        self.log_info(f"[{self.AGENT_NAME}] Agent output: \n{result.output}")
        for i, tool_call in enumerate(result.tool_calls):
            self.log_info(f"{prefix}tool_call_{i}: {tool_call}")

    def _portfolio_snapshot(self) -> dict[str, object]:
        """What the account holds right now, handed to the agent up front.

        The agent used to skip `get_positions` in most backtest runs and then "hold" a position it did
        not have, so the truth goes into the run context instead of depending on a tool call.
        Built from the account tools so it has exactly the shape the agent already reads from them,
        plus each position's share of the book: `BacktestBroker` positions carry no market value, and
        the agent must not be left to work out that 7 SHV shares are 7% of equity, not a full rotation.

        A backtest data failure (`BacktestError`, which the account tools do not catch) must not skip the
        tick unseen -- it would never reach the agent-error counter -- so it degrades to a namespaced
        error the agent can read, exactly as the same failure inside a tool call would.
        """
        tools = {tool.__name__: tool for tool in account_tools(self)}
        snapshot: dict[str, object] = {}
        for tool_name, error_key in (("get_account_balance", "balance_error"), ("get_positions", "positions_error")):
            try:
                part = tools[tool_name]()
            except (BrokerError, BacktestError) as exc:
                part = {"error": str(exc)}
            # Namespaced so a failed `get_positions` is not read as an empty book (no `positions` key at all).
            snapshot.update({error_key: part["error"]} if "error" in part else part)
        equity = snapshot.get("portfolio_value")
        for position in snapshot.get("positions", []):  # type: ignore # ty: ignore[not-iterable]
            if "market_value" not in position:
                try:
                    price = self.get_last_price(position["symbol"])
                except (BrokerError, BacktestError):
                    price = None
                if price is not None:
                    position["market_value"] = round(position["quantity"] * float(price), 2)
            if equity and "market_value" in position:
                position["pct_of_portfolio"] = round(100 * position["market_value"] / equity, 1)  # ty: ignore[unsupported-operator]
        self._track_regime(snapshot)
        return snapshot

    def _track_regime(self, snapshot: dict[str, object]) -> None:
        """Add `current_regime` and `sessions_in_regime` to `snapshot`, derived from what is actually held.

        The prompt's hysteresis ("one data print cannot flip the regime") needs the agent to know which
        regime it is in and for how long, and the agent's memory of that is exactly what went wrong
        before. The regime comes from the holdings and the configured symbols (no literals), and a
        session is a distinct trading date on which the strategy ran, so hourly live runs count once.
        A failed positions read claims nothing and leaves the counter alone.
        """
        positions = snapshot.get("positions")
        if positions is None:
            return
        parameters = self.vars.strategy_parameters
        held = {position["symbol"] for position in positions if position["quantity"] > 0}  # type: ignore # ty: ignore[not-iterable, invalid-argument-type]
        risky = bool(held & set(parameters["symbols"]))
        defensive = parameters["defensive_symbol"] in held
        regime = "mixed" if risky and defensive else "risk_on" if risky else "defensive" if defensive else "none"
        today = self.get_datetime().date()
        if regime != self.vars.regime:
            self.vars.regime, self.vars.sessions_in_regime, self.vars.regime_last_date = regime, 1, today
        elif today != self.vars.regime_last_date:
            self.vars.sessions_in_regime += 1
            self.vars.regime_last_date = today
        snapshot["current_regime"] = regime
        snapshot["sessions_in_regime"] = self.vars.sessions_in_regime

    def _backtest_iteration_is_due(self) -> bool:
        """
        Set the interval of iterations for backtesting (not to trade every day)
        """
        count = self.vars.iteration_count
        return count == 1 or count % self.vars.strategy_parameters["backtest_every_n_iterations"] == 0

    def run_backtesting(self):
        # use class strategy parameters as the strategy constructor is not run yet
        symbols = [*self.strategy_parameters["symbols"], self.strategy_parameters["defensive_symbol"]]
        return super().run_backtesting(
            data_source=AlpacaBacktestData,  # Use alpaca broker for intraday quotes (> 6 year history)
            timestep="minute",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            commission=Decimal(str(self.parameters["commission"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
