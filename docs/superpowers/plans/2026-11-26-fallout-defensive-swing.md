# fallout_defensive_swing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new strategy `fallout_defensive_swing`: hold XLP + XLU 50/50, check hourly, sell everything at +1% net of fees, buy back after a 1% dip below the sale prices, never sell on the session of the last sale (cash-account settlement).

**Architecture:** A pure decision function (`rule.decide`) over a small frozen state; a `JsonStateStore` for that state; a thin `Strategy` subclass that reads positions and prices, estimates sell fees with `TradingFeeFactory`, places fractional market orders and saves the state after fills. Registered in `Strategies`. No LLM.

**Tech Stack:** Python 3.14, `Decimal`, pytest with `tests/fakes.py` (`FakeBroker`, `FakeClock`, `et`), uv.

**Spec:** `docs/superpowers/specs/2026-11-26-fallout-defensive-swing-design.md`

## Global Constraints

- Package `src/trading_agent_framework/strategies/fallout_defensive_swing/`; enum `Strategies.FALLOUT_DEFENSIVE_SWING` (value `fallout_defensive_swing`); class `FalloutDefensiveSwingStrategy`.
- `parameters` defaults: `symbols = ["XLP", "XLU"]`, `take_profit = Decimal("0.01")`, `buyback_dip = Decimal("0.01")`, `budget = 768`, `benchmark_symbol = "SPY"`, `backtesting_start = datetime(2026, 11, 3, tzinfo=MARKET_TZ)`, `backtesting_end = datetime(2026, 11, 25, tzinfo=MARKET_TZ)`.
- `sleeptime = "60M"`, `iteration_start_time = time(10, 30)` (ticks 10:30 ... 15:30 ET).
- Money is `Decimal`. Buys sized from `self.spendable_cash()` only. Fractional quantities floored to 6 decimals (`ROUND_DOWN`), as the Rebalancer does.
- A sell is allowed only when `session > last_sale_session` (or no sale yet).
- State saved only after orders have been waited on (`wait_for_order_execution`).
- No Alpaca paper run (user's decision); the only backtest window is Nov 3 - Nov 25 2026.
- Google docstrings on every public module/class/function; no `noqa`/`type: ignore`.

## Review Focus

- Sale on the same session as the last sale -> `HOLD` even above target (test in Task 1).
- Take-profit compared **net of sell fees**: basis 768 -> target 775.68; value 776.00 - fees 0.24 = 775.76 -> `SELL`; value 775.90 - 0.24 = 775.66 -> `HOLD` (tests in Task 1 pin both sides of the line).
- Broker positions disagree with the state (restart, partial fill, manual sale) -> the tick adopts the broker's view with a warning (tests in Task 3).
- One of two sell orders not filled -> phase stays `HOLDING` with `selling=True`; next allowed tick sells the rest whatever the price (tests in Task 1 and 3).
- A price missing for one ETF (`get_last_price` -> `None`) -> the tick does nothing and logs a warning (test in Task 3).

---

### Task 1: The pure rule

**Files:**
- Create: `src/trading_agent_framework/strategies/fallout_defensive_swing/__init__.py` (empty docstring module for now; Task 3 exports the class)
- Create: `src/trading_agent_framework/strategies/fallout_defensive_swing/rule.py`
- Test: `tests/strategies/fallout_defensive_swing/__init__.py`, `tests/strategies/fallout_defensive_swing/test_rule.py`

**Interfaces:**
- Produces:
  - `class Phase(StrEnum)`: `FRESH`, `HOLDING`, `CASH`.
  - `class Action(StrEnum)`: `BUY`, `SELL`, `HOLD`.
  - `@dataclass(frozen=True) class SwingState`: `phase: Phase = Phase.FRESH`, `basis: Decimal | None = None`, `sale_prices: dict[str, Decimal] = field(default_factory=dict)`, `last_sale_session: date | None = None`, `selling: bool = False`.
  - `decide(state: SwingState, *, value: Decimal, sell_fees: Decimal, prices: Mapping[str, Decimal], session: date, take_profit: Decimal, buyback_dip: Decimal) -> Action`.

Logic: `FRESH` -> `BUY`. `HOLDING` -> `HOLD` if `last_sale_session is not None and session <= last_sale_session`; else `SELL` if `selling` or `value - sell_fees >= basis * (1 + take_profit)`; else `HOLD`. `CASH` -> `BUY` if mean over `sale_prices` of `prices[s] / sale_prices[s]` `<= 1 - buyback_dip`, else `HOLD`.

- [ ] **Step 1: Write the failing tests** (`D = Decimal`, `TODAY = date(2026, 11, 30)`, defaults `take_profit=D("0.01")`, `buyback_dip=D("0.01")`)

```python
def test_fresh_buys():
    assert _decide(SwingState()) is Action.BUY


def test_holding_below_target_holds():
    assert _decide(_holding(), value=D("775.90"), sell_fees=D("0.24")) is Action.HOLD


def test_holding_at_target_net_of_fees_sells():
    assert _decide(_holding(), value=D("776.00"), sell_fees=D("0.24")) is Action.SELL


def test_no_sale_on_the_last_sale_session():
    assert _decide(_holding(last_sale=TODAY), value=D("800")) is Action.HOLD


def test_sale_allowed_the_next_session():
    assert _decide(_holding(last_sale=TODAY - timedelta(days=1)), value=D("800")) is Action.SELL


def test_unfinished_sale_sells_the_rest_whatever_the_price():
    assert _decide(_holding(selling=True, last_sale=TODAY - timedelta(days=3)), value=D("700")) is Action.SELL


def test_unfinished_sale_still_waits_for_settlement():
    assert _decide(_holding(selling=True, last_sale=TODAY), value=D("700")) is Action.HOLD


def test_cash_above_the_dip_holds():
    assert _decide(_cash(), prices={"XLP": D("99.5"), "XLU": D("49.5")}) is Action.HOLD  # mean ratio 0.995


def test_cash_at_the_dip_buys():
    assert _decide(_cash(), prices={"XLP": D("99"), "XLU": D("49.5")}) is Action.BUY  # mean ratio 0.99
```

with `_holding(**kw)` = `SwingState(phase=Phase.HOLDING, basis=D("768"), ...)` and `_cash()` = `SwingState(phase=Phase.CASH, sale_prices={"XLP": D("100"), "XLU": D("50")}, last_sale_session=TODAY - timedelta(days=1))`; `_decide` fills the unused keyword arguments with neutral values (`value=D("768")`, `sell_fees=D(0)`, `prices={"XLP": D("100"), "XLU": D("50")}`, `session=TODAY`).

- [ ] **Step 2: Run them, expect FAIL** — `uv run pytest tests/strategies/fallout_defensive_swing/test_rule.py -q` -> `ModuleNotFoundError` / import error.
- [ ] **Step 3: Implement `rule.py`** as in Interfaces.
- [ ] **Step 4: Run, expect PASS** — same command, 9 passed.
- [ ] **Step 5: Commit** — `git add src/trading_agent_framework/strategies/fallout_defensive_swing tests/strategies/fallout_defensive_swing && git commit -m "feat(fallout_defensive_swing): pure take-profit / buy-back rule"`

---

### Task 2: The state store

**Files:**
- Create: `src/trading_agent_framework/strategies/fallout_defensive_swing/store.py`
- Test: `tests/strategies/fallout_defensive_swing/test_store.py`

**Interfaces:**
- Consumes: `SwingState`, `Phase` (Task 1); `JsonStateStore`, `InvalidState` (`strategies/common/json_state.py`; pattern: `strategies/sector_pods/store.py`).
- Produces: `class StateStore(JsonStateStore[SwingState])`, `__init__(self, path: Path, *, log: ColorLogger, fresh: bool = False)`, version 1. Encoding: `Decimal` as `str`, `date` as ISO string, `phase` as its value.

- [ ] **Step 1: Failing tests:** round trip of a `HOLDING` state with `basis=D("768.00")`, `sale_prices={"XLP": D("92.28")}`, `last_sale_session=date(2026, 11, 4)`, `selling=True` gives an equal `SwingState`; a file with `"phase": "nope"` loads as `SwingState()`; a missing file loads as `SwingState()`; `fresh=True` ignores an existing file.
- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/strategies/fallout_defensive_swing/test_store.py -q`.
- [ ] **Step 3: Implement** `_decode` (raise `InvalidState` on a bad phase/decimal/date/bool) and `_encode`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat(fallout_defensive_swing): state store`.

---

### Task 3: The strategy and its registration

**Files:**
- Create: `src/trading_agent_framework/strategies/fallout_defensive_swing/agent_fallout_defensive_swing.py`
- Modify: `src/trading_agent_framework/strategies/fallout_defensive_swing/__init__.py` (export `FalloutDefensiveSwingStrategy`)
- Modify: `src/trading_agent_framework/utils/strategy_factory.py` (enum member, `_build_fallout_defensive_swing`, `_STRATEGY_BUILDERS` entry)
- Test: `tests/strategies/fallout_defensive_swing/test_strategy.py`

**Interfaces:**
- Consumes: `decide`, `SwingState`, `Phase`, `Action` (Task 1); `StateStore` (Task 2); `Strategy` (`create_order`, `submit_order`, `wait_for_order_execution`, `get_positions`, `get_last_price`, `spendable_cash`, `state_file`, `clock`, `is_backtesting`, `log_info`/`log_warning`); `TradingFeeFactory` (`brokers/fees.py`); `AlpacaBacktestData` (`backtesting/data/alpaca.py`); `MARKET_TZ` (`utils/clock.py`).
- Produces: `class FalloutDefensiveSwingStrategy(Strategy)`, `__init__(self, broker, *, mode=TradingMode.PAPER, fees: TradingFeeFactory | None = None, **kwargs)` (`fees` defaults to `TradingFeeFactory.from_env()`); `initialize()`; `on_trading_iteration()`; `run_backtesting(**overrides)` with defaults `data_source=AlpacaBacktestData`, `timestep="minute"`, `agent_telemetry=False` (same shape as `VwapPullbackStrategy.run_backtesting`).

One tick (`on_trading_iteration`):
1. `session` = `self.clock.now().astimezone(MARKET_TZ).date()`; `held` = mix positions with quantity > 0; `prices` = `get_last_price` per symbol. Any price `None` -> `log_warning` and return.
2. Reconcile: `held` and phase is not `HOLDING` -> warn, phase `HOLDING`, `basis` = market value, `selling=False`. Nothing held and phase `HOLDING` -> warn, phase `CASH` if `sale_prices` else `FRESH`, `selling=False`. Save if changed.
3. `value` = sum(q x price); `sell_fees` = sum over held of `fees.fees(buy_shares=0, sell_shares=q, buy_value=0, sell_value=q x price).sell`.
4. `BUY`: `per = spendable_cash() / len(symbols)`; quantity `= (per / price).quantize(Decimal("0.000001"), ROUND_DOWN)` per symbol; submit each, wait each. Filled orders: `basis` = sum(filled x avg price) + sum of their `.buy` fees. Phase `HOLDING` if anything filled (else unchanged), `selling=False`. Save.
5. `SELL`: submit a market sell of each held quantity, wait each. `sale_prices` from fills (fallback: the tick's price), `last_sale_session = session`. Phase `CASH` and `selling=False` if every sell filled, else `HOLDING` with `selling=True`. Save.
6. Log one line per tick: phase, value, basis/target or dip ratio, action.

State file: `self.state_file()` with `fresh=self.is_backtesting`.

- [ ] **Step 1: Failing tests** (`FakeBroker(FakeClock(et(2026, 11, 30, 10, 30)), strategy_name="fallout_defensive_swing")`, prices in `broker.last_prices`, `project_root=tmp_path`; drive `initialize()` then `on_trading_iteration()`; the fake fills market orders the way the sector_pods/Rebalancer tests already rely on — copy their set-up):
  - first tick buys both ETFs with equal notional from `spendable_cash()`, quantities with at most 6 decimals, and saves `HOLDING` with a basis > 0;
  - holding at +1% net of fees on a later session sells both full quantities and saves `CASH` with both sale prices and today's session;
  - holding at +1% on the session of the last sale submits nothing;
  - in cash with prices 1% under the sale prices buys back;
  - restart with mix positions in the broker and a `FRESH` state file -> no buy, state `HOLDING`, a warning logged;
  - phase `HOLDING` but no positions -> state `CASH`, warning;
  - one sell not filled -> state `HOLDING` with `selling=True`;
  - a `None` price -> no order, warning;
  - `Strategies("fallout_defensive_swing")` builds a `FalloutDefensiveSwingStrategy` (`build_strategy`).
- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/strategies/fallout_defensive_swing/test_strategy.py -q`.
- [ ] **Step 3: Implement** the class, the export and the factory entry.
- [ ] **Step 4: Run, expect PASS** — then `uv run pytest tests/test_strategy_factory.py tests/strategies/test_backtest_defaults.py -q` (the factory tests iterate every `Strategies` member: they must pass unchanged).
- [ ] **Step 5: Commit** — `feat(fallout_defensive_swing): strategy and registration`.

---

### Task 4: Docs, quality gate, backtest

**Files:**
- Modify: `README.md` (new `### 📈 fallout_defensive_swing` section after `sector_pods`, same table shape: File, Model "none", Agents "none", Tools "none", Asset universe "XLP, XLU", frequency "hourly 10:30-15:30 ET", modes "backtest, live (no Alpaca paper run)", Benchmark SPY, Env file `env/.env.fallout_defensive_swing.<mode>` with `BROKER=tastytrade`, `ALPACA_DATA_*`, `TASTYTRADE_*` in live). Then a **Conditions of applicability** list copied from the spec's README section (volatile trendless market, dedicated cash account, daily review and manual stop; not for a rising market nor a long fall; longest study hold 232 days), the rule in 4 lines and the run commands `uv run agent fallout_defensive_swing backtesting` / `live`.
- Create: `src/trading_agent_framework/strategies/fallout_defensive_swing/CLAUDE.md` (the rule, the settlement guard, reconcile-on-tick, state file, study numbers and their limits).
- Modify: `src/trading_agent_framework/strategies/CLAUDE.md` and root `CLAUDE.md` (add `fallout_defensive_swing` to the per-strategy `CLAUDE.md` lists).

- [ ] **Step 1: Write the docs.**
- [ ] **Step 2: Quality gate** — `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov -q && uv run interrogate -v`. Expected: exit 0, 0 pyright errors, coverage >= 80%, interrogate >= 80%.
- [ ] **Step 3: Commit** — `docs(fallout_defensive_swing): README conditions of applicability and CLAUDE.md`.
- [ ] **Step 4: Backtest (needs the user's `env/.env.fallout_defensive_swing.backtesting` with `BROKER=tastytrade` and `ALPACA_DATA_*`; the user creates it, never create or edit env files)** — `uv run agent fallout_defensive_swing backtesting`. Expected: a report under `logs/fallout_defensive_swing/backtesting/`; compare its P&L, trades and drawdown with the study (+$40.34, 4 sales, DD -0.8%). Report the gap and its causes (next-bar fills, 5 bps slippage, 2% buffer, ticks at 10:30-15:30 vs the study's bars); do not tune the rule to close it.
