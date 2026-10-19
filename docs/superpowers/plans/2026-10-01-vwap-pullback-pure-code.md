# vwap_pullback Pure-Code Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `strategies/vwap_pullback/` into a deterministic, code-only strategy: every triggered setup is entered by code (best stage-2 score first), a trade ends on its protective stop or at the 15:50 flatten, and no LLM, LangGraph or news code remains.

**Architecture:** The per-tick LangGraph pass (classify → exit agent → entry agent) is replaced by three direct calls in `on_trading_iteration`: `Desk.reconcile` → `Scanner.scan` → `Desk.enter_triggered`. The scan now keeps the stage-2 composite score per symbol in `SessionState.scores`, which orders entries when triggers outnumber free slots. Everything that existed only for the agents (prompts, tools, graph, headlines, exit actions, review bookkeeping) is deleted; the order-handling core of `Desk` is not touched.

**Tech Stack:** Python 3.14, `uv`, `pytest`, `ruff`. No new dependency; `langgraph` is dropped from `pyproject.toml`.

**Spec:** `docs/superpowers/specs/2026-10-01-vwap-pullback-pure-code-design.md`

## Global Constraints

- Branch: `feature/vwap-pullback-pure-code` (already created; the spec is committed on it).
- Strategy name `vwap_pullback_continuation`, class `VwapPullbackStrategy`, file name `agent_vwap_pullback.py` and the `main.py` registration do not change.
- No parameter value changes: every field of `VwapPullbackParameters` that survives keeps its default.
- Do not change the behaviour of these `Desk` methods: `on_order_filled`, `on_order_canceled`, `_settle_entry`, `_settle_exit`, `_reprotect`, `_protect`, `reconcile`, `_release_stop`, `_market_sell`, `_cancel`, `flatten_all`, `_sell_unprotected`, `close_unknown_positions`, `_close_orphans`, `_archive`, `free_slots`, `session_pnl`, `breaker_tripped`, `_has_exposure`, `_pending_sell_proceeds`, `_free_quantity`, `_exit_pending`. Only the edits written in this plan are allowed inside them.
- Do not touch the framework outside the strategy package: `agents/`, `backtesting/`, `brokers/`, `core/`, `tests/fakes.py`.
- Money stays `Decimal`; no `time.sleep` / `datetime.now` in strategy code (use `strategy.get_datetime()`).
- Tests use the hand-written fakes in `tests/fakes.py`, never `MagicMock`, and never touch the network.
- Match the surrounding code: line length and comment density as in the files edited, no `# noqa` added.
- Run commands from the project root `/home/yann/projets/trading-agent-framework`.
- After every task: `uv run pytest tests/strategies/vwap_pullback/ -q` and `uv run ruff check` must both pass before the commit.
- End every commit message with the `Co-Authored-By:` trailer of the model doing the work (for example `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`).
- `TODO.md` has an uncommitted user edit: never `git add -A`; add only the files a task names (Task 5 is the only one that stages `TODO.md`).

## Review Focus

1. **`enter_triggered` called twice in the same tick** (a retry, a future caller): the second call must submit nothing, because an entered setup is frozen `IN_TRADE`. Test in Task 2.
2. **One symbol's price or account read fails mid-loop** (`BacktestError`/`BrokerError` from `get_last_price`): that symbol is refused and logged, the following triggers are still tried. Test in Task 2.
3. **A tick with no triggered setup**: no account read, no order, an empty result. Test in Task 2.
4. **Scores left over from the previous tick**: a symbol that scored last tick but is not ranked now must not keep its old score. Test in Task 1.
5. **An unscored trigger against a negatively scored one**: "no score" must rank last, not be read as `0.0` (which would beat a negative score). Test in Task 2.

---

## File Structure

| File | After this plan |
|---|---|
| `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` | Strategy lifecycle hooks; tick = reconcile → scan → enter. No agent, no graph. |
| `.../vwap_pullback/desk.py` | Orders: `enter_triggered`, `enter_long`, stops, flatten, restart cleanup. No exit actions. |
| `.../vwap_pullback/scanner.py` | Stage 1, stage 2, setups, and `state.scores`. No news. |
| `.../vwap_pullback/screening.py` | `RankedCandidate.composite` is `float \| None`. |
| `.../vwap_pullback/session.py` | `SessionState` gains `scores`; loses the headline and agent fields. |
| `.../vwap_pullback/trades.py` | `Trade`, `TradeBook`. No review flags, no trail fields. |
| `.../vwap_pullback/setups.py` | State machine. `health` removed. |
| `.../vwap_pullback/features.py` | Bar features. `Levels`, `latest_levels`, `ema_last`, `bar_atr` removed. |
| `.../vwap_pullback/parameters.py` | Agent, news and exit-action parameters removed. |
| `.../vwap_pullback/prompts.py`, `tools.py`, `graph.py`, `news.py` | Deleted. |
| `tests/strategies/vwap_pullback/test_vwap_graph.py`, `test_vwap_tools.py`, `test_vwap_news_prompts.py` | Deleted. |
| `pyproject.toml`, `uv.lock` | `langgraph` no longer a direct dependency. |
| `CLAUDE.md`, `TODO.md` | Describe a code-only strategy. |

---

### Task 1: Keep the stage-2 score of every ranked symbol

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/screening.py` (`RankedCandidate`, `rank_stage2`)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/session.py` (`SessionState`)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/scanner.py` (`Scanner.scan`)
- Test: `tests/strategies/vwap_pullback/test_vwap_screening.py`, `tests/strategies/vwap_pullback/test_vwap_scanner.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `SessionState.scores: dict[str, float]` — this tick's stage-2 composite for every tracked symbol that passes the stage-2 floor, rebuilt by every `Scanner.scan`. A tracked symbol outside the floor has no key. `RankedCandidate.composite: float | None` (`None` for a sticky symbol outside the floor).

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/vwap_pullback/test_vwap_screening.py`, replace `test_rank_stage2_keeps_sticky_symbols_even_when_they_fail_the_floor` with these two tests:

```python
def test_rank_stage2_keeps_sticky_symbols_even_when_they_fail_the_floor() -> None:
    ranked = rank_stage2([_snapshot("A"), _snapshot("HELD", rs=-0.01)], PARAMS, sticky={"HELD", "GONE"})
    assert [c.symbol for c in ranked] == ["A", "GONE", "HELD"]
    assert ranked[0].composite == 0.0  # the only symbol passing the floor: no spread to rank on
    assert ranked[1].composite is None and ranked[2].composite is None  # outside the floor: no score, not a mid-pack 0.0


def test_rank_stage2_gives_a_sticky_symbol_inside_the_floor_its_real_score() -> None:
    snapshots = [_snapshot("A", ret=0.05, rs=0.04, rvol=4.0), _snapshot("B", ret=0.01, rs=0.005, rvol=1.6)]
    ranked = rank_stage2(snapshots, dataclasses.replace(PARAMS, tracked_size=1), sticky={"B"})
    assert [c.symbol for c in ranked] == ["A", "B"]
    assert ranked[1].composite == pytest.approx(-1.0)  # below A on all three measures
```

In `tests/strategies/vwap_pullback/test_vwap_scanner.py`, add after `test_scan_builds_contexts_and_advances_setups`:

```python
def test_scan_stores_this_ticks_stage2_scores(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 9, 50, 30))
    rising = [(100 + 0.06 * i, 100 + 0.06 * (i + 1) + 0.01, 100 + 0.06 * i - 0.01, 100 + 0.06 * (i + 1), 300.0) for i in range(20)]
    broker.timestep_frames = {
        ("AAA", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), rising),
        ("SPY", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), [(400, 400, 400, 400, 1000)] * 20),
    }
    broker.news = FakeNewsProvider()
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.baselines["AAA"] = pd.Series([100.0 * (m + 1) for m in range(390)])
    state.scores = {"OLD": 1.0}  # left from the previous tick
    Scanner(strategy, PARAMS, ["AAA"]).scan(state)
    assert state.scores == {"AAA": 0.0}  # rebuilt: one symbol passes the floor, so its composite is 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_screening.py tests/strategies/vwap_pullback/test_vwap_scanner.py -q`
Expected: 2 failures. `test_rank_stage2_keeps_sticky_symbols_even_when_they_fail_the_floor` fails on `assert ranked[1].composite is None` (it is `0.0`); `test_scan_stores_this_ticks_stage2_scores` fails with `{'OLD': 1.0} == {'AAA': 0.0}`. The "real score" test passes already (it pins existing behaviour).

- [ ] **Step 3: Implement**

`screening.py` — the `RankedCandidate` dataclass becomes:

```python
@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A symbol tracked this tick, in composite-score order."""

    symbol: str
    # Mean of z(ret), z(rs), z(rvol) across the symbols passing the floor: the tracking order, and the order
    # entries are tried in. None for a sticky symbol outside the floor (it has no score this tick).
    composite: float | None
```

`screening.py` — in `rank_stage2`, replace the docstring's last sentence and the `extras` line:

```python
    """The top `tracked_size` symbols passing the floor by composite z-score, then every `sticky` symbol not already in (by name).

    Floor: RVOL at least `rvol_min`, RS above 0, last close above VWAP. A symbol with no RVOL baseline
    cannot pass it. A sticky symbol (a setup already past WATCH) stays tracked whatever its rank; it keeps
    its composite when it passes the floor and gets `None` when it does not.
    """
```

```python
    extras = [by_symbol.get(symbol, RankedCandidate(symbol=symbol, composite=None)) for symbol in sorted(set(sticky) - kept)]
```

In the same function the `sorted(candidates, key=lambda c: (-c.composite, c.symbol))` line stays as it is: every element of `candidates` passes the floor and has a float composite.

`session.py` — add this field to `SessionState`, right after `contexts`:

```python
    scores: dict[str, float] = field(default_factory=dict)  # this tick's stage-2 composite per tracked symbol passing the floor (rebuilt by every scan)
```

`scanner.py` — in `Scanner.scan`, right after `ranked = rank_stage2(snapshots, self._params, sticky)`:

```python
        # Kept for the desk: entries are tried best score first when triggers outnumber free slots.
        state.scores = {candidate.symbol: candidate.composite for candidate in ranked if candidate.composite is not None}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/vwap_pullback/ -q && uv run ruff check`
Expected: all pass, no lint error.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/screening.py src/trading_agent_framework/strategies/vwap_pullback/session.py src/trading_agent_framework/strategies/vwap_pullback/scanner.py tests/strategies/vwap_pullback/test_vwap_screening.py tests/strategies/vwap_pullback/test_vwap_scanner.py
git commit -m "feat: vwap_pullback scan keeps the stage-2 score of every ranked symbol"
```

---

### Task 2: Entries in code, a code-only tick, agents deleted

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/desk.py` (module docstring, imports, the `views` and `entries` sections)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` (whole file rewritten)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/trades.py` (`Trade.catalyst`, `Trade.reason`, `to_json`)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/session.py` (`decided`, `passes`, module docstring)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/setups.py` (`health`, the `TRIGGERED` comment)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/parameters.py` (three agent parameters)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/risk.py` (two docstrings)
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/__init__.py` (docstring)
- Delete: `src/trading_agent_framework/strategies/vwap_pullback/prompts.py`, `tools.py`, `graph.py`
- Delete: `tests/strategies/vwap_pullback/test_vwap_graph.py`, `tests/strategies/vwap_pullback/test_vwap_tools.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_desk_entries.py`, `test_vwap_desk_exits.py`, `test_vwap_strategy.py` (rewritten), `test_vwap_trades.py`, `test_vwap_setups.py`, `test_vwap_news_prompts.py`

**Interfaces:**
- Consumes: `SessionState.scores: dict[str, float]` (Task 1).
- Produces:
  - `Desk.enter_triggered() -> list[str]` — tries every `TRIGGERED` setup, best score first; returns the symbols whose entry order was submitted, in submission order.
  - `Desk.enter_long(symbol: str) -> dict[str, Any]` — the `catalyst` and `reason` parameters are gone; the return value is unchanged (`{"symbol", "quantity", "limit_price", "stop_price", "r_per_share", "status"}` or `{"error": ...}`).
  - `Trade(...)` no longer takes `catalyst` or `reason`.
  - `VwapPullbackStrategy(broker, *, mode, universe, settings=None, **kwargs)` — no `chat_model` parameter.
  - Removed: `Desk.pass_on_setup`, `Desk.awaits_decision`, `Desk.entry_due`, `Desk.planned_risk`, `setups.health`, `SessionState.decided`, `SessionState.passes`, `VwapPullbackParameters.news_calls_per_run`, `bars_calls_per_run`, `max_passes_per_symbol`.
  - Still present after this task (removed in Task 3): the exit actions of `Desk`, `Desk.exit_review_due`, `Desk.mark_reviewed`, `Trade.tp1_done`, `Trade.stop_kind`.

- [ ] **Step 1: Write the failing desk tests**

In `tests/strategies/vwap_pullback/test_vwap_desk_entries.py`:

(a) Add `from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus` (replacing the `TradeStatus`-only import) and `from trading_agent_framework.utils.errors import BacktestError`.

(b) In `Rig.open_trade`, the first line becomes `self.desk.enter_long("AAA")`.

(c) Replace every other `enter_long("AAA", "<catalyst>", "<reason>")` and `enter_long("aaa", "earnings", "clean pullback")` call in the file with the one-argument form (`enter_long("AAA")`, `enter_long("aaa")`).

(d) In `test_enter_long_sizes_the_trade_and_submits_a_limit_buy`, replace `assert trade.status is TradeStatus.PENDING and trade.catalyst == "earnings"` with `assert trade.status is TradeStatus.PENDING` and delete the last line (`assert "AAA" in rig.state.decided`).

(e) Replace `test_enter_long_refuses_an_unknown_catalyst_and_a_full_book` with:

```python
def test_enter_long_refuses_a_full_book(tmp_path: Path) -> None:
    full = Rig(tmp_path, params=dataclasses.replace(PARAMS, max_positions=0))
    assert "slot" in full.desk.enter_long("AAA")["error"]
```

(f) Delete `test_the_label_the_agent_sets_is_logged_at_debug_even_when_the_entry_is_refused` and `test_an_unknown_label_is_not_logged_as_a_label`.

(g) Replace the parametrized `test_a_refused_entry_logs_a_warning_with_the_symbol_and_the_reason` with:

```python
def test_a_refused_entry_logs_a_warning_with_the_symbol_and_the_reason(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig(tmp_path, params=dataclasses.replace(PARAMS, max_positions=0))
    with caplog.at_level(logging.DEBUG):
        error = rig.desk.enter_long("AAA")["error"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1 and "AAA" in warnings[0] and "no free position slot" in warnings[0] and error in warnings[0]
```

(h) Replace `test_entry_due_and_exit_review_due` with (the entry-agent half is gone; Task 3 deletes the rest):

```python
def test_exit_review_due(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    assert rig.desk.exit_review_due(rig.clock.now()) == []  # 100 is above VWAP 99.9 and the EMA, below +1R
    rig.clock.advance(timedelta(minutes=15).total_seconds())
    assert rig.desk.exit_review_due(rig.clock.now()) == ["AAA"]
    rig.desk.mark_reviewed(rig.clock.now())
    assert rig.desk.exit_review_due(rig.clock.now()) == []
```

(i) Delete the last test of the file, `test_a_symbol_passed_on_max_times_no_longer_wakes_the_entry_agent`.

(j) Append the new `enter_triggered` tests at the end of the file:

```python
def _add_trigger(rig: Rig, symbol: str) -> None:
    """Another triggered setup shaped like the rig's AAA (the data source needs its minute bars too)."""
    rig.broker._data_source.frames[(symbol, "minute")] = minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)
    rig.state.candidates[symbol] = CandidateInfo(symbol=symbol, daily_atr=2.0, beta=1.0)
    rig.state.setups[symbol] = Setup(symbol=symbol, state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
    rig.state.contexts[symbol] = list(rig.state.contexts["AAA"])


def test_enter_triggered_enters_every_trigger_when_slots_allow(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _add_trigger(rig, "BBB")
    assert rig.desk.enter_triggered() == ["AAA", "BBB"]  # no score this tick: symbol order
    assert {trade.symbol for trade in rig.state.book.pending()} == {"AAA", "BBB"}
    assert rig.state.setups["AAA"].state is SetupState.IN_TRADE and rig.state.setups["BBB"].state is SetupState.IN_TRADE


def test_enter_triggered_gives_the_slots_to_the_best_stage2_scores(tmp_path: Path) -> None:
    rig = Rig(tmp_path, params=dataclasses.replace(PARAMS, max_positions=3))
    for symbol in ("BBB", "CCC", "DDD"):
        _add_trigger(rig, symbol)
    # BBB is tracked but outside the stage-2 floor this tick: no score. AAA and DDD tie below zero.
    rig.state.scores = {"AAA": -0.5, "CCC": 1.5, "DDD": -0.5}
    assert rig.desk.enter_triggered() == ["CCC", "AAA", "DDD"]  # best score, then the symbol on a tie
    assert rig.state.book.get("BBB") is None  # unscored goes last, even behind a negative score
    assert rig.state.setups["BBB"].state is SetupState.TRIGGERED  # left to expire on the next bar


def test_enter_triggered_a_second_call_in_the_same_tick_submits_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.desk.enter_triggered() == ["AAA"]
    orders = len(rig.broker.tracker.get_active_orders())
    assert rig.desk.enter_triggered() == []
    assert len(rig.broker.tracker.get_active_orders()) == orders


def test_enter_triggered_without_a_trigger_reads_nothing_and_submits_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.PULLBACK)

    def no_read(*args, **kwargs):
        raise AssertionError("no account or price read is expected without a trigger")

    rig.broker.get_account = no_read
    rig.strategy.get_last_price = no_read
    assert rig.desk.enter_triggered() == []
    assert rig.broker.tracker.get_active_orders() == []


def _trip_the_breaker(rig: Rig) -> None:
    lost = Trade(symbol="ZZZ", entry_order_id="zzz-entry", planned_quantity=D(1), stop_price=D(1), r_per_share=D(1), entered_at=rig.clock.now())
    lost.realised_pnl = D("-2000")  # more than 1.5% of the 100,000 opening equity
    rig.state.book.closed.append(lost)


@pytest.mark.parametrize(
    "block",
    [
        lambda rig: setattr(rig.state, "flattened", True),
        lambda rig: rig.clock.advance(-20 * 60),  # 09:40: before the entry window
        _trip_the_breaker,
    ],
)
def test_enter_triggered_submits_nothing_when_entries_are_closed(tmp_path: Path, block) -> None:
    rig = Rig(tmp_path)
    block(rig)
    assert rig.desk.enter_triggered() == []
    assert rig.broker.tracker.get_active_orders() == []
    assert rig.state.setups["AAA"].state is SetupState.TRIGGERED


def test_enter_triggered_submits_nothing_without_a_free_slot(tmp_path: Path) -> None:
    rig = Rig(tmp_path, params=dataclasses.replace(PARAMS, max_positions=0))
    assert rig.desk.enter_triggered() == []
    assert rig.broker.tracker.get_active_orders() == []


def test_a_refused_entry_does_not_stop_the_following_ones(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _add_trigger(rig, "BBB")
    rig.state.setups["AAA"] = dataclasses.replace(rig.state.setups["AAA"], pullback_low=99.99)  # a stop tighter than the ATR band allows
    assert rig.desk.enter_triggered() == ["BBB"]
    assert rig.state.book.get("AAA") is None


def test_a_price_read_that_fails_for_one_symbol_does_not_stop_the_following_ones(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    _add_trigger(rig, "BBB")
    real_price = rig.strategy.get_last_price

    def flaky(symbol, *args, **kwargs):
        if symbol == "AAA":
            raise BacktestError("no bars for AAA")
        return real_price(symbol, *args, **kwargs)

    rig.strategy.get_last_price = flaky
    assert rig.desk.enter_triggered() == ["BBB"]
```

In `tests/strategies/vwap_pullback/test_vwap_desk_exits.py`, in `test_a_pending_full_exit_frees_its_slot_before_the_sell_fills`, remove `catalyst="earnings", reason="x", ` from the `Trade(...)` call.

In `tests/strategies/vwap_pullback/test_vwap_trades.py`, `_trade` becomes:

```python
def _trade(symbol: str = "AAA") -> Trade:
    return Trade(symbol=symbol, entry_order_id=f"{symbol}-entry", planned_quantity=D(100), stop_price=D("99.00"), r_per_share=D("1.00"), entered_at=T0)
```

and in `test_a_trade_lifecycle_records_pnl_and_closes` the last line becomes:

```python
    assert row["symbol"] == "AAA" and row["realised_pnl"] == "0.00" and "catalyst" not in row and "reason" not in row
```

- [ ] **Step 2: Write the failing strategy tests**

Replace the whole of `tests/strategies/vwap_pullback/test_vwap_strategy.py` with:

```python
from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from tests.fakes import FakeBroker, FakeClock, et, make_session

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState

DAY = date(2026, 9, 1)
TEN_AM = et(2026, 9, 1, 10, 0)


def _strategy(tmp_path: Path, *, mode: TradingMode = TradingMode.PAPER, now=TEN_AM) -> VwapPullbackStrategy:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path)
    strategy.vars.session = None
    strategy._build_components()
    return strategy


def _session() -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"))


def _record_steps(strategy: VwapPullbackStrategy) -> list[str]:
    """Replace the three tick steps by recorders; the list fills in call order."""
    steps: list[str] = []
    strategy.desk.reconcile = lambda now: steps.append("reconcile")
    strategy.scanner.scan = lambda state: steps.append("scan")
    strategy.desk.enter_triggered = lambda: steps.append("enter") or []
    return steps


def test_initialize_builds_the_desk_and_the_scanner_and_no_agent(tmp_path: Path) -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 8, 0), [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, universe=["AAA"], project_root=tmp_path)
    strategy.initialize()
    assert strategy.desk is not None and strategy.scanner is not None
    assert strategy._agents is None  # the lazy agent manager was never touched


def test_a_tick_reconciles_then_scans_then_enters_and_the_session_is_prepared_once(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    prepared: list[date] = []
    strategy.scanner.prepare_session = lambda: prepared.append(DAY) or _session()
    steps = _record_steps(strategy)
    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    assert prepared == [DAY]
    assert steps == ["reconcile", "scan", "enter"] * 2
    assert strategy.vars.session.unknown_positions_checked


def test_a_flattened_session_runs_no_tick_step(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.vars.session = _session()
    strategy.vars.session.flattened = True
    steps = _record_steps(strategy)
    strategy.on_trading_iteration()
    assert steps == []


def test_order_hooks_and_the_close_reach_the_desk(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    seen: list[str] = []
    strategy.desk.on_order_filled = lambda order, price, quantity: seen.append("filled")
    strategy.desk.on_order_canceled = lambda order: seen.append("canceled")
    strategy.desk.flatten_all = lambda reason: seen.append(reason)
    strategy.on_filled_order(None, object(), Decimal(1), Decimal(1), 1)
    strategy.on_canceled_order(object())
    strategy.before_market_closes()
    assert seen == ["filled", "canceled", "end-of-day flatten"]


def test_main_registers_the_strategy() -> None:
    from trading_agent_framework.main import AGENT_STRATEGIES

    assert "vwap_pullback_continuation" in AGENT_STRATEGIES


def test_the_strategy_package_imports_no_llm_library() -> None:
    # A fresh interpreter: this process has already imported LangChain through other tests.
    code = "import sys; import trading_agent_framework.strategies.vwap_pullback; bad = [m for m in ('langchain', 'langchain_openai', 'langgraph') if m in sys.modules]; assert not bad, bad"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_backtest_waits_in_one_minute_slices_so_the_stop_follows_the_entry_bar_by_bar(tmp_path: Path) -> None:
    # Final review I1: with the backtest clock's default infinite slice, one 5-minute tick jumps 10:00 -> 10:05 at
    # once, so the entry filled on the 10:01 bar gets its stop only at 10:05 and the 10:03 break is never checked.
    from tests.fakes import FrameDataSource, minute_ohlc
    from tests.strategies.vwap_pullback.test_vwap_desk_entries import ROWS

    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.backtesting.clock import BacktestClock
    from trading_agent_framework.strategies.vwap_pullback.features import BarContext
    from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo
    from trading_agent_framework.strategies.vwap_pullback.trades import TradeStatus

    clock = BacktestClock(start=TEN_AM, sessions=[make_session(DAY)])
    frames = {("AAA", "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)}
    broker = BacktestBroker("vwap_pullback_continuation", data_source=FrameDataSource(frames), clock=clock, budget=Decimal("100000"), timestep="minute")
    clock.on_advance = broker.on_advance
    strategy = VwapPullbackStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], project_root=tmp_path)
    strategy.initialize()
    assert strategy.clock.max_wait_slice == 60.0
    assert BacktestClock.max_wait_slice == float("inf")  # set on this run's clock only
    broker.tracker.listeners.append(strategy.executor._events)  # what executor.run() wires
    state = SessionState(day=DAY, session=make_session(DAY), bar_stamp="close", session_open_equity=Decimal("100000"))
    state.candidates["AAA"] = CandidateInfo(symbol="AAA", daily_atr=2.0, beta=1.0)
    state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
    state.contexts["AAA"] = [BarContext(time=TEN_AM, open=100, high=100.2, low=99.8, close=100, volume=5000, vwap=99.9, rs=0.01, rvol=2.0, session_open=99.0, session_high=100.2)]
    strategy.vars.session = state
    assert strategy.desk.enter_triggered() == ["AAA"]
    trade = state.book.get("AAA")
    strategy.executor.wait_until(TEN_AM + timedelta(seconds=300))  # one 5-minute tick
    assert trade.status is TradeStatus.CLOSED and trade.exit_reason == "stop"
    stop_fill = next(f for f in broker.ledger.fills if f.side.value == "sell")
    assert stop_fill.time == et(2026, 9, 1, 10, 3) and stop_fill.price == Decimal("99.30")  # the breaking bar, at the stop
```

In `tests/strategies/vwap_pullback/test_vwap_setups.py`: remove `health` from the `setups` import, delete `test_health_flags`, and replace `test_health_reports_the_pullback_depth_not_the_shallower_retracement_after_the_bounce` with:

```python
def test_max_retracement_keeps_the_pullback_depth_after_the_bounce() -> None:
    triggered = _run(B1, B2, B3, B4, B5)
    assert triggered.state is SetupState.TRIGGERED
    assert triggered.retracement < triggered.max_retracement  # the trigger bar closed higher than the pullback's low close
    assert triggered.max_retracement == pytest.approx(0.306, abs=0.001)
```

In `tests/strategies/vwap_pullback/test_vwap_news_prompts.py`: delete the two prompt tests (`test_entry_prompt_names_the_tools_the_catalysts_and_the_thresholds`, `test_exit_prompt_names_every_action_and_the_flatten_time`) and the `VwapPullbackParameters` and `prompts` imports; only `test_lean_headlines_keeps_the_newest_distinct_headlines` and the `lean_headlines` import remain.

Delete the two agent test files:

```bash
git rm tests/strategies/vwap_pullback/test_vwap_graph.py tests/strategies/vwap_pullback/test_vwap_tools.py
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/vwap_pullback/ -q`
Expected: failures in `test_vwap_desk_entries.py` (`TypeError: Desk.enter_long() missing 2 required positional arguments`, `AttributeError: 'Desk' object has no attribute 'enter_triggered'`), in `test_vwap_trades.py` and `test_vwap_desk_exits.py` (`TypeError: Trade.__init__() missing 2 required positional arguments: 'catalyst' and 'reason'`), and in `test_vwap_strategy.py` (the tick test errors with `AttributeError: 'NoneType' object has no attribute 'invoke'`, because the tick still calls `self._graph.invoke`; the initialize test fails while creating the agents or on `_agents is None`; the backtest test fails on `enter_triggered`). `test_the_strategy_package_imports_no_llm_library` may already pass; it pins the property.

- [ ] **Step 4: Implement the desk entries**

`desk.py` — module docstring, first two paragraphs become:

```python
"""Every order the vwap_pullback strategy places: entries, protective stops, exit hand-offs, the flatten.

The only module of the package that submits, cancels or modifies orders. The strategy reaches it through
`enter_triggered` every tick and through its order hooks (`on_order_filled`/`on_order_canceled`).
`enter_long` returns `{"error": ...}` instead of raising; hook paths log and never raise. A position is
never left without a stop: a stop that cannot be placed is replaced by an immediate market sell.
```

`desk.py` — delete the import line `from trading_agent_framework.strategies.vwap_pullback.prompts import CATALYSTS`.

`desk.py` — in the `views` section delete the methods `planned_risk`, `awaits_decision` and `entry_due`.

`desk.py` — replace the whole `entries` section (from the `# --- entries ---` comment through the end of `pass_on_setup`, stopping before `_has_exposure`) with:

```python
# --- entries -------------------------------------------------------------------


def enter_triggered(self) -> list[str]:
    """Try every triggered setup, best stage-2 score first; the symbols whose entry was submitted.

    The score is the filter when triggers outnumber free slots: the best-ranked take the slots. A
    triggered symbol with no score this tick (tracked only because its setup is under way) goes after
    every scored one. A refused entry is logged by `enter_long` and the next trigger is tried.
    """
    state = self.state
    triggered = [symbol for symbol, setup in state.setups.items() if setup.state is SetupState.TRIGGERED]
    if not triggered or state.flattened or not risk.in_entry_window(self._strategy.get_datetime(), self._params) or self.breaker_tripped():
        return []
    # Unscored last (False sorts before True), then the higher score, then the symbol: the same order from run to run.
    triggered.sort(key=lambda symbol: (symbol not in state.scores, -state.scores.get(symbol, 0.0), symbol))
    entered: list[str] = []
    for index, symbol in enumerate(triggered):
        if self.free_slots() <= 0:
            self._strategy.log_info(f"no free position slot for {', '.join(triggered[index:])}")
            break
        if "error" not in self.enter_long(symbol):
            entered.append(symbol)
    return entered


def enter_long(self, symbol: str) -> dict[str, Any]:
    """Enter a triggered setup: validate, size (`risk.plan_entry`), submit a marketable limit buy.

    Every rule is checked here, and a refusal comes back as `{"error": ...}` (and is logged as a warning,
    so the run log says why a triggered setup was not entered).
    The protective stop is NOT placed here: it goes in when the entry fills (`on_order_filled` -> `_protect`).
    """
    result = self._enter_long(symbol)
    if "error" in result:
        self._strategy.log_warning(f"entry {symbol.strip().upper()} refused: {result['error']}")
    return result


def _enter_long(self, symbol: str) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    state = self.state
    setup = state.setups.get(symbol)
    # Guards, cheapest first; each names the rule so the warning in the log explains itself.
    if setup is None or setup.state is not SetupState.TRIGGERED:
        current = setup.state.value if setup is not None else "untracked"
        return {"error": f"{symbol} has no triggered setup right now (state: {current}); only triggered setups can be entered"}
    if state.flattened:
        return {"error": "the session is already flattened; no more entries today"}
    now = self._strategy.get_datetime()
    if not risk.in_entry_window(now, self._params):
        return {"error": f"entries are only allowed between {self._params.no_entry_before:%H:%M} and {self._params.no_entry_after:%H:%M}"}
    if self.breaker_tripped():
        return {"error": "the daily loss limit is reached; no more entries today"}
    if self.free_slots() <= 0:
        return {"error": "no free position slot"}
    if self._has_exposure(symbol):
        return {"error": f"{symbol} already has a position or an open order"}
    info = state.candidates.get(symbol)
    if info is None:
        return {"error": f"{symbol} is not a candidate this session"}
    try:
        last = self._strategy.get_last_price(symbol)
        account = self._strategy.broker.get_account()
    except _DATA_ERRORS as exc:
        return {"error": f"price or account unavailable: {exc}"}
    if last is None or setup.trigger_close is None or setup.pullback_low is None:
        return {"error": f"no price for {symbol}"}
    try:
        plan = risk.plan_entry(
            trigger_close=setup.trigger_close,
            pullback_low=setup.pullback_low,
            last_price=last,
            daily_atr=info.daily_atr,
            equity=account.portfolio_value,
            buying_power=account.buying_power,
            cash=account.cash,
            pending_sell_proceeds=self._pending_sell_proceeds(),
            params=self._params,
        )
    except risk.EntryRefused as exc:
        return {"error": str(exc)}
    try:
        submitted = self._strategy.submit_order(self._strategy.create_order(symbol, plan.quantity, "buy", limit_price=plan.limit_price))
    except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
        return {"error": str(exc)}
    # The trade exists from submission (PENDING): it takes a slot, and the fill/cancel hooks find it by order id.
    state.book.add(
        Trade(
            symbol=symbol,
            entry_order_id=submitted.identifier,
            planned_quantity=plan.quantity,
            stop_price=plan.stop_price,
            r_per_share=plan.r_per_share,
            entered_at=now,
        )
    )
    state.setups[symbol] = mark_in_trade(setup)  # freeze the setup: one trade per symbol per session
    self._strategy.log_info(f"entry {symbol}: {plan.quantity} at limit {plan.limit_price}, stop {plan.stop_price}, R {plan.r_per_share}")
    return {
        "symbol": symbol,
        "quantity": int(plan.quantity),
        "limit_price": float(plan.limit_price),
        "stop_price": float(plan.stop_price),
        "r_per_share": float(plan.r_per_share),
        "status": "entry submitted",
    }
```

`desk.py` — the `params` property docstring becomes `"""The strategy's parameters."""`.

- [ ] **Step 5: Implement the trade, session, setups, parameters and risk edits**

`trades.py` — in `Trade`, delete the two fields `catalyst: str ...` and `reason: str ...`; in `to_json`, delete the two lines `"catalyst": self.catalyst,` and `"reason": self.reason,`.

`session.py` — delete the fields `decided` and `passes`; the module docstring's last sentence becomes:

```python
builds it, `Scanner.scan` and `Desk` mutate it.
```

`setups.py` — delete the `health` function (the last function of the file); the `TRIGGERED` line becomes:

```python
    TRIGGERED = "triggered"  # this bar resumed upward: the desk enters it now
```

`parameters.py` — delete the three lines `news_calls_per_run`, `bars_calls_per_run`, `max_passes_per_symbol`; the section comment `# --- agents (`prompts.py`, `tools.py`, `desk.Desk.exit_review_due`) ---` becomes `# --- exit reviews and headlines (`desk.Desk.exit_review_due`, `scanner.Scanner.refresh_headlines`) ---`; in the module docstring replace `the prompts, the screens, the state machine and the` with `the screens, the state machine and the`.

`risk.py` — the module docstring sentence starting `The agent never picks a size or a price:` becomes:

```python
Size and prices are always the code's: `Desk.enter_long` calls `plan_entry`, and any rule it breaks comes
back as the text of an `EntryRefused`.
```

and the `EntryRefused` docstring becomes `"""Why `plan_entry` refused; `Desk.enter_long` returns the message as `{"error": ...}` and logs it."""`.

- [ ] **Step 6: Rewrite the strategy and delete the agent modules**

```bash
git rm src/trading_agent_framework/strategies/vwap_pullback/prompts.py src/trading_agent_framework/strategies/vwap_pullback/tools.py src/trading_agent_framework/strategies/vwap_pullback/graph.py
```

Replace the whole of `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` with:

```python
"""VwapPullbackStrategy: an intraday VWAP pullback continuation strategy, in code only (no LLM).

Every tick (5 minutes): `Desk.reconcile` settles what the order hooks may have missed, `Scanner.scan`
advances the setups, `Desk.enter_triggered` enters the triggered ones (best stage-2 score first). A trade
ends on its protective stop or at the 15:50 flatten. Code owns everything (`Desk`): sizing, the protective
stop on every fill, the loss limit, the entry window and the flatten.
See docs/superpowers/specs/2026-10-01-vwap-pullback-pure-code-design.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError


class VwapPullbackStrategy(Strategy):
    """Wires `Scanner` and `Desk` into the framework's lifecycle hooks.

    Hook map: `before_market_opens` prepares the session (stage 1); `on_trading_iteration` runs one tick
    (reconcile, scan, entries); `on_filled_order`/`on_canceled_order` forward to the desk (protective stops,
    trade bookkeeping); `before_market_closes` flattens at 15:50.
    """

    sleeptime = "5M"  # one tick per 5-minute bar
    minutes_before_closing = 10  # before_market_closes, and so the flatten, runs at 15:50

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.MONTH)[0],
        "backtesting_end": backtest_window(PredefinedWindow.MONTH)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 75,  # 70 daily bars for stage 1, plus the RVOL baseline sessions
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: VwapPullbackParameters | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or VwapPullbackParameters()
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None

    # --- lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the desk and the scanner (once per run)."""
        self.vars.session = None
        if self.is_backtesting:
            # A backtest clock jumps a whole 5-minute tick at once: an entry filled on the tick's first bar would get
            # its stop (placed by the fill hook) only at the tick's end, with the bars in between never checked
            # against it. One-minute slices let the executor dispatch fills, and so place stops, bar by bar.
            # Set on this run's clock instance only: other strategies' clocks keep the class's infinite slice.
            self.clock.max_wait_slice = 60.0
        self._build_components()
        self.log_info(f"VwapPullbackStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def _build_components(self) -> None:
        """Create the desk and the scanner (separate from `initialize` so tests can build them on their own)."""
        self.desk = Desk(self, self.settings, trade_log=self._trade_log_path)
        self.scanner = Scanner(self, self.settings, self.universe, benchmark=self.parameters["benchmark_symbol"], preload=self._preload if self.is_backtesting else None)

    def before_market_opens(self) -> None:
        """Run stage 1 before the open, so the first tick can scan right away."""
        self._ensure_session()

    def on_trading_iteration(self) -> None:
        """One tick: make sure the session is prepared, run the restart check once, then reconcile, scan and enter."""
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the last minute bar be published
        state = self._ensure_session()
        # No session (preparation failed; retried next tick) or already flattened: nothing to do this tick.
        if state is None or state.flattened:
            return
        assert self.desk is not None and self.scanner is not None
        # First tick of a session: cancel/close what a previous run of this strategy left behind (restart).
        if not state.unknown_positions_checked:
            self.desk.close_unknown_positions()
        # Reconcile first: expired/rejected entries and missing stops are settled before the scan and the entries.
        self.desk.reconcile(self.get_datetime())
        self.scanner.scan(state)
        self.desk.enter_triggered()

    def before_market_closes(self) -> None:
        """15:50 (`minutes_before_closing`): sell everything this strategy holds; the strategy is strictly intraday."""
        if self.desk is not None:
            self.desk.flatten_all("end-of-day flatten")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        """Order hook (executor thread): an entry fill gets its protective stop, an exit fill is booked."""
        if self.desk is not None:
            self.desk.on_order_filled(order, price, quantity)

    def on_canceled_order(self, order: Order) -> None:
        """Order hook: settle an expired entry, re-place a stop cancelled from outside, re-protect after an exit dies."""
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    def _ensure_session(self) -> SessionState | None:
        """This session's state, prepared on first need: `before_market_opens` does not run when a live run starts mid-session."""
        today = self.get_datetime().astimezone(MARKET_TZ).date()
        state = self.vars.session
        if state is None or state.day != today:
            assert self.scanner is not None
            try:
                self.vars.session = self.scanner.prepare_session()
            except (BrokerError, BacktestError) as exc:
                self.log_error(f"session preparation failed, skipping this tick: {exc}")
                self.vars.session = None
        return self.vars.session

    # --- backtesting ---------------------------------------------------------------------

    def _trade_log_path(self) -> Path | None:
        """`trades.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "trades.jsonl"

    def _preload(self, assets: Sequence[Asset], timestep: str) -> None:
        """Backtests only: batch-load `assets` over the data source's own window before the scanner reads them.

        The window must equal the one `run_backtesting` builds the data source with (backtest start minus the
        warm-up, to the end): a source that caches one frame per asset would otherwise keep a shorter frame.
        """
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.warmup import warmup_calendar_days

        if not isinstance(self.broker, BacktestBroker):
            return
        start = self.parameters["backtesting_start"] - timedelta(days=warmup_calendar_days(self.parameters["warmup_trading_days"]))
        self.broker.preload_bars(assets, start, self.parameters["backtesting_end"], timestep)

    def run_backtesting(self):
        """Backtest over the class `parameters` window on Alpaca minute bars (only the benchmark preloaded)."""
        # class parameters: the same window the data source is built with, so preload_bars matches it
        return super().run_backtesting(
            data_source=AlpacaBacktestData,  # minute bars with enough history (Yahoo keeps ~30 days of minutes)
            timestep="minute",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=False,  # no agent runs: nothing to record
        )
```

`src/trading_agent_framework/strategies/vwap_pullback/__init__.py` — the docstring (first line) becomes:

```python
"""Intraday VWAP pullback continuation: a code-only strategy (no LLM) over a Python setup scanner."""
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/vwap_pullback/ -q && uv run ruff check`
Expected: all pass, no lint error. If `ruff` reports an unused import in `desk.py` or `agent_vwap_pullback.py`, remove exactly that import and re-run.

Then the whole suite, since `main.py` imports the strategy: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/ tests/strategies/vwap_pullback/
git commit -m "feat: vwap_pullback enters triggers in code; both agents and the tick graph removed"
```

---

### Task 3: Remove the exit actions (stop and flatten only)

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/desk.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/trades.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/features.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/parameters.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_desk_exits.py` (rewritten), `test_vwap_desk_entries.py`, `test_vwap_trades.py`, `test_vwap_features.py`

**Interfaces:**
- Consumes: `Desk.enter_long(symbol)` and `Trade(...)` without `catalyst`/`reason` (Task 2).
- Produces:
  - `Trade.to_json()` returns exactly these keys: `symbol`, `entered_at`, `closed_at`, `entry_price`, `filled_quantity`, `stop_price`, `r_per_share`, `realised_pnl`, `realised_r`, `exit_reason`.
  - `Desk.last_close(symbol) -> Decimal | None` (unchanged signature, now reads the last scanned bar directly).
  - `Desk._submit_stop(trade, quantity) -> bool` always builds a plain stop at `trade.stop_level`.
  - Removed from `Desk`: `take_partial_profit`, `tighten_stop`, `replace_stop_with_trailing`, `exit_position`, `hold`, `_restored`, `_split`, `_open_trade`, `exit_review_due`, `mark_reviewed`, `_flags`, `minutes_to_flatten`, `levels`.
  - Removed from `trades.py`: `Trade.tp1_done`, `stop_kind`, `trail_price`, `last_review_at`, `review_flags`, `Trade.unrealised_r`, `trade_flags`, `exit_review_due`.
  - Removed from `features.py`: `Levels`, `latest_levels`, `ema_last`, `bar_atr`.
  - Removed from `VwapPullbackParameters`: `ema_length`, `tp1_fraction_band`, `trail_atr_band`.

The order of the steps matters: the tests are first moved off the exit actions and run against the OLD code (Step 2), which proves the hand-built states are the ones the actions used to produce, before any code is deleted.

- [ ] **Step 1: Move the tests off the exit actions**

Replace the whole of `tests/strategies/vwap_pullback/test_vwap_desk_exits.py` with:

```python
from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig

from trading_agent_framework.strategies.vwap_pullback.desk import STOPPED_OUT
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus


def _open(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)
    rig.open_trade()
    return rig


def test_releasing_a_stop_that_already_filled_reports_it_and_sells_nothing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    rig.advance(120)  # the stop fills in the broker; the hook has not reached the desk yet
    assert rig.desk._release_stop(trade) == STOPPED_OUT
    assert trade.exit_order_ids == []


def test_a_stop_that_cannot_be_placed_reports_false(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk._submit_stop(trade, D(10_000)) is False  # more than is held: refused, and so is the market-sell fallback


def test_a_pending_full_exit_frees_its_slot_before_the_sell_fills(tmp_path: Path) -> None:
    # Final review I3 (spec §4): pending entries count as taken, pending full exits as freed.
    rig = _open(tmp_path)
    for symbol in ("BBB", "CCC", "DDD"):
        rig.state.book.add(
            Trade(
                symbol=symbol,
                entry_order_id=f"entry-{symbol}",
                planned_quantity=D(10),
                stop_price=D(90),
                r_per_share=D(1),
                entered_at=rig.clock.now(),
                status=TradeStatus.OPEN,
                quantity=D(10),
            )
        )
    assert rig.desk.free_slots() == 0
    trade = rig.state.book.get("AAA")
    assert rig.desk._release_stop(trade) is None
    sell = rig.desk._market_sell(trade, D(249), "flatten")
    assert sell is not None and sell.is_active() and trade.status is TradeStatus.OPEN
    assert rig.desk.free_slots() == 1


def test_releasing_a_stop_that_already_ended_part_filled_books_the_partial(tmp_path: Path) -> None:
    # Final review M1: a stop already cancelled after a partial fill, its hook not seen yet, when the flatten releases it.
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    stop.filled_quantity, stop.avg_fill_price = D(100), D("99.30")
    rig.broker.cancel_order(stop)
    assert rig.desk._release_stop(trade) is None
    assert trade.quantity == D(149) and trade.realised_pnl == D("-70.00") and trade.stop_order_id is None
```

In `tests/strategies/vwap_pullback/test_vwap_desk_entries.py`:

(a) Delete `test_exit_review_due`, and change the import `from datetime import date, datetime, timedelta` to `from datetime import date, datetime` (that test was the last user of `timedelta`).

(b) Replace the helper `_after_tp1` with (same state the partial-profit action left: a working market sell of 124 beside a stop resized to the other 125):

```python
def _after_partial_sell(rig: Rig):
    """A working market sell of 124 shares beside a stop resized to the other 125, built with the desk's own helpers."""
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    assert rig.desk._release_stop(trade) is None
    sell = rig.desk._market_sell(trade, D(124), "partial sell")
    assert sell is not None and rig.desk._submit_stop(trade, D(125))
    assert sell.quantity == D(124) and rig.strategy.get_order(trade.stop_order_id).quantity == D(125)
    return trade, sell
```

(c) Replace every `_after_tp1(rig)` call with `_after_partial_sell(rig)` (six tests: `test_a_zero_fill_cancelled_tp1_sell_resizes_the_stop_to_every_share`, `test_a_partly_filled_cancelled_tp1_sell_books_the_fill_and_resizes_the_stop`, `test_flattened_never_sells_the_shares_an_unconfirmed_stop_still_covers`, `test_reconcile_settles_an_exit_sell_that_ended_in_error`, `test_flattened_sells_only_what_a_partly_filled_unconfirmed_stop_does_not_cover`, `test_a_filled_but_unbooked_tp1_sell_still_counts_as_not_free`). The local variable name `tp1` in those tests may stay.

(d) In `test_flatten_sells_the_other_shares_when_a_tp1_sell_is_working_and_the_stop_cancel_is_late`, replace its first six lines (from `rig = Rig(tmp_path)` through the `assert tp1.is_active() ...` line) with:

```python
    rig = Rig(tmp_path)
    trade, tp1 = _after_partial_sell(rig)
    stop = rig.strategy.get_order(trade.stop_order_id)
    assert tp1.is_active() and tp1.quantity == D(124) and stop.quantity == D(125)
```

(e) In `test_free_slots_counts_a_filled_but_unbooked_full_exit_as_freed`, replace the line `rig.desk.exit_position("AAA", "downgrade")` and the following `trade = ...` line with:

```python
    trade = rig.state.book.get("AAA")
    assert rig.desk._release_stop(trade) is None
    assert rig.desk._market_sell(trade, D(249), "flatten") is not None
```

- [ ] **Step 2: Run the moved tests against the old code**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_desk_entries.py tests/strategies/vwap_pullback/test_vwap_desk_exits.py -q`
Expected: all pass. This is the safety net: the order-handling tests no longer depend on the exit actions. If one fails here, stop and fix the test helper, not the desk.

- [ ] **Step 3: Write the failing tests for the removals**

In `tests/strategies/vwap_pullback/test_vwap_trades.py`:

- the import line becomes `from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeBook, TradeStatus`; delete `import dataclasses` and the `VwapPullbackParameters` import and the `PARAMS = ...` line;
- replace `test_unrealised_pnl_and_r` with:

```python
def test_unrealised_pnl() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.unrealised_pnl(D("101.50")) == D("150.00")
```

- delete `test_exit_review_is_due_on_a_new_flag_a_new_headline_or_elapsed_time`;
- add:

```python
def test_to_json_has_exactly_the_trade_log_fields() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    trade.exit_reason = "stop"
    trade.record_exit_fill(D(100), D("99.00"), et(2026, 9, 1, 11, 0))
    assert trade.to_json() == {
        "symbol": "AAA",
        "entered_at": T0.isoformat(),
        "closed_at": et(2026, 9, 1, 11, 0).isoformat(),
        "entry_price": "100.00",
        "filled_quantity": "100",
        "stop_price": "99.00",
        "r_per_share": "1.00",
        "realised_pnl": "-100.00",
        "realised_r": -1.0,
        "exit_reason": "stop",
    }
```

In `tests/strategies/vwap_pullback/test_vwap_features.py`: remove `bar_atr`, `ema_last` and `latest_levels` from the import list and delete `test_ema_bar_atr_and_latest_levels`.

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_trades.py -q`
Expected: `test_to_json_has_exactly_the_trade_log_fields` FAILS (the dict still has `tp1_done` and `stop_kind`).

- [ ] **Step 4: Delete the exit code**

`trades.py`:

- module docstring first line becomes `"""Pure trade records: one `Trade` per entry and the session's `TradeBook`.`;
- imports: `from datetime import datetime` (drop `timedelta`); delete the `VwapPullbackParameters` import;
- in `Trade`, delete the fields `stop_kind`, `trail_price`, `tp1_done`, `last_review_at`, `review_flags`; the `stop_level` comment becomes `# the working stop's level (the initial stop: nothing moves it)`; the `__post_init__` comment becomes `# The working stop starts at the initial stop.`;
- delete the method `unrealised_r`;
- in `to_json`, delete the lines `"tp1_done": self.tp1_done,` and `"stop_kind": self.stop_kind,`;
- delete the module-level functions `trade_flags` and `exit_review_due`.

`features.py`:

- module docstring first line becomes `"""Pure intraday features over minute bars: session slicing, 5-minute contexts, VWAP, RVOL, RS, ATR.`;
- in the `BarContext` docstring replace `the state machine and the agents get` with `the state machine gets`;
- delete the `Levels` dataclass and the functions `bar_atr`, `ema_last`, `latest_levels` (`_true_range` and `daily_atr` stay).

`parameters.py`: delete the lines `ema_length`, `tp1_fraction_band`, `trail_atr_band`; the `atr_length` comment becomes `# ATR window (daily bars)`; the `bar_minutes` comment becomes `# bar size the state machine reasons on`.

`desk.py`:

- module docstring: its first line becomes `"""Every order the vwap_pullback strategy places: entries, protective stops, the flatten.`;
- imports: `from datetime import datetime`; `from decimal import Decimal`; delete the `features` import line (`Levels, latest_levels`); the `trades` import becomes `from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus`;
- the `STOPPED_OUT` comment becomes:

```python
# `_release_stop`'s answer when the stop has already filled: the trade is out, nothing else to do.
STOPPED_OUT = "already_stopped_out"
```

- class docstring's `Sections:` sentence becomes `Sections: views (read-only), entries, order events (hooks), order helpers, session boundaries (flatten, restart).`;
- delete the methods `levels`, `minutes_to_flatten`, `_flags`, `exit_review_due`, `mark_reviewed`, `_open_trade`;
- `last_close` becomes:

```python
    def last_close(self, symbol: str) -> Decimal | None:
        """Latest 5-minute close as Decimal (no I/O: from the last scan); None without bars."""
        contexts = self.state.contexts.get(symbol)
        return Decimal(str(contexts[-1].close)) if contexts else None
```

- in `on_order_filled`: the docstring's second paragraph becomes `Entry fill -> the trade opens and gets its protective stop. Exit fill (stop or flatten sell) -> shares booked out, the order's id dropped from the unbooked list, the trade archived when nothing is left. A buy of ours with no trade at all is sold at once (`_sell_orphan_fill`).`; the line setting the exit reason becomes `trade.exit_reason = trade.exit_reason or "stop"`;
- in `_record_stop_partial`, the same line becomes `trade.exit_reason = trade.exit_reason or "stop"`;
- `_pending_sell_proceeds` docstring becomes `"""What working market/limit sells should bring in (a stop only sells if triggered)."""`;
- `_submit_stop` becomes:

```python
    def _submit_stop(self, trade: Trade, quantity: Decimal) -> bool:
        """Place the trade's protective stop for `quantity`: True when placed; False when it fell back to a market sell (or that failed too)."""
        if quantity <= 0:
            return False
        order = self._strategy.create_order(trade.symbol, quantity, "sell", stop_price=trade.stop_level)
        try:
            submitted = self._strategy.submit_order(order)
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"stop for {trade.symbol} could not be placed ({exc}); selling {quantity} now")
            self._market_sell(trade, quantity, "protective stop failed")
            return False
        trade.stop_order_id = submitted.identifier
        return True
```

- delete the whole `# --- exits (the exit agent's actions) ---` section: the section comment, the two-line comment under it, and the methods `take_partial_profit`, `tighten_stop`, `replace_stop_with_trailing`, `exit_position`, `hold`, `_restored`, `_split` (everything up to, not including, `# --- session boundaries ---`).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/vwap_pullback/ -q && uv run ruff check`
Expected: all pass, no lint error. If `ruff` reports an unused import (`Any`, `timedelta`, `ROUND_*`, `OrderType`, ...), remove exactly that import and re-run; do not remove an import that is still used.

Then: `grep -n "trail\|tp1\|stop_kind\|review\|levels(\|agent" src/trading_agent_framework/strategies/vwap_pullback/desk.py src/trading_agent_framework/strategies/vwap_pullback/trades.py src/trading_agent_framework/strategies/vwap_pullback/features.py | grep -v "import"`
Expected: prints nothing. A hit means a reference or a comment was missed: remove or reword it. (`scanner.py`, `session.py` and `parameters.py` still mention headlines and reviews: Task 4 removes those.)

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/ tests/strategies/vwap_pullback/
git commit -m "refactor: vwap_pullback trades end on the stop or the flatten; exit actions removed"
```

---

### Task 4: Remove the news fetch

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/scanner.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/session.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/parameters.py`
- Delete: `src/trading_agent_framework/strategies/vwap_pullback/news.py`, `tests/strategies/vwap_pullback/test_vwap_news_prompts.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_scanner.py`

**Interfaces:**
- Consumes: `Trade(...)` as left by Task 3 (no `catalyst`, `reason`, `stop_kind`).
- Produces: `Scanner.scan(state)` makes no news call. Removed: `Scanner.refresh_headlines`, `SessionState.headlines`, `headlines_fetched_at`, `new_headline`, `VwapPullbackParameters.exit_review_minutes`, `headlines_per_symbol`, the module `vwap_pullback/news.py`.

- [ ] **Step 1: Write the failing test**

In `tests/strategies/vwap_pullback/test_vwap_scanner.py`:

- add the imports `from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus`;
- delete `test_refresh_headlines_fetches_active_setups_and_flags_new_ones`;
- in `test_scan_builds_contexts_and_advances_setups`, delete its last line (`assert broker.news.calls == []  # headlines are only fetched ...`);
- the `setups` import becomes `from trading_agent_framework.strategies.vwap_pullback.setups import SetupState` if `Setup` has no other use in the file after the deletion (check with `grep -n "Setup(" tests/strategies/vwap_pullback/test_vwap_scanner.py`);
- add:

```python
def test_scan_fetches_no_news(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 9, 50, 30))
    rising = [(100 + 0.06 * i, 100 + 0.06 * (i + 1) + 0.01, 100 + 0.06 * i - 0.01, 100 + 0.06 * (i + 1), 300.0) for i in range(20)]
    broker.timestep_frames = {
        ("AAA", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), rising),
        ("SPY", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), [(400, 400, 400, 400, 1000)] * 20),
    }
    broker.news = FakeNewsProvider()
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.baselines["AAA"] = pd.Series([100.0 * (m + 1) for m in range(390)])
    # An open trade: the symbol whose headlines used to be fetched on every scan.
    state.book.add(
        Trade(
            symbol="AAA",
            entry_order_id="aaa-entry",
            planned_quantity=Decimal(1),
            stop_price=Decimal(99),
            r_per_share=Decimal(1),
            entered_at=et(2026, 9, 2, 9, 45),
            status=TradeStatus.OPEN,
            quantity=Decimal(1),
        )
    )
    Scanner(strategy, PARAMS, ["AAA"]).scan(state)
    assert broker.news.calls == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_scanner.py::test_scan_fetches_no_news -q`
Expected: FAIL — `broker.news.calls` holds one call for `AAA`.

- [ ] **Step 3: Implement**

```bash
git rm src/trading_agent_framework/strategies/vwap_pullback/news.py tests/strategies/vwap_pullback/test_vwap_news_prompts.py
```

`scanner.py`:

- module docstring becomes:

```python
"""Session preparation (stage 1) and the per-tick scan (stage 2, setups).

The I/O side of candidate selection: bars come in through the strategy and broker, go through the pure
`features`/`screening`/`setups` modules, and land in `SessionState`. Price reads go through the strategy
(and so, in a backtest, through `BacktestBroker._source_bars`, the no-look-ahead gate).
"""
```

- delete the import `from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines`;
- delete the constant `_NEWS_LOOKBACK`;
- the `Scanner` class docstring becomes `"""Finds and tracks setups: stage 1 once per session, stage 2 + setup state machine every tick."""`;
- the `scan` docstring becomes `"""One tick: contexts for every candidate, stage-2 ranking and scores, setups advanced."""`;
- delete the last line of `scan` (`self.refresh_headlines(state, now)`);
- delete the method `refresh_headlines`.

`session.py`: delete the fields `headlines`, `headlines_fetched_at`, `new_headline`.

`parameters.py`: delete the section comment `# --- exit reviews and headlines (`desk.Desk.exit_review_due`, `scanner.Scanner.refresh_headlines`) ---` and the lines `exit_review_minutes` and `headlines_per_symbol`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/vwap_pullback/ -q && uv run ruff check`
Expected: all pass, no lint error. If `ruff` reports an unused import in `scanner.py` or `session.py` (`timedelta`, `datetime`, ...), remove exactly that import and re-run. `BrokerError` stays in `scanner.py`: `prepare_session` raises it.

Then: `grep -rn "headline\|news" src/trading_agent_framework/strategies/vwap_pullback/`
Expected: prints nothing.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/ tests/strategies/vwap_pullback/
git commit -m "refactor: vwap_pullback no longer fetches news"
```

---

### Task 5: Drop the `langgraph` dependency and update the docs

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Modify: `CLAUDE.md`
- Modify: `TODO.md`

**Interfaces:**
- Consumes: the package as left by Tasks 2-4 (no file imports `langgraph`).
- Produces: nothing other tasks rely on.

- [ ] **Step 1: Confirm nothing imports `langgraph`**

Run: `grep -rn "langgraph" src/ tests/ scripts/`
Expected: one hit only, the module-name string in `tests/strategies/vwap_pullback/test_vwap_strategy.py` (`test_the_strategy_package_imports_no_llm_library`). Any `import langgraph` or `from langgraph` hit means stop and report it.

- [ ] **Step 2: Remove the dependency**

In `pyproject.toml`, delete the line `    "langgraph>=1.2,<2",` from the `dependencies` list.

Run: `uv sync`
Expected: completes; `uv.lock` is rewritten. `langgraph` may stay in `uv.lock` as a dependency of `langchain` (LangChain's agents are built on it): that is expected, only the direct requirement goes.

- [ ] **Step 3: Update `CLAUDE.md`**

In the `strategies/` bullet of the Architecture section, replace this text:

```
`vwap_pullback/` (`VwapPullbackStrategy`, registered as `"vwap_pullback_continuation"`): strictly intraday, 5-minute ticks. Pure `features`/`screening`/`setups`/`risk`/`trades`; `Scanner` (stage 1 before the open from the cross_momentum universe file, stage 2 and the setup state machine each tick), `Desk` (the only order code: code-sized entries, a protective stop on every fill, exit hand-offs, 15:50 flatten), `tools.py` (no raw order tools for the agents), `graph.py` (per-tick LangGraph: classify → exit agent → entry agent, agents only when due). Closed trades go to `trades.jsonl` in the run directory.
```

with:

```
`vwap_pullback/` (`VwapPullbackStrategy`, registered as `"vwap_pullback_continuation"`): strictly intraday, 5-minute ticks, code only (no LLM, no news). Pure `features`/`screening`/`setups`/`risk`/`trades`; `Scanner` (stage 1 before the open from the cross_momentum universe file, stage 2 and the setup state machine each tick, the stage-2 scores kept in `SessionState.scores`), `Desk` (the only order code: `enter_triggered` tries every triggered setup, best stage-2 score first when triggers outnumber free slots; code-sized entries, a protective stop on every fill, 15:50 flatten). A tick is reconcile → scan → enter; a trade ends on its stop or at the flatten. Closed trades go to `trades.jsonl` in the run directory.
```

In the gotcha that starts `**vwap_pullback never leaves a position without a stop**`, replace `` `Desk` cancels a stop and waits for the cancel before any other exit sell `` with `` `Desk` cancels a stop and waits for the cancel before the flatten sell ``. Leave the rest of that paragraph as it is.

- [ ] **Step 4: Update `TODO.md`**

In the `### VWAP Pullback continuation` list, strike through the two finished items (the other lines of the file, including the user's uncommitted additions, stay as they are):

```
* ~~Remove entry agent: it is effectless~~
* ~~Change agentic architecture for : Make the agent "agentic" in the right place (cf chatgpt)~~ => agents removed, the strategy is code only
```

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q && uv run ruff check`
Expected: the full suite passes, no lint error.

Run: `(cd src/trading_agent_framework/strategies/vwap_pullback && grep -n -i "agent\|llm\|prompt" *.py | grep -v "trading_agent_framework\|agent_telemetry\|no LLM")`
Expected: prints nothing (the filter drops the import lines, the `agent_telemetry=False` line and the two "no LLM" docstrings). Any hit is leftover agent wording: reword that comment to describe what the code does now.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md TODO.md
git commit -m "chore: drop the langgraph dependency; docs describe the code-only vwap_pullback"
```

---

### Task 6: Check on real data (window A)

No code change. Needs `env/.env.vwap_pullback_continuation.backtesting` (present) with `ALPACA_DATA_*`; no `LLM_*` variable is needed any more. Network access to Alpaca is required; about 20 minutes.

**Interfaces:**
- Consumes: the finished branch.
- Produces: a run directory under `logs/vwap_pullback_continuation/backtesting/` and a short report to the user.

- [ ] **Step 1: Run the backtest**

The class `parameters` use `PredefinedWindow.MONTH`, which is window A (2026-08-25 to 2026-09-23, $10,000).

Run: `uv run agent vwap_pullback_continuation backtesting`
Expected: it finishes without an error and prints the run directory. A missing `LLM_BASE_URL` must not matter.

- [ ] **Step 2: Read the result**

```bash
RUN=$(ls -d logs/vwap_pullback_continuation/backtesting/*/ | tail -1)
python3 -c "import json,sys; m=json.load(open('$RUN/metrics.json')); print({k: m[k] for k in m if 'return' in k.lower() or 'sharpe' in k.lower() or 'drawdown' in k.lower()})"
wc -l "$RUN/trades.jsonl"
head -1 "$RUN/trades.jsonl"
grep -c "no free position slot for" "$RUN/backtest.log"
grep -il "llm\|agent run" "$RUN/backtest.log" || echo "no LLM activity in the log"
```

Expected: a total return and a trade count; each `trades.jsonl` line has exactly the ten fields of the spec §5; no LLM activity.

- [ ] **Step 3: Compare with the earlier no-LLM baseline and report**

The earlier baseline for window A was +1.56% with 26 trades. Find that run to compare trade by trade:

```bash
grep -l '"baseline"' logs/vwap_pullback_continuation/backtesting/*/variant.json
```

For the baseline run whose `variant.json` window starts `2026-08-25`, compare the `(symbol, entered_at)` pairs of its `trades.jsonl` with the new run's. Report to the user, as measured: total return, trade count, run time, and every trade present in one run and not the other with its cause (the count from `grep -c "no free position slot for"` says how many ticks the score ordering decided; a difference on a tick without that line is not explained by the ordering and must be investigated before the branch is called done).
