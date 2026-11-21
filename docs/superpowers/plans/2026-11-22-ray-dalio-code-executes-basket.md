# ray_dalio: code executes the disagreement basket -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The basket the thoughtful-disagreement agent submits is what the account holds; the trader agent is removed.

**Architecture:** A new `submit_basket` tool (pure checks, a plain dict as storage) is given to the disagreement
agent. The disagreement node writes the accepted basket to the graph state; a plain-code `rebalance` node (where the
trader node was) hands it to the shared `target_portfolio` + `Rebalancer` (`strategies/common/`), which sells what
the basket leaves out, buys, and parks the rest in SHV. The whole session stays one graph run.

**Tech Stack:** Python 3.14, LangGraph, pytest, the repo's `Rebalancer`.

**Spec:** `docs/superpowers/specs/2026-11-22-ray-dalio-code-executes-basket-design.md`

**Branch:** work on `feat/ray-dalio-code-executes-basket` off `main`; its first commit is the spec and this plan.
Before every commit run `/ponytail:ponytail-review` on the staged changes (user rule) and stage files by path only.

## Global Constraints

- Plain functions, dicts and lists. The only new class allowed is the 3-field frozen dataclass of Task 2 (pyright
  rejects a `SimpleNamespace` for `Rebalancer`'s `RebalanceParams` protocol: checked 2026-11-22).
- Weights are fractions of the portfolio (`0.35` = 35%).
- At least 3 symbols per basket, SHV counting as one (`MIN_POSITIONS` in `prompts.py`).
- Non-SHV weights add up to at most `1 - BUY_BUFFER` = 0.98 (`utils/helpers.BUY_BUFFER`).
- Rebalance band 0.03, smallest trade 0.005 of the book, parking symbol `"SHV"`.
- No valid basket in a session: log a warning, send no order.
- Every new public function gets a Google docstring; gate: ruff, ruff format, pytest --cov >= 80%, interrogate >= 80%, pyright.

## Review Focus

1. The agent writes percentages (`{"TQQQ": 35}`) instead of fractions: refused, and the error says weights are fractions. (Task 1 test.)
2. The agent calls `submit_basket` twice in one run, the first accepted: the last accepted basket is the one traded, nothing of the first is left. (Task 1 test.)
3. A holding the basket leaves out (the original TMF bug) is sold. (Task 2 test.)
4. The agent never submits an accepted basket: no order and a warning; the previous session's basket is not reused. (Task 2 test.)
5. A basket's SHV weight is not counted in the 0.98 sum and SHV is never traded as a basket stock (it is the parking remainder). (Task 1 and Task 2 tests.)

---

### Task 1: `submit_basket` tool

**Files:**
- Create: `src/trading_agent_framework/strategies/ray_dalio/basket.py`
- Test: `tests/strategies/ray_dalio/test_basket.py`

**Interfaces:**
- Consumes: `UNIVERSE`, `MIN_POSITIONS` from `ray_dalio/prompts.py`; `BUY_BUFFER` from `utils/helpers.py`.
- Produces:
  - `basket_problem(weights: dict[str, float]) -> str | None` -- the first reason the basket is refused, or `None`.
  - `basket_tool(store: dict[str, float]) -> Callable[..., dict[str, Any]]` -- returns the tool function
    `submit_basket(weights: dict[str, float], reason: str) -> dict[str, Any]`. Accepted: `store` is cleared and
    filled with `weights`, returns `{"accepted": True, "basket": weights}`. Refused: `store` untouched, returns
    `{"error": <basket_problem's text>}`. Its docstring is what the agent reads: say weights are fractions, the basket
    is executed exactly, holdings left out are sold and the unallocated rest is parked in SHV.

- [ ] **Step 1: Write the failing tests** in `tests/strategies/ray_dalio/test_basket.py`

```python
GOOD = {"TQQQ": 0.35, "UPRO": 0.25, "TMF": 0.25, "SHV": 0.15}

def test_a_good_basket_passes():
    assert basket_problem(GOOD) is None

def test_shv_is_not_counted_in_the_98_percent():
    assert basket_problem({"TQQQ": 0.49, "UPRO": 0.49, "SHV": 0.5}) is None

def test_refusals_say_why():
    assert "XYZ" in basket_problem({"XYZ": 0.3, "TQQQ": 0.3, "SHV": 0.3})
    assert "above 0" in basket_problem({"TQQQ": 0.0, "UPRO": 0.3, "SHV": 0.3})
    assert "at least 3" in basket_problem({"TQQQ": 0.5, "SHV": 0.5})
    assert "0.98" in basket_problem({"TQQQ": 0.5, "UPRO": 0.5, "SHV": 0.0001})

def test_percentages_are_refused_and_the_error_says_fractions():
    assert "fraction" in basket_problem({"TQQQ": 35, "UPRO": 25, "SHV": 40})

def test_an_accepted_basket_replaces_the_stored_one():
    store: dict[str, float] = {}
    submit = basket_tool(store)
    assert submit(weights={"TQQQ": 0.4, "UPRO": 0.4, "SOXL": 0.1}, reason="first")["accepted"] is True
    assert submit(weights=GOOD, reason="second")["accepted"] is True
    assert store == GOOD

def test_a_refused_basket_leaves_the_store_alone():
    store = dict(GOOD)
    assert "error" in basket_tool(store)(weights={"XYZ": 1.0}, reason="bad")
    assert store == GOOD

def test_the_tool_is_named_submit_basket():
    assert basket_tool({}).__name__ == "submit_basket"
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/ray_dalio/test_basket.py -q` -- Expected: FAIL (`ModuleNotFoundError: ...ray_dalio.basket`).

- [ ] **Step 3: Implement `basket.py`.** Checks in this order: unknown symbol, weight not above 0, fewer than
  `MIN_POSITIONS` symbols, non-SHV sum above `1 - float(BUY_BUFFER)` (message names 0.98 and says weights are
  fractions, e.g. 0.35 for 35%). A module docstring of two lines saying why this exists (the trader ignored the basket).

- [ ] **Step 4: Run** the same command -- Expected: 7 passed.

- [ ] **Step 5: Commit** `git add src/trading_agent_framework/strategies/ray_dalio/basket.py tests/strategies/ray_dalio/test_basket.py && git commit -m "feat(ray_dalio): submit_basket tool with plain checks"`

---

### Task 2: the disagreement's basket is traded; the trader is removed

**Files:**
- Modify: `src/trading_agent_framework/strategies/ray_dalio/agent_ray_dalio.py`
- Modify: `src/trading_agent_framework/strategies/ray_dalio/prompts.py`
- Delete: `src/trading_agent_framework/strategies/ray_dalio/readiness.py`, `tests/strategies/ray_dalio/test_readiness.py`
- Modify: `tests/strategies/ray_dalio/test_ray_dalio_strategy.py`, `tests/strategies/ray_dalio/test_prompts.py`
- Modify: `src/trading_agent_framework/strategies/ray_dalio/CLAUDE.md`, `README.md` (the `ray_dalio` section, line ~333)

**Interfaces:**
- Consumes: `basket_tool(store)` from Task 1; `target_portfolio(weights, *, cash_buffer)` (`common/portfolio.py`);
  `Rebalancer(strategy, params)` and `.rebalance(target) -> list[PlacedOrder]` (`common/rebalancer.py`).
- Produces: `RayDalioStrategy` with 4 agents (`*ANALYSTS`, `DISAGREEMENT`); `TRADER`, `ORDER_READINESS_RULE`,
  `SIZING_RULE` and `readiness.py` no longer exist.

Decisions:
- In `agent_ray_dalio.py`, at module level:
  `@dataclass(frozen=True) class _RebalanceSettings: parking_symbol: str = "SHV"; rebalance_band: float = 0.03; min_trade_pct: float = 0.005`
  with a one-line comment: the `Rebalancer` needs these three named settings, and pyright refuses a plain namespace.
- `initialize`: `self._basket: dict[str, float] = {}`; the disagreement agent's tools are `[*self._research_tools(), basket_tool(self._basket)]`;
  `self._rebalancer = Rebalancer(self, _RebalanceSettings())`. Delete `_ORDER_TOOLS` and the trader creation.
- `DalioState` gains `basket: dict[str, float]`. `REBALANCE = "rebalance"` (node name, in `agent_ray_dalio.py`).
- The disagreement node empties `self._basket` before running the agent and writes `{"disagreement": output, "basket": dict(self._basket)}`
  (a small wrapper around `_node`, or a `_disagreement_node` method: the builder's pick, keep it readable).
- `_rebalance_node(self, state) -> dict`: if `state.get("basket")` is empty, `log_warning("no basket submitted: holding the current book")`;
  otherwise log `basket: {...}` on one line and `self._rebalancer.rebalance(target_portfolio({s: w for s, w in basket.items() if s != "SHV"}, cash_buffer=float(BUY_BUFFER)))`.
  Returns `{}`.
- `_build_graph`: `... -> DISAGREEMENT -> REBALANCE -> END`. `on_trading_iteration` is unchanged.
- `prompts.py`: delete `TRADER`, `ORDER_READINESS_RULE`, `SIZING_RULE` and the trader's two prompts. Append to the
  disagreement's system and task prompts: end by calling `submit_basket` with the basket as fractions of the portfolio;
  it is executed exactly as submitted (holdings left out are sold, the unallocated rest goes to SHV); on an error, fix
  the basket and call it again.

- [ ] **Step 1: Update the tests first** in `test_ray_dalio_strategy.py`:
  - `_NAMES = (*ANALYSTS, DISAGREEMENT)`; `_TOOLS`: drop `submit_order`, add `submit_basket`.
  - Delete `test_only_the_trader_has_order_tools_and_they_are_guarded` and `test_the_trader_has_no_unguarded_order_tools`.
  - Rename/adjust: `test_initialize_creates_the_four_agents...`; `test_an_iteration_runs_analysts_then_disagreement`
    (no trader run; disagreement last); `test_an_analyst_error_stops_the_iteration_before_any_order` (`broker.submitted == []`);
    `test_disagreement_and_rebalance_run_on_the_strategy_thread` (record `threading.get_ident()` inside the
    rebalance node, e.g. by wrapping `strategy._rebalancer.rebalance`).
  - Give `_FakeHandle` an optional `on_run(tools)` hook so a test can make the disagreement agent call its
    `submit_basket` (take it from `_tools(agents)[DISAGREEMENT]`). Set up the broker like
    `tests/strategies/sector_pods/test_sector_pods_graph.py::_harness`: portfolio 10,000, every fund 50, SHV 100,
    `buying_power` 1,000,000, `held` as `Position(..., side=PositionSide.LONG)`.
  - New tests:

```python
def test_only_the_disagreement_agent_can_submit_a_basket(...):
    assert [n for n, t in _tools(agents).items() if "submit_basket" in t] == [DISAGREEMENT]
    assert not any("submit_order" in t for t in _tools(agents).values())

def test_a_holding_the_basket_leaves_out_is_sold(...):  # held TMF 40 shares; basket TQQQ/UPRO/SHV
    sells = [o for o in broker.submitted if o.side.value == "sell"]
    assert [o.asset.symbol for o in sells] == ["TMF"]

def test_the_basket_is_bought_and_the_rest_parked_in_shv(...):  # basket {"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3}, nothing held
    assert {o.asset.symbol for o in broker.submitted if o.side.value == "buy"} == {"TQQQ", "UPRO", "SHV"}

def test_no_basket_means_no_order_and_a_warning(...):  # caplog
    assert broker.submitted == [] and "no basket submitted" in caplog.text

def test_a_basket_is_not_reused_the_next_session(...):  # session 1 submits, session 2 does not
    assert len(broker.submitted) == <orders of session 1> and "no basket submitted" in caplog.text
```

  In `test_prompts.py`: drop `TRADER`, `ORDER_READINESS_RULE`, `SIZING_RULE` imports and
  `test_only_the_trader_is_told_about_orders_and_sizing`, `test_sizing_rule_points_at_spendable`; the agent set is
  `{*ANALYSTS, DISAGREEMENT}`; add `test_only_the_disagreement_agent_is_told_to_submit_a_basket` (`"submit_basket"`
  in its system and task prompt, in no analyst's).

- [ ] **Step 2: Run** `uv run pytest tests/strategies/ray_dalio -q` -- Expected: FAIL (import errors on `TRADER`, no `submit_basket`).

- [ ] **Step 3: Implement** the decisions above in `agent_ray_dalio.py` and `prompts.py`; `git rm` the readiness module and its test.

- [ ] **Step 4: Run** `uv run pytest tests/strategies/ray_dalio -q` -- Expected: all pass.

- [ ] **Step 5: Docs.** `ray_dalio/CLAUDE.md`: new graph line, agents table (disagreement gets `submit_basket`, no
  trader row), replace the readiness and "tool calls run one at a time" and threading "known gap" bullets with one
  bullet: orders are placed by `Rebalancer` in the plain-code `rebalance` node, on the strategy thread (never add
  streaming, `subgraphs=True` or `step_timeout`); the basket reaches it through the graph state; why (first backtest: the trader
  kept TMF in 50 of 103 sessions the basket left it out); SHV is the remainder; no basket = hold. Keep the `get_orders`
  bullet only if `get_orders` is still given to an agent (it is not: delete it). README `ray_dalio` section: "a disagreement round whose basket the code trades".

- [ ] **Step 6: Gate.** `uv run ruff check && uv run ruff format --check && uv run pyright && uv run interrogate src | tail -1 && uv run pytest --cov -q`
  -- Expected: all clean, coverage >= 80%, interrogate >= 80%.

- [ ] **Step 7: Commit** the files above by path (`git add` each, `git rm` already staged): `git commit -m "feat(ray_dalio): code trades the disagreement basket; trader agent removed"`
