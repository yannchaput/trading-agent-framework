# ray_dalio Idea-Meritocracy Leveraged ETF Team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `ray_dalio` strategy that ports lumibot's leveraged-ETF idea-meritocracy team (3 analysts, a disagreement agent, a trader) as a LangGraph graph of LangChain agents, with the same prompts and logic.

**Architecture:** A `StateGraph` over a 5-slot `TypedDict`: growth, inflation and liquidity analysts in parallel from `START`, then `disagreement`, then `trader`. Each node runs an agent created by `AgentManager.create` in `initialize`. The trader places its own orders through the existing trading tools, behind a strategy-local order-readiness guard. Two FRED tools (`get_fred_latest`, `get_fred_snapshot`) are added to `agents/tools/macro.py`.

**Tech Stack:** Python 3.14, uv, LangGraph (`langgraph.graph.StateGraph`), LangChain via `AgentManager`, pytest with hand-written fakes.

**Spec:** `docs/superpowers/specs/2026-11-20-ray-dalio-meritocracy-design.md`

## Global Constraints

- Branch `feat/ray-dalio-meritocracy` from `main`; Task 1's commit also adds the spec and this plan.
- Strategy name `ray_dalio`, enum member `Strategies.RAY_DALIO`, package `strategies/ray_dalio/`.
- No new classes or types beyond `RayDalioStrategy` and the state `DalioState(TypedDict, total=False)`. Helpers are functions.
- Do not import or copy anything from `strategies/sector_pods/` or `agents/graph.py` / `agents/submit.py`.
- `langgraph` is imported inside the function that builds the graph (as `sector_pods/graph.py:279` does), not at module top.
- Every data tool's cutoff is `strategy.clock.now().date()`; no wall clock (`datetime.now`, `time.sleep`) anywhere.
- Tests never touch the network: FRED via a fake client, agents via fake handles, `FRED_API_KEY` / `SEC_EDGAR_USER_AGENT` set with `monkeypatch.setenv`.
- Backtest costs: the framework default (`Strategy.backtesting_slippage` = 5 bps per side + broker fees). Set nothing.
- Quality gate per task: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest` (use `uv sync --extra dashboard` first so the dashboard tests run), at the end `uv run pytest --cov` >= 80% and `uv run interrogate -v` >= 80%. Google docstrings on every public function. No `# noqa` / `# type: ignore` added.
- Before every commit: `/ponytail:ponytail-review` on the staged diff; fix every "Must fix".

## Review Focus

1. FRED returns `NaN` for days with no observation (DGS10 on holidays): `get_fred_latest` must return the last non-NaN value, not `NaN`. Test in Task 1.
2. A snapshot string like `"DGS10, ,UNRATE "`: ids are stripped and empty ones skipped. Test in Task 1.
3. The trader prices `"tqqq"` and orders `"TQQQ"` (or the reverse): the price check counts, case-insensitively. Test in Task 2.
4. A readiness check whose tool returned `{"error": ...}` does not count as done. Test in Task 2.
5. A refused order (`submit_order` returned `{"error": ...}`) is not an order: the account and positions checks stay valid. Test in Task 2.

---

### Task 1: FRED `get_fred_latest` and `get_fred_snapshot`

**Files:**
- Modify: `src/trading_agent_framework/agents/tools/macro.py` (inside `macro_tools`, return list)
- Modify: `tests/agents/tools/test_macro_tools.py` (the two `[tool] = macro_tools(...)` unpackings at lines 43 and 134 become a lookup by name)
- Modify: `src/trading_agent_framework/agents/CLAUDE.md` (the macro tool line: name the two new tools)
- Add: the spec and this plan

**Interfaces:**
- Produces: `macro_tools(strategy, *, fred_client_factory=None)` now returns `[get_fred_series, get_fred_latest, get_fred_snapshot]`.
  - `get_fred_latest(series_id: str) -> dict[str, Any]` -> `{"series_id", "as_of", "date", "value"}` or `{"series_id", "error"}`.
  - `get_fred_snapshot(series_ids: list[str] | str) -> dict[str, Any]` -> `{"as_of", "series": [<one get_fred_latest row per id>]}` or `{"error"}` when more than `MAX_SNAPSHOT_SERIES = 10` ids or none.

- [ ] **Step 1: Branch and update the existing tests' unpacking**

`git switch -c feat/ray-dalio-meritocracy`. In `test_macro_tools.py`, replace `[tool] = macro_tools(...)` with `tool = {t.__name__: t for t in macro_tools(...)}["get_fred_series"]` (both places). Run `uv run pytest tests/agents/tools/test_macro_tools.py -q`: PASS (unchanged behaviour).

- [ ] **Step 2: Write the failing tests** (same file, reuse `_FakeFred` and the 2026-09-14 clock)

```python
def test_get_fred_latest_returns_the_last_non_nan_observation():
    # index 2026-09-10, 2026-09-11, 2026-09-12 ; values 4.1, 4.2, nan
    assert latest("DGS10") == {"series_id": "DGS10", "as_of": "2026-09-14", "date": "2026-09-11", "value": 4.2}
    # the fake recorded realtime_start == realtime_end == date(2026, 9, 14) and observation_end == date(2026, 9, 14)


def test_get_fred_latest_reports_an_empty_series():
    assert latest("DGS10") == {"series_id": "DGS10", "error": "no observation on or before 2026-09-14"}


def test_get_fred_latest_reports_a_fetch_failure():  # a client whose get_series raises
    assert result["series_id"] == "DGS10" and "Failed to fetch FRED series 'DGS10'" in result["error"]


def test_get_fred_snapshot_accepts_a_comma_string_with_blanks():
    result = snapshot("DGS10, ,UNRATE ")
    assert result["as_of"] == "2026-09-14"
    assert [row["series_id"] for row in result["series"]] == ["DGS10", "UNRATE"]


def test_get_fred_snapshot_accepts_a_list_and_keeps_a_failed_row():
    # a fake that raises for "BAD" only
    assert [("error" in row) for row in snapshot(["DGS10", "BAD"])["series"]] == [False, True]


def test_get_fred_snapshot_refuses_more_than_ten_or_no_ids():
    assert "error" in snapshot([f"S{i}" for i in range(11)]) and "error" in snapshot(" , ")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/agents/tools/test_macro_tools.py -q`. Expected: FAIL (`KeyError: 'get_fred_latest'`).

- [ ] **Step 4: Implement the two tools in `macro_tools`**

`get_fred_latest` calls `_fetch_series(fred_client_factory(), series_id, None, cutoff)`, then `series.dropna()`, and takes the last item. `get_fred_snapshot` normalizes a string with `split(",")`, strips each id and skips empty ones, then builds one `get_fred_latest` row per id. Docstrings: lumibot's descriptions ("Get the latest FRED macro observation available as of the current date." and "Get the latest values of several FRED series at once, e.g. FEDFUNDS,DGS10,CPIAUCSL,UNRATE (a list or a comma-separated string, at most 10).").

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_macro_tools.py tests/strategies/sector_pods -q`. Expected: PASS (sector_pods' pods now carry two more tools, and its tests pin only `search_news`/`get_fred_series`).

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/macro.py tests/agents/tools/test_macro_tools.py src/trading_agent_framework/agents/CLAUDE.md docs/superpowers/specs/2026-11-20-ray-dalio-meritocracy-design.md docs/superpowers/plans/2026-11-20-ray-dalio-meritocracy.md
git commit -m "feat(macro): get_fred_latest and get_fred_snapshot tools (Task 1)"
```

---

### Task 2: Order-readiness guard

**Files:**
- Create: `src/trading_agent_framework/strategies/ray_dalio/__init__.py` (docstring only for now; Task 4 adds the export)
- Create: `src/trading_agent_framework/strategies/ray_dalio/readiness.py`
- Test: `tests/strategies/ray_dalio/test_readiness.py`

**Interfaces:**
- Consumes: `current_run_id()` from `trading_agent_framework.memory.tools`; tests use `agent_call_context(run_id=...)` from the same module, as `tests/strategies/test_news_builtin_grounding.py` does.
- Produces: `require_readiness_before_orders(tools: list[Callable[..., dict[str, Any]]]) -> list[Callable[..., dict[str, Any]]]`. It returns the same list in the same order: `get_account_balance`, `get_positions` and `get_last_price` are wrapped to record a successful call, `submit_order` is wrapped to check, and everything else is unchanged. It raises `ValueError` if any of the four names is missing. It uses `functools.wraps` so LangChain sees the original names, signatures and docstrings (as `news_builtin/grounding.py` does).

Rule (lumibot's `ORDER_READINESS_REQUIRED`): within one run (`current_run_id()`), `submit_order(symbol, ...)` runs only if
(a) `get_account_balance` and (b) `get_positions` succeeded (no `"error"` key) since the last **accepted** order (or
since the start of the run), and (c) `get_last_price` succeeded for `symbol` in this run (case-insensitive). Otherwise it
returns `{"error": "ORDER_READINESS_REQUIRED: before submitting an order, call <missing, comma-separated> in this same agent run."}`
naming each missing call (`get_account_balance`, `get_positions`, `get_last_price(symbol='TQQQ')`). An accepted order (a
result with no `"error"`) clears (a) and (b); the prices stay valid. A new run id starts empty.

- [ ] **Step 1: Write the failing tests** (fake tools record their calls; `submit_order` fake returns `{"identifier": "o1"}` or a set error)

```python
def test_an_order_without_any_check_is_refused_and_names_all_three():
    ...
    # error starts "ORDER_READINESS_REQUIRED" and contains "get_account_balance", "get_positions", "get_last_price(symbol='TQQQ')"; the inner submit_order was not called


def test_an_order_after_all_three_checks_is_placed(): ...
def test_the_price_check_is_per_symbol(): ...  # priced UPRO, ordering TQQQ -> refused, names only get_last_price
def test_the_price_check_ignores_case(): ...  # priced "tqqq", ordering "TQQQ" -> placed
def test_a_failed_check_does_not_count(): ...  # get_positions returned {"error": "x"} -> refused, names get_positions
def test_an_accepted_order_requires_fresh_account_and_positions(): ...  # 2nd order refused naming balance+positions, not the price
def test_a_refused_order_keeps_the_checks(): ...  # submit returns {"error": "insufficient"} -> next order goes straight through
def test_checks_do_not_carry_over_to_a_new_run(): ...  # checks in run-1, order in run-2 -> refused
def test_missing_tool_raises(): ...  # list without get_positions -> ValueError
def test_wrapped_tools_keep_their_names_and_signatures(): ...  # [t.__name__ for t in guarded] unchanged; inspect.signature(submit_order) == original
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/ray_dalio/test_readiness.py -q`. Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `require_readiness_before_orders` in `readiness.py`**

State: one closure dict holding the run id and its record: `account: bool`, `positions: bool`, `priced: set[str]` of uppercased symbols. It is reset whenever `current_run_id()` differs from the stored one. Read `symbol` from the call with `inspect.signature(...).bind_partial(...)`, the way `grounding._requested_full_content` reads `include_content`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/ray_dalio/test_readiness.py -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/ray_dalio tests/strategies/ray_dalio
git commit -m "feat(ray_dalio): lumibot's order-readiness guard for the trader's tools (Task 2)"
```

---

### Task 3: Prompts

**Files:**
- Create: `src/trading_agent_framework/strategies/ray_dalio/prompts.py`
- Test: `tests/strategies/ray_dalio/test_prompts.py`

**Interfaces:**
- Produces: `UNIVERSE: tuple[str, ...]` (54), `MIN_POSITIONS = 3`, `ANALYSTS: dict[str, str]` (agent name -> state slot, in this order: `{"growth_agent": "growth", "inflation_agent": "inflation", "debt_liquidity_agent": "liquidity"}`), `DISAGREEMENT = "thoughtful_disagreement"`, `TRADER = "trader"`, `SYSTEM_PROMPTS: dict[str, str]` and `TASK_PROMPTS: dict[str, str]` keyed by those five names, `CONTEXT: dict[str, object]` (`universe` as a list, `min_positions`, `data_expectation`, `leverage_expectation`).

Exact copy. This is lumibot `ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py` @ `38e23c72248dd1555387b1487ca7b676ef6e95ca` with the spec's substitutions already applied. Module docstring: credit "Adapted from LumiBot (MIT License, Lumiwealth), example_strategies/ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py" and add the spec's disclaimer (not affiliated with Ray Dalio or Bridgewater).

```python
UNIVERSE = (
    "TQQQ",
    "QLD",
    "SQQQ",
    "QID",
    "UPRO",
    "SSO",
    "SPXU",
    "SDS",
    "UDOW",
    "DDM",
    "SDOW",
    "DXD",
    "TNA",
    "UWM",
    "TZA",
    "TWM",
    "FNGU",
    "FNGD",
    "TECL",
    "TECS",
    "SOXL",
    "SOXS",
    "FAS",
    "FAZ",
    "CURE",
    "RXD",
    "LABU",
    "LABD",
    "ERX",
    "ERY",
    "GUSH",
    "DRIP",
    "TMF",
    "UBT",
    "TBT",
    "TTT",
    "UGL",
    "GLL",
    "AGQ",
    "ZSL",
    "UCO",
    "SCO",
    "NUGT",
    "DUST",
    "YINN",
    "YANG",
    "EDC",
    "EDZ",
    "EURL",
    "EUO",
    "YCS",
    "DRN",
    "SRS",
    "SHV",
)

ORDER_READINESS_RULE = (
    "Immediately before every buy or sell order, call get_account_balance, get_positions, and get_last_price for the "
    "exact ordered symbol in this same agent run, then call submit_order. Blind orders are rejected with "
    "ORDER_READINESS_REQUIRED when those readiness calls are missing."
)
DATA_USAGE_RULE = (
    "Use the built-in tools by their real names: get_fred_snapshot, get_fred_latest, and get_fred_series for macro "
    "evidence; search_news for recent market, sector, rates, and ETF-proxy headlines; SEC tools only when sector or "
    "company fundamentals are relevant. Keep tool use bounded: at most one FRED snapshot or short series request and one "
    "search_news call per agent run; set search_news limit <= 5; do not paginate or repeatedly re-check the same evidence."
)
LEVERAGE_RULE = (
    "This is a leveraged ETF strategy. Use only symbols from the leveraged universe plus SHV/cash-like exposure. Stay "
    "biased toward diversified 3x exposure when evidence supports risk-taking; use 2x as a risk-down choice; use inverse "
    "leveraged ETFs only with explicit downside or hedge evidence. Hold at least three positions in normal conditions, "
    "and do not make a single all-in bet. SHV/cash may count as one position when the model refuses every reasonable "
    "leveraged setup."
)
SIZING_RULE = (  # this repo's rule for every agent that buys (strategies/CLAUDE.md), not in lumibot
    "Size every buy against the SMALLER of buying_power and cash plus the proceeds of the sells you submitted in this run, never against buying_power alone."
)
_RULES = DATA_USAGE_RULE + " " + LEVERAGE_RULE

SYSTEM_PROMPTS = {
    "growth_agent": (
        "Argue which leveraged ETFs win if growth improves. First call get_fred_snapshot or get_fred_latest for growth, "
        "liquidity, rates, and credit context, then call search_news for broad market and ETF-proxy headlines. Use exact "
        "symbols from the universe only. " + _RULES
    ),
    "inflation_agent": (
        "Argue which leveraged ETFs win or lose if inflation and rates surprise. First call get_fred_snapshot or "
        "get_fred_latest for CPI, inflation expectations, Treasury yields, and policy-rate context, then call search_news "
        "for rates, commodities, and market headlines. Use exact symbols from the universe only. " + _RULES
    ),
    "debt_liquidity_agent": (
        "Argue from debt, liquidity, currency, credit, and policy pressure. First call FRED tools such as "
        "get_fred_snapshot/get_fred_latest, then call search_news for market stress and ETF-proxy headlines. Use exact "
        "symbols from the universe only. " + _RULES
    ),
    "thoughtful_disagreement": (
        "Challenge all views with thoughtful disagreement. Identify the best diversified leveraged basket after stress "
        "testing. Check whether upstream agents actually used get_fred_snapshot/get_fred_latest and search_news. Do not "
        "re-call tools unless upstream evidence is entirely absent; if you must, make only one short FRED call and one "
        "search_news call with limit <= 5. " + _RULES
    ),
    "trader": (
        "Build a Ray Dalio-style idea-meritocracy leveraged ETF basket. This is not All Weather and not a one-ETF "
        "momentum bet. Use the specialists' disagreement process, reconcile overrides, and size at least three positions "
        "in normal conditions. Favor 3x ETFs for high-conviction sleeves, use 2x when conviction or drawdown risk is lower, "
        "and use inverse ETFs only as carefully justified hedge or downside exposure. SHV/cash-like exposure may count as "
        "one position only when evidence is too weak for full leveraged risk. " + _RULES + " " + ORDER_READINESS_RULE
    ),
}

TASK_PROMPTS = {
    "growth_agent": (
        "Call get_fred_snapshot or get_fred_latest and search_news, then rank the strongest leveraged ETFs from a growth-regime view. Prefer diversified 3x exposure when evidence supports it."
    ),
    "inflation_agent": (
        "Call get_fred_snapshot or get_fred_latest and search_news, then rank the strongest leveraged ETFs from an inflation-and-rates view. Explain any inverse or 2x risk-down preference."
    ),
    "debt_liquidity_agent": (
        "Call get_fred_snapshot or get_fred_latest and search_news, then rank the strongest leveraged ETFs from a "
        "debt-and-liquidity view. Explain whether liquidity argues for 3x risk, 2x risk-down, inverse exposure, or SHV ballast."
    ),
    "thoughtful_disagreement": (
        "Challenge the growth, inflation, and liquidity views. Prefer a diversified leveraged basket of at least three symbols unless risk evidence argues for SHV/cash-like ballast."
    ),
    "trader": (
        "Rebalance into a diversified leveraged ETF basket of at least three positions. Bias toward 3x ETFs, use 2x only "
        "as a risk-down choice, and use inverse ETFs only with explicit hedge/downside evidence. Before each order, call "
        "get_account_balance, get_positions, and get_last_price for the exact ordered symbol in this same run. "
        + SIZING_RULE
        + " Explain which specialist advice you accepted or rejected and cite FRED/news evidence used."
    ),
}

CONTEXT = {
    "universe": list(UNIVERSE),
    "min_positions": MIN_POSITIONS,
    "data_expectation": "Use get_fred_snapshot/get_fred_latest and search_news commonly; the run log records the actual tool calls.",
    "leverage_expectation": "Use only leveraged ETFs plus SHV/cash escape, hold at least three positions, and bias toward diversified 3x exposure.",
}
```

- [ ] **Step 1: Write the failing tests**

```python
def test_universe_is_lumibots_54_symbols_with_shv():
    assert len(UNIVERSE) == 54 == len(set(UNIVERSE)) and UNIVERSE[-1] == "SHV"


def test_every_agent_has_a_system_and_a_task_prompt():
    assert set(SYSTEM_PROMPTS) == set(TASK_PROMPTS) == {*ANALYSTS, DISAGREEMENT, TRADER}


def test_no_lumibot_tool_name_is_left():  # in every prompt and in CONTEXT's strings
    for gone in ("alpaca_news", "account_portfolio", "account_positions", "market_last_price", "orders_submit_order", "list_fred_series", "LumiBot"):
        ...


def test_only_the_trader_is_told_about_orders_and_sizing():
    assert ORDER_READINESS_RULE in SYSTEM_PROMPTS[TRADER] and SIZING_RULE in TASK_PROMPTS[TRADER]
    assert all("submit_order" not in SYSTEM_PROMPTS[name] for name in (*ANALYSTS, DISAGREEMENT))
```

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/strategies/ray_dalio/test_prompts.py -q`: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Write `prompts.py`** as given above.
- [ ] **Step 4: Run them to verify they pass.** Same command: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(ray_dalio): lumibot's prompts with this framework's tool names (Task 3)"`

---

### Task 4: Strategy, graph, registration, docs

**Files:**
- Create: `src/trading_agent_framework/strategies/ray_dalio/agent_ray_dalio.py`
- Modify: `src/trading_agent_framework/strategies/ray_dalio/__init__.py` (export `RayDalioStrategy`, shaped like `sector_pods/__init__.py`)
- Modify: `src/trading_agent_framework/utils/strategy_factory.py` (`RAY_DALIO = auto()` in alphabetical order, `_build_ray_dalio`, the `_STRATEGY_BUILDERS` entry)
- Create: `src/trading_agent_framework/strategies/ray_dalio/CLAUDE.md`
- Modify: `src/trading_agent_framework/strategies/CLAUDE.md` (index), the root `CLAUDE.md` (strategy list in Architecture)
- Test: `tests/strategies/ray_dalio/test_ray_dalio_strategy.py`, `tests/test_strategy_factory.py`, `tests/strategies/test_backtest_defaults.py`

**Interfaces:**
- Consumes: Task 1 `macro_tools`; Task 2 `require_readiness_before_orders`; Task 3 `ANALYSTS`, `DISAGREEMENT`, `TRADER`, `SYSTEM_PROMPTS`, `TASK_PROMPTS`, `CONTEXT`, `UNIVERSE`; the existing `news_tools`, `fundamentals_tools`, `market_data_tools`, `account_tools`, `indicator_tools`, `trading_tools` (`agents/tools/`), `require_news_provider(broker, who)` (`strategies/common/news.py`), `backtest_window(PredefinedWindow.MONTH_LATEST)`.
- Produces:
  - `class DalioState(TypedDict, total=False)`: `date: str`, `growth: str`, `inflation: str`, `liquidity: str`, `disagreement: str`.
  - `class RayDalioStrategy(Strategy)`, with `parameters = {"backtesting_start"/"backtesting_end": backtest_window(PredefinedWindow.MONTH_LATEST), "benchmark_symbol": "SPY", "warmup_trading_days": 10, "budget": 10000}` (news_builtin's values, window aside).
  - `initialize(self) -> None`: `sleeptime = "1D"`; `require_news_provider(self.broker, "ray_dalio's agents need")`; `self.agents.create(name=..., system_prompt=SYSTEM_PROMPTS[name], tools=...)` for the five agents, in the order growth, inflation, debt_liquidity, disagreement, trader; `self._graph = self._build_graph()`.
  - `_research_tools(self) -> list[Callable[..., dict[str, Any]]]`: new instances on every call: `macro_tools`, `news_tools`, `fundamentals_tools`, `market_data_tools`, `account_tools`, `indicator_tools`. The trader gets `require_readiness_before_orders([*self._research_tools(), *trading_tools(self)])`.
  - `_node(self, name: str, slot: str | None, upstream: tuple[str, ...]) -> Callable[[DalioState], dict[str, str]]`: runs `self.agents[name].run(TASK_PROMPTS[name], context={"date": state["date"], **CONTEXT, **{k: state[k] for k in upstream}})`, logs `[<name>] <output>` with `log_info`, and returns `{slot: result.output}` (`{}` for the trader).
  - `_build_graph(self) -> Any`: nodes `growth`/`inflation`/`liquidity` (`upstream=()`), `disagreement` (`upstream=("growth", "inflation", "liquidity")`, slot `"disagreement"`) and `trader` (upstream: those four, slot `None`). Edges: `START` to each analyst, `[analysts] -> "disagreement" -> "trader" -> END`. Returns `graph.compile()`.
  - `on_trading_iteration(self) -> None`: `self._graph.invoke({"date": self.get_datetime().date().isoformat()})`. No try/except: an agent error goes to the executor (logged; it runs again next session, as lumibot does).
  - `run_backtesting(self, **overrides)`: defaults `preload_assets=[Asset(symbol=s) for s in (*UNIVERSE, "SPY")]`, `timestep="day"`, `agent_telemetry=True`. This mirrors `SectorPodsStrategy.run_backtesting` (no `data_source`: Yahoo is the default). Copy the pattern, not the code.

- [ ] **Step 1: Write the failing tests**

Fakes: a `_FakeAgents` with `create(**kwargs)` recording the kwargs, and `__getitem__(name)` returning a per-name `_FakeHandle`. That handle records `(task_prompt, context)` and returns `AgentRunResult(output=f"{name} view", tool_calls=[])`, or raises a preset error. Copy the shape of `tests/strategies/test_news_builtin.py:77-95`. Build the strategy with `RayDalioStrategy(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="ray_dalio"), mode=TradingMode.PAPER, project_root=tmp_path)`, then set `strategy._agents = cast(AgentManager, agents)`. Use `monkeypatch.setenv` for `FRED_API_KEY` and `SEC_EDGAR_USER_AGENT`, and stub `broker.news_provider` as `test_news_builtin.py:199` does.

```python
def test_initialize_creates_the_five_agents_with_lumibots_prompts():
    ...
    # names in order; system_prompt == SYSTEM_PROMPTS[name]; strategy.sleeptime == "1D"


def test_only_the_trader_has_order_tools_and_they_are_guarded():
    ...
    # "submit_order" absent from the four others; the trader's submit_order without checks returns ORDER_READINESS_REQUIRED (call it inside agent_call_context)


def test_each_agent_gets_its_own_news_and_macro_tool_instances(): ...  # `is not` between agents for search_news and get_fred_snapshot
def test_every_tool_a_prompt_names_is_in_that_agents_tools():
    ...
    # for each of get_fred_snapshot, get_fred_latest, get_fred_series, search_news, get_account_balance, get_positions, get_last_price, submit_order: if named in the agent's system or task prompt, it is in its tools


def test_an_iteration_runs_analysts_then_disagreement_then_trader():
    ...
    # each of the five ran once with TASK_PROMPTS[name]; analysts' context == {"date": "2026-09-14", **CONTEXT}
    # disagreement's context adds growth/inflation/liquidity == "<agent> view"; trader's adds disagreement too; trader ran last


def test_an_analyst_error_stops_the_iteration_before_the_trader(): ...  # inflation raises AgentError -> invoke raises it, the trader never ran
def test_initialize_fails_without_a_news_provider(): ...  # ConfigurationError
```

In `tests/test_strategy_factory.py`, add a case that `build_strategy(Strategies.RAY_DALIO, ...)` returns a `RayDalioStrategy`, following the file's existing cases. In `tests/strategies/test_backtest_defaults.py`, add a case like the sector_pods/news_builtin ones: preload = the 54 + SPY, `timestep == "day"`, telemetry on.

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/strategies/ray_dalio tests/test_strategy_factory.py tests/strategies/test_backtest_defaults.py -q`: FAIL (import error / missing member).
- [ ] **Step 3: Implement `agent_ray_dalio.py`, the `__init__` export and the factory entry** per the Interfaces block.
- [ ] **Step 4: Run them to verify they pass.** Same command: PASS.
- [ ] **Step 5: Docs.** `strategies/ray_dalio/CLAUDE.md`: what it is (link the spec and this plan; lumibot source + commit; disclaimer); the graph sketch; agents and tools table; the readiness rule; the threading note (parallel analysts on worker threads, read-only tools; the trader single-node on the strategy thread; never add streaming / `subgraphs=True` / `step_timeout`); env (`LLM_*` with `LLM_REASONING_EFFORT=high`, `ALPACA_NEWS_*`, `FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`, broker keys in `env/.env.ray_dalio.<mode>`); gotchas (run on an account holding only this strategy: the trader may sell any position; news before 2026-04-29 is empty in backtests; tickers not verified tradable). Add one line to the `strategies/CLAUDE.md` index and the root `CLAUDE.md` strategy list.
- [ ] **Step 6: Full gate.** `uv sync --extra dashboard && uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov && uv run interrogate -v`. Expected: all exit 0, coverage >= 80%, interrogate >= 80%. Report the numbers.
- [ ] **Step 7: Commit** `git commit -m "feat(ray_dalio): idea-meritocracy leveraged ETF team as a LangGraph graph (Task 4)"`

---

After Task 4: the user runs a first short backtest themselves (`uv run agent ray_dalio backtesting`, needs the LLM server and `env/.env.ray_dalio.backtesting`).
