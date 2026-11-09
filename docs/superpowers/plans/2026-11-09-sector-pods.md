# Sector Pods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A weekly `sector_pods` strategy: five parallel sector-pod agents, a risk manager and a portfolio manager wired as a LangGraph `StateGraph` over a shared state, with code executing the PM's weights through `Rebalancer`.

**Architecture:** Reusable graph plumbing (`AgentNode`, `SubmitSlot`, `merge_dicts`) goes in `agents/graph.py`. The strategy package `strategies/sector_pods/` holds the sleeve table, pure facts maths, submit schemas and validators, prompts, the JSON state store, the graph construction and the `Strategy` subclass. Each node takes the state, picks its slots, passes them to its agent's `run(context=...)` and returns its findings as a state update.

**Tech Stack:** Python 3.14, uv, LangGraph 1.2 (`StateGraph`, already installed with langchain 1.4), the project's `AgentManager`/`AgentHandle`, `Rebalancer`, `JsonStateStore`, `RunLog`, the `news_tools`/`macro_tools`/`memory_tools` factories, Yahoo daily bars.

**Spec:** `docs/superpowers/specs/2026-11-09-sector-pods-design.md`

## Global Constraints

- Paths below are relative to the repo root; `src/` means `src/trading_agent_framework/`.
- Python 3.14, `uv` only. No new third-party package: add `"langgraph>=1.2,<2"` to `[project.dependencies]` (already installed through langchain, now imported directly).
- LangGraph/LangChain imports stay inside function bodies in `agents/` and in the strategy (never at module level), as the rest of `agents/` does.
- Modules that define tool functions read by LangChain (`strategies/sector_pods/handoff.py`) have NO `from __future__ import annotations` (LangChain reads real annotations).
- No `time.sleep` / `datetime.now` in strategy code: `strategy.clock.now()`, `strategy.sleep()`.
- Money stays `Decimal` through `Rebalancer`; weights are floats (fractions of portfolio value), as in the other strategies. No new float boundary.
- Tests never touch the network; hand-written fakes (`tests/fakes.py`, fake agents whose `run` calls the real submit tools), no `MagicMock`.
- Pods: Technology & Communications `XLK, XLC, SMH, IGV, FDN`; Financials & Real Estate `XLF, KRE, KIE, IYR`; Healthcare `XLV, XBI, IHI, XPH, IHF`; Energy, Utilities, Materials & Industrials `XLE, XOP, OIH, XLU, XLB, XME, XLI, ITA, IYT`; Consumer & Housing `XLY, XLP, XHB, XRT`. Parking `SHV`, benchmark `SPY`.
- Order limits: weight >= 0, sleeve symbols only, <= 0.25 per ETF, <= 0.40 per pod, total <= 1 - cash_buffer (0.02). A sleeve whose pod is unavailable, and a held ETF absent from the week's facts: weight <= current weight (keep or trim), unheld ones not allowed.
- Pod scores are integers -2..+2, one per symbol of the pod's facts that week.
- Weekly: `sleeptime = "1D"`, review on `rebalance_weekday` 1 (Tuesday) at `"12:00"` ET; a completed week is not run again after a restart.
- 3 abandoned weeks in a row raise `FatalStrategyError` in a backtest only.
- Each code-step's public module/class/function gets a Google-style docstring (interrogate >= 80% on `src/`).
- Quality gate before each commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, the task's tests, then before the final commit the full `uv run pytest --cov` (>= 80%) and `uv run interrogate -v` (>= 80%). Run `uv sync --extra dashboard` (plain sync skips ~290 dashboard tests).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Pods really run concurrently and every other node (`load_previous`, `market_facts`, `risk_manager`, `portfolio_manager`, `execute`) runs on the caller's (strategy) thread: the root rule "strategy code runs on one thread" depends on it (verified with LangGraph 1.2.16: parallel tasks run on `ThreadPoolExecutor` workers, a single-task superstep runs on the calling thread). Pinned in Task 6.
2. A pod that fails (exception, no submission, loop) must never cause a sale of its sleeve and never a buy in it: keep-or-trim enforced by the PM's validator AND re-checked in `execute`. Pinned in Tasks 4 and 6.
3. Shared tool state across parallel pods: `news_tools`' `RunMemo` keeps ONE run id and holds its lock across the HTTP call, so pods sharing one instance would serialize and wipe each other's cache. Each pod gets its own `news_tools(self)` and `macro_tools(self)` instance. Pinned in Task 7.
4. A held ETF whose bars are missing that week (Yahoo gap, XLC before mid-2019 with the 200-day need) or whose pod failed: omitting it is weight 0, a sale, which keep-or-trim allows. So the PM must know which holdings nobody researched: its context carries `unresearched` (those held symbols with their current weights) and its prompt says an omitted symbol is sold and an unresearched holding can only be kept or cut. Pinned in Tasks 4, 5 and 6.
5. An abandoned week (PM unavailable, broker error) must not overwrite last completed week's `previous` state and must not mark the day completed except for the streak; a restart on the same Tuesday after a completed week must not trade again. Pinned in Task 6.

---

### Task 1: Move the daily-bars sources to `strategies/common/`

bull_bear's `market_data.py` (`DailySeries`, `DailyBars`, `YahooBars`, `GateBars`, `DataUnavailable`, `HISTORY_BARS`) is generic; sector_pods needs the same Yahoo-in-paper/live, gate-in-backtest split. Move it, do not copy it.

**Files:**
- Move: `src/strategies/bull_bear/market_data.py` -> `src/strategies/common/daily_bars.py` (`git mv`; module docstring generalized: "a strategy's daily bars", keep the IEX-volume reason)
- Move: `tests/strategies/bull_bear/test_bull_bear_market_data.py` -> `tests/strategies/common/test_common_daily_bars.py`
- Modify: `src/strategies/bull_bear/agent_bull_bear.py`, `src/strategies/bull_bear/pipeline.py`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py` (imports only)
- Modify: `src/strategies/bull_bear/CLAUDE.md`, `src/strategies/common/CLAUDE.md` (name the new location: `common/daily_bars.py`)

**Interfaces:**
- Produces: `trading_agent_framework.strategies.common.daily_bars` with the unchanged names `HISTORY_BARS = 300`, `DataUnavailable`, `DailySeries(closes, volumes, traded_closes=None, traded_volumes=None)`, `DailyBars.load(universe, held) -> dict[str, DailySeries]`, `YahooBars(source, *, today, sleep, warn, retry_delays, min_coverage)`, `GateBars(fetch, *, today, warn)`.

- [ ] **Step 1:** `git mv` both files; update every import (`grep -rn "bull_bear.market_data\|from .market_data" src tests scripts` must print nothing afterwards).
- [ ] **Step 2:** Run `uv run pytest tests/strategies/bull_bear tests/strategies/common -q` -> all pass, same count as before the move.
- [ ] **Step 3:** Gate (`ruff check`, `ruff format --check`, `pyright`) and commit `refactor: move bull_bear's daily-bars sources to strategies/common/daily_bars.py`.

---

### Task 2: Graph plumbing in `agents/graph.py`

**Files:**
- Create: `src/agents/graph.py`
- Create: `tests/agents/test_agent_graph.py`
- Modify: `pyproject.toml` (add `"langgraph>=1.2,<2"`; `uv lock`), `src/agents/CLAUDE.md` (one Architecture bullet for `graph.py` plus the thread facts of Review Focus 1)

**Interfaces:**
- Consumes: `AgentRunResult` (`agents/results.py`), `AgentError`/`AgentLoopError` (where `manager.py` imports them from), the logger `manager.py` uses.
- Produces:
  - `merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]` -- reducer for a slot written by parallel nodes (`{**left, **right}`, `None` as empty).
  - `class SubmitSlot`: `__init__(self, validate: Callable[[Any, Mapping[str, Any]], str | None])`; `arm(state: Mapping[str, Any]) -> None` (clears the value, keeps `state` for the validator); `submit(payload: Any) -> dict[str, Any]` (`{"error": msg}` when `validate` returns a message, else records and returns `{"status": "recorded"}`; a second valid submit in the same arming replaces the first); `value -> Any | None`.
  - `class AgentLike(Protocol)`: `run(task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None, max_steps: int | None = None) -> AgentRunResult` (what `AgentHandle` already is).
  - `@dataclass(frozen=True) class AgentNode`: fields `name: str`, `agent: AgentLike`, `slot: SubmitSlot`, `submit_tool: str`, `task: str`, `reads: Callable[[Mapping[str, Any]], Mapping[str, Any]]`, `write: Callable[[Any | None], dict[str, Any]]`, `tool_budget: int | None = None`, `max_steps: int | None = None`; `__call__(state: Mapping[str, Any]) -> dict[str, Any]`.
  - `AgentNode.__call__` behaviour: `slot.arm(state)`; `agent.run(task, context=reads(state), tool_budget=..., max_steps=...)`; if `slot.value` is set -> `write(value)`. Otherwise (no submission, or an `AgentError` other than `AgentLoopError`) log a WARNING `[<name>] no valid <submit_tool>: retrying with it forced` and run ONCE more with `force_tool=submit_tool` and the task followed by `"\n\nYour previous attempt ended without a valid <submit_tool> call. Call <submit_tool> now."`; still nothing -> `{**write(None), "errors": [{"node": name, "error": <reason>}]}` with a WARNING. An `AgentLoopError` skips the retry (a fresh run of a looping model loops again). Reasons are short strings: `"no valid submission"`, `"agent failed: <exc>"`, `"loop: <exc>"`.

- [ ] **Step 1: Write the failing tests** (`tests/agents/test_agent_graph.py`, with a local `ScriptedAgent` whose `run` pops the next script and calls it with `(slot, context, force_tool)`):
  - `test_merge_dicts_combines_parallel_writes`: `merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}`, `merge_dicts(None, {"b": 2}) == {"b": 2}`.
  - `test_submit_slot_refuses_invalid_payload`: validator returns `"bad"` for `{"x": 0}` -> `submit({"x": 0}) == {"error": "bad"}` and `slot.value is None`; valid -> `{"status": "recorded"}`, `slot.value == payload`; `arm({})` clears it.
  - `test_submit_slot_validator_sees_armed_state`: validator asserts on `state["allowed"]`.
  - `test_node_writes_the_submission`: agent submits `{"v": 1}` -> node returns `write({"v": 1})`; agent called once with `context == reads(state)`.
  - `test_node_retries_once_with_tool_forced`: first run submits nothing, second submits -> two runs, second has `force_tool == "submit_x"` and its task ends with `"Call submit_x now."`; result is the write.
  - `test_node_marks_unavailable_after_failed_retry`: both runs submit nothing -> returns `{**write(None), "errors": [{"node": "n", "error": "no valid submission"}]}`.
  - `test_node_retries_after_agent_error`: first run raises `AgentError("boom")`, second submits -> write.
  - `test_node_does_not_retry_a_loop`: first run raises `AgentLoopError` -> one run only, errors entry starts with `"loop:"`.
- [ ] **Step 2:** `uv run pytest tests/agents/test_agent_graph.py -v` -> FAIL (module missing).
- [ ] **Step 3:** Implement `src/agents/graph.py` per the Interfaces block (no LangGraph import needed here: nodes are plain callables, reducers plain functions).
- [ ] **Step 4:** Tests pass; gate; commit `feat(agents): AgentNode, SubmitSlot and merge_dicts for LangGraph shared-state graphs`.

---

### Task 3: Sleeve table, params and market facts (pure)

**Files:**
- Create: `src/strategies/sector_pods/__init__.py` (exports `SectorPodsStrategy` once Task 7 adds it; empty docstring module for now), `src/strategies/sector_pods/pods.py`, `src/strategies/sector_pods/params.py`, `src/strategies/sector_pods/facts.py`
- Create: `tests/strategies/sector_pods/__init__.py` if sibling test dirs have one, `tests/strategies/sector_pods/test_sector_pods_facts.py`, `tests/strategies/sector_pods/test_sector_pods_params.py`

**Interfaces:**
- Consumes: `compute_return_from_prices(closes, lookback, skip=0) -> float | None` (a FRACTION, not percent) and `annualized_volatility(closes, window) -> float | None` (fraction) from `strategies/common/scoring.py`; `parse_rebalance_time` from `strategies/common/sessions.py`.
- Produces:
  - `pods.py`: `@dataclass(frozen=True) Pod(key: str, label: str, focus: str, sleeve: tuple[str, ...])`; `PODS: tuple[Pod, ...]` in the order tech, financials, healthcare, energy, consumer, keys `"tech_pod"`, `"financials_pod"`, `"healthcare_pod"`, `"energy_pod"`, `"consumer_pod"`, sleeves from Global Constraints; `focus` one sentence each (tech: rates, growth and liquidity; financials: yield curve, credit spreads, policy rates; healthcare: defensive growth, drug pricing and policy; energy: oil and gas prices, inflation, dollar, industrial cycle, utilities' rate sensitivity; consumer: income, inflation, housing, retail demand); `ALL_SYMBOLS: tuple[str, ...]` (27, sleeve order); `pod_of(symbol) -> Pod`.
  - `params.py`: `@dataclass(frozen=True, slots=True) SectorPodsParams` with `rebalance_weekday: int = 1`, `rebalance_time: str = "12:00"`, `cash_buffer: float = 0.02`, `rebalance_band: float = 0.03`, `min_trade_pct: float = 0.005`, `parking_symbol: str = "SHV"`, `max_etf_weight: float = 0.25`, `max_pod_weight: float = 0.40`, `pod_tool_budget: int = 4`, `pm_memory_budget: int = 3`, `max_agent_steps: int = 40`, `agent_temperature: float | None = 0.3`, `thinking_token_budget: int | None = 16384`, `max_output_tokens: int | None = 24576`, `reason_max_chars: int = 200`, `summary_max_chars: int = 400`, `max_consecutive_abandoned: int = 3`, `yahoo_retry_delays: tuple[float, ...] = (60.0, 180.0)`, `min_yahoo_coverage: float = 0.9`; `__post_init__` raises `ConfigurationError` when a weight/cap is outside (0, 1], `max_etf_weight > max_pod_weight`, or `rebalance_weekday` not in 0..4. Satisfies `RebalanceParams` (`parking_symbol`, `rebalance_band`, `min_trade_pct`, `cash_buffer`).
  - `facts.py`: `MIN_BARS = 201` (200-day average + 1); `etf_facts(closes: Sequence[float], spy: Sequence[float]) -> dict[str, float] | None` -> `None` when `len(closes) < MIN_BARS`, else keys `return_1w_pct` (5 sessions), `return_1m_pct` (21), `return_3m_pct` (63), `return_6m_pct` (126), `relative_1m_pct`, `relative_3m_pct` (ETF minus SPY return, points), `volatility_20d_pct` (annualized), `vs_200d_avg_pct` (last close vs simple mean of the last 200 closes), all `round(x * 100, 1)`; a relative key is omitted when SPY lacks the bars. `outcome(weights: Mapping[str, float], pods: Mapping[str, Mapping[str, Any]], facts: Mapping[str, Mapping[str, Mapping[str, float]]], spy_1w_pct: float | None) -> dict[str, Any]` -> `{"window": "last 5 sessions", "holdings": {symbol: return_1w_pct}, "portfolio_1w_pct": weight-weighted mean of those returns (SHV/missing ignored), "spy_1w_pct": ..., "pods": {pod_key: {"positive_minus_negative_1w_pct": mean return_1w_pct of last week's +1/+2 symbols minus mean of its -1/-2 symbols}}}`; a pod with no positive or no negative scores, or unavailable last week, gets `None`. Comment the shortcut: `# shortcut: the outcome uses the last 5 sessions, not the sessions since the previous review; exact after a holiday week or an abandoned week only if DailySeries gains dates`.
- [ ] **Step 1: Write the failing tests:**
  - `test_pods_cover_27_unique_symbols`: `len(ALL_SYMBOLS) == 27 == len(set(ALL_SYMBOLS))`, `"SHV" not in ALL_SYMBOLS`, `pod_of("XLU").key == "energy_pod"`, `pod_of("IYR").key == "financials_pod"`.
  - `test_etf_facts_needs_201_bars`: 200 closes -> `None`.
  - `test_etf_facts_values`: `closes = [100.0] * 200 + [110.0]`, `spy = [100.0] * 201` -> `return_1w_pct == 10.0`, `return_6m_pct == 10.0`, `relative_1m_pct == 10.0`, `vs_200d_avg_pct == round((110 / 100.05 - 1) * 100, 1)` (mean of the last 200 closes: 199 x 100 + 110).
  - `test_etf_facts_omits_relative_without_spy_history`: `spy = [100.0] * 10` -> no `relative_1m_pct` key, `return_1m_pct` present.
  - `test_outcome_scores_pods_and_portfolio`: weights `{"XLK": 0.2, "XLE": 0.2}`, facts with `return_1w_pct` 2.0 / -1.0, last week's tech pod scored XLK +2 and SMH -1 (SMH `return_1w_pct` -3.0) -> `portfolio_1w_pct == 0.5`, `pods["tech_pod"]["positive_minus_negative_1w_pct"] == 5.0`, a pod without negatives -> `None`.
  - `test_params_reject_bad_caps`: `SectorPodsParams(max_etf_weight=0.5, max_pod_weight=0.4)` raises `ConfigurationError`.
- [ ] **Step 2:** Run -> FAIL. **Step 3:** Implement. **Step 4:** Pass; gate; commit `feat(sector_pods): sleeve table, params and market facts`.

---

### Task 4: Submit schemas, validators and submit tools

**Files:**
- Create: `src/strategies/sector_pods/handoff.py` (no `from __future__ import annotations`)
- Create: `tests/strategies/sector_pods/test_sector_pods_handoff.py`

**Interfaces:**
- Consumes: `SubmitSlot` (Task 2), `PODS`, `pod_of`, `SectorPodsParams` (Task 3).
- Produces (TypedDicts are the tool schemas the model sees):
  - `class Score(TypedDict): symbol: str; score: int; reason: str`
  - `class Objection(TypedDict): symbols: list[str]; kind: str; severity: str; reason: str` with `OBJECTION_KINDS = ("crowding", "macro", "drawdown", "reversal", "other")`, `SEVERITIES = ("low", "high")`.
  - `class Weight(TypedDict): symbol: str; weight: float; reason: str`
  - `UNAVAILABLE = {"unavailable": True}` (the marker an `AgentNode.write(None)` stores).
  - `validate_pod(pod: Pod, params) -> Callable[[Any, Mapping[str, Any]], str | None]`: payload `{"scores": [...], "sector_view": str}`; every symbol of `state["facts"][pod.key]` exactly once and nothing else (message names the missing/unknown/duplicate symbols), `score` an int in -2..2 (a bool is not an int), `reason` non-empty and <= `reason_max_chars`, `sector_view` non-empty and <= `summary_max_chars`.
  - `validate_risk(params)`: payload `{"objections": [...], "summary": str}`; objections may be empty; each symbol in `ALL_SYMBOLS`, `kind` in `OBJECTION_KINDS`, `severity` in `SEVERITIES`, reason length as above.
  - `check_weights(weights: Mapping[str, float], state: Mapping[str, Any], params) -> str | None` (also called by `execute` in Task 6): symbols in `ALL_SYMBOLS`; `0 <= w <= max_etf_weight`; per pod sum <= `max_pod_weight`; total <= `1 - cash_buffer` (1e-9 tolerance); keep-or-trim: for a symbol whose pod's `state["pods"][key]` is `UNAVAILABLE` or missing, or that is absent from `state["facts"]`, `w <= state["holdings"].get(symbol, 0.0) + 1e-9` (message: `"<SYM>: its pod gave no view this week, so it can only be kept at up to its current weight (<x>) or cut"`).
  - `validate_pm(params)`: payload `{"weights": [...], "overrides": str, "summary": str}`; each symbol at most once, reason length, then ONE `check_weights({symbol: weight for every entry}, state, params)` (the pod and total caps need the whole book).
  - `unresearched(state: Mapping[str, Any]) -> dict[str, float]`: the held symbols (`state["holdings"]`, SHV excluded) whose pod is `UNAVAILABLE`/missing in `state["pods"]` or that are absent from `state["facts"]`, with their current weight; the same rule `check_weights` applies, so both use it.
  - `pod_submit_tool(slot: SubmitSlot) -> Callable[..., dict[str, Any]]` named `submit_pod_view(scores: list[Score], sector_view: str)`; `risk_submit_tool(slot)` named `submit_risk_review(objections: list[Objection], summary: str)`; `pm_submit_tool(slot)` named `submit_target_weights(weights: list[Weight], overrides: str, summary: str)`. Each forwards a dict to `slot.submit`; one-line docstrings (they reach the model). Submit tool names exported as `POD_SUBMIT`, `RISK_SUBMIT`, `PM_SUBMIT`.
- [ ] **Step 1: Write the failing tests** (state fixtures: tech facts for XLK, SMH only; holdings `{"XLF": 0.10}`; `pods` with `financials_pod: UNAVAILABLE`):
  - `test_pod_view_must_cover_exactly_its_facts`: missing SMH -> error naming `SMH`; extra `XLE` -> error naming `XLE`; duplicate -> error.
  - `test_pod_score_range_and_type`: `3`, `-3`, `True`, `1.5` refused; `-2` and `2` accepted.
  - `test_reason_length_limits`: reason of 201 chars refused, 200 accepted (default params).
  - `test_risk_review_kinds_and_symbols`: kind `"liquidity"` refused, symbol `"SPY"` refused, empty objections accepted.
  - `test_weights_caps`: XLK 0.26 refused; XLK 0.25 + SMH 0.20 refused (tech pod 0.45 > 0.40); total 0.99 refused; total 0.98 within caps accepted.
  - `test_keep_or_trim_for_unavailable_pod`: XLF 0.10 accepted, 0.0 accepted, 0.11 refused; KRE 0.05 refused (unheld, pod unavailable).
  - `test_keep_or_trim_for_symbol_without_facts`: holdings `{"XLC": 0.08}`, tech pod available but XLC not in facts -> XLC 0.08 accepted, 0.09 refused.
  - `test_submit_tools_record_into_their_slot`: `pod_submit_tool(slot)(scores=[...], sector_view="x")` on a valid payload returns `{"status": "recorded"}` and `slot.value["scores"][0]["symbol"] == "XLK"`; tool `__name__ == "submit_pod_view"`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass; gate; commit `feat(sector_pods): validated submit tools for pods, risk and PM`.

---

### Task 5: Prompts

**Files:**
- Create: `src/strategies/sector_pods/prompts.py`
- Create: `tests/strategies/sector_pods/test_sector_pods_prompts.py`

**Interfaces:**
- Consumes: `Pod`, `PODS`, `SectorPodsParams`, the submit names `POD_SUBMIT`/`RISK_SUBMIT`/`PM_SUBMIT`.
- Produces: `pod_system(pod: Pod, params) -> str`, `RISK_SYSTEM: str` (or `risk_system(params)`), `pm_system(params) -> str`, and task strings `POD_TASK`, `RISK_TASK`, `PM_TASK`. Written fresh (not lumibot's or bull_bear's text). Content each must carry:
  - pod: its label, focus and sleeve (from the table, never hard-coded twice); what the context holds (`facts` = its sleeve's rows, `previous` = its own last view, `regime`, `date`); tools: `search_news` once or twice with the sleeve tickers and a `limit` of at most 5, `get_fred_series` once or twice for its macro focus; a score -2..+2 for every symbol in `facts`, one-line reason, justify a change of 2 points or more vs `previous`; end with `submit_pod_view`.
  - risk: reads `pods`, `facts`, `previous`, `regime`, `holdings`; no research; challenge crowding (the same bet across names/pods), macro contradictions between pods, drawdown and reversal risk; objections with kind/severity; end with `submit_risk_review`.
  - PM: `search_memory` first for relevant lessons; read `pods`, `risk`, `outcome`, `previous`, `holdings`, `unresearched`, `regime`; weights with a reason each, caps stated with the params' numbers (25% per ETF, 40% per pod, at most 98% in total, the rest parked in SHV); an omitted symbol is sold; a sleeve whose pod gave no view (or a holding without facts) can only be kept or cut; spread across several sectors in normal conditions, a small book only with clear risk-off evidence; say which objections it accepted or overrode; `remember_lesson` only when last week's outcome taught something specific; end with `submit_target_weights`.
- [ ] **Step 1: Write the failing tests:**
  - `test_pod_prompt_lists_its_sleeve`: for every pod, every sleeve symbol and the label appear in `pod_system(pod, params)`, and no other pod's exclusive symbol does.
  - `test_prompts_name_their_submit_tool`: `POD_SUBMIT` in each pod prompt, `RISK_SUBMIT` in the risk prompt, `PM_SUBMIT` in the PM prompt.
  - `test_pm_prompt_states_the_caps_from_params`: `"25%"`, `"40%"`, `"98%"` in `pm_system(SectorPodsParams())`; with `max_etf_weight=0.2` it says `"20%"`.
  - `test_no_prompt_contains_forced`: the word `forced` appears in no system prompt or task (the retry line in `AgentNode` is the only place it may appear).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass; gate; commit `feat(sector_pods): pod, risk and PM prompts`.

---

### Task 6: State store, graph state and the graph

**Files:**
- Create: `src/strategies/sector_pods/store.py`, `src/strategies/sector_pods/graph.py`
- Create: `tests/strategies/sector_pods/test_sector_pods_store.py`, `tests/strategies/sector_pods/test_sector_pods_graph.py`

**Interfaces:**
- Consumes: `JsonStateStore`, `InvalidState` (`common/json_state.py`); `RunLog`; `Rebalancer`, `PlacedOrder`; `target_portfolio(weights, *, cash_buffer)`; `DailySeries`, `DailyBars`, `DataUnavailable` (Task 1); `AgentNode`, `SubmitSlot`, `merge_dicts` (Task 2); `etf_facts`, `outcome`, `PODS` (Task 3); `check_weights`, `UNAVAILABLE` (Task 4); `BrokerError`, `BacktestError`.
- Produces:
  - `store.py`: `@dataclass(frozen=True) WeekState(last_completed: str | None = None, abandoned_streak: int = 0, previous: dict[str, Any] = {})` (`field(default_factory=dict)`; `previous` = `{"date", "pods", "risk", "pm"}` of the last completed week); `StateStore(JsonStateStore[WeekState])` with `STATE_VERSION = 1`, `__init__(path, *, log, fresh=False)`; decode validates `last_completed` (ISO date string or None), `abandoned_streak` (`is_int`, >= 0) and `previous` (a dict).
  - `graph.py`:
    - `class PodsState(TypedDict, total=False)`: `date: str`, `regime: str`, `holdings: dict[str, float]`, `previous: dict[str, Any]`, `facts: dict[str, dict[str, dict[str, float]]]` (pod key -> symbol -> row), `outcome: dict[str, Any]`, `pods: Annotated[dict[str, Any], merge_dicts]`, `risk: dict[str, Any]`, `pm: dict[str, Any]`, `errors: Annotated[list[dict[str, str]], operator.add]`, `orders: list[dict[str, Any]]`, `abandoned: str | None` (the reason, None when completed).
    - `@dataclass(frozen=True) GraphParts(load_previous, market_facts, pods: Mapping[str, AgentNode], risk_manager: AgentNode, portfolio_manager: AgentNode, execute)` where the three code nodes are `Callable[[PodsState], dict[str, Any]]`.
    - `build_graph(parts: GraphParts) -> Any` (the compiled graph): LangGraph imported inside; nodes `load_previous` -> `market_facts` -> each `PODS` key (loop) -> `risk_manager` (one `add_edge([pod keys], "risk_manager")` fan-in) -> `portfolio_manager` -> `execute` -> `END`. This function is the readable flow: no other logic in it.
    - `MarketFacts(bars: DailyBars, *, regime: Callable[[], int | None], holdings: Callable[[], dict[str, float]], today: Callable[[], date])` callable node: loads `ALL_SYMBOLS + ("SPY",)` with `held` = holdings' keys; `facts[pod.key][symbol] = etf_facts(...)` for symbols with a result; `regime` as `"bullish" / "neutral" / "bearish" / "unknown"`; `outcome(previous["pm"] weights, previous["pods"], facts, spy 1w)` when `previous` has a PM, else `{}`. `DataUnavailable` -> `{"abandoned": "data: <msg>"}`. The graph stays linear (no conditional edges): `skip_if_abandoned(node) -> node` (a small helper in `graph.py`) wraps each agent node and returns `{}` without running it when `state.get("abandoned")`. A pod whose `facts[pod.key]` is empty is not run either and writes `{"pods": {key: UNAVAILABLE}}` (`skip_pod_without_facts(key, node)`, same file).
    - Node `reads` (the slots each agent sees; strategy wiring in Task 7 passes these): pod -> `{"date", "regime", "facts": facts[key], "previous": previous["pods"][key] or None}`; risk -> `{"date", "regime", "holdings", "facts", "pods", "previous": previous["risk"] or None}`; PM -> `{"date", "regime", "holdings", "unresearched": unresearched(state), "facts", "outcome", "pods", "risk", "previous": previous["pm"] or None}`. Exposed as `pod_reads(key)`, `risk_reads`, `pm_reads`.
    - `LoadPrevious(store: StateStore)` node: `{"previous": store.load().previous}`.
    - `Execute(store: StateStore, rebalancer: Rebalancer, log: RunLog, params, *, on_warning: Callable[[str], None], on_error: Callable[[str], None])` node: if `state["abandoned"]` or `state["pm"]` is `UNAVAILABLE`/missing -> abandoned week; else `check_weights` again (a failure is an abandoned week, logged as an error, no order), `rebalancer.rebalance(target_portfolio(weights, cash_buffer=params.cash_buffer))`, a `BrokerError`/`BacktestError` -> abandoned week with `orders` = what `rebalancer.placed` holds. Completed week: save `WeekState(last_completed=date, abandoned_streak=0, previous={"date", "pods", "risk", "pm"})`. Abandoned week: save the loaded state with `abandoned_streak + 1` only. Always append one `RunLog` record: `{"date", "regime", "facts", "pods", "risk", "pm", "orders", "errors", "abandoned", "abandoned_streak"}`. Returns `{"orders": [...], "abandoned": reason_or_None}`.
- [ ] **Step 1: Write the failing tests:**
  - store: `test_store_round_trips_week_state`, `test_store_rejects_negative_streak` (loads empty with a warning), `test_store_fresh_wipes`.
  - graph (fake bars giving 260 closes per symbol, fake agents per node that call the real submit tools of Task 4, a real `Rebalancer` over `tests.fakes.FakeBroker` as bull_bear's pipeline tests do, `tmp_path` store and run log):
    - `test_graph_runs_pods_in_parallel`: each pod's fake agent waits on a shared `threading.Barrier(5, timeout=5)` before submitting -> the run completes (a sequential run would raise `BrokenBarrierError`).
    - `test_single_nodes_run_on_the_calling_thread`: the risk, PM agents and the `execute` rebalancer record `threading.current_thread()`; all equal the test's thread; each pod's thread differs from it.
    - `test_agents_see_only_their_slots`: the tech pod's context has keys `{"date", "regime", "facts", "previous"}` and `facts` holds tech symbols only; the risk context has `pods` with 5 entries; the PM context has `pods`, `risk`, `outcome`, `holdings`, `unresearched`.
    - `test_pm_sees_unresearched_holdings`: holdings XLF 0.10 with the financials pod failing -> PM context `unresearched == {"XLF": 0.10}`.
    - `test_completed_week_trades_and_saves_previous`: PM weights XLK 0.2, XLV 0.2 -> FakeBroker receives buys for XLK, XLV and SHV; store `last_completed == date`, `previous["pm"]["weights"]` saved; run log line has `abandoned is None`.
    - `test_failed_pod_keeps_or_trims_its_sleeve`: holdings XLF 0.10, the financials pod raises twice -> `pods["financials_pod"] == UNAVAILABLE`, `errors` names `financials_pod`; a PM that tries XLF 0.15 is refused (tool returns error) then submits XLF 0.10 -> no XLF order.
    - `test_pm_unavailable_abandons_without_orders`: PM never submits -> no order, `abandoned` set, store `abandoned_streak == 1`, `previous` unchanged from before the run.
    - `test_data_unavailable_abandons_before_agents`: bars raise `DataUnavailable` -> no agent `run` call, no order, streak + 1.
    - `test_broker_error_during_rebalance_is_logged_and_abandoned`: FakeBroker raising `BrokerError` on the second order -> run log `orders` has the first, `abandoned` starts with `"execution"`.
    - `test_previous_week_reaches_agents_and_outcome`: a store holding last week's PM weights and pod scores -> the PM context `outcome["portfolio_1w_pct"]` is a number and the tech pod context `previous` is last week's tech view.
- [ ] **Step 2:** FAIL. **Step 3:** Implement `store.py` then `graph.py`. **Step 4:** Pass; gate; commit `feat(sector_pods): shared-state graph, market facts, execution and weekly state`.

---

### Task 7: `SectorPodsStrategy` and registration

**Files:**
- Create: `src/strategies/sector_pods/strategy.py`; export from `src/strategies/sector_pods/__init__.py`
- Modify: `src/utils/strategy_factory.py` (`Strategies.SECTOR_PODS`, `_build_sector_pods(broker, mode)` -> `SectorPodsStrategy(broker, mode=mode)`)
- Create: `tests/strategies/sector_pods/test_sector_pods_strategy.py`, `tests/strategies/sector_pods/test_sector_pods_backtest.py`

**Interfaces:**
- Consumes: everything above; `Strategy`; `news_tools(strategy)`, `macro_tools(strategy, *, fred_client_factory=None)`, `memory_tools(store)`, `only(tools, names)`; `YahooDailyBars` (paper/live), `GateBars` (backtest); `backtest_window(PredefinedWindow.SEMI_DECADE_LATEST)`; `MARKET_TZ`; `FatalStrategyError`, `ConfigurationError`, `BrokerError`.
- Produces: `class SectorPodsStrategy(Strategy)`: `sleeptime = "1D"`; `parameters = {"backtesting_start"/"backtesting_end": backtest_window(PredefinedWindow.SEMI_DECADE_LATEST), "benchmark_symbol": "SPY", "warmup_trading_days": 300, "budget": 10000}` plus `max_etf_weight`, `max_pod_weight`, `thinking_token_budget`, `max_output_tokens` mirrored in `__init__` (recorded in `settings.json`); `__init__(broker, *, mode=TradingMode.PAPER, settings: SectorPodsParams | None = None, bars_source: BarsSource | None = None, fred_client_factory=None, **kwargs)`; `iteration_start_time = parse_rebalance_time(settings.rebalance_time)`.
  - `initialize()`: requires a news provider (`broker.news_provider()` not None, `BrokerError` included) and builds every agent; any `ConfigurationError` (no news provider, no `LLM_MODEL`, no `FRED_API_KEY`) -> `FatalStrategyError`. Agents: one per pod named by its key (`tools=[*news_tools(self), *macro_tools(self, fred_client_factory=...), submit]`, a NEW `news_tools`/`macro_tools` call per pod, `exempt_tools=[POD_SUBMIT]`, `final_tools=[POD_SUBMIT]`), `risk_manager` (submit tool only), `portfolio_manager` (`only(memory_tools(self.memory), {"search_memory", "remember_lesson"})` + submit, exempt and final = `PM_SUBMIT`); all with `temperature`, `max_output_tokens`, `thinking_token_budget` from params. `AgentNode`s get `tool_budget` (`pod_tool_budget`, `pm_memory_budget`, none for risk) and `max_steps=max_agent_steps`; the graph is built once and kept.
  - `on_trading_iteration()`: not `rebalance_weekday` -> log and return; `store.load().last_completed == today` -> log `"[sector_pods] the week of <date> already completed: not run again"` and return; else `graph.invoke({"date": today.isoformat()})`; when the result is abandoned and `is_backtesting` and the streak >= `max_consecutive_abandoned` -> `FatalStrategyError`.
  - `run_backtesting(**overrides)`: defaults `preload_assets=[Asset(s) for s in (*ALL_SYMBOLS, "SHV", "SPY")]`, `timestep="day"`, `agent_telemetry=True`; the data source stays the framework default (`YahooBacktestData`).
  - Holdings for the graph: `Rebalancer.current_weights()`.
- [ ] **Step 1: Write the failing tests:**
  - `test_skips_other_weekdays` and `test_completed_week_not_run_again` (FakeClock on a Wednesday / on a Tuesday with `last_completed` = that Tuesday -> graph not invoked).
  - `test_initialize_without_news_provider_is_fatal` and `test_initialize_without_fred_key_is_fatal` (`monkeypatch.delenv("FRED_API_KEY")`, no factory).
  - `test_each_pod_gets_its_own_news_and_macro_tools`: a recording fake `AgentManager.create` (as `tests/strategies/bull_bear/test_bull_bear_backtest.py` does) -> 7 agents named `tech_pod ... consumer_pod, risk_manager, portfolio_manager`; the five pods' `search_news` tool objects are pairwise distinct (`is not`); the PM's tools are exactly `search_memory`, `remember_lesson`, `submit_target_weights`; the risk manager's only tool is `submit_risk_review`.
  - `test_factory_builds_sector_pods`: `build_strategy(Strategies.SECTOR_PODS, FakeBroker(...), TradingMode.BACKTESTING)` is a `SectorPodsStrategy`.
  - `test_backtest_end_to_end` (`test_sector_pods_backtest.py`, modelled on `test_bull_bear_backtest.py`: `FakeBacktestDataSource` with 300+ warmup sessions of bars for the 27 ETFs, SHV and SPY over ~3 weeks of sessions, `FakeNewsProvider`, a fake FRED factory, fake manager whose handles call the real submit tools): two Tuesdays traded, the run directory's `weeks.jsonl` (`RunLog(self.run_file("weeks.jsonl"), what="week log")`) has 2 lines with `abandoned` None, the second line's PM context saw `previous`.
  - `test_backtest_aborts_after_three_abandoned_weeks`: a PM that never submits over 3 Tuesdays -> `run_backtesting` raises `BacktestError` (the runner's wrap of `FatalStrategyError`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Pass; gate; commit `feat(sector_pods): SectorPodsStrategy, weekly graph run and factory registration`.

---

### Task 8: Documentation and final gate

**Files:**
- Create: `src/strategies/sector_pods/CLAUDE.md` (architecture: the graph diagram, slot table, who reads what, keep-or-trim, failure handling, state file `data/sector_pods_state_<mode>.json`, `weeks.jsonl`; gotchas: per-pod tool instances and why, pods on worker threads / everything else on the strategy thread, the 5-session outcome shortcut, env needs `LLM_*`, `ALPACA_NEWS_*`, `FRED_API_KEY`)
- Modify: `CLAUDE.md` (root: add `sector_pods` to the `strategies/CLAUDE.md` bullet list; add under "Strategy code runs on one thread": "a LangGraph graph's parallel nodes run on worker threads: they may only call read-only research tools; every single node, including order code, runs on the strategy thread"); `src/strategies/CLAUDE.md` (add `sector_pods/` to the list); `README.md` (a `sector_pods` section in the same table format as bull_bear: File, Model, Agents, Tools, Asset universe, Agent frequency, Trading modes, Benchmark, Env file `env/.env.sector_pods.<mode>`: `LLM_*`, `ALPACA_NEWS_*`, `FRED_API_KEY`, broker keys in paper/live; one paragraph; inspired-by link and "not affiliated with Citadel"); the `uv run agent` strategy list wherever README/CLAUDE.md enumerate strategy names.

- [ ] **Step 1:** Write the docs.
- [ ] **Step 2: Full gate:** `uv sync --extra dashboard`, `uv run ruff check` (0 errors), `uv run ruff format --check` (exit 0), `uv run pyright` (0 errors), `uv run pytest --cov` (all pass, total >= 80%, the new modules covered), `uv run interrogate -v` (>= 80%). Report the numbers.
- [ ] **Step 3:** Commit `docs(sector_pods): strategy CLAUDE.md, README section, thread rule`.
- [ ] **Step 4:** Tell the user to create `env/.env.sector_pods.backtesting` (and `.paper`/`.live` when needed) with `LLM_*`, `ALPACA_NEWS_*`, `FRED_API_KEY` (gitignored, never created or read by the agent), then `uv run agent sector_pods backtesting`; a 2-week smoke window first.
