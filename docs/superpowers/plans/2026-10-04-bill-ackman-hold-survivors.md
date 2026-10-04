# Bill Ackman: Hold Every Surviving Holding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The trader can no longer drop a held stock that has not failed twice, and the book can hold 8 stocks
instead of 5.

**Architecture:** The pipeline's `required` list (today: first fails) becomes every holding in the trader's allowed
set. The existing `validate_portfolio` check then refuses a portfolio that drops one, and the agent corrects itself
in the same run. Only the error text, a warning's wording, the trader prompt and the `max_positions` default change.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-04-bill-ackman-hold-survivors-design.md`

## Global Constraints

- New validator error, exactly: `{symbol} is held and has not failed twice: keep it with a weight of at least {min_weight}`
- Truncation warning: `[bill_ackman] {n} holdings to keep but max_positions is {max_positions}: requiring only {...}; left to the trader's choice: {...}`
- `required` = `[symbol for symbol in holdings if symbol in allowed]` (`holdings` is `Rebalancer.holdings()`: sorted, parking excluded).
- `AckmanParams.max_positions` default 8. `max_weight` 0.35, `min_weight` 0.05 and `rebalance_band` 0.05 are unchanged.
- No change to `hysteresis.py`, `rebalancer.py`, `state.py` (`STATE_VERSION` stays 2), or the trader context's shape.
- Tests never touch the network. Run with `uv run pytest`, lint with `uv run ruff check`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A held position with no SEC data (`no_data`, e.g. an ETF) that survives: now required. Task 1 rewrites `test_a_position_the_screen_does_not_know_can_be_dropped_by_the_trader_and_is_then_sold` to pin this.
2. A held symbol that is cooling down: outside `allowed`, so not required, and still sold. Task 1 adds a `required == []` assertion to `test_a_cooling_holding_is_judged_but_not_allowed_and_is_sold`.
3. A forced exit while a new idea survives: neither allowed nor required, and sold by code. Task 1 adds `test_a_forced_exit_is_neither_allowed_nor_required`.
4. More holdings to keep than `max_positions` (paper/live restart with a lowered cap): the first ones in holdings order are required, the rest are left to the trader, with a warning. Task 1 adds `test_holdings_to_keep_beyond_max_positions_are_capped_in_holdings_order`.
5. A required holding shrunk to exactly `min_weight` is accepted. The existing `test_a_required_holding_may_be_shrunk_to_the_minimum_weight` already covers it and is left unchanged.

---

### Task 1: Every held name in the allowed set is required

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/handoff.py:211` (error message)
- Modify: `src/trading_agent_framework/strategies/bill_ackman/pipeline.py:203-210` (`required`, warning)
- Modify: `src/trading_agent_framework/strategies/bill_ackman/prompts.py:63-65` (trader prompt)
- Modify: `CLAUDE.md` ("Ackman hysteresis and open orders" bullet)
- Test: `tests/strategies/bill_ackman/test_ackman_handoff.py`, `tests/strategies/bill_ackman/test_ackman_pipeline.py`, `tests/strategies/bill_ackman/test_ackman_prompts.py`

**Interfaces:**
- Consumes: `validate_portfolio(..., required: Collection[str])`, `HandoffRecorder.expect_portfolio(allowed, required=)` (unchanged signatures).
- Produces: the trader context's `required` list now includes held survivors. The `reviews.jsonl` `required` field follows it.

- [ ] **Step 1: Update the handoff tests to the new message**

In `tests/strategies/bill_ackman/test_ackman_handoff.py`:

```python
def test_a_required_holding_must_stay_in_the_portfolio() -> None:
    with pytest.raises(HandoffError, match="CCC is held and has not failed twice: keep it with a weight of at least 0.05"):
        validate_portfolio([_position("AAA", 0.3)], allowed=["AAA", "CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def test_an_empty_portfolio_is_refused_while_a_holding_is_required() -> None:
    with pytest.raises(HandoffError, match="CCC is held and has not failed twice"):
        validate_portfolio([], allowed=["CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)
```

In `test_the_recorder_checks_flips_and_required_holdings_against_what_it_was_armed_with`, replace
`assert "CCC failed once" in tools["submit_portfolio"]([_position("AAA", 0.3)])["error"]` with:

```python
    assert "CCC is held and has not failed twice" in tools["submit_portfolio"]([_position("AAA", 0.3)])["error"]
```

- [ ] **Step 2: Update and add the pipeline tests**

In `tests/strategies/bill_ackman/test_ackman_pipeline.py`:

(a) In `test_a_pending_fail_the_trader_drops_is_refused_and_the_holding_is_kept`, replace
`assert "HHH failed once and is kept until a second consecutive fail" in h.trader.calls[1]["task"]` with:

```python
    assert "HHH is held and has not failed twice" in h.trader.calls[1]["task"]
```

(b) Replace the whole `test_a_position_the_screen_does_not_know_can_be_dropped_by_the_trader_and_is_then_sold` with:

```python
def test_a_held_position_the_screen_does_not_know_is_required_once_it_survives(tmp_path: Path) -> None:
    # A dedicated account is assumed, so any position is reviewed; an ETF has no SEC data and reaches the short seller with a reduced sheet.
    screen = FakeScreen([], holding_rejections={"HHH": "no_data"})
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.short_seller.steps = [judges(HHH="survive")]
    h.trader.steps = [holds(), holds(HHH=0.3)]  # an empty portfolio is refused: HHH survived and is held

    assert h.run().completed

    assert h.trader.calls[0]["context"]["required"] == ["HHH"]
    assert "HHH is held and has not failed twice" in h.trader.calls[1]["task"]
    assert ("HHH", "sell", 40.0) in h.orders  # shrunk from 50% to 30% (5,000 -> 3,000 at $50), not sold out
```

(c) In `test_a_failure_after_a_sell_went_out_abandons_the_review_and_logs_the_orders_already_sent`, HHH can no
longer be dropped. Make it a forced exit instead (its state already has `fail_counts={"HHH": 1}`). Replace the
two step lines with:

```python
    h.short_seller.steps = [judges(AAA="survive", HHH="fail")]
    h.trader.steps = [holds(AAA=0.3)]  # HHH fails a second time: the rebalancer sells it, then sizes the buys
```

The assertions stay as they are: the orders are still `[("HHH", "sell", 100.0)]`, and the state is not saved.

(d) In `test_more_pending_fails_than_max_positions_are_capped_so_the_trader_can_still_comply`, replace
`assert any("pending fail" in message and left_out in message for message in warnings)` with:

```python
    assert any("holdings to keep" in message and left_out in message for message in warnings)
```

(e) At the end of `test_a_cooling_holding_is_judged_but_not_allowed_and_is_sold`, add:

```python
    assert h.trader.calls[0]["context"]["required"] == []  # cooling: not allowed, so not required
```

(f) Add these tests after `test_a_pending_fail_the_trader_drops_is_refused_and_the_holding_is_kept`:

```python
def test_a_held_survivor_is_required_and_a_portfolio_that_drops_it_is_refused(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="survive")]
    h.trader.steps = [holds(AAA=0.3), holds(AAA=0.3, HHH=0.2)]

    with caplog.at_level(logging.INFO):
        h.run()

    assert h.trader.calls[0]["context"]["required"] == ["HHH"]  # the held survivor; AAA, a new survivor, is not
    assert h.trader.calls[1]["force_tool"] == "submit_portfolio"
    assert "HHH is held and has not failed twice" in h.trader.calls[1]["task"]
    assert ("HHH", "sell", 60.0) in h.orders  # shrunk from 50% to 20%, not sold out
    assert any("[trader] 2 allowed: AAA, HHH; required: HHH; forced exits: none" in record.getMessage() for record in caplog.records)
    (line,) = h.log_lines()
    assert line["required"] == ["HHH"]


def test_a_forced_exit_is_neither_allowed_nor_required(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, state=ReviewState(fail_counts={"HHH": 1}))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="fail")]
    h.trader.steps = [holds(AAA=0.3)]

    h.run()

    context = h.trader.calls[0]["context"]
    assert [entry["symbol"] for entry in context["allowed"]] == ["AAA"]
    assert context["required"] == [] and context["forced_exits"] == ["HHH"]
    assert ("HHH", "sell", 100.0) in h.orders


def test_holdings_to_keep_beyond_max_positions_are_capped_in_holdings_order(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    screen = FakeScreen([], holdings=[_candidate("AAA"), _candidate("BBB"), _candidate("CCC")])
    h = _harness(tmp_path, screen, held={"AAA": 10, "BBB": 10, "CCC": 10}, params=AckmanParams(max_positions=2))
    h.short_seller.steps = [judges(AAA="survive", BBB="fail", CCC="survive")]
    keeps_required: Step = lambda tools, ctx: tools["submit_portfolio"]([{"symbol": s, "weight": 0.1, "reason": "kept"} for s in ctx["required"]])  # noqa: E731
    h.trader.steps = [keeps_required]

    with caplog.at_level(logging.WARNING):
        outcome = h.run()

    assert outcome.completed is True and len(h.trader.calls) == 1
    assert h.trader.calls[0]["context"]["required"] == ["AAA", "BBB"]  # survivors and pending fails alike, in holdings order
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("3 holdings to keep but max_positions is 2" in message and "left to the trader's choice: CCC" in message for message in warnings)
```

- [ ] **Step 3: Update the prompt test**

In `tests/strategies/bill_ackman/test_ackman_prompts.py`, replace
`test_the_trader_must_keep_required_holdings_and_is_no_longer_told_to_let_go` with:

```python
def test_the_trader_must_keep_every_required_holding_and_is_no_longer_told_to_let_go() -> None:
    assert "let go" not in prompts.TRADER_SYSTEM
    assert "failed once while held" not in prompts.TRADER_SYSTEM
    assert "current holdings that have not failed twice" in prompts.TRADER_SYSTEM
    assert "required" in prompts.TRADER_SYSTEM and "at least at the minimum weight" in prompts.TRADER_SYSTEM
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_handoff.py tests/strategies/bill_ackman/test_ackman_pipeline.py tests/strategies/bill_ackman/test_ackman_prompts.py -q`

Expected failures:
- The handoff tests and `test_a_pending_fail_...`, because of the old message.
- `test_a_held_position_the_screen_does_not_know_...`, because `required == []`.
- `test_a_held_survivor_is_required_...`, for the same reason.
- `test_holdings_to_keep_beyond_...`, because `required == []` and the warning reads "pending fails".
- `test_more_pending_fails_...`, because of the warning wording.
- The prompt test.

`test_a_forced_exit_is_neither_allowed_nor_required`, the cooling test and the reworked
`test_a_failure_after_a_sell_went_out_...` already pass, since they pin behaviour that does not change.

- [ ] **Step 5: Change the validator message**

In `src/trading_agent_framework/strategies/bill_ackman/handoff.py`, `validate_portfolio`:

```python
    held = {position.symbol for position in positions}
    for symbol in required:
        if symbol not in held:
            raise HandoffError(f"{symbol} is held and has not failed twice: keep it with a weight of at least {min_weight}")
```

and its docstring's last sentence becomes: `Every symbol in `required` (a holding that has not failed twice) must be held.`

- [ ] **Step 6: Require every held name in the allowed set**

In `src/trading_agent_framework/strategies/bill_ackman/pipeline.py`, replace lines 203-210 with:

```python
        required = [symbol for symbol in holdings if symbol in allowed]  # held and not failed twice: may be shrunk, not dropped
        if len(required) > params.max_positions:  # the trader could not hold them all: require the first max_positions
            left_out = required[params.max_positions :]
            strategy.log_warning(
                f"[bill_ackman] {len(required)} holdings to keep but max_positions is {params.max_positions}: "
                f"requiring only {', '.join(required[: params.max_positions])}; left to the trader's choice: {', '.join(left_out)}"
            )
            required = required[: params.max_positions]
```

- [ ] **Step 7: Update the trader prompt**

In `src/trading_agent_framework/strategies/bill_ackman/prompts.py`, `TRADER_SYSTEM`, replace:

```python
    "and reason, a pending_fail_count (a holding that failed once), its current_weight and its fact sheet. The names in "
    "required failed once while held: each must stay in your portfolio at least at the minimum weight; you may reduce it, "
    "and code sells it if it fails again. Names in forced_exits are sold by code and are not allowed.\n"
```

with:

```python
    "and reason, a pending_fail_count (a holding that failed once), its current_weight and its fact sheet. The names in "
    "required are your current holdings that have not failed twice: each must stay in your portfolio at least at the "
    "minimum weight; you may reduce it. Code sells a holding when it fails twice in a row. Names in forced_exits are sold "
    "by code and are not allowed.\n"
```

- [ ] **Step 8: Run the bill_ackman tests**

Run: `uv run pytest tests/strategies/bill_ackman/ -q`
Expected: all pass.

`test_ackman_strategy.py` and `test_ackman_backtest.py` drive whole reviews. If a test there fails with "is held and
has not failed twice", it relied on the trader dropping a held survivor. Fix it the way Step 2 (c) does: keep the
holding in the scripted portfolio, or make it fail where the test is about selling. Never loosen the rule.

- [ ] **Step 9: Update CLAUDE.md**

In the "**Ackman hysteresis and open orders.**" bullet, replace:

`A holding at its first fail is `required`: `submit_portfolio` refuses a portfolio without it at `min_weight` or more (the trader may shrink it, not drop it; its old prompt told it to let go of a failed name, and it dropped every pending fail).`

with:

`Every holding in the allowed set is `required` -- a survivor as well as a first fail -- and `submit_portfolio` refuses a portfolio without it at `min_weight` or more: the trader may shrink a holding, never drop it, so a holding leaves only through two consecutive fails, a quality-gate rejection (a code `fail`) or a cooldown (its old prompt told it to let go of a failed name, and it dropped every pending fail; before 2026-10-04 survivors were droppable and the trader swapped them for newer ideas every review, churning the book).`

- [ ] **Step 10: Lint and the full suite**

Run: `uv run ruff check && uv run pytest -q`
Expected: no lint errors, all tests pass.

- [ ] **Step 11: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/handoff.py src/trading_agent_framework/strategies/bill_ackman/pipeline.py src/trading_agent_framework/strategies/bill_ackman/prompts.py tests/strategies/bill_ackman/ CLAUDE.md
git commit -m "$(cat <<'EOF'
feat: bill_ackman requires every holding that has not failed twice

The trader swapped held survivors for newer ideas every review; a holding
now leaves only through two consecutive fails, a quality-gate rejection or
a cooldown.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `max_positions` 5 -> 8

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/parameters.py:15`
- Test: `tests/strategies/bill_ackman/test_ackman_params.py:17`, `tests/strategies/bill_ackman/test_ackman_pipeline.py:257`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AckmanParams().max_positions == 8`, which reaches the trader's `constraints.max_positions` and `validate_portfolio`.

- [ ] **Step 1: Update the tests**

`tests/strategies/bill_ackman/test_ackman_params.py`, in `test_the_defaults_are_the_spec_values`:

```python
    assert (params.research_top_n, params.max_positions) == (8, 8)
```

`tests/strategies/bill_ackman/test_ackman_pipeline.py`, in `test_the_agents_get_the_context_the_design_promises`:

```python
    assert trader["constraints"]["max_positions"] == 8 and trader["constraints"]["max_total_weight"] == pytest.approx(0.98)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_params.py::test_the_defaults_are_the_spec_values "tests/strategies/bill_ackman/test_ackman_pipeline.py::test_the_agents_get_the_context_the_design_promises" -q`
Expected: 2 failed (`5 != 8`).

- [ ] **Step 3: Change the default**

`src/trading_agent_framework/strategies/bill_ackman/parameters.py`:

```python
    max_positions: int = 8  # stocks in the target portfolio; equal to research_top_n, so every surviving idea can be held
```

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run ruff check && uv run pytest -q`
Expected: all pass. `__post_init__` accepts 8 x 0.05 = 0.40 <= 0.98.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bill_ackman/parameters.py tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "$(cat <<'EOF'
feat: bill_ackman holds up to 8 stocks, one per researched idea

With every holding protected, 5 slots would block new survivors until a
double fail freed one.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### After both tasks (manual, by the user)

Rerun the 1Y backtest: `uv run agent bill_ackman backtesting`. In the new run's `backtest.log`, check that every
"not in the target portfolio" sell is of a name that was not a held survivor in that review, and that no surviving
name is bought and then sold. Compare return, turnover and drawdown in the dashboard against `2026-10-03_231803`
and, if it has finished, `2026-10-04_203136`.
