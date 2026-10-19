# Bill Ackman: hold every surviving holding, 8 positions

## Problem

In backtest `2026-10-04_203136` the trader sold holdings the short seller had just passed. At the 2025-10-06
review the book held CROX, OMC, PYPL, AKAM and ALSN. The short seller passed seven names and failed CROX, which made
CROX `required`. Under `max_positions = 5` that left four slots for seven survivors. The trader swapped AKAM
(rank 6) and ALSN (rank 7) for HESM and CMCSA, and the rebalancer sold both as "not in the target portfolio". Seven
such sells happened in the first seven reviews. CMCSA was bought on 10/06 and sold in full on 10/13. This matches
the churn noted for the 2026-10-03 run.

The cause is that a `survive` verdict only makes a stock *eligible*. Only first fails are protected (`required`),
and nothing protects a healthy holding.

## Goal

Less turnover, with the same selection. A holding leaves the book only through the existing exits: two
consecutive fails (`forced_exit_fails`), or a quality-gate rejection, which code turns into a `fail`. The trader
still sets every weight.

Success: on a rerun of the same 1Y window, no "not in the target portfolio" sell of a stock that survived that
review, and no buy-then-sell round trip of a surviving name.

## Design

### 1. Every held name in the allowed set is required

`pipeline.py` step 7 currently does this:

```python
required = [symbol for symbol in outcome.pending if symbol in allowed]
```

It becomes every holding the trader may choose, in `holdings` order:

```python
required = [symbol for symbol in holdings if symbol in allowed]
```

`holdings` means the held stocks the pipeline already passes to `apply_verdicts`. A holding falls into one of
three groups:

- a survivor: in `allowed`, so now required;
- a pending (first) fail: in `allowed`, so required, as today;
- a forced exit: not in `allowed`, so not required and sold by code, as today.

A held symbol that is cooling down is outside `allowed`. That happens after a forced exit whose sell did not fill,
and `test_a_cooling_holding_is_judged_but_not_allowed_and_is_sold` covers it. Such a symbol is therefore not
required, and the rebalancer sells it as today.

`holdings` comes from `Rebalancer.holdings()`, sorted alphabetically and excluding the parking symbol. A held
position the screen has no SEC data for (`no_data`, e.g. an ETF) still reaches the short seller. If it survives it is
required like any other holding, where today the trader could drop it.

The existing truncation (`pipeline.py:204`) keeps its rule. When `len(required) > max_positions` it requires the
first `max_positions`, in holdings order, and logs a warning. The warning's "pending fails" becomes "holdings to
keep", since the list now holds survivors too. Holdings number at most `max_positions` after any
completed review, so this only fires when `max_positions` was lowered between runs (paper/live restart).

### 2. The validator's message covers both cases

`validate_portfolio` keeps its check: every `required` symbol must be present, and its weight already has to be
at least `min_weight`. Only the error text changes, because "failed once" is now wrong for a survivor:

```
{symbol} is held and has not failed twice: keep it with a weight of at least {min_weight}
```

It goes back to the model as `{"error": ...}` as before, and `_run_stage`'s forced retry quotes it.

### 3. The trader prompt

In `TRADER_SYSTEM`, the sentence "The names in required failed once while held: each must stay in your portfolio at
least at the minimum weight; you may reduce it, and code sells it if it fails again." becomes:

> The names in required are your current holdings that have not failed twice: each must stay in your portfolio at
> least at the minimum weight; you may reduce it. Code sells a holding when it fails twice in a row.

"Prefer changing little: keep a current holding near its current weight unless the ranking or the verdict gives a
reason to change it." stays. It now only guides weights, since dropping a holding is no longer an option.

The context shape is unchanged. Each allowed entry already carries `verdict`, `pending_fail_count` and
`current_weight`, so the model can see why a name is required.

### 4. `max_positions` 5 -> 8

This equals `research_top_n` (8), so every idea that survives in one review can be held. Protection has a cost: once
the book holds 8 stocks that have not failed twice, no newcomer can enter until a holding fails twice in a row or is
rejected on quality (there is no displacement rule, by choice). Watch the rerun for reviews where
`len(required) == max_positions`. The weights still fit, since
8 x `min_weight` 0.05 = 0.40 <= `max_total_weight` 0.98, and `__post_init__` already checks that.
`max_weight` (0.35), `min_weight` (0.05) and `rebalance_band` (0.05) are unchanged.

## Out of scope

- No change to `hysteresis.py`, the cooldowns, `Rebalancer` or `StateStore` (`STATE_VERSION` stays 2: the state
  shape does not change).
- No displacement rule (rank or yield margin). A held survivor is never displaced by a better newcomer.
- `reviews.jsonl`'s `required` field keeps its name. It now lists survivors as well as pending fails.

## Testing

Unit tests, with no network:

- `test_ackman_params.py`: the default `max_positions` is 8.
- `test_ackman_handoff.py`: the three `required` tests and the recorder test expect the new message.
- `test_ackman_pipeline.py`:
  - A held survivor is in the trader's `context["required"]` and in the `[trader] ... required:` log line.
  - A portfolio that drops it is refused, and the forced retry's task quotes the new message.
  - A new (unheld) survivor is not required.
  - A forced exit is neither allowed nor required.
  - The existing truncation test still passes, and the constraints assertion expects `max_positions == 8`.
  - Tests that assumed a held survivor could be dropped are updated to keep it.
- `test_ackman_prompts.py`: the trader prompt says required names are current holdings that have not failed
  twice, and keeps "at least at the minimum weight".

Manual: rerun the 1Y bill_ackman backtest. Check `backtest.log` for "not in the target portfolio" sells and compare
turnover and return against `2026-10-04_203136` (if it completes) and `2026-10-03_231803`.

## Docs

- `CLAUDE.md`, "Ackman hysteresis and open orders": every holding that is not a forced exit is `required`, not
  only first fails. The bill_ackman architecture paragraph does not mention `max_positions` and needs no change.
- `docs/superpowers/specs/2026-10-03-bill-ackman-stability-design.md` is a historical record and is left as is.
