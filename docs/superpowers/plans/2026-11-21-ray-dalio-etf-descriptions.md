# ray_dalio ETF descriptions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every ray_dalio agent sees each universe symbol's signed daily leverage and what it tracks, so it stops guessing funds from memory.

**Architecture:** `UNIVERSE` in `strategies/ray_dalio/prompts.py` becomes a `dict[str, str]` (symbol -> description) in today's order; `CONTEXT["universe"]` carries that dict to every agent; `LEVERAGE_RULE` gains one sentence. Callers iterate `UNIVERSE` for symbols, which a dict does unchanged.

**Tech Stack:** Python 3.14, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-21-ray-dalio-etf-descriptions-design.md`

## Global Constraints

- The 54 symbols and their order stay exactly as today; SHV stays last.
- Description format: `"<+N|-N>x daily[ (inverse)], <underlying> (<issuer>)"`; SHV is `"1x, 0-1 year US Treasury bills (iShares), the cash sleeve"`.
- Pass the code quality gate (ruff check, ruff format --check, pytest, pytest --cov >= 80%, interrogate >= 80%); no `noqa` or ignores.
- Run commands from the worktree; it has no `.venv`, so sync first: `uv sync --extra dashboard`.

## Review Focus

- A leverage sign typed wrong (e.g. `+2x` for GLL): the agents would trust it fully, worse than today. Pinned by the spot-check test and by checking every inverse fund's description says `-` and `(inverse)`.
- A description that names an old lumibot tool name: `test_no_lumibot_tool_name_is_left` must see dict values (Task 1, step 1).
- The context grows by ~1k tokens per call: acceptable (peak 30k of 65k); no test.
- `UNIVERSE[-1]` style indexing elsewhere: grep found only `test_prompts.py:27`.

---

### Task 1: Describe every fund in the universe

**Files:**
- Modify: `src/trading_agent_framework/strategies/ray_dalio/prompts.py:8-63` (`UNIVERSE`), `:80-86` (`LEVERAGE_RULE`), `:147` (`CONTEXT["universe"]`), module docstring if it calls `UNIVERSE` a tuple
- Modify: `tests/strategies/ray_dalio/test_prompts.py:16-27`
- Modify: `src/trading_agent_framework/strategies/ray_dalio/CLAUDE.md` (Architecture bullet on `prompts.py`, Gotchas line "The 54 tickers were taken from lumibot and not verified tradable")

**Interfaces:**
- Produces: `UNIVERSE: dict[str, str]`; `CONTEXT["universe"]: dict[str, str]` (a copy of `UNIVERSE`).
- Unchanged consumers: `agent_ray_dalio.py:114` `(*UNIVERSE, "SPY")`, `tests/strategies/ray_dalio/test_ray_dalio_strategy.py:154` `[*UNIVERSE, "SPY"]`.

Leverage and underlying for every symbol (checked against Yahoo `longName` on 2026-11-21; issuer in brackets: P = ProShares, D = Direxion):

| Symbols | Leverage, underlying |
|---|---|
| TQQQ +3, QLD +2, SQQQ -3, QID -2 | Nasdaq-100 (P) |
| UPRO +3, SSO +2, SPXU -3, SDS -2 | S&P 500 (P) |
| UDOW +3, DDM +2, SDOW -3, DXD -2 | Dow Jones Industrial Average (P) |
| TNA +3 (D), UWM +2 (P), TZA -3 (D), TWM -2 (P) | Russell 2000 small caps |
| FNGU +3, FNGD -3 | NYSE FANG+ index, BMO MicroSectors ETNs (notes, not ETFs) |
| TECL +3, TECS -3 | US technology sector (D) |
| SOXL +3, SOXS -3 | semiconductors (D) |
| FAS +3, FAZ -3 | US financials (D) |
| CURE +3 (D), RXD -2 (P) | US health care |
| LABU +3, LABD -3 | S&P biotech (D) |
| ERX +2, ERY -2 | US energy sector (D) |
| GUSH +2, DRIP -2 | S&P oil and gas exploration and production (D) |
| TMF +3 (D), UBT +2 (P), TBT -2 (P), TTT -3 (P) | 20+ year US Treasuries |
| UGL +2, GLL -2 | gold bullion (P) |
| AGQ +2, ZSL -2 | silver (P) |
| UCO +2, SCO -2 | Bloomberg WTI crude oil futures (P) |
| NUGT +2, DUST -2 | gold miners (D) |
| YINN +3, YANG -3 | FTSE China 50 (D) |
| EDC +3, EDZ -3 | MSCI emerging markets (D) |
| EURL +3 | FTSE Developed Europe (D) |
| EUO -2 | euro vs the US dollar (P; rises when the dollar strengthens) |
| YCS -2 | yen vs the US dollar (P; rises when the dollar strengthens) |
| DRN +3 (D), SRS -2 (P) | US real estate |
| SHV 1x | 0-1 year US Treasury bills (iShares), the cash sleeve |

- [ ] **Step 1: Update the two existing tests and write the failing new ones** in `tests/strategies/ray_dalio/test_prompts.py`

`_context_strings()` gains a `dict` branch appending `str(key)` and `str(value)` for each item. `test_universe_is_lumibots_54_symbols_with_shv` uses `list(UNIVERSE)[-1] == "SHV"`. New tests:

```python
def test_every_symbol_has_a_description_and_the_agents_get_them():
    assert all(isinstance(text, str) and text.strip() for text in UNIVERSE.values())
    assert CONTEXT["universe"] == UNIVERSE


def test_the_funds_the_agents_misread_are_described_right():
    expected = {"TMF": "+3x", "TWM": "-2x", "UGL": "+2x", "GLL": "-2x", "SSO": "+2x", "UPRO": "+3x", "TQQQ": "+3x", "SHV": "1x"}
    for symbol, leverage in expected.items():
        assert UNIVERSE[symbol].startswith(leverage), symbol


def test_every_inverse_fund_says_so():
    for symbol, text in UNIVERSE.items():
        assert text.startswith("-") == ("(inverse)" in text), symbol


def test_the_leverage_rule_points_at_the_descriptions():
    assert "never from memory" in LEVERAGE_RULE  # import LEVERAGE_RULE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/ray_dalio/test_prompts.py -v`
Expected: FAIL (`UNIVERSE` is a tuple: `.values()` AttributeError; `LEVERAGE_RULE` lacks the sentence).

- [ ] **Step 3: Implement in `prompts.py`**

- `UNIVERSE: dict[str, str]` with the table above, same order, format from Global Constraints, e.g. `"TMF": "+3x daily, 20+ year US Treasuries (Direxion)"`, `"TWM": "-2x daily (inverse), Russell 2000 small caps (ProShares)"`.
- Append to `LEVERAGE_RULE`: `"Read each fund's leverage, direction and underlying from the universe in the context, never from memory; a fund and its inverse held together cancel out and only pay fees."`
- `CONTEXT["universe"] = dict(UNIVERSE)`.
- Update `strategies/ray_dalio/CLAUDE.md`: `prompts.py` -- `UNIVERSE` (54 symbols mapped to signed leverage, underlying and issuer, checked against Yahoo 2026-11-21; ERX/ERY, GUSH/DRIP, NUGT/DUST were 3x before 2020, so the descriptions are wrong for earlier windows).

- [ ] **Step 4: Run the ray_dalio tests and the quality gate**

Run: `uv run pytest tests/strategies/ray_dalio -v` -> all pass.
Run: `uv run ruff check && uv run ruff format --check && uv run pytest --cov -q && uv run interrogate -v` -> exit 0, coverage >= 80%, interrogate >= 80%.

- [ ] **Step 5: Commit** (after `/ponytail:ponytail-review` on the staged changes)

```bash
git add src/trading_agent_framework/strategies/ray_dalio/prompts.py src/trading_agent_framework/strategies/ray_dalio/CLAUDE.md tests/strategies/ray_dalio/test_prompts.py
git commit -m "feat(ray_dalio): describe every ETF's leverage and underlying to the agents"
```
