# ray_dalio: describe every ETF in the universe

## Why

In the 2026-11-21 backtest (3 Sep 2024 to 29 Jan 2025, 104 sessions), ray_dalio returned +6.3% vs SPY +9.7%. The prompt
lists the 54 bare tickers, so the agents guess each fund from memory and get many wrong: TMF called 2x (41 of 103
disagreement outputs; it is 3x), UGL "3x gold" (2x), SSO 3x / UPRO 2x (swapped), TWM "3x S&P 500" (it is -2x Russell
2000), and on 2024-10-02 the book held UGL 15% + GLL 16% (2x gold and -2x gold: they cancel and pay fees). The user's
priority is raw return, then drawdown; the SHV safety sleeve stays.

## Goals / non-goals

- Goal: every agent sees, for every symbol, its signed daily leverage and what it tracks.
- Non-goals: changing the basket rules, the SHV sleeve, the trader's obedience to the disagreement node, or adding an
  order guard against holding a fund and its inverse (approach 3, deferred until a new run shows the pattern again).

## Approaches

1. **Describe each fund in the universe (picked).** Static data, one test, no new tool calls, ~1k extra tokens per call
   (peak use was 30k of the 65k cap).
2. A per-symbol lookup tool: more code and relies on the agent calling it, which is exactly what fails today. Dropped.
3. Descriptions plus an order guard refusing a fund and its inverse together: a second change; deferred.

## Design

- `prompts.py`: `UNIVERSE` becomes a dict `{symbol: description}` in today's order (each fund next to its inverse), e.g.
  `"TMF": "+3x daily, ICE 20+ year US Treasuries (Direxion)"`, `"TWM": "-2x daily (inverse), Russell 2000 small caps
  (ProShares)"`, `"SHV": "1x, 0-1 year US Treasury bills (iShares), the cash sleeve"`. Iterating a dict yields its keys, so
  `(*UNIVERSE, "SPY")` in the backtest preload is unchanged.
- `CONTEXT["universe"]` becomes `dict(UNIVERSE)`, and `LEVERAGE_RULE` gains one sentence: read each fund's leverage,
  direction and underlying from the universe in the context, never from memory; a fund and its inverse cancel out.
- Source: Yahoo `longName` checked for all 54 on 2026-11-21 (ProShares: Ultra = 2x, UltraShort = -2x, UltraPro = 3x,
  UltraPro Short = -3x; Direxion states it). ERX/ERY, GUSH/DRIP, NUGT/DUST are 2x today (they were 3x before 2020), so the
  descriptions hold for windows from 2020 on; FNGU/FNGD are BMO ETNs, not ETFs.
- Tests (`tests/strategies/ray_dalio/test_prompts.py`): every symbol has a non-empty description; spot checks pinning the
  misread funds (TMF +3x, TWM -2x, UGL +2x, GLL -2x, SSO +2x, UPRO +3x); the 54 symbols and their order are unchanged.
  Two existing tests need adjusting: `test_universe_is_lumibots_54_symbols_with_shv` indexes `UNIVERSE[-1]` (a dict
  cannot: use `list(UNIVERSE)[-1]`), and `_context_strings()` reads only list and str values, so it gains a dict branch
  (keys and values) or `test_no_lumibot_tool_name_is_left` silently stops checking the universe.
- Docs: `strategies/ray_dalio/CLAUDE.md` mentions the descriptions.

## Left to the builder

Exact description wording, within the format above.

## Judging it

The running backtest keeps the old prompts. Judge the change with a new run on the same window; run-to-run noise is
large, so one rerun is a signal, not proof.
