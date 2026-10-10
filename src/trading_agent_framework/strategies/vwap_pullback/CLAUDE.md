# strategies/vwap_pullback/ -- intraday, code only

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- `vwap_pullback/` (`VwapPullbackStrategy`, registered as `"vwap_pullback_continuation"`): strictly intraday, 5-minute ticks, code only (no LLM, no news). Pure `features`/`screening`/`setups`/`risk`/`trades`; `Scanner` (stage 1 before the open from the cross_momentum universe file, stage 2 and the setup state machine each tick, the stage-2 scores kept in `SessionState.scores`), `Desk` (the only order code: `enter_triggered` tries every triggered setup, best stage-2 score first when triggers outnumber free slots; code-sized entries, a protective stop on every fill, 15:50 flatten). A tick is reconcile → scan → enter; a trade ends on its stop or at the flatten. Closed trades go to `trades.jsonl` in the run directory.

## Gotchas

- **vwap_pullback backtests charge 5 bps of slippage per side** (the `slippage` class parameter of `VwapPullbackStrategy`, passed to `run_backtesting`; overridable). The zero-slippage 5Y run (IEX) had +0.007R expectancy per trade, i.e. no edge before costs.
- **vwap_pullback never leaves a position without a stop**: `Desk` cancels a stop and waits for the cancel before the flatten sell (both brokers refuse a sell above held minus pending sells), replaces a stop it cannot place with an immediate market sell, and `reconcile` re-places a stop that errored or was cancelled from outside. It flattens and restart-closes only positions this strategy owns (a shared account's other positions are left alone). Its backtests wait in 60 s slices (`initialize` sets `max_wait_slice = 60.0` on that run's `BacktestClock` instance, not the class): a 5-minute tick is otherwise one clock jump, so an entry filled on the tick's first bar got its stop (placed by the fill hook the executor dispatches between slices) only at the tick's end, and the bars in between were never checked against it. Its backtests read minute bars through `YearChunkedData` (the 5Y window fetched in one go was OOM-killed at 58 GB): the scanner's `_preload` still passes the whole window and the wrapper narrows a minute `load()` to the current year; a minute cutoff before the current chunk's fetch window raises `BacktestDataError`.
