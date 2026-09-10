# `env/`

This directory holds the environment files that `trading_agent_framework.config.env`
loads credentials and configuration from. **Only two files in this directory are
committed to git: `.env.example` and this `README.md`.** Everything else matching
`env/.env.*` is gitignored (see `/env/*` with `!/env/README.md` and
`!/env/.env.example` in the repository `.gitignore`) because those files carry real
secrets or point at real accounts.

## Naming convention

Strategy- and mode-specific env files follow:

```
env/.env.{strategy_name}.{live|paper|backtesting}
```

For example, a strategy named `momentum` running in paper mode reads from
`env/.env.momentum.paper`. `{strategy_name}` should match the `strategy_name` you
pass to `load_strategy_env`, and the mode suffix must be one of the three values in
`trading_agent_framework.config.env.TRADING_MODES`: `live`, `paper`, `backtesting`.

## Resolution order

`trading_agent_framework.config.env.resolve_env_file` (called by
`load_strategy_env`) looks for the first file that exists, in this order:

1. `env/.env.{strategy_name}.{trading_mode}` -- the strategy- and mode-specific file.
2. `env/.env` -- a shared fallback for all strategies/modes.
3. `.env` at the repository root -- a last-resort fallback.

The first candidate found is loaded (via `python-dotenv`, with `override=True`) and
its path is returned. If none of the three exist, `load_strategy_env` raises
`ConfigurationError`.

## Getting started

Copy `.env.example` to a strategy- and mode-specific file (or to `env/.env` for a
shared default) and fill in real values:

```bash
cp env/.env.example env/.env.momentum.paper
```

Never commit the copy -- it is already covered by the `env/.env.*` gitignore
pattern, so a plain `git add` will not pick it up.
