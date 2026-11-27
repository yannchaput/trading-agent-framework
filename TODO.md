# TODOS

## General

* Add a skill to review, challenge and validate the design of a strategy
* Set the plugins claude-api and build-evals to enable evals
* Implement a skill scanning the auto memory and session traces to get a finding
* Set up a mlflow server for agent evaluation
* ~~Add a parameter start/stop date and time window to the CLI for the backtest~~
* ~~Fix pyright errors (consume tokens)~~
* ~~Update claude.md so that simplify skill is always run before commit as pony-tail. Check if a hook would be possible. Add documentation by default~~
* ~~create a PR template~~
* ~~ Migrate projects to the 2nd hard drive (with the models)~~
* ~~Enable IBKR broker~~
* ~~Build a log "purge" batch for logs more than 1 month old (except backstesting). Same for 'cache' (inclusing backtesting)~~
* ~~Cache LLM calls during backtesting so reruns against the same period are deterministic and free (design spec section 1.3 -- currently every backtest rerun calls the live LLM again)~~ => Too complex
* ~~Add agent call telemetry (token counts, latency) to backtesting settings.json and `llm_stats.sqlite`, shown in the dashboard's Parameters tab (no cache hits: LLM call caching was dropped as too complex)~~
* ~~Create a batch to run all the smoke tests altogether~~
* ~~Add cash available during backtesting~~
* ~~Update pyproject.toml to make dashboard an "extra" dependency~~
* ~~Add a gloabl factory/builder for strategy in main.py~~
* ~~Handle difference of reasoning token budget payload attribute between vllm and llama.cpp~~

## BUGS & Refactoring

* ~~Review `uv check` errors and fix or silent them~~
* ~~Replace every instance variable `self.` with `self.vars` to avoid namespace collisions~~
* Update tools comments with parameters + check if tool annotation is recommended
* Leverage Langchain native middleware like "Tool error " or "Tool call" limit

## Skills

* Install [Alpaca skills](https://alpaca.markets/blog/alpaca-launches-skills-library-for-ai-agents/)
* Wait-what (mattpockock)
* ~~poneytail~~

## PEA Manager

* ~~Tester Fable 5.2~~
* ~~Utiliser Fable pour revoir le skill~~

## Tests

* Test Typesafe.AI.
* [Deep Agents](https://reference.langchain.com/python/deepagents)
* [Deep Agents tutorial](https://docs.langchain.com/oss/python/deepagents/data-analysis)

## Evolutions

### Dashboard

* Plot the SPY line plot on a side by side view.
* ~~Add a column for the broker~~
* ~~REplace VIX, ADX etc with a "regime" indicator: 1 =bullish, neutral =0, bearish = -1~~
* ~~Add a column with slippage and fees in dashboard~~
* ~~Track P&L as an indicator and display it in the trade chart. Compute should happen at the dashboard level to track entry point (buy)~~
* ~~Review  Agent table pivot format to be consistent with all strategies~~
* ~~Dashboard: Live/Paper tabs list Alpaca strategies only; add Tastytrade accounts~~
* ~~Draw SPY daily P&L in the live/paper P&L plot~~

#### News fetch

* ~~Update the news_search tool: all news earlier than April 2026 are fetched from GEDLT cache. After it's Alpaca.~~
* ~~Add a warning in the logs where no news are returned~~


### Cross sectional momentum

* ~~Add a market regime filter to cross sectional momentum strategy.if bearish -> defensive else cross momentum.~~
* ~~When a nex position is taken, put a stop order at 1 x ATR (confirm threshold)~~ => Poor results
* ~~Restrain the number of positions to a hard cap of 20: do not buy if more than 20~~ => Poor results
* ~~Compute the residual volatility compared to the market (cf chatgpt thread)~~

### Bull Bear

From the 2026-11-24 assessment (docs/), ranked by priority:

1. ~~Leakage check, no GPU: split the 5Y and COVID judge-vs-top-10 gap and the bull conviction spread before/after Qwen 3.8's knowledge edge (~2025-04). Edge only before it = memory, not skill.~~ => No sign of memory: no edge in 2020-2023 (COVID, 2022), edge back in 2026 after the cutoff; the 2Y pre/post gap is one bad stretch (2025-05..12). Item 2 still needed for 2024-09..2025-04.
2. Anonymized replay: `anonymize` flag on the fact sheet (ticker -> "Stock 07", no `company`, keep sector/industry so the model does not guess), judge + bull + bear on ~30 pre-2025 reviews, with a rerun arm. Keep/drop rule fixed first (e.g. keeps >= 70% of the judge-minus-top-10 gap).
3. Paper run after the model's cutoff (>= 3 months): score the judge against the rank-only top-10 shadow book rebuilt from `reviews.jsonl` (no code: ranks are logged). The only truly out-of-sample proof.
4. Survivorship-free universe: rebuild the point-in-time history with delisted names from EODHD and measure the residual bias on the no-LLM book.
5. Long stress test of the rank-only engine (no LLM, 2005-2026: 2008, 2015-16, 2018 Q4, 2022): worst drawdown of the base book, to know the real risk before live money.
6. Score variants on the no-LLM book (shared with cross_momentum): vol-scaled or residual momentum, 52-week-high proximity, skip the last month in R(3). Rule fixed first, one untouched holdout window.
7. Average 2-3 bull/bear ratings per stock (majority vote), the one idea left open by the judge-signal study; GPU replay with a rule fixed first.
8. Point-in-time sector: SEC SIC code instead of yfinance's sector of today.
9. Move the run history out of `bull_bear/CLAUDE.md` (31 KB loaded every session) into `docs/`, keep the rules.
10. Move the order-logging hooks copied in bull_bear and sector_pods to `strategies/common`.

* ~~Test with news off~~
* ~~Refactor bull bear into Langgraph like "pod_sectors"~~
* ~~Add the "character count" guardrails before submitting a note.~~
* ~~Merge the trim utilities of common in handoff with the one of bull-bear in "pipeline".~~
* ~~Verifier si le tools search_memory a ete utilisé et pourquoi, relevance of the answer vs query~~
* ~~Retest a 1Y backtest with news 'off'~~
* ~~Bump up research_tool_budget to 5 with news on~~
* ~~Enable compaction of the state in langchain with middleware~~
* ~~Enrich logging, specifically the research agent~~
* ~~Show P&L in the log when a stock is sold~~

### Citadel Pod Sector

* ~~Simplify Langgraph code (cf Ray Dalio)~~
* ~~Log the AgentRunResult~~
* ~~Run the fist baseline (on 1 month)~~

### Ray Dalio

* Set guardrails if they are missing
* Add traling stop orders to have a daily cap loss. Make sure the stop order is gain is counted in buying power.
* Optimize and fix bugs of the strategy

### LLM benchmark

* Gemma 4 26B A4B MoE local second model on llama.cpp
* Deepseek v4.1 Flash with [Unsloth](https://unsloth.ai/docs/models/deepseek-v4.1-flash#usage-guide)
* Install Strata (better of MoE) and update the framework to run Strata along with vllm - test deepseek
* Test deepseek 4.1 flash : [X](https://x.com/NFT_Chen/status/2113854991958176118)
* Test Sage 1.5
* Test Nanbeige 4.3
* Test Qwen 3.8 35B: [X](https://x.com/Oluwaphilemon1/status/2114912917157974508)
* ~~Run Qwen 3.8 on both servers and compare~~
* ~~Add a cli parameter to select scenarios to run~~
* ~~Test new scenarios from bull bear and insider~~
* ~~Add mathematical and tradinc calculation problems to the scenarios~~
* ~~Implement Qwen 3.8 27B with DFlash2: [Hugging Face](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)~~
~~Test Qwen 3.8-27B~~*
* ~~Log a frontmatter before each model benchmark~~
* ~~ Update with scenarios from bill ackman and vwap pullback continuation ~~

### NAS_KELT

* To study

### Time Machine

* https://github.com/Jakeschincariol/day-trader-skill
* Prends juste le mécanisme Time Machine parce que le reste est une strat différente

``` code
Ce qui est demandé : chaque mardi, juste avant d'envoyer les ordres, cross_momentum et bull_bear simulent le portefeuille qu'elles s'apprêtent à acheter. Elles calculent la même chose pour le portefeuille qu'elles détiennent. Deux méthodes, dans l'esprit de Time Machine :

	1.	Monte Carlo : 5 000 rejeux des 20 prochaines séances, tirés des rendements réels de l'année écoulée des titres visés, avec une graine fixe. On obtient la fourchette de résultats (le pire cas sur 20 et le cas médian), la probabilité de perdre 5 % ou 10 %, et la baisse maximale probable.
	2.	Rejeu historique : les mêmes pondérations appliquées à chaque période réelle de 20 séances de l'année écoulée, avec le pire résultat, le médian et le meilleur.

Garde-fous du brief :

	•	La simulation sert uniquement d'information : elle ne change aucune pondération et ne bloque aucun ordre.
	•	En cas d'erreur, elle produit un simple avertissement et le rééquilibrage continue.
	•	Elle est désactivée en backtest, pour que les résultats de P1 et P2 ne changent pas.
	•	Les résultats vont dans les logs et dans un fichier :
	•	pour cross_momentum, dans un nouveau previews.jsonl ;
	•	pour bull_bear, dans reviews.jsonl, à côté des verdicts du juge.

La suite :

	1.	Au retour du worker, je relis le code et je lance ruff, pyright et les tests.
	2.	Je vous propose ensuite de fusionner. Il faudra alors redémarrer les deux stratégies pour charger le nouveau code, avant mardi.
```

## Archives

### VWAP Pullback continuation

* ~~Add indicators on equity curve~~
* ~~Remove entry agent: it is effectless~~
* ~~Change agentic architecture for : Make the agent "agentic" in the right place (cf chatgpt)~~ => agents removed, the strategy is code only

### Bill Ackman

* ~~Update claude.md with exempt_tools and budget_tools~~
* ~~Update dashboard to display tempearature~~
* ~~Log output of each agent to understand what was their decision~~

### MIGRATION

* ~~Alpaca broker~~
* ~~Orchestration framework (scheduling, lifecycle,...)~~
* ~~Data and accountig tools~~
* ~~Agentic framework - langchain foundation~~
* ~~Tools~~
* ~~Memory~~
* ~~Account configuration (no short sell, no margin trading)~~
* ~~Backtesting (vectorbt)~~
* ~~Cache management~~ => not needed
* ~~dashboard~~
* ~~Fix `Bar.empty`, `bars.pandas_df`.~~
* ~~Add warmup_trading_days to AlpacaBacktestDataSource~~

### News sentiment analysis

* ~~Find the root cause of having the same decision to remember called several times in the same iteration: help with log and the memory database.~~
* ~~Investigate "current_regime": None and defensive posture~~
* ~~Add a log in executor to log the execution hour : usefull for intraday.~~
* ~~Log the system prompt in 'initialize' method~~
* ~~Fix prompt to avoid chineese character with GLM models~~

### Market regime

Goal: detect a regime shift at the very start (one day or less) to change the daily strategy.
Details: `docs/Market regim early detection work list.md`. Check the box once a paper is tested.

* [X] Shu, Yu & Mulvey (2024), Statistical Jump Model: [arXiv 2402.05272](https://arxiv.org/abs/2402.05272)
* [x] Yi et al. (2026), LLM proposals validated by VAR/PCMCI (lag +0.9 day, Treasuries): [arXiv 2605.30363](https://arxiv.org/abs/2605.30363)
* [x] Boukardagha (2026), Wasserstein HMM, strictly causal: [arXiv 2603.04441](https://arxiv.org/abs/2603.04441)
* [x] Theory: quickest detection in latent factor models (2026): [arXiv 2610.20366](https://arxiv.org/pdf/2610.20366)
* [x] Theory: Li et al. (2025), Sparse Jump Model + MPC: [MDPI](https://www.mdpi.com/2227-7390/13/17/2837)
* [x] Confirm the 3 high-numbered arXiv IDs (2605.30363, 2610.20366, 2610.27751) resolve
* [x] HHMM (arXiv 2007.14874): NO, misses 11/14 falls, needs look-ahead
* [x] MS-TVTP weekly exit: NO, exits ~6 weeks after the trough
* [x] Leveraged ETF switch (UPRO/TQQQ): vol-scaled and 200-day PASS, graded regime FAIL
* [x] Current SPY/VIX regime: too late to gate leverage

#### Yi et al. (2026), LLM proposals validated by VAR/PCMCI (lag +0.9 day, Treasuries)

The idea is to use a LLM or a ML model on the market crisis news to detect if sentiment dropped during those periods
* Use daily on a 2Y window
* Use intraday on the current period
* Use a tranformer sentiment analysis model from HF rather than Qwen.
* [x] Spike "news + VIX" (news drift and CUSUM, both orderings): NO. News is calm before falls and coincident with VIX; its shift at a fall start is +0.1 to +0.5 sd, too small to detect within a week. Median switch 5 sessions / -4% after the high (VIX alone does the same). Holdout 2025-2026 kept untouched. Spec: `docs/superpowers/specs/2026-11-22-news-vix-early-regime-spike-design.md`
* [ ] (Low priority after the NO above) Follow-up, pre-2020 titles (2015-2019 have no titles in GDELT): rebuild pseudo-titles from URL slugs (78% of 2017 URLs have a 5+ word slug). Validate first on 2020-2024: FinBERT score of slug vs title, and the `dev` alerts from slugs only must match the title run. Needs a slug variant of the ticker filter ("NASDAQ OLLI" without parentheses). If valid, 2015-2019 (5 falls) becomes a second untouched window.