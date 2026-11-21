"""Prompts and the leveraged-ETF universe for the ray_dalio idea-meritocracy strategy.

Adapted from LumiBot (MIT License, Lumiwealth), example_strategies/ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py
(commit 38e23c72248dd1555387b1487ca7b676ef6e95ca), with this framework's tool names. This strategy is not affiliated
with Ray Dalio or Bridgewater; the name only describes the idea-meritocracy process.
"""

# Each fund's daily leverage (negative = inverse), what it tracks and its issuer, checked against Yahoo on 2026-11-21.
# The agents read these instead of guessing a ticker from memory. ERX/ERY, GUSH/DRIP and NUGT/DUST were 3x before 2020.
UNIVERSE = {
    "TQQQ": "+3x daily, Nasdaq-100 (ProShares)",
    "QLD": "+2x daily, Nasdaq-100 (ProShares)",
    "SQQQ": "-3x daily (inverse), Nasdaq-100 (ProShares)",
    "QID": "-2x daily (inverse), Nasdaq-100 (ProShares)",
    "UPRO": "+3x daily, S&P 500 (ProShares)",
    "SSO": "+2x daily, S&P 500 (ProShares)",
    "SPXU": "-3x daily (inverse), S&P 500 (ProShares)",
    "SDS": "-2x daily (inverse), S&P 500 (ProShares)",
    "UDOW": "+3x daily, Dow Jones Industrial Average (ProShares)",
    "DDM": "+2x daily, Dow Jones Industrial Average (ProShares)",
    "SDOW": "-3x daily (inverse), Dow Jones Industrial Average (ProShares)",
    "DXD": "-2x daily (inverse), Dow Jones Industrial Average (ProShares)",
    "TNA": "+3x daily, Russell 2000 small caps (Direxion)",
    "UWM": "+2x daily, Russell 2000 small caps (ProShares)",
    "TZA": "-3x daily (inverse), Russell 2000 small caps (Direxion)",
    "TWM": "-2x daily (inverse), Russell 2000 small caps (ProShares)",
    "FNGU": "+3x daily, NYSE FANG+ index (BMO MicroSectors ETN, a note, not an ETF)",
    "FNGD": "-3x daily (inverse), NYSE FANG+ index (BMO MicroSectors ETN, a note, not an ETF)",
    "TECL": "+3x daily, US technology sector (Direxion)",
    "TECS": "-3x daily (inverse), US technology sector (Direxion)",
    "SOXL": "+3x daily, semiconductors (Direxion)",
    "SOXS": "-3x daily (inverse), semiconductors (Direxion)",
    "FAS": "+3x daily, US financials (Direxion)",
    "FAZ": "-3x daily (inverse), US financials (Direxion)",
    "CURE": "+3x daily, US health care (Direxion)",
    "RXD": "-2x daily (inverse), US health care (ProShares)",
    "LABU": "+3x daily, S&P biotech (Direxion)",
    "LABD": "-3x daily (inverse), S&P biotech (Direxion)",
    "ERX": "+2x daily, US energy sector (Direxion)",
    "ERY": "-2x daily (inverse), US energy sector (Direxion)",
    "GUSH": "+2x daily, S&P oil and gas exploration and production (Direxion)",
    "DRIP": "-2x daily (inverse), S&P oil and gas exploration and production (Direxion)",
    "TMF": "+3x daily, 20+ year US Treasuries (Direxion)",
    "UBT": "+2x daily, 20+ year US Treasuries (ProShares)",
    "TBT": "-2x daily (inverse), 20+ year US Treasuries (ProShares)",
    "TTT": "-3x daily (inverse), 20+ year US Treasuries (ProShares)",
    "UGL": "+2x daily, gold bullion (ProShares)",
    "GLL": "-2x daily (inverse), gold bullion (ProShares)",
    "AGQ": "+2x daily, silver (ProShares)",
    "ZSL": "-2x daily (inverse), silver (ProShares)",
    "UCO": "+2x daily, Bloomberg WTI crude oil futures (ProShares)",
    "SCO": "-2x daily (inverse), Bloomberg WTI crude oil futures (ProShares)",
    "NUGT": "+2x daily, gold miners (Direxion)",
    "DUST": "-2x daily (inverse), gold miners (Direxion)",
    "YINN": "+3x daily, FTSE China 50 (Direxion)",
    "YANG": "-3x daily (inverse), FTSE China 50 (Direxion)",
    "EDC": "+3x daily, MSCI emerging markets (Direxion)",
    "EDZ": "-3x daily (inverse), MSCI emerging markets (Direxion)",
    "EURL": "+3x daily, FTSE Developed Europe (Direxion)",
    "EUO": "-2x daily (inverse), the euro against the US dollar, so it rises when the dollar strengthens (ProShares)",
    "YCS": "-2x daily (inverse), the yen against the US dollar, so it rises when the dollar strengthens (ProShares)",
    "DRN": "+3x daily, US real estate (Direxion)",
    "SRS": "-2x daily (inverse), US real estate (ProShares)",
    "SHV": "1x, 0-1 year US Treasury bills (iShares), the cash sleeve",
}
MIN_POSITIONS = 3
ANALYSTS = {"growth_agent": "growth", "inflation_agent": "inflation", "debt_liquidity_agent": "liquidity"}
DISAGREEMENT = "thoughtful_disagreement"
TRADER = "trader"

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
    "leveraged setup. Read each fund's leverage, direction and underlying from the universe in the context, never from "
    "memory; a fund and its inverse held together cancel out and only pay fees."
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
    "universe": UNIVERSE,
    "min_positions": MIN_POSITIONS,
    "data_expectation": "Use get_fred_snapshot/get_fred_latest and search_news commonly; the run log records the actual tool calls.",
    "leverage_expectation": "Use only leveraged ETFs plus SHV/cash escape, hold at least three positions, and bias toward diversified 3x exposure.",
}
