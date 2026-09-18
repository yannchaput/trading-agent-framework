### Cross Sectional Momentum Strategy Parameters
CONFIG = {
    # ── Liquidity / tradability filters ────────
    "min_price": 10.0,
    "min_dollar_volume": 20_000_000,
    "max_volatility": 0.80,
    "min_market_cap": 2_000_000_000,  # pre-filtered by batch, gate here for safety
    "min_trading_days": 250,
    # ── Rebalance schedule ─────────────────────
    "rebalance_frequency": "weekly",
    # As per research, Tuesday (1) is the best day to rebalance the portfolio
    "day_of_week": 1,  # 0=Monday, 4=Friday
    # ── Momentum scoring ──────────────────────
    "w_12m": 0.50,
    "w_6m": 0.30,
    "w_3m": 0.20,
    "skip_days": 21,
    # ── Portfolio construction ──────────────────
    "top_n": 20,
    "sell_rank_threshold": 35,
    "max_position_pct": 0.10,
    "min_position_pct": 0.02,
    # ── Volatility sizing ───────────────────────
    "volatility_window": 20,
    # ── Risk diagnostics (V2.3) ──────────────────
    "enable_diagnostics": True,  # set to True to enable risk diagnostics
    "risk_diagnostics": {
        "enabled": True,
        "benchmark": "SPY",
        "volatility_lookbacks": [20, 63],
        "correlation_lookbacks": [20, 63],
        "beta_lookback_days": 63,
        "min_overlap_observations": 10,
    },
    "volatility_targeting": {
        "enabled": True,
        "target_volatility": 0.20,
        "min_exposure": 0.40,
        "max_exposure": 1.00,
    },
    "breadth_overlay": {
        "enabled": True,
        "sma_window": 100,
    },
}
