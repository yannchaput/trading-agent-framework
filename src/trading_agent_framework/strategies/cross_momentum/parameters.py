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
    # ── Momentum deterioration (V2.2-A: Signal A — ATR drawdown) ─
    "deterioration_atr_period": 20,
    "deterioration_atr_high_lookback": 20,
    "deterioration_atr_multiple": 3.0,
    # ── Momentum deterioration (V2.2-B: Signal B — Rank velocity) ─
    "deterioration_rank_lookback_days": 5,
    "deterioration_rank_delta_threshold": 30,
    # ── Momentum deterioration (V2.2-C: Signal C — Relative strength break) ─
    "deterioration_relative_strength_lookback_days": 5,
    "deterioration_relative_return_threshold": -0.08,
    "deterioration_benchmark": "SPY",
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
    # ── Market regime filter (V2) ─────────────────
    "regime_sma_window": 200,  # SMA lookback for regime filter
    "regime_buffer_pct": 0.02,  # ±2% buffer around SMA to avoid whipsaw
    "regime_bull_multiplier": 1.00,  # 100% exposure in bull regime
    "regime_neutral_multiplier": 0.70,  # 70% exposure in neutral regime
    "regime_bear_multiplier": 0.30,  # 30% exposure in bear regime
    "regime_data_days": 300,  # days of SPY data to fetch
    # ── Momentum Breadth Exposure Overlay (V2.4) ────────
    "BREADTH_BANDS": [
        (0.65, 1.00),  # >= 65% → full exposure
        (0.50, 0.85),  # 50–65% → 85% exposure
        (0.35, 0.60),  # 35–50% → 60% exposure
        (0.00, 0.35),  # < 35% → 35% exposure
    ],
    "breadth_overlay": {
        "enabled": True,
        "sma_window": 100,
    },
    # ── Momentum breadth overlay (V3.3) ──────────────────────────────────────────
    # Each band is (min_breadth, exposure_multiplier).
    # Breadth = fraction of eligible universe with positive momentum score.
    # Bands are evaluated from top to bottom; first match wins.
    "MOMENTUM_BREADTH_BANDS": [
        (0.40, 1.00),  # >= 40% → full exposure
        (0.25, 0.75),  # 25–40% → 75% exposure
        (0.15, 0.50),  # 15–25% → 50% exposure
        (0.00, 0.25),  # < 15% → 25% exposure
    ],
}
