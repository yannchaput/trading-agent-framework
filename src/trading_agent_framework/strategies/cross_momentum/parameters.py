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
    "day_of_week": 2,  # 0=Monday, 4=Friday
    # Market time (ET, HH:MM, from 09:30 to before 15:59) of the daily iteration, so of the weekly rebalance: past the opening's wide spreads,
    # well before the close. The decision reads completed sessions only, so the hour changes execution, not signals.
    "rebalance_time": "12:00",
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
    # Share of the estimated buying cash never spent on a rebalance. Orders are sized at the last close
    # but fill at a later open (plus fees), so the estimate is never exact; this reserve lets the buys land.
    "cash_buffer_pct": 0.05,
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
    # ── Parking sleeve ─────────────────────────
    # Capital the exposure legs take out of stocks is parked in this T-bill ETF instead of idle cash.
    # Trims and parking orders smaller than min_trade_pct of the portfolio are skipped (per-order fees).
    "parking": {
        "symbol": "SHV",
        "min_trade_pct": 0.01,
    },
    # ── Breadth overlay (market regime) ─────────
    # Share of the scored stocks closing above their own sma_window-day SMA. thresholds are the breadth levels
    # below which the strategy enters step 1 / step 2; exposures[step] scales the stock weights. Cutting is
    # immediate; re-risking to a less defensive step needs that step's threshold plus `hysteresis`.
    # Fewer than min_stocks valid stocks leaves the leg neutral.
    "breadth_overlay": {
        "enabled": True,
        "sma_window": 100,
        "min_stocks": 50,
        "thresholds": (0.50, 0.30),
        "exposures": (1.0, 0.7, 0.4),
        "hysteresis": 0.05,
    },
}
