"""The trader's weights to a target portfolio (pure): each stock's weight, and the remainder parked in SHV."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    weights: dict[str, float]  # stock -> fraction of portfolio value
    parking_weight: float  # the parking instrument's share: 1 - cash_buffer - sum(weights)


def target_portfolio(weights: Mapping[str, float], *, cash_buffer: float) -> TargetPortfolio:
    """Nothing is rescaled: a weight the trader chose is the weight targeted. The cash buffer stays as cash.

    Raises `ValueError` when the weights exceed `1 - cash_buffer` (the submit tool already refuses that; this
    keeps the invariant if another caller builds a target).
    """
    total = sum(weights.values())
    investable = 1 - cash_buffer
    if total > investable + _EPS:
        raise ValueError(f"stock weights sum to {total:.4f}, above the investable {investable:.4f}")
    return TargetPortfolio(weights=dict(weights), parking_weight=max(0.0, investable - total))
