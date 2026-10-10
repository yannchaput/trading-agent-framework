from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.common.scoring import score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _stock(closes: list[float], *, held: bool = False) -> DebateStock:
    row = score_stock("AAA", closes, [1e6] * len(closes), CONFIG)
    assert row is not None
    return DebateStock(rank=3, row=row, held=held)


def test_a_row_reports_the_momentum_facts_in_rounded_percentages() -> None:
    sheet = fact_sheet_row(_stock([100.0 + i for i in range(300)], held=True), sector="Technology", weight=0.1234)

    assert sheet["symbol"] == "AAA" and sheet["momentum_rank"] == 3
    assert sheet["return_12m_skip_1m_pct"] == 200.0
    assert sheet["return_6m_skip_1m_pct"] == 50.0
    assert sheet["return_3m_pct"] == 18.8  # 0.1875
    assert sheet["return_1m_pct"] == 5.6  # (399 - 378) / 378
    assert sheet["drawdown_from_52w_high_pct"] == 0.0  # the last close is the high
    assert sheet["vs_sma200_pct"] == 33.2  # 399 / mean(200..399) - 1
    assert sheet["momentum_score"] == pytest.approx(1.188, abs=1e-3)
    assert (sheet["sector"], sheet["held"], sheet["weight_pct"]) == ("Technology", True, 12.3)
    assert isinstance(sheet["volatility_pct"], float)


def test_the_drawdown_is_measured_from_the_52_week_high() -> None:
    # The last close falls 10% from 398 (a 50% fall would push the 20-day volatility over max_volatility and filter the stock out)
    closes = [100.0 + i for i in range(299)] + [398.0 * 0.9]

    sheet = fact_sheet_row(_stock(closes), sector="UNKNOWN", weight=0.0)

    assert sheet["drawdown_from_52w_high_pct"] == pytest.approx(-10.0, abs=0.05)
    assert sheet["weight_pct"] == 0.0
