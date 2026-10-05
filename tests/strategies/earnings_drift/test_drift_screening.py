from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest
from tests.fakes import et
from tests.strategies.earnings_drift.drift_helpers import make_candidate, make_features

from trading_agent_framework.strategies.earnings_drift.fact_sheet import fact_sheet
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.screening import REJECT_REASONS, gate
from trading_agent_framework.strategies.earnings_drift.surprise import Surprise

PARAMS = DriftParams()
BEAT = Surprise(D("1.52"), D("1.20"))


def test_a_good_reaction_passes() -> None:
    assert gate(BEAT, make_features(), PARAMS, held=False) is None


@pytest.mark.parametrize(
    ("surprise", "features", "held", "reason"),
    [
        (BEAT, make_features(), True, "already_held"),
        (None, make_features(), False, "no_surprise_data"),
        (Surprise(D("1.00"), D("1.00")), make_features(), False, "eps_miss"),
        (BEAT, None, False, "no_bars"),
        (BEAT, make_features(abnormal_pct=0.0299), False, "weak_reaction"),
        (BEAT, make_features(hold_ratio=0.49), False, "faded"),
        (BEAT, make_features(hold_ratio=None), False, "faded"),
        (BEAT, make_features(close_location=0.49), False, "faded"),
        (BEAT, make_features(rel_volume=1.99), False, "low_volume"),
        (BEAT, make_features(close=9.99), False, "illiquid"),
        (BEAT, make_features(dollar_volume_20d=19_999_999.0), False, "illiquid"),
    ],
)
def test_each_gate(surprise, features, held: bool, reason: str) -> None:  # noqa: ANN001
    assert gate(surprise, features, PARAMS, held=held) == reason


def test_boundaries_pass_and_the_first_failure_wins() -> None:
    edge = make_features(abnormal_pct=0.03, hold_ratio=0.5, close_location=0.5, rel_volume=2.0, close=10.0, dollar_volume_20d=20_000_000.0)
    assert gate(BEAT, edge, PARAMS, held=False) is None
    assert gate(None, make_features(abnormal_pct=0.0), PARAMS, held=True) == "already_held"
    assert gate(BEAT, make_features(abnormal_pct=0.0, rel_volume=0.5), PARAMS, held=False) == "weak_reaction"
    assert REJECT_REASONS == ("already_held", "no_surprise_data", "eps_miss", "no_bars", "weak_reaction", "faded", "low_volume", "illiquid")


def test_fact_sheet_is_lean_and_labelled_by_code() -> None:
    candidate = make_candidate("AAA", day=date(2026, 9, 2), accepted_at=et(2026, 9, 1, 16, 5), runup_60d_pct=None)
    sheet = fact_sheet(candidate, max_quantity=13)
    assert sheet == {
        "symbol": "AAA",
        "reported_at": "2026-09-01 16:05",
        "timing": "after_close",
        "eps": {"actual": 1.52, "estimate": 1.2, "surprise_pct": 26.7, "result": "BEAT"},
        "sales": {"actual": 1_100_000_000.0, "estimate": 1_000_000_000.0, "surprise_pct": 10.0, "result": "BEAT"},
        "reaction": {"gap_pct": 6.0, "return_pct": 8.0, "abnormal_pct": 7.5, "hold_ratio": 0.8, "close_location": 0.9, "rel_volume": 3.0},
        "context": {"runup_20d_pct": 2.0, "runup_60d_pct": None, "atr14_pct": 2.0, "close": 100.0, "reaction_low": 95.0},
        "max_quantity": 13,
        "filing": {"accession_number": "0000000000-26-AAA"},
        "headlines": [{"time": "2026-09-02 07:01", "headline": "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate"}],
    }


def test_fact_sheet_without_sales() -> None:
    sheet = fact_sheet(make_candidate("BBB", surprise=Surprise(D("0.50"), D("0.40"))), max_quantity=0)
    assert sheet["sales"] is None and sheet["eps"]["result"] == "BEAT"
