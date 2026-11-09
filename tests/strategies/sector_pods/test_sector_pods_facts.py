"""Tests for the sleeve table and market facts of sector_pods."""

from trading_agent_framework.strategies.sector_pods.facts import etf_facts, outcome, week_return_pct
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, pod_of


def _score(symbol, score):
    return {"symbol": symbol, "score": score, "reason": "x"}


def test_week_return_pct_needs_only_six_closes():
    assert week_return_pct([100.0, 100.0, 100.0, 100.0, 100.0]) is None
    assert week_return_pct([100.0] * 5 + [103.0]) == 3.0


def test_week_return_pct_matches_the_etf_facts_value_and_rounding():
    closes = [100.0 + index * 0.37 for index in range(260)]
    facts = etf_facts(closes, closes)
    assert facts is not None
    assert week_return_pct(closes) == facts["return_1w_pct"]


def test_week_return_pct_ignores_a_non_positive_starting_close():
    assert week_return_pct([0.0] * 5 + [10.0]) is None


def test_pods_cover_unique_symbols():
    assert len(ALL_SYMBOLS) == len(set(ALL_SYMBOLS)) == 27
    assert "SHV" not in ALL_SYMBOLS
    assert pod_of("XLU").key == "energy_pod"
    assert pod_of("IYR").key == "financials_pod"


def test_etf_facts_needs_201_bars():
    assert etf_facts([100.0] * 200, [100.0] * 200) is None


def test_etf_facts_values():
    facts = etf_facts([100.0] * 200 + [110.0], [100.0] * 201)
    assert facts is not None
    assert facts["return_1w_pct"] == 10.0
    assert facts["return_6m_pct"] == 10.0
    assert facts["relative_1m_pct"] == 10.0
    assert facts["vs_200d_avg_pct"] == round((110 / 100.05 - 1) * 100, 1)


def test_etf_facts_omits_relative_without_spy_history():
    facts = etf_facts([100.0] * 200 + [110.0], [100.0] * 10)
    assert facts is not None
    assert "relative_1m_pct" not in facts
    assert "return_1m_pct" in facts


def test_outcome_scores_pods_and_portfolio():
    facts = {
        "tech_pod": {"XLK": {"return_1w_pct": 2.0}, "SMH": {"return_1w_pct": -3.0}},
        "energy_pod": {"XLE": {"return_1w_pct": -1.0}},
    }
    pods = {
        "tech_pod": {"scores": [_score("XLK", 2), _score("SMH", -1)], "sector_view": "v"},
        "energy_pod": {"scores": [_score("XLE", 1)], "sector_view": "v"},
    }
    result = outcome({"XLK": 0.2, "XLE": 0.2}, pods, facts, 0.7)
    assert result["portfolio_1w_pct"] == 0.2
    assert result["invested_weight"] == 0.4
    assert result["spy_1w_pct"] == 0.7
    assert result["pods"]["tech_pod"]["positive_minus_negative_1w_pct"] == 5.0
    assert result["pods"]["energy_pod"]["positive_minus_negative_1w_pct"] is None


def test_outcome_unavailable_pods_score_none_and_unknown_symbols_are_ignored():
    facts = {"tech_pod": {"XLK": {"return_1w_pct": 2.0}}}
    pods = {"tech_pod": None, "energy_pod": {"unavailable": True}}
    result = outcome({"XLK": 0.2, "SHV": 0.5, "XYZ": 0.1}, pods, facts, None)
    assert result["portfolio_1w_pct"] == 0.4
    assert result["invested_weight"] == 0.2
    assert result["positions_1w_pct"] == {"XLK": 2.0}
    assert len(result["pods"]) == 5
    assert all(v["positive_minus_negative_1w_pct"] is None for v in result["pods"].values())
