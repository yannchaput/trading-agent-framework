from __future__ import annotations

from trading_agent_framework.strategies.ray_dalio.basket import basket_problem, basket_tool

GOOD = {"TQQQ": 0.35, "UPRO": 0.25, "TMF": 0.25, "SHV": 0.15}


def test_a_good_basket_passes():
    assert basket_problem(GOOD) is None


def test_shv_is_not_counted_in_the_98_percent():
    assert basket_problem({"TQQQ": 0.49, "UPRO": 0.49, "SHV": 0.5}) is None


def test_refusals_say_why():
    assert "XYZ" in str(basket_problem({"XYZ": 0.3, "TQQQ": 0.3, "SHV": 0.3}))
    assert "above 0" in str(basket_problem({"TQQQ": 0.0, "UPRO": 0.3, "SHV": 0.3}))
    assert "at least 3" in str(basket_problem({"TQQQ": 0.5, "SHV": 0.5}))
    assert "0.98" in str(basket_problem({"TQQQ": 0.5, "UPRO": 0.5, "SHV": 0.0001}))


def test_a_stock_weight_the_rebalancer_would_never_buy_is_refused():
    assert "0.03" in str(basket_problem({"TQQQ": 0.9, "UPRO": 0.03, "TMF": 0.05}))  # at the rebalance band: never bought from an empty position
    assert basket_problem({"TQQQ": 0.9, "UPRO": 0.05, "SHV": 0.01}) is None  # SHV is the remainder, so its own weight is free


def test_a_nan_weight_is_refused():
    assert "above 0" in str(basket_problem({"TQQQ": float("nan"), "UPRO": 0.3, "SHV": 0.3}))


def test_percentages_are_refused_and_the_error_says_fractions():
    assert "fraction" in str(basket_problem({"TQQQ": 35, "UPRO": 25, "SHV": 40}))


def test_an_accepted_basket_replaces_the_stored_one():
    store: dict[str, float] = {}
    submit = basket_tool(store)
    assert submit(weights={"TQQQ": 0.4, "UPRO": 0.4, "SOXL": 0.1}, reason="first")["accepted"] is True
    assert submit(weights=GOOD, reason="second")["accepted"] is True
    assert store == GOOD


def test_a_refused_basket_leaves_the_store_alone():
    store = dict(GOOD)
    assert "error" in basket_tool(store)(weights={"XYZ": 1.0}, reason="bad")
    assert store == GOOD


def test_the_tool_is_named_submit_basket():
    assert basket_tool({}).__name__ == "submit_basket"
