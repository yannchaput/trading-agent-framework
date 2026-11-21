from __future__ import annotations

from trading_agent_framework.strategies.ray_dalio.basket import basket_problem, basket_tool


def _ignore(message: str) -> None:
    pass


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
    submit = basket_tool(store, _ignore)
    assert submit(weights={"TQQQ": 0.4, "UPRO": 0.4, "SOXL": 0.1}, reason="first")["accepted"] is True
    assert submit(weights=GOOD, reason="second")["accepted"] is True
    assert store == GOOD


def test_a_refused_basket_leaves_the_store_alone():
    store = dict(GOOD)
    assert "error" in basket_tool(store, _ignore)(weights={"XYZ": 1.0}, reason="bad")
    assert store == GOOD


def test_the_tool_is_named_submit_basket():
    assert basket_tool({}, _ignore).__name__ == "submit_basket"


def test_the_tool_logs_an_accepted_basket_with_its_reason():
    lines: list[str] = []
    basket_tool({}, lines.append)(weights=GOOD, reason="rates are falling")
    assert len(lines) == 1 and "rates are falling" in lines[0] and "TQQQ" in lines[0]


def test_a_refused_basket_logs_nothing():
    lines: list[str] = []
    basket_tool({}, lines.append)(weights={"XYZ": 1.0}, reason="bad")
    assert lines == []


def test_two_calls_in_one_turn_are_serialized_by_the_agent_manager():
    assert vars(basket_tool({}, _ignore))["mutates_trading"] is True


def test_a_basket_that_is_all_stock_says_the_98_percent_limit_not_fractions():
    message = str(basket_problem({"TQQQ": 0.4, "UPRO": 0.3, "TMF": 0.3}))
    assert "0.98" in message and "fraction" not in message
