from __future__ import annotations

import pytest

from trading_agent_framework.brokers.alpaca.symbols import from_alpaca_symbol, to_alpaca_symbol


@pytest.mark.parametrize(
    ("dash", "dot"),
    [("BRK-A", "BRK.A"), ("BRK-B", "BRK.B"), ("HEI-A", "HEI.A"), ("MOG-A", "MOG.A"), ("BF-B", "BF.B")],
)
def test_to_alpaca_symbol_converts_our_dash_share_class_suffix_to_alpacas_dot_form(dash: str, dot: str) -> None:
    assert to_alpaca_symbol(dash) == dot


def test_to_alpaca_symbol_leaves_a_plain_symbol_unchanged() -> None:
    assert to_alpaca_symbol("AAPL") == "AAPL"


def test_to_alpaca_symbol_ignores_a_multi_letter_suffix() -> None:
    # Not a share class (e.g. a warrant); only a single trailing letter is a share class.
    assert to_alpaca_symbol("ABC-WT") == "ABC-WT"


@pytest.mark.parametrize(
    ("dot", "dash"),
    [("BRK.A", "BRK-A"), ("BRK.B", "BRK-B"), ("HEI.A", "HEI-A"), ("MOG.A", "MOG-A"), ("BF.B", "BF-B")],
)
def test_from_alpaca_symbol_converts_alpacas_dot_share_class_suffix_to_our_dash_form(dot: str, dash: str) -> None:
    assert from_alpaca_symbol(dot) == dash


def test_from_alpaca_symbol_leaves_a_plain_symbol_unchanged() -> None:
    assert from_alpaca_symbol("AAPL") == "AAPL"


@pytest.mark.parametrize("symbol", ["BRK-A", "BRK-B", "HEI-A", "MOG-A", "BF-B"])
def test_symbol_translation_round_trips(symbol: str) -> None:
    assert from_alpaca_symbol(to_alpaca_symbol(symbol)) == symbol
