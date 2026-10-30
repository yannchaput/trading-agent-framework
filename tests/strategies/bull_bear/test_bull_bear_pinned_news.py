from __future__ import annotations

import inspect
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.pinned_news import pin_search_news


def _search_news(symbols: str = "", start: str | None = None, end: str | None = None, limit: int = 10, include_content: bool = False) -> dict[str, Any]:
    """Search recent news headlines and summaries, optionally filtered to symbols."""
    _search_news.calls.append({"symbols": symbols, "start": start, "end": end, "limit": limit, "include_content": include_content})  # type: ignore[attr-defined]
    return _search_news.result  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _reset() -> None:
    _search_news.calls = []  # type: ignore[attr-defined]
    _search_news.result = {"count": 0, "articles": []}  # type: ignore[attr-defined]


def _pinned(stock: str | None = "GGAL"):
    return pin_search_news(_search_news, stock=lambda: stock)


def _searched() -> list[str]:
    return [call["symbols"] for call in _search_news.calls]  # type: ignore[attr-defined]


@pytest.mark.parametrize("given", ["", "   ", ",", " , "])
def test_an_empty_symbol_is_pinned_to_the_stock_being_researched(given: str) -> None:
    result = _pinned()(symbols=given)

    assert _searched() == ["GGAL"]  # not the market-wide news an empty filter returns
    assert result["symbols_searched"] == "GGAL"


@pytest.mark.parametrize("given", ["Grupo Galicia", "Gallo", "Hesai", "iShares JPMorgan EM Bond", "ggal"])
def test_a_company_name_or_a_lower_case_ticker_is_pinned_to_the_stock_being_researched(given: str) -> None:
    result = _pinned()(symbols=given)

    assert _searched() == ["GGAL"]
    assert result["symbols_searched"] == "GGAL"


@pytest.mark.parametrize("given", ["GGAL", "NVDA", "NVDA,AMD", "BRK.B", "BF-B", "SPY, QQQ"])
def test_tickers_pass_through_untouched_and_without_a_marker(given: str) -> None:
    result = _pinned()(symbols=given)

    assert _searched() == [given]  # another stock's ticker (a competitor, the market) is the model's call
    assert "symbols_searched" not in result


def test_a_name_among_tickers_is_dropped_and_the_tickers_are_kept() -> None:
    result = _pinned()(symbols="NVDA,Nvidia Corp,AMD")

    assert _searched() == ["NVDA,AMD"]
    assert result["symbols_searched"] == "NVDA,AMD"


def test_the_other_arguments_are_forwarded() -> None:
    _pinned()(symbols="", start="2026-01-01", end="2026-01-31", limit=5, include_content=True)

    assert _search_news.calls == [{"symbols": "GGAL", "start": "2026-01-01", "end": "2026-01-31", "limit": 5, "include_content": True}]  # type: ignore[attr-defined]


def test_without_a_stock_being_researched_the_call_is_left_as_it_is() -> None:
    result = _pinned(stock=None)(symbols="")

    assert _searched() == [""]
    assert "symbols_searched" not in result


def test_an_error_result_gets_no_marker() -> None:
    _search_news.result = {"error": "tool budget of 5 calls spent"}  # type: ignore[attr-defined]

    result = _pinned()(symbols="")

    assert result == {"error": "tool budget of 5 calls spent"}


def test_the_wrapper_is_the_same_tool_to_the_agent() -> None:
    wrapper = _pinned()

    assert wrapper.__name__ == _search_news.__name__  # the strategy test checks the real tool keeps the name `search_news`
    assert wrapper.__doc__ == _search_news.__doc__
    assert inspect.signature(wrapper) == inspect.signature(_search_news)  # the schema the model sees is unchanged
