from __future__ import annotations

from trading_agent_framework.brokers import alpaca
from trading_agent_framework.brokers.alpaca import broker as broker_module
from trading_agent_framework.brokers.alpaca import stream as stream_module


def test_alpaca_package_reexports_match_source_modules() -> None:
    assert alpaca.AlpacaBroker is broker_module.AlpacaBroker
    assert alpaca.AlpacaTradeStream is stream_module.AlpacaTradeStream
