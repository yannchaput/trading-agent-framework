from __future__ import annotations

from trading_agent_framework import config
from trading_agent_framework.config import env as env_module


def test_config_reexports_match_source_module() -> None:
    assert config.load_strategy_env is env_module.load_strategy_env
    assert config.AlpacaCredentials is env_module.AlpacaCredentials
    assert config.TRADING_MODES is env_module.TRADING_MODES
