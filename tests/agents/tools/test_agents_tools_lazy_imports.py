from __future__ import annotations

import subprocess
import sys

import pytest

from trading_agent_framework.agents import tools
from trading_agent_framework.agents.tools import account as account_module
from trading_agent_framework.agents.tools import indicators as indicators_module
from trading_agent_framework.agents.tools import macro as macro_module
from trading_agent_framework.agents.tools import market_data as market_data_module
from trading_agent_framework.agents.tools import prebuilt as prebuilt_module
from trading_agent_framework.agents.tools import trading as trading_module


def test_light_factories_are_eagerly_reexported() -> None:
    assert tools.PrebuiltTools is prebuilt_module.PrebuiltTools
    assert tools.account_tools is account_module.account_tools
    assert tools.indicator_tools is indicators_module.indicator_tools
    assert tools.macro_tools is macro_module.macro_tools
    assert tools.market_data_tools is market_data_module.market_data_tools
    assert tools.trading_tools is trading_module.trading_tools


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        tools.does_not_exist  # noqa: B018


def test_lazy_attribute_resolves_to_the_real_factory_in_process() -> None:
    """Exercises the __getattr__ lazy-import branch itself (import + cache), as opposed to
    the subprocess tests below which check the *absence* of `alpaca`/`httpx` from
    `sys.modules` before either attribute is ever touched.
    """
    from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools as direct_fundamentals
    from trading_agent_framework.agents.tools.news import news_tools as direct_news

    assert tools.news_tools is direct_news
    assert tools.fundamentals_tools is direct_fundamentals
    # Second access hits the globals() cache set by __getattr__, not the _LAZY branch again.
    assert tools.news_tools is direct_news


def test_dir_includes_lazy_and_eager_names() -> None:
    names = dir(tools)
    assert "news_tools" in names
    assert "fundamentals_tools" in names
    assert "PrebuiltTools" in names
    assert "trading_tools" in names


def test_importing_agents_tools_package_does_not_import_alpaca_or_httpx() -> None:
    """Must run in a subprocess.

    By the time this test runs, other test modules in the same pytest session have already
    imported `alpaca`/`httpx` (and therefore pandas) dozens of times, so checking
    `sys.modules` in-process would pass or fail for reasons unrelated to whether
    `agents/tools/__init__.py` itself is import-light. A fresh interpreter is the only way
    to make the assertion meaningful.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.agents.tools\n"
            "import sys\n"
            "assert 'alpaca' not in sys.modules\n"
            "assert 'httpx' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_news_tools_lazily_imports_alpaca() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trading_agent_framework.agents.tools import news_tools\n"
            "import sys\n"
            "assert 'alpaca' in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_fundamentals_tools_lazily_imports_httpx() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trading_agent_framework.agents.tools import fundamentals_tools\n"
            "import sys\n"
            "assert 'httpx' in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
