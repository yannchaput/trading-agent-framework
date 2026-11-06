"""Every package of `trading_agent_framework` exposes `__version__ = get_version("trading_agent_framework")`.

The check is static (the `__init__.py` source is parsed, never imported) so it also covers packages that need an
optional dependency, such as the dashboard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "trading_agent_framework"
INIT_FILES = sorted(path for path in PACKAGE_ROOT.rglob("__init__.py") if "__pycache__" not in path.parts)


def _is_the_version_assignment(node: ast.stmt) -> bool:
    """`__version__ = get_version("trading_agent_framework")`, exactly."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__version__"):
        return False
    call = node.value
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "get_version"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "trading_agent_framework"
        and not call.keywords
    )


def _imports_get_version(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "get_version" and alias.asname is None for alias in node.names)
        for node in ast.walk(tree)
    )


def test_every_package_directory_has_an_init_file() -> None:
    """A directory of modules without `__init__.py` would be a namespace package that this test (and the version) cannot see."""
    missing = [
        directory.relative_to(PACKAGE_ROOT).as_posix()
        for directory in sorted(PACKAGE_ROOT.rglob("*"))
        if directory.is_dir() and "__pycache__" not in directory.parts and any(directory.glob("*.py")) and not (directory / "__init__.py").exists()
    ]
    assert missing == []


@pytest.mark.parametrize("init_file", INIT_FILES, ids=lambda path: path.relative_to(PACKAGE_ROOT.parent).parent.as_posix())
def test_the_package_defines_its_version(init_file: Path) -> None:
    tree = ast.parse(init_file.read_text(encoding="utf-8"))

    assert any(_is_the_version_assignment(node) for node in tree.body), f'{init_file} lacks __version__ = get_version("trading_agent_framework")'
    assert _imports_get_version(tree), f"{init_file} does not import get_version"
