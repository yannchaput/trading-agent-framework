from __future__ import annotations

from pathlib import Path

from trading_agent_framework.dashboard.cli import THEME_ARGS, build_command

APP = Path("/x/app.py")


def test_the_dark_theme_is_passed_as_flags() -> None:
    command = build_command("py", APP, [])

    assert command[:5] == ["py", "-m", "streamlit", "run", str(APP)]
    assert "--theme.base" in command and command[command.index("--theme.base") + 1] == "dark"
    assert command[command.index("--theme.backgroundColor") + 1] == "#0b0e14"
    assert "--" not in command


def test_user_args_come_after_the_defaults_so_they_override() -> None:
    command = build_command("py", APP, ["--theme.base", "light"])

    assert command[-2:] == ["--theme.base", "light"]
    assert command.index("--theme.base") < len(command) - 2  # the default is still there, earlier


def test_benchmark_dir_is_forwarded_to_the_script_after_a_double_dash() -> None:
    command = build_command("py", APP, ["--benchmark-dir", "/r", "--server.port", "8502"])

    assert command[-3:] == ["--", "--benchmark-dir", "/r"]
    assert command.index("--server.port") < command.index("--")


def test_theme_args_are_flag_value_pairs() -> None:
    assert len(THEME_ARGS) % 2 == 0
    assert all(flag.startswith("--theme.") for flag in THEME_ARGS[::2])
