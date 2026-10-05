"""CLI entry point invoked by `uv run dashboard`.

`uv run dashboard [--benchmark-dir PATH] [streamlit options...]`
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from trading_agent_framework.dashboard.benchmark_reader import BENCHMARK_DIR_FLAG, split_benchmark_dir

# Streamlit reads .streamlit/config.toml from the working directory, not the app's, so the dark
# trading theme travels as flags. User-supplied args come later and win.
THEME_ARGS = [
    "--theme.base",
    "dark",
    "--theme.backgroundColor",
    "#0b0e14",
    "--theme.secondaryBackgroundColor",
    "#131722",
    "--theme.textColor",
    "#d1d4dc",
    "--theme.primaryColor",
    "#22d3ee",
]


def build_command(python: str, app_file: Path, argv: Sequence[str]) -> list[str]:
    """The `streamlit run` command; --benchmark-dir goes to the app as a script argument (after `--`)."""
    rest, benchmark_dir = split_benchmark_dir(argv)
    # Default to hot-reload on source changes; user-supplied args override.
    command = [python, "-m", "streamlit", "run", str(app_file), *THEME_ARGS, "--server.runOnSave", "true", *rest]
    if benchmark_dir is not None:
        command += ["--", BENCHMARK_DIR_FLAG, benchmark_dir]
    return command


def main():
    import subprocess
    import sys

    try:
        # Fail fast if dashboard dependencies are missing, rather than failing later in the Streamlit runtime.
        import streamlit as st  # noqa: F401
    except ImportError as e:
        raise ImportError("Dashboard dependencies are missing. Please reinstall with the dashboard extra:\n    pip install trading_agent_framework[dashboard]") from e

    app_file = Path(__file__).parent / "app.py"
    try:
        sys.exit(subprocess.run(build_command(sys.executable, app_file, sys.argv[1:])).returncode)
    except KeyboardInterrupt:
        print("Exit gracefully.")


if __name__ == "__main__":
    main()
