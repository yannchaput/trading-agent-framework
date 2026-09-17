"""CLI entry point invoked by `uv run dashboard`."""


def main():
    import subprocess
    import sys
    from pathlib import Path

    app_file = Path(__file__).parent / "app.py"
    # Default to hot-reload on source changes; user-supplied args override.
    default_args = ["--server.runOnSave", "true"]
    args = [sys.executable, "-m", "streamlit", "run", str(app_file), *default_args, *sys.argv[1:]]
    try:
        sys.exit(subprocess.run(args).returncode)
    except KeyboardInterrupt:
        print("Exit gracefully.")


if __name__ == "__main__":
    main()
