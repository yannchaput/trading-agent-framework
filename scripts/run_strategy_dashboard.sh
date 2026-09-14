#!/usr/bin/env bash
set -euo pipefail

echo "### Running strategy dashboard ###"

VERSION=0.1.3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEEL="$PROJECT_DIR/dist/lumibot_trading_agent-${VERSION}-py3-none-any.whl"

echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Wheel path: $WHEEL"

if [[ ! -f "$WHEEL" ]]; then
    echo "Wheel not found: $WHEEL"
    exit 1
fi

# Make sure we're in the project directory to run the command
cd "$PROJECT_DIR"
echo "We are in the project directory: $(pwd)"

# Install the wheel (and its dependencies) if not already present
uv pip install -q --upgrade "$WHEEL"
uv run python -m lumibot_trading_agent.dashboard.cli
