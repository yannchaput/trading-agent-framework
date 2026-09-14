#!/usr/bin/env bash
set -euo pipefail

VERSION=0.1.3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEEL="$PROJECT_DIR/dist/lumibot_trading_agent-$VERSION-py3-none-any.whl"

echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Wheel path: $WHEEL"

if [ $# -ne 2 ]; then
    echo "illegal number of parameters"
    echo "Usage: <command> [strategy_name] [backtest|paper|live]"
    exit 1

fi
STRATEGY="${1:-macro_risk}"
MODE_INPUT="${2:-backtest}"

case "$MODE_INPUT" in
    backtest|backtesting)   
        MODE="backtesting" ;;
    paper|paper_trading)    
        MODE="paper" ;;
    live|live_trading)      
        MODE="live" ;;
    *)
        echo "Usage: <command> [strategy_name] [backtest|paper|live]"
        echo "Unknown trading mode: $MODE_INPUT"
        exit 1
        ;;
esac

if [[ ! -f "$WHEEL" ]]; then
    echo "Wheel not found: $WHEEL"
    exit 1
fi

OUTPUT_FILE="./logs/agent_$MODE.log"
echo "Script output will be logged in $OUTPUT_FILE"

# Make sure we're in the project directory to run the command
cd "$PROJECT_DIR"
echo "We are in the project directory: $(pwd)"

# Install the wheel (and its dependencies) if not already present
uv pip install -q --upgrade "$WHEEL"

echo "Running $STRATEGY in $MODE mode from wheel..."
# Run under a pseudo-terminal (`script`) so ANSI colors (rich + Lumibot's
# termcolor-based log_message) are preserved; a pipe via `tee` would make both
# libraries strip them. We send the typescript to /dev/null and pipe script's
# stdout through tee instead, so the log file stays free of script's own
# "Script started/done" header/footer. `-e` + `pipefail` propagate the child's
# exit code.
script -qefc "uv run python -m lumibot_trading_agent.main \"$STRATEGY\" \"$MODE\"" /dev/null | tee "$OUTPUT_FILE"
