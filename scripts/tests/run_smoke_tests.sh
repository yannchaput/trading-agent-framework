#!/usr/bin/env bash
set -uo pipefail

echo -e "### Running smoke tests ###\n"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

results=()
failures=0

for test in "$SCRIPT_DIR"/smoke_*.py; do
    name="$(basename "$test")"
    echo "--- $name ---"
    output="$(uv run python "$test" 2>&1)"
    status=$?
    echo "$output"
    if (( status != 0 )); then
        results+=("FAIL  $name")
        failures=$((failures + 1))
    elif [[ "$output" == SKIP:* || "$output" == *$'\n'SKIP:* ]]; then
        results+=("SKIP  $name")
    else
        results+=("PASS  $name")
    fi
    echo
done

echo "### Summary ###"
printf '%s\n' "${results[@]}"
echo

if (( failures > 0 )); then
    echo "$failures test(s) failed."
    exit 1
fi

echo "All smoke tests passed (some may have been skipped; see summary)."
