#!/usr/bin/env bash
set -euo pipefail

# Purge stale generated files older than one month:
#   - live/paper run logs: logs/<strategy>/{live,paper}/<ts>_{live,paper}
#     (backtesting logs are never touched)
#   - dated cross_momentum universe archives: data/universe/us_stock_universe_YYYY-MM-DD.json
#     (the undated us_stock_universe.json is what cross_momentum reads: never touched)
#   - tool caches: .pytest_cache, .ruff_cache, .VSCodeCounter
# "Last update" of a directory is the newest mtime of anything inside it.
#
# Usage: run_purge_batch.sh [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN=0
case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    *)
        echo "Usage: $(basename "$0") [--dry-run]"
        exit 1
        ;;
esac

cd "$PROJECT_DIR"

CUTOFF_EPOCH="$(date -d '1 month ago' +%s)"
echo "### Purging files not updated since $(date -d "@$CUTOFF_EPOCH" '+%Y-%m-%d %H:%M') ###"
[[ "$DRY_RUN" -eq 1 ]] && echo "(dry run: nothing is deleted)"

DELETED=0

# Newest mtime (epoch seconds) of a path and, for a directory, of everything inside it.
last_update() {
    find "$1" -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1
}

purge() {
    local path="$1"
    if [[ "$(last_update "$path")" -ge "$CUTOFF_EPOCH" ]]; then
        return
    fi
    echo "delete: $path"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        rm -rf -- "$path"
    fi
    DELETED=$((DELETED + 1))
}

# Live/paper run logs. Only the live/ and paper/ folders are scanned, so backtesting/ is out of reach.
for mode in live paper; do
    for run in logs/*/"$mode"/*_"$mode"; do
        [[ -d "$run" && ! -L "$run" ]] || continue
        purge "$run"
    done
done

# Dated universe archives only (the glob needs a digit after the underscore, which excludes the undated latest file).
for archive in data/universe/us_stock_universe_[0-9]*.json; do
    [[ -f "$archive" ]] || continue
    purge "$archive"
done

# Tool caches.
for cache in .pytest_cache .ruff_cache .VSCodeCounter; do
    [[ -d "$cache" ]] || continue
    purge "$cache"
done

echo "### Done: $DELETED item(s) $([[ "$DRY_RUN" -eq 1 ]] && echo 'would be ')deleted ###"
