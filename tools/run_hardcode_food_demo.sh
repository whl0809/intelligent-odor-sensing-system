#!/usr/bin/env bash
# One-terminal launcher for the TGS + SVM41 hard-coded food demonstration.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

uart_device="${1:-/dev/ttyUSB0}"
frames="${FRAMES:-120}"
state_file="runtime/display_state.json"

mkdir -p runtime
bash tools/run_co5300_dashboard.sh &
dashboard_pid=$!

cleanup() {
  kill "$dashboard_pid" 2>/dev/null || true
  wait "$dashboard_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m enose acquire-classify \
  --config config/rpi5.toml \
  --uart "$uart_device" \
  --frames "$frames" \
  --display-state "$state_file"
