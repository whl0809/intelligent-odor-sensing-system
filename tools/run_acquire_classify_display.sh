#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  VENV_PYTHON="$VIRTUAL_ENV/bin/python"
else
  VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Activate the project virtual environment first." >&2
  exit 1
fi

STATE_FILE="$REPO_ROOT/runtime/display_state.json"
mkdir -p "$REPO_ROOT/runtime"
cp "$REPO_ROOT/config/display_state.example.json" "$STATE_FILE.tmp"
mv "$STATE_FILE.tmp" "$STATE_FILE"

sudo -v
sudo python3 tools/co5300_dashboard.py \
  --state-file "$STATE_FILE" \
  --init config/co5300_init.json \
  --gpiochip auto \
  --clk 21 \
  --sio0 20 \
  --sio1 19 \
  --sio2 16 \
  --sio3 26 \
  --cs 18 \
  --rst 25 \
  --te 24 \
  --half-period-us 5 \
  --chunk-bytes 1024 \
  --refresh-seconds 1 &
dashboard_pid=$!

cleanup() {
  sudo kill "$dashboard_pid" 2>/dev/null || true
  wait "$dashboard_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$VENV_PYTHON" -m enose acquire-classify \
  --config config/rpi5.toml \
  --display-state "$STATE_FILE" \
  "$@"
