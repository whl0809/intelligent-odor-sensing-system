#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec sudo python3 tools/co5300_dashboard.py \
  --state-file config/display_state.example.json \
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
  --once \
  --hold
