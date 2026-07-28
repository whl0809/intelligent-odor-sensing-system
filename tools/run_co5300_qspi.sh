#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec sudo python3 tools/co5300_qspi_test.py \
  --init config/co5300_init.json \
  --gpiochip auto \
  --pattern bars \
  --half-period-us 2 \
  --chunk-bytes 2048 \
  --hold
