#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXTRA_ROOT="$REPO_ROOT/extras/food_freshness"
cd "$REPO_ROOT"

exec sudo python3 "$EXTRA_ROOT/tools/co5300_qspi_test.py" \
  --init "$EXTRA_ROOT/config/co5300_init.json" \
  --gpiochip auto \
  --pattern bars \
  --half-period-us 2 \
  --chunk-bytes 2048 \
  --hold
