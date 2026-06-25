#!/usr/bin/env bash
set -euo pipefail

mkdir -p /huyang2/double-loop/artifacts/launch
LOG=/huyang2/double-loop/artifacts/launch/gdn_triton_direct_tiny.log
PID=/huyang2/double-loop/artifacts/launch/gdn_triton_direct_tiny.pid

nohup bash /huyang2/double-loop/artifacts/launch/run_gdn_triton_direct_tiny.sh > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$PID"
cat "$PID"
