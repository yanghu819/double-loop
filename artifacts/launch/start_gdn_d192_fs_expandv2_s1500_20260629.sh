#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
SOURCE_SHA="${SOURCE_SHA:?set SOURCE_SHA}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SHORT="${SOURCE_SHA:0:7}"
RUN_NAME="${RUN_NAME:-gdn-d192-official-sudoku-fs-expandv2-s1500-${RUN_STAMP}-${SHORT}}"
SCRIPT="$ROOT/artifacts/launch/gdn_d192_fs_expandv2_s1500_20260629.sh"
LOG="$ROOT/artifacts/logs/${RUN_NAME}.log"
PID="$ROOT/artifacts/logs/${RUN_NAME}.pid"

mkdir -p "$ROOT/artifacts/logs"
chmod +x "$SCRIPT"
SOURCE_SHA="$SOURCE_SHA" RUN_STAMP="$RUN_STAMP" RUN_NAME="$RUN_NAME" WT_OVERRIDE="${WT_OVERRIDE:-}" GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  setsid bash "$SCRIPT" > "$LOG.nohup" 2>&1 < /dev/null &
p=$!
sleep 1
echo "$p" > "$PID"
echo "pid=$p"
echo "run_name=$RUN_NAME"
ps -p "$p" -o pid,etime,cmd --no-headers || true
