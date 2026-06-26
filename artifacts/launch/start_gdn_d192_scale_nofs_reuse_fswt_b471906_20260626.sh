#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
SCRIPT="$ROOT/artifacts/launch/gdn_d192_scale_nofs_reuse_fswt_b471906_20260626.sh"
LOG="$ROOT/artifacts/logs/gdn_d192_scale_nofs_reuse_fswt_b471906_20260626.log"
PID="$ROOT/artifacts/logs/gdn_d192_scale_nofs_reuse_fswt_b471906_20260626.pid"

mkdir -p "$ROOT/artifacts/logs"
chmod +x "$SCRIPT"
setsid bash "$SCRIPT" > "$LOG.nohup" 2>&1 < /dev/null &
p=$!
sleep 1
echo "$p" > "$PID"
echo "pid=$p"
ps -p "$p" -o pid,etime,cmd --no-headers || true
