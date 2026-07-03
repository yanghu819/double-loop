#!/usr/bin/env bash
set -euo pipefail

LAUNCH_DIR="/huyang2/double-loop/artifacts/launch/gdn-transition-bridgehard-d224l12-step9000-20260703T0604Z-679f0f0"
set -a
source "$LAUNCH_DIR/launch.env"
set +a

cd "$WORKTREE"
mkdir -p "$LAUNCH_DIR"
mkdir -p runs
cp "$LAUNCH_DIR/launch.env" "runs/$RUN_NAME.launch.env" 2>/dev/null || true

nohup ./run.sh full >"$LAUNCH_DIR/nohup.out" 2>&1 &
echo "$!" >"$LAUNCH_DIR/pid.txt"
echo "pid=$(cat "$LAUNCH_DIR/pid.txt")"
echo "log=$LAUNCH_DIR/nohup.out"
echo "run_dir=$WORKTREE/runs/$RUN_NAME"
