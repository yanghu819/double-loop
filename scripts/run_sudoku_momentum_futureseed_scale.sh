#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${MOMENTUM_SUDOKU_CONFIG:-$REPO_ROOT/configs/sudoku/gdn3_momentum_futureseed_scale.env}"
set -a
source "$CONFIG"
set +a

: "${EXPECTED_GPU_NAME:?set EXPECTED_GPU_NAME}"
: "${EXPECTED_GPU_UUID:?set EXPECTED_GPU_UUID}"
: "${EXPECTED_SOURCE_SHA:?set EXPECTED_SOURCE_SHA}"
: "${SOURCE_REMOTE_REF:?set SOURCE_REMOTE_REF}"

export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST_ROOT/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TMPDIR="$PERSIST_ROOT/.cache/tmp"
export PYTHON_BIN="${PYTHON_BIN:-$PERSIST_ROOT/.cache/venvs/p-gdn3-034-torch271/bin/python}"
export FLA_SOURCE_ROOT="${FLA_SOURCE_ROOT:-$PERSIST_ROOT/.cache/fla-versions/9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e}"
export FLA_EXPECTED_SOURCE_SHA="${FLA_EXPECTED_SOURCE_SHA:-9c8e42e762fce087c27b673af4922795d9edb85e}"
export MDN_REPO_ROOT="${MDN_REPO_ROOT:-$PERSIST_ROOT/.cache/external/MomentumDeltaNet-c6e77fa}"
export MDN_FLA_ROOT="${MDN_FLA_ROOT:-$MDN_REPO_ROOT/flash-linear-attention}"
export MDN_EXPECTED_SHA
export FLA_DISABLE_BACKEND_DISPATCH FLA_CONV_BACKEND FLA_USE_CUDA_GRAPH
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-$PERSIST_ROOT/.cache/python-extra-pylib}"
export PYTHONPATH="$REPO_ROOT/experiments/rwkv_fs_sudoku:$REPO_ROOT:$FLA_SOURCE_ROOT:$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUNS_ROOT" \
  "$PERSIST_ROOT/artifacts/launch/p-gdn3-071" \
  "$(dirname "$MDN_REPO_ROOT")"

if [[ ! -d "$MDN_REPO_ROOT/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$MDN_REPO_URL" "$MDN_REPO_ROOT"
fi
if [[ "$(git -C "$MDN_REPO_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$MDN_EXPECTED_SHA" ]]; then
  git -C "$MDN_REPO_ROOT" fetch --depth=1 origin "$MDN_EXPECTED_SHA"
  git -C "$MDN_REPO_ROOT" checkout --detach --force "$MDN_EXPECTED_SHA"
fi
[[ "$(git -C "$MDN_REPO_ROOT" rev-parse HEAD)" == "$MDN_EXPECTED_SHA" ]]
[[ -z "$(git -C "$MDN_REPO_ROOT" status --porcelain --untracked-files=no)" ]]

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits)"
[[ "$VISIBLE_GPU" == "0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME" ]]
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')" == 1 ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
[[ -d "$OFFICIAL_SUDOKU_DATA_DIR" ]]
[[ -x "$PYTHON_BIN" ]]
[[ -d "$FLA_SOURCE_ROOT" ]]
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-071 requires a detached source worktree.\n' >&2
  exit 8
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_SHA" == "$EXPECTED_SOURCE_SHA" ]]
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
REMOTE_SHA="$(git -C "$REPO_ROOT" ls-remote origin "$SOURCE_REMOTE_REF" | awk '{print $1}')"
[[ "$REMOTE_SHA" == "$GIT_SHA" ]]
if (( FULL_BATCH * GRAD_ACCUM_STEPS != 128 )); then
  printf 'P-GDN3-071 requires effective batch 128.\n' >&2
  exit 9
fi
[[ "$BACKBONE" == momentum ]]
[[ "$MOMENTUM_FUTURE_SEED_TRANSPORT" == full_state ]]

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-gdn3-071.lock"
flock -n 9

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LAUNCH_DIR="$PERSIST_ROOT/artifacts/launch/p-gdn3-071/${TIMESTAMP}-${GIT_SHA:0:7}"
mkdir -p "$LAUNCH_DIR"
STATUS=0
PHASE=contract
GPU_SAMPLER_PID=""

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
    sleep 5
  done
}

finalize() {
  local shell_status="$?"
  trap - EXIT
  set +e
  [[ "$STATUS" != 0 ]] || STATUS="$shell_status"
  if [[ -n "$GPU_SAMPLER_PID" ]]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  fi
  if [[ "$STATUS" != 0 && ! -f "$LAUNCH_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$LAUNCH_DIR/abort.json" "$STATUS" "$PHASE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "reason": "P-GDN3-071 contract, probe, or formal integrity failure",
    "scientific_failure": False,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  printf '%s\n' "$STATUS" > "$LAUNCH_DIR/status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$LAUNCH_DIR/completed_at.txt"
  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$LAUNCH_DIR/gpu_after.txt"
  find "$LAUNCH_DIR" -type f ! -name artifacts.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > "$LAUNCH_DIR/artifacts.sha256"
  exit "$STATUS"
}
trap finalize EXIT
trap 'STATUS=130; exit 130' INT
trap 'STATUS=143; exit 143' TERM

printf '%s\n' "$$" > "$LAUNCH_DIR/pid.txt"
ps -o pgid= -p "$$" | tr -d ' ' > "$LAUNCH_DIR/pgid.txt"
printf '%s\n' "$GIT_SHA" > "$LAUNCH_DIR/git_sha.txt"
printf '%s\n' "$REMOTE_SHA" > "$LAUNCH_DIR/github_readback_sha.txt"
printf '%s\n' "$MDN_EXPECTED_SHA" > "$LAUNCH_DIR/external_momentum_sha.txt"
cp "$CONFIG" "$LAUNCH_DIR/launch.env"
git -C "$REPO_ROOT" archive --format=tar.gz --output="$LAUNCH_DIR/source_snapshot.tar.gz" HEAD
date -u +%Y-%m-%dT%H:%M:%SZ > "$LAUNCH_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits > "$LAUNCH_DIR/gpu_before.txt"
sample_gpu >> "$LAUNCH_DIR/gpu_samples.csv" &
GPU_SAMPLER_PID="$!"

set +e
timeout --signal=TERM --kill-after=30 1800 \
  "$PYTHON_BIN" -u "$REPO_ROOT/experiments/rwkv_fs_sudoku/check_momentum_futureseed_cuda.py" \
  --out "$LAUNCH_DIR/contract.json" \
  --expected_gpu_name "$EXPECTED_GPU_NAME" \
  --expected_gpu_uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$LAUNCH_DIR/contract.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
  > "$LAUNCH_DIR/contract_status.txt"
if [[ "$STATUS" != 0 || ! -s "$LAUNCH_DIR/contract.json" ]]; then
  [[ "$STATUS" != 0 ]] || STATUS=66
  exit "$STATUS"
fi

PHASE=probe
PROBE_NAME="p-gdn3-071-momentum-sudoku-probe-${TIMESTAMP}-${GIT_SHA:0:7}"
set +e
(
  export RUN_NAME="$PROBE_NAME"
  export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$PROBE_NAME/checkpoints"
  export FULL_STEPS=2
  export HOLE_STAGES=51-55:2
  export FULL_EVAL_N=8
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_HOLES_LIST=53
  export EVAL_CHECKPOINT_STEPS=2
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=2
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
  cd "$REPO_ROOT"
  ./run.sh full
) 2>&1 | tee "$LAUNCH_DIR/probe.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'run=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
  > "$LAUNCH_DIR/probe_status.txt"
if [[ "$STATUS" != 0 \
  || ! -s "$RUNS_ROOT/$PROBE_NAME/output/futureseed_loop_seed52.json" \
  || ! -s "$PERSIST_ROOT/models/$PROBE_NAME/checkpoints/train_state_step000002.pt" ]]; then
  [[ "$STATUS" != 0 ]] || STATUS=66
  exit "$STATUS"
fi
sha256sum \
  "$RUNS_ROOT/$PROBE_NAME/output/futureseed_loop_seed52.json" \
  "$PERSIST_ROOT/models/$PROBE_NAME/checkpoints/train_state_step000002.pt" \
  > "$LAUNCH_DIR/probe_artifacts.sha256"

PHASE=formal
FORMAL_NAME="p-gdn3-071-momentum-sudoku-d192l10-s12000-${TIMESTAMP}-${GIT_SHA:0:7}"
printf '%s\n' "$FORMAL_NAME" > "$LAUNCH_DIR/formal_run_name.txt"
export RUN_NAME="$FORMAL_NAME"
export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$FORMAL_NAME/checkpoints"
mkdir -p "$TRAIN_CHECKPOINT_DIR"
cd "$REPO_ROOT"
set +e
./run.sh full 2>&1 | tee "$LAUNCH_DIR/formal.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'run=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
  > "$LAUNCH_DIR/formal_status.txt"
exit "$STATUS"
