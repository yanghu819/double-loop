#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEFAULT_CONFIG="$REPO_ROOT/configs/retrieval/zoology_owner_local_momentum.env"
CONFIG="${PGDN3068_CONFIG:-$DEFAULT_CONFIG}"
# shellcheck disable=SC1090
source "$CONFIG"

: "${EXPECTED_GPU_NAME:?set EXPECTED_GPU_NAME}"
: "${EXPECTED_GPU_UUID:?set EXPECTED_GPU_UUID}"
: "${EXPECTED_SOURCE_SHA:?set EXPECTED_SOURCE_SHA}"
: "${MATCHED_INIT:?set MATCHED_INIT}"
: "${FROZEN_SCORE:?set FROZEN_SCORE}"
: "${FROZEN_CASES:?set FROZEN_CASES}"

export CUDA_VISIBLE_DEVICES=0
export PATH="$ENV_ROOT/bin:$PATH"
export CUDA_HOME="$ENV_ROOT"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export LIBRARY_PATH="$ENV_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions_olm280"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton_olm280"
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT"
export PERSIST_ROOT PYTHON_BIN ZOOLOGY_ROOT ZOOLOGY_SHA FLA_SOURCE_ROOT
export FLA_EXPECTED_SOURCE_SHA FLA_DISABLE_BACKEND_DISPATCH FLA_CONV_BACKEND
export MDN_REPO_ROOT MDN_FLA_ROOT MDN_EXPECTED_SHA
export WANDB_MODE=disabled

mkdir -p "$(dirname "$MDN_REPO_ROOT")"
if [[ ! -d "$MDN_REPO_ROOT/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$MDN_REPO_URL" "$MDN_REPO_ROOT"
fi
if [[ "$(git -C "$MDN_REPO_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$MDN_EXPECTED_SHA" ]]; then
  git -C "$MDN_REPO_ROOT" fetch --depth=1 origin "$MDN_EXPECTED_SHA"
  git -C "$MDN_REPO_ROOT" checkout --detach --force "$MDN_EXPECTED_SHA"
fi
[[ "$(git -C "$MDN_REPO_ROOT" rev-parse HEAD)" == "$MDN_EXPECTED_SHA" ]]
[[ -z "$(git -C "$MDN_REPO_ROOT" status --porcelain --untracked-files=no)" ]]

[[ -x "$PYTHON_BIN" ]]
VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
[[ "$VISIBLE_GPU" == "0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$ZOOLOGY_SHA" ]]
[[ -z "$(git -C "$ZOOLOGY_ROOT" status --porcelain --untracked-files=no)" ]]
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-068 requires a detached source worktree.\n' >&2
  exit 8
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_SHA" == "$EXPECTED_SOURCE_SHA" ]]
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
REMOTE_SHA="$(
  env -u LD_LIBRARY_PATH git -C "$REPO_ROOT" ls-remote \
    origin "$SOURCE_REMOTE_REF" | awk '{print $1}'
)"
[[ "$REMOTE_SHA" == "$GIT_SHA" ]]
[[ -f "$MATCHED_INIT" && -f "$FROZEN_SCORE" && -f "$FROZEN_CASES" ]]
"$PYTHON_BIN" - <<'PY'
import torch, triton
assert torch.__version__.split('+')[0].startswith('2.8.'), torch.__version__
assert tuple(map(int, triton.__version__.split('.')[:2])) >= (3, 4), triton.__version__
assert torch.version.cuda is not None
PY
[[ "$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')" == "$("$PYTHON_BIN" -c 'import torch; print(torch.version.cuda)')" ]]

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-gdn3-068.lock"
flock -n 9

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-068-owner-local-momentum-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
mkdir -p "$OUT_DIR"
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
  if [[ "$STATUS" != 0 && "$STATUS" != 2 && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$PHASE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "reason": "P-GDN3-068 integrity or infrastructure failure",
    "scientific_failure": False,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
  printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
  find "$RUN_DIR" -type f ! -name artifacts.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256"
  printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
  exit "$STATUS"
}
trap finalize EXIT
trap 'STATUS=130; exit 130' INT
trap 'STATUS=143; exit 143' TERM

LAUNCHER_PID="$$"
LAUNCHER_PGID="$(ps -o pgid= -p "$$" | tr -d ' ')"
[[ "$LAUNCHER_PID" == "$LAUNCHER_PGID" ]]
printf '%s\n' "$LAUNCHER_PID" > "$RUN_DIR/pid.txt"
printf '%s\n' "$LAUNCHER_PGID" > "$RUN_DIR/pgid.txt"
cp "$CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$REMOTE_SHA" > "$RUN_DIR/github_readback_sha.txt"
printf '%s\n' "$MDN_EXPECTED_SHA" > "$RUN_DIR/external_mdn_sha.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
git -C "$REPO_ROOT" archive --format=tar.gz \
  --output="$RUN_DIR/source_snapshot.tar.gz" HEAD
sample_gpu >> "$RUN_DIR/gpu_samples.csv" &
GPU_SAMPLER_PID="$!"

set +e
timeout --signal=TERM --kill-after=30 "$CONTRACT_WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m scripts.check_zoology_owner_local_momentum \
  --matched-init "$MATCHED_INIT" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --output "$RUN_DIR/contract.json" \
  2>&1 | tee "$RUN_DIR/contract.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' \
  "${PIPE[0]}" "${PIPE[1]}" "$STATUS" > "$RUN_DIR/contract_status.txt"
[[ "$STATUS" != 0 || -s "$RUN_DIR/contract.json" ]] || STATUS=66
[[ "$STATUS" == 0 ]] || exit "$STATUS"

PHASE=formal
set +e
timeout --signal=TERM --kill-after=30 "$FORMAL_WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.momentum_owner_local_endpoint \
  --output-dir "$OUT_DIR" \
  --matched-init "$MATCHED_INIT" \
  --frozen-score "$FROZEN_SCORE" \
  --frozen-cases "$FROZEN_CASES" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$RUN_DIR/formal.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'endpoint=%s\ntee=%s\ncombined=%s\n' \
  "${PIPE[0]}" "${PIPE[1]}" "$STATUS" > "$RUN_DIR/formal_status.txt"
if [[ -f "$OUT_DIR/comparison.json" ]]; then
  PHASE=decision
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  if [[ "$STATUS" == 0 ]]; then
    set +e
    "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["registered_gate"]["passed"] else 2)' "$RUN_DIR/score.json"
    STATUS="$?"
    set -e
  fi
fi
exit "$STATUS"
