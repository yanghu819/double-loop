#!/usr/bin/env bash
set -euo pipefail

PLAN_ID="P-FS2-006"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
EXPECTED_UUID="${EXPECTED_UUID:-}"
EXPECTED_SOURCE_SHA="${EXPECTED_SOURCE_SHA:-}"
SOURCE_BRANCH="${SOURCE_BRANCH:-codex/futureseed2-receiver-recommit-20260813}"
EXPECTED_FLA_SHA="9c8e42e762fce087c27b673af4922795d9edb85e"
EXPECTED_FLA_WHEEL_SHA256="0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
FLA_PINNED_ROOT="$PERSIST_ROOT/.cache/fla-versions/$EXPECTED_FLA_SHA-0280db310981915e"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-$PERSIST_ROOT/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt}"
PARENT_CHECKPOINT_SHA256="6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da"
PARENT_SOURCE_SHA="9f2ee8d1738032bc5f09b55db0b81d507780b376"
DATA_DIR="${DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"
UNSEEN_INDEX_PATH="${UNSEEN_INDEX_PATH:-$DATA_DIR/indices/unseen-after-gdn3-position-qk-step003000-seed52-b128-51-64.npy}"
UNSEEN_INDEX_SHA256="8675395b75899aa6c9f3a899791df9b71f042d87a168426eafdf89269258bdd7"
UNSEEN_MANIFEST_PATH="${UNSEEN_MANIFEST_PATH:-$DATA_DIR/indices/unseen-after-gdn3-position-qk-step003000-seed52-b128-51-64.json}"
UNSEEN_MANIFEST_SHA256="7be5a07d15b7b9dbd3127491e6d75cda43877756d218dd452fd5b2056b36dde5"
ENDPOINT="$REPO_ROOT/experiments/rwkv_fs_sudoku/diagnose_futureseed_receiver_live_contrast_cuda.py"
RUN_NAME="${RUN_NAME:-p-fs2-006-receiver-live-contrast-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$REPO_ROOT" rev-parse --short=7 HEAD)}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
SCORE_PATH="$RUN_DIR/score.json"
CACHE_ROOT="$PERSIST_ROOT/.cache/p-fs2-006"
WALL_BUDGET_SEC=2400

PHASE="launcher"
ABORT_REASON="launcher exited before a classified terminal state"
TERMINAL=0

record_integrity_abort() {
  local exit_status=$?
  trap - EXIT
  if [[ "$TERMINAL" != "1" ]]; then
    mkdir -p "$RUN_DIR"
    printf '%s\n' "$exit_status" > "$RUN_DIR/status"
    printf '{"plan_id":"%s","run_name":"%s","exit_code":%d,"phase":"%s","reason":"%s","failure_class":"integrity","integrity_failure":true,"scientific_failure":false,"timestamp_utc":"%s"}\n' \
      "$PLAN_ID" "$RUN_NAME" "$exit_status" "$PHASE" "$ABORT_REASON" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/abort.json"
  fi
  exit "$exit_status"
}
trap record_integrity_abort EXIT

integrity_fail() {
  local exit_status="$1"
  shift
  ABORT_REASON="$*"
  exit "$exit_status"
}

if [[ -z "$EXPECTED_UUID" || -z "$EXPECTED_SOURCE_SHA" ]]; then
  integrity_fail 2 "EXPECTED_UUID and EXPECTED_SOURCE_SHA are required"
fi
if [[ -e "$RUN_DIR" ]]; then
  integrity_fail 32 "formal run directory already exists"
fi
CURRENT_PGID="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
if [[ "$CURRENT_PGID" != "$$" ]]; then
  integrity_fail 29 "formal launcher must be started in a dedicated setsid process group"
fi
exec 9>"$PERSIST_ROOT/.gpu0-model.lock"
if ! flock -n 9; then
  integrity_fail 30 "GPU0 model lock is already held"
fi

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  integrity_fail 3 "$PLAN_ID requires CUDA_VISIBLE_DEVICES=0"
fi
export CUDA_VISIBLE_DEVICES=0

PHASE="gpu_contract"
VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits)"
if [[ "$VISIBLE_GPU" != "0, $EXPECTED_UUID, NVIDIA A100-SXM4-80GB, 81920" ]]; then
  integrity_fail 4 "unexpected visible GPU contract"
fi
COMPUTE_BEFORE="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" \
  || integrity_fail 26 "failed to query existing GPU compute processes"
if [[ -n "$COMPUTE_BEFORE" ]]; then
  integrity_fail 5 "$PLAN_ID refuses to overlap an existing GPU compute process"
fi

PHASE="source_contract"
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  integrity_fail 6 "$PLAN_ID requires a detached source worktree"
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  integrity_fail 7 "$PLAN_ID requires a clean source worktree"
fi
SOURCE_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$SOURCE_HEAD" != "$EXPECTED_SOURCE_SHA" ]]; then
  integrity_fail 20 "formal source HEAD does not match EXPECTED_SOURCE_SHA"
fi
if [[ ! -f "$ENDPOINT" ]]; then
  integrity_fail 8 "diagnostic endpoint is missing"
fi
REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url origin)"
if [[ "$REMOTE_URL" != "https://github.com/yanghu819/double-loop.git" ]]; then
  integrity_fail 31 "formal origin URL is not the GitHub code source"
fi
REMOTE_HEAD="$(timeout 60 git -C "$REPO_ROOT" ls-remote origin "refs/heads/$SOURCE_BRANCH" | awk '{print $1}')" \
  || integrity_fail 24 "GitHub branch readback failed or timed out"
if [[ "$REMOTE_HEAD" != "$SOURCE_HEAD" ]]; then
  integrity_fail 21 "GitHub branch readback does not match formal source HEAD"
fi

PHASE="parent_contract"
if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
  integrity_fail 9 "parent checkpoint is missing"
fi
ACTUAL_PARENT_SHA="$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_PARENT_SHA" != "$PARENT_CHECKPOINT_SHA256" ]]; then
  integrity_fail 10 "parent checkpoint SHA mismatch"
fi
if [[ ! -f "$UNSEEN_INDEX_PATH" ]] || [[ "$(sha256sum "$UNSEEN_INDEX_PATH" | awk '{print $1}')" != "$UNSEEN_INDEX_SHA256" ]]; then
  integrity_fail 22 "parent-specific unseen index is missing or has the wrong SHA"
fi
if [[ ! -f "$UNSEEN_MANIFEST_PATH" ]] || [[ "$(sha256sum "$UNSEEN_MANIFEST_PATH" | awk '{print $1}')" != "$UNSEEN_MANIFEST_SHA256" ]]; then
  integrity_fail 23 "parent-specific unseen manifest is missing or has the wrong SHA"
fi

FLA_SOURCE_SHA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
if [[ ! -f "$FLA_SOURCE_SHA_MARKER" ]]; then
  integrity_fail 11 "pinned FLA source SHA marker is missing"
fi
ACTUAL_FLA_SHA="$(tr -d '[:space:]' < "$FLA_SOURCE_SHA_MARKER")"
if [[ "$ACTUAL_FLA_SHA" != "$EXPECTED_FLA_SHA" ]]; then
  integrity_fail 12 "pinned FLA source SHA mismatch"
fi
if [[ ! -d "$FLA_PINNED_ROOT/fla" ]]; then
  integrity_fail 27 "pinned FLA wheel tree is missing"
fi
FLA_WHEEL="$PERSIST_ROOT/wheelhouse/flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
if [[ ! -f "$FLA_WHEEL" ]] || [[ "$(sha256sum "$FLA_WHEEL" | awk '{print $1}')" != "$EXPECTED_FLA_WHEEL_SHA256" ]]; then
  integrity_fail 28 "pinned FLA wheel is missing or has the wrong SHA"
fi
if ! command -v timeout >/dev/null 2>&1; then
  integrity_fail 13 "GNU timeout is required"
fi

export PERSIST_ROOT
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TMPDIR="$CACHE_ROOT/tmp"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export FLA_PINNED_ROOT
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experiments/rwkv_fs_sudoku:$FLA_PINNED_ROOT:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"
export FLA_EXPECTED_SOURCE_SHA="$EXPECTED_FLA_SHA"
export FLA_SOURCE_SHA_MARKER
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_STRICT_OFFICIAL=1
unset FLA_SOURCE_ROOT

mkdir -p "$RUN_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" \
  "$TORCH_HOME" "$HF_HOME"
printf '%s\n' "$SOURCE_HEAD" > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source.patch"
printf '%s\n' "$VISIBLE_GPU" > "$RUN_DIR/gpu_before.csv"
printf '%s\n' "$COMPUTE_BEFORE" > "$RUN_DIR/compute_before.csv"
printf '%s\n' "$ACTUAL_PARENT_SHA  $PARENT_CHECKPOINT" > "$RUN_DIR/parent_checkpoint.sha256"
printf '%s\n' "$ACTUAL_FLA_SHA" > "$RUN_DIR/fla_source_SHA.txt"
printf '%s\n' "$$" > "$RUN_DIR/launcher_pid.txt"
printf '%s\n' "$CURRENT_PGID" > "$RUN_DIR/launcher_pgid.txt"
printf 'plan_id=%s\nsource_head=%s\nsource_branch=%s\nparent_source_sha=%s\nparent_checkpoint_sha256=%s\nunseen_index_sha256=%s\nunseen_manifest_sha256=%s\nfla_source_sha=%s\nexpected_gpu_uuid=%s\nwall_budget_sec=%s\n' \
  "$PLAN_ID" "$SOURCE_HEAD" "$SOURCE_BRANCH" "$PARENT_SOURCE_SHA" "$PARENT_CHECKPOINT_SHA256" \
  "$UNSEEN_INDEX_SHA256" "$UNSEEN_MANIFEST_SHA256" \
  "$EXPECTED_FLA_SHA" "$EXPECTED_UUID" "$WALL_BUDGET_SEC" > "$RUN_DIR/provenance.env"

PYTHON_BIN="${PYTHON_BIN:-$PERSIST_ROOT/official_eqr_compare/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  integrity_fail 14 "configured Python executable is missing"
fi

PHASE="diagnostic"
cd "$REPO_ROOT"
set +e
timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
  "$PYTHON_BIN" "$ENDPOINT" \
    --checkpoint "$PARENT_CHECKPOINT" \
    --checkpoint-sha256 "$PARENT_CHECKPOINT_SHA256" \
    --parent-source-sha "$PARENT_SOURCE_SHA" \
    --data-dir "$DATA_DIR" \
    --unseen-index-path "$UNSEEN_INDEX_PATH" \
    --unseen-index-sha256 "$UNSEEN_INDEX_SHA256" \
    --unseen-manifest-path "$UNSEEN_MANIFEST_PATH" \
    --unseen-manifest-sha256 "$UNSEEN_MANIFEST_SHA256" \
    --out "$SCORE_PATH" \
    --expected-gpu-uuid "$EXPECTED_UUID" \
    --batch-size 64 \
    --eval-seed 52051 \
    --discovery-per-range 128 \
    --holdout-per-range 128 \
    --discovery-seed 5205101 \
    --holdout-seed 5205102 \
    --blank-ranges 51-55 56-60 61-64 \
    --blank-weight 8 \
    2>&1 | tee "$RUN_DIR/run.log"
PIPELINE_STATUS=("${PIPESTATUS[@]}")
set -e
ENDPOINT_STATUS="${PIPELINE_STATUS[0]}"
TEE_STATUS="${PIPELINE_STATUS[1]}"
printf '%s\n' "$ENDPOINT_STATUS" > "$RUN_DIR/endpoint_exit_code.txt"

if [[ "$TEE_STATUS" != "0" ]]; then
  integrity_fail 15 "failed to persist diagnostic log"
fi
if [[ "$ENDPOINT_STATUS" == "124" || "$ENDPOINT_STATUS" == "137" || "$ENDPOINT_STATUS" == "143" ]]; then
  integrity_fail 16 "diagnostic exceeded the 2400 second wall budget"
fi
if [[ "$ENDPOINT_STATUS" != "0" ]]; then
  integrity_fail 25 "diagnostic endpoint exited nonzero: $ENDPOINT_STATUS"
fi
if [[ ! -s "$SCORE_PATH" ]]; then
  integrity_fail 17 "diagnostic did not produce score.json"
fi

PHASE="decision_contract"
DECISION_STATUS="$({
  "$PYTHON_BIN" - "$SCORE_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

decision = payload.get("decision")
candidates = [payload.get("status")]
if isinstance(decision, dict):
    candidates.extend((decision.get("status"), decision.get("outcome")))

for candidate in candidates:
    if candidate in {"completed_rejected", "admitted"}:
        print(candidate)
        raise SystemExit(0)
raise SystemExit(2)
PY
} 2>/dev/null)" || integrity_fail 18 "score.json lacks an admitted or completed_rejected terminal decision"

if [[ "$DECISION_STATUS" != "completed_rejected" && "$DECISION_STATUS" != "admitted" ]]; then
  integrity_fail 19 "score.json contains an invalid terminal decision"
fi

printf '%s\n' "$DECISION_STATUS" > "$RUN_DIR/decision_status.txt"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader > "$RUN_DIR/gpu_after.csv"
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
  > "$RUN_DIR/compute_after.csv" || true
sha256sum "$SCORE_PATH" "$RUN_DIR/run.log" "$RUN_DIR/source_HEAD.txt" \
  "$RUN_DIR/source.patch" "$RUN_DIR/gpu_before.csv" "$RUN_DIR/gpu_after.csv" \
  "$RUN_DIR/provenance.env" "$RUN_DIR/endpoint_exit_code.txt" \
  "$RUN_DIR/decision_status.txt" > "$RUN_DIR/hashes.sha256"
rm -f "$RUN_DIR/abort.json"
printf '0\n' > "$RUN_DIR/status"
TERMINAL=1
trap - EXIT
exit 0
