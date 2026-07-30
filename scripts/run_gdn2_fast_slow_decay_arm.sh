#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-}"
PHASE="${2:-formal}"
if [[ "$ARM" != "external_identity" && "$ARM" != "positive_causal" ]]; then
  printf 'usage: %s external_identity|positive_causal [smoke|formal]\n' "$0" >&2
  exit 2
fi
if [[ "$PHASE" != "smoke" && "$PHASE" != "formal" ]]; then
  printf 'phase must be smoke or formal\n' >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-FSMC-001 is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing; run setup.sh first.\n' >&2
  exit 4
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHON_EXTRA_PATH
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1
export UPDATE_LEADERBOARD=0
export GDN2_FAST_SLOW_DECAY_MODE="$ARM"

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/gdn2_fast_slow_decay_probe.env"
set +a

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
CONTRACT_JSON="${FAST_SLOW_CONTRACT_JSON:-$PERSIST_ROOT/artifacts/gdn2-fast-slow-decay/$GIT_SHA/cuda_contract.json}"
"$PYTHON_BIN" - "$CONTRACT_JSON" "$GIT_SHA" <<'PY'
import json
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Fast-Slow CUDA contract is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit(f"Fast-Slow CUDA contract did not pass: {payload.get('status')}")
if payload.get("git_sha") != expected_sha:
    raise SystemExit(
        f"Fast-Slow CUDA contract SHA mismatch: {payload.get('git_sha')} != {expected_sha}"
    )
gpu_uuids = [
    row.strip()
    for row in subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    if row.strip()
]
if gpu_uuids != [payload.get("gpu_uuid")]:
    raise SystemExit(
        f"Fast-Slow CUDA contract GPU mismatch: {gpu_uuids} != "
        f"{[payload.get('gpu_uuid')]}"
    )
PY

if [[ "$PHASE" == "smoke" ]]; then
  export HOLE_STAGES=46-50:500,51-55:3500,51-60:4000,51-64:1002
  export FULL_STEPS=9002
  export FULL_EVAL_N=8
  export EVAL_HOLES_LIST=53
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_CHECKPOINT_STEPS=9002
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
else
  READY_JSON="${FAST_SLOW_READY_JSON:-$PERSIST_ROOT/artifacts/gdn2-fast-slow-decay/$GIT_SHA/formal_ready.json}"
  "$PYTHON_BIN" - "$READY_JSON" "$CONTRACT_JSON" "$GIT_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

ready_path = pathlib.Path(sys.argv[1])
contract_path = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
if not ready_path.is_file():
    raise SystemExit(f"Fast-Slow formal-ready manifest is missing: {ready_path}")
payload = json.loads(ready_path.read_text(encoding="utf-8"))
checks = {
    "status": payload.get("status") == "passed",
    "git_sha": payload.get("git_sha") == expected_sha,
    "contract_hash": payload.get("cuda_contract_sha256")
    == hashlib.sha256(contract_path.read_bytes()).hexdigest(),
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit(f"Fast-Slow formal-ready checks failed: {failed}")
for key in (
    "official_fla_gdn2",
    "pure_math_log",
    "external_identity_smoke",
    "positive_causal_smoke",
):
    row = payload.get("evidence", {}).get(key, {})
    path = pathlib.Path(row.get("path", ""))
    if not path.is_file():
        raise SystemExit(f"Fast-Slow formal-ready evidence missing: {key}: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
        raise SystemExit(f"Fast-Slow formal-ready evidence hash mismatch: {key}")
PY
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$PHASE" == "smoke" ]]; then
  DEFAULT_NAME="gdn2-fast-slow-${ARM}-smoke2-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  DEFAULT_NAME="gdn2-fast-slow-${ARM}-s9100-${TIMESTAMP}-${GIT_SHA:0:7}"
fi
export RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"
export EXACT_TRAIN_PID_FILE="${EXACT_TRAIN_PID_FILE:-$PERSIST_ROOT/artifacts/launch/$RUN_NAME.pid}"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUNS_ROOT" \
  "$TRAIN_CHECKPOINT_DIR" \
  "$(dirname "$EXACT_TRAIN_PID_FILE")"

cd "$REPO_ROOT"
exec ./run.sh full
