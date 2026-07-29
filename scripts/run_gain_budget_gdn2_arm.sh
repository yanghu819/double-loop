#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-}"
if [[ "$ARM" != "external_identity" && "$ARM" != "decay_funded" ]]; then
  printf 'usage: %s external_identity|decay_funded\n' "$0" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-GAIN-001 is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
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
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing; run setup.sh first.\n' >&2
  exit 4
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHON_EXTRA_PATH
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export REQUIRE_GAIN_BUDGET_CONTRACT=1
export SMOKE_DONE=1
export SKIP_SETUP=1
export GDN2_GAIN_BUDGET_MODE="$ARM"
export UPDATE_LEADERBOARD=0

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/gain_budget_gdn2_probe.env"
set +a

if [[ "${GAIN_BUDGET_FULL_STACK_PROBE:-0}" == "1" ]]; then
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
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
export GAIN_BUDGET_CONTRACT_JSON="${GAIN_BUDGET_CONTRACT_JSON:-$PERSIST_ROOT/artifacts/gain-budget-gdn2/$GIT_SHA/cuda_contract.json}"
"$PYTHON_BIN" - "$GAIN_BUDGET_CONTRACT_JSON" "$GIT_SHA" <<'PY'
import json
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
if not path.is_file():
    raise SystemExit(f"Gain-Budget CUDA contract is missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "passed":
    raise SystemExit(f"Gain-Budget CUDA contract did not pass: {payload.get('status')}")
if payload.get("git_sha") != expected_sha:
    raise SystemExit(
        f"Gain-Budget CUDA contract SHA mismatch: {payload.get('git_sha')} != {expected_sha}"
    )
gpu_rows = [
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
if gpu_rows != [payload.get("gpu_uuid")]:
    raise SystemExit(
        f"Gain-Budget CUDA contract GPU mismatch: {gpu_rows} != "
        f"{[payload.get('gpu_uuid')]}"
    )
PY
if [[ "${GAIN_BUDGET_PREFLIGHT:-0}" != "1" ]]; then
  export GAIN_BUDGET_FORMAL_READY_JSON="${GAIN_BUDGET_FORMAL_READY_JSON:-$PERSIST_ROOT/artifacts/gain-budget-gdn2/$GIT_SHA/formal_ready.json}"
  "$PYTHON_BIN" - \
    "$GAIN_BUDGET_FORMAL_READY_JSON" \
    "$GAIN_BUDGET_CONTRACT_JSON" \
    "$GIT_SHA" \
    "$PERSIST_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ready_path = pathlib.Path(sys.argv[1])
contract_path = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
persistent_root = pathlib.Path(sys.argv[4]).resolve()
if not ready_path.is_file():
    raise SystemExit(f"Gain-Budget formal-ready manifest is missing: {ready_path}")
payload = json.loads(ready_path.read_text(encoding="utf-8"))


def resolve_evidence(path_value: str, label: str) -> pathlib.Path:
    path = pathlib.Path(path_value).resolve()
    if not path.is_file():
        raise SystemExit(f"Gain-Budget {label} is missing: {path}")
    if not path.is_relative_to(persistent_root):
        raise SystemExit(
            f"Gain-Budget {label} escaped persistent root: {path}"
        )
    return path


cpu_log = resolve_evidence(payload.get("cpu_properties_log", ""), "CPU log")
dataset_manifest_path = resolve_evidence(
    payload.get("dataset_manifest", ""),
    "dataset manifest",
)
dataset_manifest = json.loads(
    dataset_manifest_path.read_text(encoding="utf-8")
)
dataset_files_valid = True
for relative_path, row in dataset_manifest.get("files", {}).items():
    data_path = resolve_evidence(row.get("path", ""), f"dataset file {relative_path}")
    dataset_files_valid = dataset_files_valid and (
        data_path.stat().st_size == row.get("size_bytes")
        and sha256_file(data_path) == row.get("sha256")
    )

smoke_checks = {}
for expected_mode in ("external_identity", "decay_funded"):
    row = payload.get("smokes", {}).get(expected_mode, {})
    result_path = resolve_evidence(
        row.get("result_path", ""),
        f"{expected_mode} smoke result",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    train = result.get("metrics", {}).get("train", {})
    semantic = (
        train.get("resume_train_checkpoint", {})
        .get("semantic_contract", {})
    )
    smoke_checks[f"{expected_mode}_smoke_hash"] = (
        sha256_file(result_path) == row.get("result_sha256")
    )
    smoke_checks[f"{expected_mode}_smoke_mode"] = (
        result.get("args", {}).get("gdn2_gain_budget_mode")
        == expected_mode
    )
    smoke_checks[f"{expected_mode}_smoke_resume"] = (
        semantic.get("matched") is True
    )
    if expected_mode == "decay_funded":
        smoke_checks["budget_smoke_mechanism"] = (
            train.get("gain_budget_clipped_frac", 0.0) > 0.0
            and train.get("gain_budget_infeasible_frac", 1.0) == 0.0
            and train.get("gain_budget_step_bound_max", float("inf"))
            <= 1.0001
            and train.get("gain_budget_delta_error_max", float("inf"))
            <= 5e-6
        )

checks = {
    "status": payload.get("status") == "passed",
    "git_sha": payload.get("git_sha") == expected_sha,
    "contract_hash": payload.get("cuda_contract_sha256")
    == sha256_file(contract_path),
    "cpu_log_hash": payload.get("cpu_properties_log_sha256")
    == sha256_file(cpu_log),
    "dataset_manifest_hash": payload.get("dataset_manifest_sha256")
    == sha256_file(dataset_manifest_path),
    "dataset_files": bool(dataset_manifest.get("files"))
    and dataset_files_valid,
    **smoke_checks,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit(
        f"Gain-Budget formal-ready manifest failed checks: {failed}"
    )
PY
fi
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${GAIN_BUDGET_FULL_STACK_PROBE:-0}" == "1" ]]; then
  DEFAULT_NAME="gain-budget-gdn2-${ARM}-smoke2-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  DEFAULT_NAME="gain-budget-gdn2-${ARM}-s9100-${TIMESTAMP}-${GIT_SHA:0:7}"
fi
export RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUNS_ROOT" \
  "$TRAIN_CHECKPOINT_DIR"

cd "$REPO_ROOT"
exec ./run.sh full
