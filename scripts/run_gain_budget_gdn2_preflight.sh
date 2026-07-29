#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
GPU1_UUID="${GPU1_UUID:-}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-GAIN-001 preflight requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 2
fi
if [[ -z "$GPU1_UUID" ]]; then
  printf 'Set GPU1_UUID from the AIStation GPU1 helper before preflight.\n' >&2
  exit 3
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT
export PYTHON_BIN
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_ROOT/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$PERSIST_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"

SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' \
  ':(exclude).venv' \
  ':(exclude)artifacts' \
  ':(exclude)models' \
  ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing Gain-Budget preflight with dirty source:\n%s\n' "$SOURCE_STATUS" >&2
  exit 5
fi

mkdir -p \
  "$PERSIST_ROOT/.cache" \
  "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" \
  "$PERSIST_ROOT/runs" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"

PERSIST_ROOT="$PERSIST_ROOT" "$REPO_ROOT/setup.sh"
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'setup.sh did not write the persistent Python path marker.\n' >&2
  exit 6
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHON_EXTRA_PATH
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
ARTIFACT_DIR="$PERSIST_ROOT/artifacts/gain-budget-gdn2/$GIT_SHA"
CONTRACT_JSON="$ARTIFACT_DIR/cuda_contract.json"
READY_JSON="$ARTIFACT_DIR/formal_ready.json"
CPU_LOG="$ARTIFACT_DIR/cpu_properties.log"
DATA_MANIFEST="$ARTIFACT_DIR/dataset_manifest.json"
OFFICIAL_DATA_DIR="${OFFICIAL_SUDOKU_DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"
mkdir -p "$ARTIFACT_DIR"

"$PYTHON_BIN" - "$DATA_MANIFEST" "$OFFICIAL_DATA_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output = pathlib.Path(sys.argv[1])
data_root = pathlib.Path(sys.argv[2]).resolve()
relative_paths = [
    pathlib.Path(split) / filename
    for split in ("train", "test")
    for filename in ("all__inputs.npy", "all__labels.npy")
]
files = {}
for relative_path in relative_paths:
    path = data_root / relative_path
    if not path.is_file():
        raise SystemExit(f"Official Sudoku data file is missing: {path}")
    files[str(relative_path)] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
payload = {
    "schema_version": "official_sudoku_dataset.v1",
    "data_root": str(data_root),
    "files": files,
}
tmp = output.with_suffix(output.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(output)
PY

"$PYTHON_BIN" "$REPO_ROOT/experiments/rwkv_fs_sudoku/test_gain_budget_gdn2.py" \
  2>&1 | tee "$CPU_LOG"

"$PYTHON_BIN" \
  "$REPO_ROOT/experiments/rwkv_fs_sudoku/check_gain_budget_gdn2_cuda.py" \
  --wheel "$REPO_ROOT/wheelhouse/flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl" \
  --expected-gpu-uuid "$GPU1_UUID" \
  --output "$CONTRACT_JSON"

SMOKE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
IDENTITY_RUN="gain-budget-gdn2-external-identity-smoke2-$SMOKE_STAMP-${GIT_SHA:0:7}"
BUDGET_RUN="gain-budget-gdn2-decay-funded-smoke2-$SMOKE_STAMP-${GIT_SHA:0:7}"

GAIN_BUDGET_PREFLIGHT=1 \
GAIN_BUDGET_FULL_STACK_PROBE=1 \
GAIN_BUDGET_CONTRACT_JSON="$CONTRACT_JSON" \
RUN_NAME="$IDENTITY_RUN" \
"$REPO_ROOT/scripts/run_gain_budget_gdn2_arm.sh" external_identity

GAIN_BUDGET_PREFLIGHT=1 \
GAIN_BUDGET_FULL_STACK_PROBE=1 \
GAIN_BUDGET_CONTRACT_JSON="$CONTRACT_JSON" \
RUN_NAME="$BUDGET_RUN" \
"$REPO_ROOT/scripts/run_gain_budget_gdn2_arm.sh" decay_funded

"$PYTHON_BIN" - \
  "$READY_JSON" \
  "$CONTRACT_JSON" \
  "$CPU_LOG" \
  "$DATA_MANIFEST" \
  "$GIT_SHA" \
  "$GPU1_UUID" \
  "$PERSIST_ROOT/runs/$IDENTITY_RUN" \
  "$PERSIST_ROOT/runs/$BUDGET_RUN" <<'PY'
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
cpu_log_path = pathlib.Path(sys.argv[3])
dataset_manifest_path = pathlib.Path(sys.argv[4])
git_sha = sys.argv[5]
gpu_uuid = sys.argv[6]
run_rows = (
    ("external_identity", pathlib.Path(sys.argv[7])),
    ("decay_funded", pathlib.Path(sys.argv[8])),
)
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract.get("status") != "passed" or contract.get("git_sha") != git_sha:
    raise SystemExit("CUDA contract is not passed for this exact source SHA")
if contract.get("gpu_uuid") != gpu_uuid:
    raise SystemExit("CUDA contract is not bound to the pre-registered GPU1 UUID")

smokes = {}
for expected_mode, run_dir in run_rows:
    result_paths = sorted(
        (run_dir / "output").glob("futureseed_loop_seed*.json")
    )
    if len(result_paths) != 1:
        raise SystemExit(
            f"Expected one smoke result for {expected_mode}, got {result_paths}"
        )
    result_path = result_paths[0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("args", {}).get("gdn2_gain_budget_mode") != expected_mode:
        raise SystemExit(f"Smoke mode mismatch for {expected_mode}")
    train = result.get("metrics", {}).get("train", {})
    semantic = (
        train.get("resume_train_checkpoint", {})
        .get("semantic_contract", {})
    )
    if semantic.get("matched") is not True:
        raise SystemExit(f"Exact resume contract failed for {expected_mode}")
    if expected_mode == "decay_funded":
        if train.get("gain_budget_clipped_frac", 0.0) <= 0.0:
            raise SystemExit("Gain-Budget smoke did not activate projection")
        if train.get("gain_budget_infeasible_frac", 1.0) != 0.0:
            raise SystemExit("Gain-Budget smoke encountered an infeasible budget")
        if train.get("gain_budget_step_bound_max", float("inf")) > 1.0001:
            raise SystemExit("Gain-Budget smoke violated its numerical bound")
        if train.get("gain_budget_delta_error_max", float("inf")) > 5e-6:
            raise SystemExit("Gain-Budget smoke failed erase-strength preservation")
    smokes[expected_mode] = {
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "semantic_resume": semantic,
        "train_gain_budget_clipped_frac": train.get(
            "gain_budget_clipped_frac"
        ),
        "train_gain_budget_infeasible_frac": train.get(
            "gain_budget_infeasible_frac"
        ),
        "train_gain_budget_step_bound_max": train.get(
            "gain_budget_step_bound_max"
        ),
        "train_gain_budget_delta_error_max": train.get(
            "gain_budget_delta_error_max"
        ),
    }

payload = {
    "status": "passed",
    "git_sha": git_sha,
    "gpu_uuid": gpu_uuid,
    "cpu_properties_log": str(cpu_log_path),
    "cpu_properties_log_sha256": sha256_file(cpu_log_path),
    "dataset_manifest": str(dataset_manifest_path),
    "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    "cuda_contract": str(contract_path),
    "cuda_contract_sha256": sha256_file(contract_path),
    "smokes": smokes,
}
tmp = ready_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(ready_path)
print(f"formal_ready={ready_path}")
PY

printf 'Gain-Budget formal preflight passed: %s\n' "$READY_JSON"
