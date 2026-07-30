#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-FSMC-001 preflight is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ -z "${GPU1_UUID:-}" ]]; then
  printf 'GPU1_UUID is required and must come from the AIStation GPU1 probe.\n' >&2
  exit 4
fi
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing; run setup.sh first.\n' >&2
  exit 5
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
ARTIFACT_DIR="$PERSIST_ROOT/artifacts/gdn2-fast-slow-decay/$GIT_SHA"
CONTRACT_JSON="$ARTIFACT_DIR/cuda_contract.json"
mkdir -p "$ARTIFACT_DIR" "$TMPDIR"
VISIBLE_UUID="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ "$VISIBLE_UUID" != "$GPU1_UUID" ]]; then
  printf 'GPU UUID mismatch: visible=%s expected GPU1=%s\n' "$VISIBLE_UUID" "$GPU1_UUID" >&2
  exit 6
fi

# CPU is used only for deterministic FIR algebra properties, never model smoke.
(
  cd "$REPO_ROOT/experiments/rwkv_fs_sudoku"
  "$PYTHON_BIN" -m pytest test_fast_slow_decay_gdn2.py -q
) | tee "$ARTIFACT_DIR/pure_math.log"

(
  cd "$REPO_ROOT/experiments/rwkv_fs_sudoku"
  "$PYTHON_BIN" check_fla_delta_backbones.py \
    --backbone gdn2 \
    --check all \
    --out "$ARTIFACT_DIR/official_fla_gdn2.json"
) | tee "$ARTIFACT_DIR/official_fla_gdn2.log"

(
  cd "$REPO_ROOT/experiments/rwkv_fs_sudoku"
  "$PYTHON_BIN" check_fast_slow_decay_gdn2_cuda.py \
    --out "$CONTRACT_JSON" \
    --git-sha "$GIT_SHA"
) | tee "$ARTIFACT_DIR/cuda_contract.log"

EXPECTED_CHECKPOINT="$PERSIST_ROOT/models/gdn2-futureseed-d192l10-s12000-20260726T131301Z-42102bd/checkpoints/train_state_step009000.pt"
EXPECTED_HASH="606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7"
ACTUAL_HASH="$(sha256sum "$EXPECTED_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
  printf 'checkpoint hash mismatch: %s != %s\n' "$ACTUAL_HASH" "$EXPECTED_HASH" >&2
  exit 7
fi

PREFLIGHT_ID="$(date -u +%Y%m%dT%H%M%SZ)-${GIT_SHA:0:7}"
CONTROL_SMOKE_NAME="gdn2-fast-slow-preflight-${PREFLIGHT_ID}-external-identity"
CANDIDATE_SMOKE_NAME="gdn2-fast-slow-preflight-${PREFLIGHT_ID}-positive-causal"
RUN_NAME="$CONTROL_SMOKE_NAME" FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" external_identity smoke
RUN_NAME="$CANDIDATE_SMOKE_NAME" FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" positive_causal smoke

CONTROL_SMOKE_RESULT="$PERSIST_ROOT/runs/$CONTROL_SMOKE_NAME/output/futureseed_loop_seed52.json"
CANDIDATE_SMOKE_RESULT="$PERSIST_ROOT/runs/$CANDIDATE_SMOKE_NAME/output/futureseed_loop_seed52.json"

"$PYTHON_BIN" - \
  "$CONTRACT_JSON" \
  "$ARTIFACT_DIR/official_fla_gdn2.json" \
  "$ARTIFACT_DIR/pure_math.log" \
  "$CONTROL_SMOKE_RESULT" \
  "$CANDIDATE_SMOKE_RESULT" \
  "$ARTIFACT_DIR/formal_ready.json" \
  "$GIT_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
official_path = pathlib.Path(sys.argv[2])
math_path = pathlib.Path(sys.argv[3])
control_path = pathlib.Path(sys.argv[4])
candidate_path = pathlib.Path(sys.argv[5])
out = pathlib.Path(sys.argv[6])
git_sha = sys.argv[7]
contract = json.loads(contract_path.read_text(encoding="utf-8"))
control = json.loads(control_path.read_text(encoding="utf-8"))
candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
control_resume = (
    control["metrics"]["train"]["resume_train_checkpoint"]["semantic_contract"]
)
candidate_resume = (
    candidate["metrics"]["train"]["resume_train_checkpoint"]["semantic_contract"]
)
candidate_diag = candidate["metrics"]["train"]["fast_slow_decay"]
checks = {
    "control_mode": control["args"]["gdn2_fast_slow_decay_mode"]
    == "external_identity",
    "candidate_mode": candidate["args"]["gdn2_fast_slow_decay_mode"]
    == "positive_causal",
    "control_resume": control_resume["matched"] is True,
    "candidate_resume": candidate_resume["matched"] is True,
    "candidate_active": candidate_diag["gdn2_fast_slow_enabled"] == 1.0,
    "candidate_tv_reduced": candidate_diag["gdn2_fast_slow_tv_ratio"] < 1.0,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit(f"Fast-Slow full-stack smoke checks failed: {failed}")

def evidence(path: pathlib.Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "status": "passed",
    "git_sha": git_sha,
    "cuda_contract": str(contract_path),
    "cuda_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "gpu_uuid": contract["gpu_uuid"],
    "identity": contract["identity"],
    "candidate": contract["candidate"],
    "benchmark": contract["benchmark"],
    "smoke_checks": checks,
    "evidence": {
        "official_fla_gdn2": evidence(official_path),
        "pure_math_log": evidence(math_path),
        "external_identity_smoke": evidence(control_path),
        "positive_causal_smoke": evidence(candidate_path),
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
