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
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing; run setup.sh first.\n' >&2
  exit 4
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
ARTIFACT_DIR="$PERSIST_ROOT/artifacts/gdn2-fast-slow-decay/$GIT_SHA"
CONTRACT_JSON="$ARTIFACT_DIR/cuda_contract.json"
mkdir -p "$ARTIFACT_DIR" "$TMPDIR"

# CPU is used only for deterministic FIR algebra properties, never model smoke.
(
  cd "$REPO_ROOT/experiments/rwkv_fs_sudoku"
  "$PYTHON_BIN" -m pytest test_fast_slow_decay_gdn2.py -q
) | tee "$ARTIFACT_DIR/pure_math.log"

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
  exit 5
fi

FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" external_identity smoke
FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" positive_causal smoke

"$PYTHON_BIN" - "$CONTRACT_JSON" "$ARTIFACT_DIR/formal_ready.json" "$GIT_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
git_sha = sys.argv[3]
contract = json.loads(contract_path.read_text(encoding="utf-8"))
payload = {
    "status": "passed",
    "git_sha": git_sha,
    "cuda_contract": str(contract_path),
    "cuda_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    "identity": contract["identity"],
    "candidate": contract["candidate"],
    "benchmark": contract["benchmark"],
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
