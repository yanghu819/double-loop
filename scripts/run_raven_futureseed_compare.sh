#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
TIMESTAMP="${RAVEN_COMPARE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GROUP_NAME="${RAVEN_COMPARE_GROUP:-raven-futureseed-vs-gdn2-${TIMESTAMP}-${GIT_SHA:0:7}}"
GROUP_DIR="$PERSIST_ROOT/runs/$GROUP_NAME"
mkdir -p "$GROUP_DIR"

GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d '[:space:]')"
CONTRACT_JSON="$GROUP_DIR/raven_cuda_contract.json"
CUDA_VISIBLE_DEVICES=0 \
FLA_DISABLE_BACKEND_DISPATCH=1 \
FLA_CONV_BACKEND=triton \
FLA_EXPECTED_SOURCE_SHA=31d15f7554bd5df05d3da6f75e09146279d2b1a8 \
FLA_SOURCE_ROOT="$PERSIST_ROOT/.cache/fla-upstream-31d15f7" \
PYTHONPATH="$PERSIST_ROOT/.cache/fla-upstream-31d15f7:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}" \
/opt/conda/bin/python "$REPO_ROOT/experiments/rwkv_fs_sudoku/check_raven_futureseed_cuda.py" \
  --out "$CONTRACT_JSON" \
  --expected_gpu_uuid "$GPU_UUID" \
  > "$GROUP_DIR/raven_cuda_contract.log" 2>&1

GDN2_RUN="${GROUP_NAME}-gdn2"
RAVEN_RUN="${GROUP_NAME}-raven"
PERSIST_ROOT="$PERSIST_ROOT" "$REPO_ROOT/scripts/run_raven_futureseed_arm.sh" gdn2 "$GDN2_RUN"
PERSIST_ROOT="$PERSIST_ROOT" "$REPO_ROOT/scripts/run_raven_futureseed_arm.sh" raven "$RAVEN_RUN"

/opt/conda/bin/python "$REPO_ROOT/scripts/compare_raven_futureseed.py" \
  --gdn2 "$PERSIST_ROOT/runs/$GDN2_RUN" \
  --raven "$PERSIST_ROOT/runs/$RAVEN_RUN" \
  --contract "$CONTRACT_JSON" \
  --out_dir "$GROUP_DIR"
