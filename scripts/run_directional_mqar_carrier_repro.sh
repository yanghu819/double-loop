#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/huyang2/double-loop/worktrees/p-diag-carrier-002-77e5539}"
EXPECTED_SOURCE_SHA="${EXPECTED_SOURCE_SHA:-77e5539fc0ef74231cab658bd610b24254fdcfa7}"
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-GPU-d2877fe4-641c-fe64-2a74-8abca47c292f}"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
ZOOLOGY_ROOT="${ZOOLOGY_ROOT:-$PERSIST_ROOT/repos/zoology-official}"
FLA_SOURCE_ROOT="${FLA_SOURCE_ROOT:-$PERSIST_ROOT/.cache/fla-versions/9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e}"
REFERENCE_RUN="${REFERENCE_RUN:-$PERSIST_ROOT/runs/zoology-mqar-length-endpoints-20260804T085500Z-4b90962}"
RUN_NAME="${RUN_NAME:-p-diag-carrier-002-historical-order-$(date -u +%Y%m%dT%H%M%SZ)-77e5539}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"

export CUDA_VISIBLE_DEVICES=0
export FLA_EXPECTED_SOURCE_SHA="9c8e42e762fce087c27b673af4922795d9edb85e"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export UV_CACHE_DIR="$PERSIST_ROOT/.cache/uv"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export PYTHONPATH="$SOURCE_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

actual_source_sha="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
if [[ "$actual_source_sha" != "$EXPECTED_SOURCE_SHA" ]]; then
  printf 'source SHA mismatch: %s\n' "$actual_source_sha" >&2
  exit 3
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --short)" ]]; then
  printf 'historical source worktree is dirty\n' >&2
  exit 4
fi
if [[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" != "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb" ]]; then
  printf 'Zoology SHA mismatch\n' >&2
  exit 5
fi

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)
if [[ "${#gpu_rows[@]}" != "1" ]]; then
  printf 'expected one visible GPU, got %s\n' "${#gpu_rows[@]}" >&2
  exit 6
fi
gpu_uuid="${gpu_rows[0]%%,*}"
gpu_name="${gpu_rows[0]#*, }"
if [[ "$gpu_uuid" != "$EXPECTED_GPU_UUID" || "$gpu_name" != "NVIDIA A100-SXM4-80GB" ]]; then
  printf 'GPU contract mismatch: %s / %s\n' "$gpu_uuid" "$gpu_name" >&2
  exit 7
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  printf 'refusing to overlap an existing GPU compute process\n' >&2
  exit 8
fi

gdn2_sha="$(sha256sum "$FLA_SOURCE_ROOT/fla/layers/gdn2.py" | awk '{print $1}')"
if [[ "$gdn2_sha" != "4d001b6a8903cc7acb42b908ab3ade0f1edca0a17275630fc9c080c0232bf910" ]]; then
  printf 'pinned GDN2 source hash mismatch\n' >&2
  exit 9
fi

mkdir -p "$RUN_DIR/output"
printf '%s\n' "$EXPECTED_SOURCE_SHA" > "$RUN_DIR/source_HEAD.txt"
git -C "$SOURCE_ROOT" status --short > "$RUN_DIR/source_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"

"$PYTHON_BIN" - "$RUN_DIR/contract.json" "$SOURCE_ROOT" "$FLA_SOURCE_ROOT" <<'PY'
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from fla.layers.gdn2 import GatedDeltaNet2
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA

output, source_root, fla_root = map(Path, sys.argv[1:])
gdn2_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
if not gdn2_path.is_relative_to(fla_root.resolve()):
    raise RuntimeError(f"GDN2 import escaped pinned FLA root: {gdn2_path}")
device = torch.cuda.get_device_properties(0)
contract = {
    "cuda_device_count": torch.cuda.device_count(),
    "device_name": device.name,
    "device_uuid": str(device.uuid),
    "source_root": str(source_root.resolve()),
    "fla_root": str(fla_root.resolve()),
    "gdn2_source": str(gdn2_path),
    "gdn2_source_sha256": hashlib.sha256(gdn2_path.read_bytes()).hexdigest(),
    "pinned_fla_sha": PINNED_FLA_SHA,
    "official_gdn2_class": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
}
output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_length_endpoint \
  --output-dir "$RUN_DIR/output" \
  --length64-reference-run "$REFERENCE_RUN" \
  --max-epochs 10 --batch-size 32 \
  2>&1 | tee "$RUN_DIR/formal.log"
status="${PIPESTATUS[0]}"
set -e

if [[ -f "$RUN_DIR/output/comparison.json" ]]; then
  cp "$RUN_DIR/output/comparison.json" "$RUN_DIR/score.json"
fi
printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
find "$RUN_DIR" -maxdepth 4 -type f ! -name artifact_sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifact_sha256.tmp"
mv "$RUN_DIR/artifact_sha256.tmp" "$RUN_DIR/artifact_sha256.txt"
exit "$status"
