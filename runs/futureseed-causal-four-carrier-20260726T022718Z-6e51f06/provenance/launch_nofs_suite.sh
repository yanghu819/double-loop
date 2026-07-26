#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
WT="$ROOT/artifacts/worktrees/fs-causal-6e51f06"
LAUNCH="$ROOT/artifacts/launch/nofs-causal-20260726T022718Z-6e51f06"
CONFIG="$LAUNCH/backbone_benchmark_nofs.env"
BASE_CONFIG="$WT/configs/sudoku/backbone_benchmark.env"
PREFLIGHT="$ROOT/runs/sudoku-backbone-preflight-nofs-causal-20260726T022718Z-6e51f06"
EXPECTED_SHA=6e51f067a18d936bda7f3d588b7b3f32625b2b18

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="$ROOT"
export RUNS_ROOT="$ROOT/runs"
export PYTHON_BIN=/opt/conda/bin/python
export PYTHON_EXTRA_PATH="$ROOT/.cache/python-extra-pylib"
export XDG_CACHE_HOME="$ROOT/.cache"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export TMPDIR="$ROOT/.cache/tmp"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export BENCHMARK_CONFIG="$CONFIG"
export BENCHMARK_STEPS=500
export BENCHMARK_EASY_STEPS=100

mkdir -p "$LAUNCH" "$ROOT/runs" "$ROOT/models" "$ROOT/.cache/tmp"
ulimit -c 0

actual_sha="$(git -C "$WT" rev-parse HEAD)"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
  printf 'Wrong detached source: %s != %s\n' "$actual_sha" "$EXPECTED_SHA" >&2
  exit 10
fi
if [[ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]]; then
  printf 'Detached source is dirty.\n' >&2
  exit 11
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  printf 'GPU1-only contract violated.\n' >&2
  exit 12
fi

expected_config="$LAUNCH/expected_nofs.env"
sed 's/^FUTURE_SEED_SCALE=1$/FUTURE_SEED_SCALE=0/' "$BASE_CONFIG" >"$expected_config"
sed '1s/.*/# State-matched four-backbone causal FutureSeed control. Parameters and runtime are measured, not padded./' \
  "$expected_config" >"$expected_config.tmp"
mv "$expected_config.tmp" "$expected_config"
if ! cmp -s "$expected_config" "$CONFIG"; then
  printf 'No-FutureSeed config differs outside the preregistered scale/comment change.\n' >&2
  diff -u "$expected_config" "$CONFIG" >&2 || true
  exit 13
fi

{
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'source_sha=%s\n' "$actual_sha"
  printf 'source_dirty=false\n'
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'gpu='
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  printf 'baseline_config_sha256='
  sha256sum "$BASE_CONFIG" | cut -d' ' -f1
  printf 'nofs_config_sha256='
  sha256sum "$CONFIG" | cut -d' ' -f1
} >"$LAUNCH/launch_manifest.txt"

"$WT/scripts/run_sudoku_baseline_preflight.sh" "$PREFLIGHT"

run_one() {
  local backbone="$1"
  local run_name="sudoku-backbone-${backbone}-nofs-s500-sharedinit-20260726T022718Z-6e51f06"
  printf '\n[causal-suite] start backbone=%s run=%s utc=%s\n' \
    "$backbone" "$run_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$WT/scripts/run_sudoku_backbone_benchmark.sh" "$backbone" "$run_name"
  printf '[causal-suite] complete backbone=%s run=%s utc=%s\n' \
    "$backbone" "$run_name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# Highest-information carriers first; all four are binding.
run_one rwkv
run_one gdn2
run_one gdn
run_one kda

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LAUNCH/launch_manifest.txt"
printf '[causal-suite] all controls complete\n'
