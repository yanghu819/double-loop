#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STEPS="${BENCHMARK_STEPS:-500}"
SUITE_ID="${SUITE_ID:-${TIMESTAMP}-${GIT_SHA:0:7}}"
PREFLIGHT_DIR="$RUNS_ROOT/sudoku-backbone-preflight-${SUITE_ID}"
SUMMARY_DIR="$RUNS_ROOT/sudoku-backbone-benchmark-${SUITE_ID}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This suite is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0
export BENCHMARK_STEPS="$STEPS"
export PERSIST_ROOT
export RUNS_ROOT
mkdir -p "$RUNS_ROOT"

"$REPO_ROOT/scripts/run_sudoku_baseline_preflight.sh" "$PREFLIGHT_DIR"

for backbone in rwkv gdn gdn2 kda; do
  run_name="sudoku-backbone-${backbone}-s${STEPS}-${SUITE_ID}"
  case "$backbone" in
    rwkv) RWKV_RUN="$run_name" ;;
    gdn) GDN_RUN="$run_name" ;;
    gdn2) GDN2_RUN="$run_name" ;;
    kda) KDA_RUN="$run_name" ;;
  esac
  "$REPO_ROOT/scripts/run_sudoku_backbone_benchmark.sh" "$backbone" "$run_name"
done

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
"$PYTHON_BIN" "$REPO_ROOT/scripts/summarize_sudoku_backbone_benchmark.py" \
  --repo "$REPO_ROOT" \
  --runs-root "$RUNS_ROOT" \
  --rwkv-run "$RWKV_RUN" \
  --gdn-run "$GDN_RUN" \
  --gdn2-run "$GDN2_RUN" \
  --kda-run "$KDA_RUN" \
  --preflight "$PREFLIGHT_DIR/backbone_contract.json" \
  --shared-shell-gate "$PREFLIGHT_DIR/shared_shell_initialization_gate.json" \
  --fla-gate "$PREFLIGHT_DIR/fla_kernel_gate.json" \
  --rwkv7-gate "$PREFLIGHT_DIR/rwkv7_official_frontend_gate.json" \
  --steps "$STEPS" \
  --out-dir "$SUMMARY_DIR"

printf 'suite_summary=%s\n' "$SUMMARY_DIR"
