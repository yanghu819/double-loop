#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/huyang2/double-loop/official_eqr_sudoku_repro_20260623}"
REPO="${REPO:-${BASE}/eqr-official}"
VENV="${VENV:-/huyang2/double-loop/official_eqr_compare/.venv}"
CKPT="${CKPT:-${BASE}/downloaded_checkpoints/sudoku-extreme/eqr.pth}"
OUT="${OUT:-${BASE}/artifacts/sudoku_quick5_20260623}"
SEEDS="${SEEDS:-1 2 3 4}"
STAMP="${STAMP:-20260623_quick5}"

mkdir -p "${OUT}/logs" "${OUT}/metrics"

cd "${REPO}"
. "${VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export DISABLE_COMPILE=1

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[config] BASE=${BASE}"
echo "[config] REPO=${REPO}"
echo "[config] CKPT=${CKPT}"
echo "[config] SEEDS=${SEEDS}"
sha256sum "${CKPT}" | tee "${OUT}/checkpoint.sha256"
git rev-parse HEAD | tee "${OUT}/eqr_official.sha"

for seed in ${SEEDS}; do
  if [[ -s "${OUT}/metrics/seed${seed}.json" ]]; then
    echo "[seed ${seed}] skip existing ${OUT}/metrics/seed${seed}.json $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}/queue.log"
    continue
  fi
  suffix="sudoku_lite_D64_B128_N0.5_S1_seed${seed}_${STAMP}"
  log="${OUT}/logs/seed${seed}.log"
  echo "[seed ${seed}] launch $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}/queue.log"
  python evaluate.py \
    eval_yaml=config/eval/sudoku_lite_noise05.yaml \
    checkpoint="${CKPT}" \
    seed="${seed}" \
    suffix="${suffix}" \
    force_rerun=true \
    >"${log}" 2>&1
  metric_path="$(find "${BASE}/downloaded_checkpoints/eval_preds" -path "*seed${seed}_${STAMP}*/eval_metrics_step_50000.json" -print | sort | tail -n 1)"
  if [[ -z "${metric_path}" ]]; then
    echo "[seed ${seed}] metrics not found" | tee -a "${OUT}/queue.log"
    exit 3
  fi
  cp "${metric_path}" "${OUT}/metrics/seed${seed}.json"
  echo "[seed ${seed}] done metric=${metric_path} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${OUT}/queue.log"
done

missing=()
for seed in 1 2 3 4; do
  if [[ ! -s "${OUT}/metrics/seed${seed}.json" ]]; then
    missing+=("${seed}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "[summary] missing seed metrics: ${missing[*]}; rerun with SEEDS=\"${missing[*]}\" after GPU restart" | tee -a "${OUT}/queue.log"
  exit 0
fi

python - <<'PY' "${OUT}" "${BASE}"
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
base = Path(sys.argv[2])
metric_dir = out / "metrics"
seed0_candidates = sorted((base / "downloaded_checkpoints" / "eval_preds").glob("*seed0_rerun_20260623T081821Z*/eval_metrics_step_50000.json"))
if not seed0_candidates:
    raise SystemExit("seed0 metrics not found")
seed_files = [(0, seed0_candidates[-1])] + [(seed, metric_dir / f"seed{seed}.json") for seed in range(1, 5)]
keys = [
    "convergence_top_k/cumulative_exact_acc_top1",
    "convergence_top_k/exact_accuracy",
    "majority_vote/exact_accuracy",
    "different_init/any_correct",
]
rows = []
for seed, path in seed_files:
    data = json.loads(path.read_text())
    metrics = data.get("metrics", data)
    row = {"seed": seed, "path": str(path)}
    for key in keys:
        row[key] = float(metrics[key])
    rows.append(row)

summary = {
    "seeds": [row["seed"] for row in rows],
    "n": len(rows),
    "metrics": {},
    "rows": rows,
}
for key in keys:
    vals = [row[key] for row in rows]
    summary["metrics"][key] = {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["seed", *keys, "path"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
PY

echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
