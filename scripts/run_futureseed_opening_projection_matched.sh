#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM_SCRIPT="$REPO_ROOT/scripts/run_futureseed_opening_projection_arm.sh"
CHECKER="$REPO_ROOT/experiments/rwkv_fs_sudoku/check_futureseed_opening_projection_cuda.py"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
PYTHON_BIN="${PYTHON_BIN:-$PERSIST_ROOT/official_eqr_compare/.venv/bin/python}"
EXPECTED_UUID="${EXPECTED_UUID:?P-LOOP-002 requires EXPECTED_UUID for the admitted GPU}"
PUSHED_REF="${PUSHED_REF:?P-LOOP-002 requires PUSHED_REF for exact remote readback}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-LOOP-002 matched run requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 2
fi
if [[ ! -x "$ARM_SCRIPT" || ! -f "$CHECKER" ]]; then
  printf 'P-LOOP-002 arm launcher is not executable: %s\n' "$ARM_SCRIPT" >&2
  exit 3
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 3
fi
if [[ ! -r "$PERSIST_ROOT/.cache/fla-source-sha" ]] || \
  [[ "$(<"$PERSIST_ROOT/.cache/fla-source-sha")" != "9c8e42e762fce087c27b673af4922795d9edb85e" ]]; then
  printf 'P-LOOP-002 pinned FLA marker is missing or mismatched.\n' >&2
  exit 3
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null || \
  [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'P-LOOP-002 matched run requires a clean detached worktree.\n' >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0 PERSIST_ROOT RUNS_ROOT PYTHON_BIN EXPECTED_UUID
export FLA_EXPECTED_SOURCE_SHA=9c8e42e762fce087c27b673af4922795d9edb85e
export FLA_SOURCE_SHA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
unset FLA_SOURCE_ROOT
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
REMOTE_SHA="$(
  timeout 30 git -C "$REPO_ROOT" ls-remote --refs "$REMOTE_NAME" "$PUSHED_REF" \
    | awk 'NR == 1 {print $1}'
)"
if [[ -z "$REMOTE_SHA" || "$REMOTE_SHA" != "$GIT_SHA" ]]; then
  printf 'P-LOOP-002 remote readback mismatch: HEAD=%s %s=%s\n' \
    "$GIT_SHA" "$PUSHED_REF" "${REMOTE_SHA:-missing}" >&2
  exit 4
fi
TIMESTAMP="${MATCH_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATCH_ID="${MATCH_ID:-p-loop-002-matched-${TIMESTAMP}-${GIT_SHA:0:7}}"
PROBE_NAME="${PROBE_RUN_NAME:-${MATCH_ID}-probe-s3001}"
CONTROL_NAME="${CONTROL_RUN_NAME:-${MATCH_ID}-canonical-s3100}"
CANDIDATE_NAME="${CANDIDATE_RUN_NAME:-${MATCH_ID}-opening-projection-s3100}"
COMPARISON_DIR="${COMPARISON_DIR:-$RUNS_ROOT/${MATCH_ID}-comparison}"
CONTRACT_DIR="${CONTRACT_DIR:-$PERSIST_ROOT/artifacts/p-loop-002/$MATCH_ID}"
CONTRACT_JSON="$CONTRACT_DIR/cuda_contract.json"
CONTRACT_LOG="$CONTRACT_DIR/cuda_contract.log"
STATUS_PATH="$CONTRACT_DIR/sequence.status"
ABORT_PATH="$CONTRACT_DIR/abort.json"
PHASE=preflight

on_exit() {
  status=$?
  printf '%s\n' "$status" > "$STATUS_PATH"
  if [[ "$status" -ne 0 ]]; then
    pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
    "$PYTHON_BIN" - "$ABORT_PATH" "$status" "$PHASE" "$GIT_SHA" \
      "$EXPECTED_UUID" "$$" "$pgid" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path, status, phase, sha, uuid, pid, pgid = sys.argv[1:]
pathlib.Path(path).write_text(
    json.dumps(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "plan_id": "P-LOOP-002",
            "status": int(status),
            "phase": phase,
            "scientific_failure": False,
            "source_sha": sha,
            "gpu_uuid": uuid,
            "wrapper_pid": int(pid),
            "wrapper_pgid": int(pgid),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  fi
}
mkdir -p "$CONTRACT_DIR"
trap on_exit EXIT

for path in \
  "$RUNS_ROOT/$PROBE_NAME" "$RUNS_ROOT/$CONTROL_NAME" \
  "$RUNS_ROOT/$CANDIDATE_NAME" "$COMPARISON_DIR"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to reuse P-LOOP-002 matched path: %s\n' "$path" >&2
    exit 5
  fi
done

mkdir -p \
  "${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache/p-loop-002/xdg}" \
  "${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/p-loop-002/triton}" \
  "${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/p-loop-002/torchinductor}" \
  "${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/p-loop-002/torch_extensions}" \
  "${TMPDIR:-$PERSIST_ROOT/.cache/p-loop-002/tmp}"
PHASE=cuda_contract
EXPECTED_GPU_UUID="$EXPECTED_UUID" \
CONTRACT_OUTPUT="$CONTRACT_JSON" \
FLA_DISABLE_BACKEND_DISPATCH=1 \
FLA_CONV_BACKEND=triton \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$REPO_ROOT/experiments/rwkv_fs_sudoku:$PERSIST_ROOT/.cache/fla-active:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}" \
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache/p-loop-002/xdg}" \
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/p-loop-002/triton}" \
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/p-loop-002/torchinductor}" \
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/p-loop-002/torch_extensions}" \
TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/p-loop-002/tmp}" \
  "$PYTHON_BIN" "$CHECKER" > "$CONTRACT_LOG" 2>&1
PHASE=contract_verification

"$PYTHON_BIN" - "$CONTRACT_JSON" "$EXPECTED_UUID" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = payload.get("identity", {})
topology = payload.get("gate_topology_and_backward", {})
checks = {
    "status": payload.get("status") == "passed",
    "gpu_uuid": payload.get("gpu_uuid") == sys.argv[2],
    "fla_source": payload.get("fla_source_sha")
    == "9c8e42e762fce087c27b673af4922795d9edb85e",
    "zero_inference_delta": identity.get("forward_exact_identity") is True,
    "state_identity": identity.get("state_dict_exact_identity") is True,
    "official_backward": topology.get("chunk_gdn2_backward_count") == 12,
    "active_receivers": topology.get("active_receiving_tensor_count") == 11,
    "active_parameters": topology.get("active_receiving_parameter_count") == 88,
}
if not all(checks.values()):
    raise SystemExit(f"P-LOOP-002 strict CUDA contract failed: {checks}")
print(json.dumps({"contract": "passed", "checks": checks}, sort_keys=True))
PY

printf 'P-LOOP-002 sequence: probe -> canonical control -> opening projection.\n'
PHASE=production_probe
RUN_NAME="$PROBE_NAME" "$ARM_SCRIPT" opening_projection probe

PROBE_RESULT="$RUNS_ROOT/$PROBE_NAME/output/futureseed_loop_seed52.json"
"$PYTHON_BIN" - "$PROBE_RESULT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
projection = payload["metrics"]["train"]["future_seed_gradient_projection"]
checks = {
    "steps": projection.get("steps") == 1,
    "active_tensor_count": projection.get("active_tensor_count") == 11,
    "active_parameter_count": projection.get("active_parameter_count") == 88,
    "inactive_tensor_count": projection.get("inactive_tensor_count") == 1,
    "norm_relative_error": projection.get("norm_relative_error_max", 1) < 1e-5,
    "global_norm_relative_error": projection.get(
        "final_global_grad_norm_relative_error_max", 1
    ) < 2e-5,
    "post_dot": projection.get("post_opening_continuation_dot_min", -1) >= -1e-6,
}
if not all(checks.values()):
    raise SystemExit(f"P-LOOP-002 step3001 production probe failed: {checks}")
print(json.dumps({"probe": "passed", "projection": projection}, sort_keys=True))
PY

PHASE=canonical_control
RUN_NAME="$CONTROL_NAME" "$ARM_SCRIPT" canonical formal
PHASE=opening_projection_candidate
RUN_NAME="$CANDIDATE_NAME" "$ARM_SCRIPT" opening_projection formal

CONTROL_RESULT="$RUNS_ROOT/$CONTROL_NAME/output/futureseed_loop_seed52.json"
CANDIDATE_RESULT="$RUNS_ROOT/$CANDIDATE_NAME/output/futureseed_loop_seed52.json"
CONTROL_CHECKPOINT="$PERSIST_ROOT/models/$CONTROL_NAME/checkpoints/train_state_step003100.pt"
CANDIDATE_CHECKPOINT="$PERSIST_ROOT/models/$CANDIDATE_NAME/checkpoints/train_state_step003100.pt"
mkdir -p "$COMPARISON_DIR"
PHASE=matched_decision

"$PYTHON_BIN" - \
  "$CONTROL_RESULT" "$CANDIDATE_RESULT" \
  "$CONTROL_CHECKPOINT" "$CANDIDATE_CHECKPOINT" \
  "$CONTRACT_JSON" "$COMPARISON_DIR/decision.json" \
  "$GIT_SHA" "$EXPECTED_UUID" "$PUSHED_REF" <<'PY'
from __future__ import annotations

import hashlib
import html
import json
import math
import pathlib
import statistics
import sys
from datetime import datetime, timezone

(
    control_path,
    candidate_path,
    control_checkpoint,
    candidate_checkpoint,
    contract_path,
    output_path,
    git_sha,
    gpu_uuid,
    pushed_ref,
) = sys.argv[1:]


def load(path: str):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_blank(payload, label: str, loop: int):
    row = payload["metrics"]["official_eval_by_blank_range"][label]["eval_clean"][f"loop{loop}"]
    return float(row["label_exact"]), float(row["blank_acc"])


def load_cases(payload, label: str):
    row = payload["metrics"]["case_bank"]["holes"][f"official_{label}"]
    path = pathlib.Path(row["all_cases_json"])
    if not path.is_file():
        raise SystemExit(f"missing all-cases export: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def wrong_correction(cases, first: int = 3, last: int = 5) -> float:
    rows = cases["cases"]
    return statistics.fmean(
        float(row["loops"][f"loop{first}"]["wrong_total_count"])
        - float(row["loops"][f"loop{last}"]["wrong_total_count"])
        for row in rows
    )


def board_html(values, labels, clues, title):
    cells = []
    for value, target, clue in zip(values, labels, clues):
        css = "clue" if clue else "correct" if value == target else "wrong"
        cells.append(f'<span class="{css}">{int(value)}</span>')
    return (
        f'<figure><figcaption>{html.escape(title)}</figcaption>'
        f'<div class="board">{"".join(cells)}</div></figure>'
    )


def hardest_case_html(label, control_cases, candidate_cases):
    control_by_id = {row["case_id"]: row for row in control_cases["cases"]}
    candidate_by_id = {row["case_id"]: row for row in candidate_cases["cases"]}
    shared = sorted(control_by_id.keys() & candidate_by_id.keys())
    case_id = max(
        shared,
        key=lambda key: (
            max(
                control_by_id[key]["loops"]["loop5"]["wrong_total_count"],
                candidate_by_id[key]["loops"]["loop5"]["wrong_total_count"],
            ),
            abs(
                candidate_by_id[key]["loops"]["loop5"]["wrong_total_count"]
                - control_by_id[key]["loops"]["loop5"]["wrong_total_count"]
            ),
        ),
    )
    control_case = control_by_id[case_id]
    candidate_case = candidate_by_id[case_id]
    boards = [
        board_html(
            control_case["label"],
            control_case["label"],
            control_case["clue_mask"],
            "solution",
        )
    ]
    for arm, case in (("canonical", control_case), ("opening projection", candidate_case)):
        for loop in range(1, 6):
            row = case["loops"][f"loop{loop}"]
            boards.append(
                board_html(
                    row["prediction"],
                    case["label"],
                    case["clue_mask"],
                    f"{arm} loop{loop} wrong={row['wrong_total_count']}",
                )
            )
    return (
        f'<section><h2>{html.escape(label)} hardest shared board</h2>'
        f'<p><code>{html.escape(case_id)}</code></p>'
        f'<div class="boards">{"".join(boards)}</div></section>'
    )


control = load(control_path)
candidate = load(candidate_path)
contract_evidence = load(contract_path)
if control["args"].get("future_seed_gradient_mode") != "canonical":
    raise SystemExit("control is not canonical")
if candidate["args"].get("future_seed_gradient_mode") != "opening_projection":
    raise SystemExit("candidate is not opening_projection")
ignored_matched_args = {
    "future_seed_gradient_mode",
    "resume_allow_future_seed_gradient_upgrade",
    "out_dir",
    "train_checkpoint_dir",
}
control_args = {
    key: value for key, value in control["args"].items()
    if key not in ignored_matched_args
}
candidate_args = {
    key: value for key, value in candidate["args"].items()
    if key not in ignored_matched_args
}
if control_args != candidate_args:
    changed = sorted(
        key for key in control_args.keys() | candidate_args.keys()
        if control_args.get(key) != candidate_args.get(key)
    )
    raise SystemExit(f"matched arm argument drift: {changed}")
for arm, payload in (("control", control), ("candidate", candidate)):
    if payload["args"].get("seed") != 52:
        raise SystemExit("matched seed drift")
    if payload["args"].get("steps") != 3100:
        raise SystemExit("matched endpoint drift")
    contract = payload["metrics"]["train"]["resume_train_checkpoint"]["semantic_contract"]
    resume = payload["metrics"]["train"]["resume_train_checkpoint"]
    if (
        not contract.get("matched")
        or contract.get("saved_at_step") != 3000
        or resume.get("saved_at_step") != 3000
    ):
        raise SystemExit("matched exact-resume evidence missing")
    accepted_upgrade = bool(contract.get("accepted_future_seed_gradient_upgrade"))
    if accepted_upgrade != (arm == "candidate"):
        raise SystemExit(f"{arm} gradient-upgrade acceptance mismatch")
if (
    control["metrics"]["train"].get("parameter_count")
    != candidate["metrics"]["train"].get("parameter_count")
    or control["metrics"]["train"].get("trainable_parameter_count")
    != candidate["metrics"]["train"].get("trainable_parameter_count")
):
    raise SystemExit("training-only candidate changed parameter topology")

ranges = ("b51_55", "b56_60", "b61_64")
official = {}
for label in ranges:
    official[label] = {"control": {}, "candidate": {}}
    for loop in range(1, 6):
        for arm, payload in (("control", control), ("candidate", candidate)):
            exact, blank = exact_blank(payload, label, loop)
            official[label][arm][f"loop{loop}"] = {"exact": exact, "blank": blank}

hard_control = statistics.fmean(official[label]["control"]["loop5"]["exact"] for label in ranges)
hard_candidate = statistics.fmean(official[label]["candidate"]["loop5"]["exact"] for label in ranges)
blank_deltas = {
    label: official[label]["candidate"]["loop5"]["blank"]
    - official[label]["control"]["loop5"]["blank"]
    for label in ranges
}
mixed_control = float(control["metrics"]["eval_clean"]["loop5"]["label_exact"])
mixed_candidate = float(candidate["metrics"]["eval_clean"]["loop5"]["label_exact"])

case_metrics = {}
case_payloads = {}
for label in ranges:
    control_cases = load_cases(control, label)
    candidate_cases = load_cases(candidate, label)
    if control_cases["data_hash"] != candidate_cases["data_hash"]:
        raise SystemExit(f"same-board data hash mismatch for {label}")
    control_ids = [row["case_id"] for row in control_cases["cases"]]
    candidate_ids = [row["case_id"] for row in candidate_cases["cases"]]
    if control_ids != candidate_ids:
        raise SystemExit(f"same-board ordering mismatch for {label}")
    control_correction = wrong_correction(control_cases)
    candidate_correction = wrong_correction(candidate_cases)
    case_metrics[label] = {
        "data_hash": control_cases["data_hash"],
        "control_loop3_to5_wrong_correction": control_correction,
        "candidate_loop3_to5_wrong_correction": candidate_correction,
        "delta": candidate_correction - control_correction,
    }
    case_payloads[label] = (control_cases, candidate_cases)

projection = candidate["metrics"]["train"]["future_seed_gradient_projection"]
activation_checks = {
    "activation_rate_ge_0_10": float(projection.get("activation_rate", 0)) >= 0.10,
    "active_removed_opening_fraction_ge_0_05": float(
        projection.get("active_removed_opening_fraction_mean", 0)
    ) >= 0.05,
    "post_dot_ge_neg_1e_6": float(
        projection.get("post_opening_continuation_dot_min", -math.inf)
    ) >= -1e-6,
    "norm_relative_error_lt_1e_5": float(
        projection.get("norm_relative_error_max", math.inf)
    ) < 1e-5,
    "exact_11_active": (
        float(projection.get("active_tensor_count", 0)) == 11
        and float(projection.get("active_parameter_count", 0)) == 88
        and float(projection.get("inactive_tensor_count", 0)) == 1
    ),
}

control_train = control["metrics"]["train"]
candidate_train = candidate["metrics"]["train"]
control_elapsed = float(control_train["train_sec"])
candidate_elapsed = float(candidate_train["train_sec"])
control_alloc = float(control_train["cuda_max_memory_allocated_mb"])
candidate_alloc = float(candidate_train["cuda_max_memory_allocated_mb"])
elapsed_overhead = candidate_elapsed / control_elapsed - 1.0
allocated_overhead = candidate_alloc / control_alloc - 1.0
cost_checks = {
    "elapsed_overhead_lt_0_25": elapsed_overhead < 0.25,
    "peak_alloc_overhead_lt_0_10": allocated_overhead < 0.10,
    "zero_inference_delta": (
        contract_evidence.get("status") == "passed"
        and contract_evidence.get("identity", {}).get("forward_exact_identity") is True
        and contract_evidence.get("identity", {}).get("state_dict_exact_identity") is True
    ),
}

primary_checks = {
    "hard_macro_exact_delta_ge_0_02": hard_candidate - hard_control >= 0.02,
    "all_blank_regressions_le_0_01": min(blank_deltas.values()) >= -0.01,
}
control_6164_exact = official["b61_64"]["control"]["loop5"]["exact"]
candidate_6164_exact = official["b61_64"]["candidate"]["loop5"]["exact"]
alternate_checks = {
    "mixed_exact_delta_ge_0_03": mixed_candidate - mixed_control >= 0.03,
    "61_64_exact_non_regression": candidate_6164_exact >= control_6164_exact,
    "61_64_blank_regression_le_0_005": blank_deltas["b61_64"] >= -0.005,
    "61_64_late_correction_delta_ge_0_5": case_metrics["b61_64"]["delta"] >= 0.5,
    "56_60_late_correction_not_weaker": case_metrics["b56_60"]["delta"] >= 0.0,
}
primary_pass = all(primary_checks.values())
alternate_pass = all(alternate_checks.values())
accepted = (
    all(activation_checks.values())
    and all(cost_checks.values())
    and (primary_pass or alternate_pass)
)

payload = {
    "status": "done" if accepted else "discarded",
    "decision": "keep" if accepted else "close_without_rescue",
    "plan_id": "P-LOOP-002",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_sha": git_sha,
    "pushed_ref": pushed_ref,
    "gpu_uuid": gpu_uuid,
    "control": control_path,
    "candidate": candidate_path,
    "hashes": {
        "control_metrics_sha256": sha256(control_path),
        "candidate_metrics_sha256": sha256(candidate_path),
        "control_checkpoint_sha256": sha256(control_checkpoint),
        "candidate_checkpoint_sha256": sha256(candidate_checkpoint),
        "cuda_contract_sha256": sha256(contract_path),
    },
    "activation": projection,
    "activation_checks": activation_checks,
    "official": official,
    "hard_macro_loop5_exact": {
        "control": hard_control,
        "candidate": hard_candidate,
        "delta": hard_candidate - hard_control,
    },
    "official_loop5_blank_deltas": blank_deltas,
    "mixed_loop5_exact": {
        "control": mixed_control,
        "candidate": mixed_candidate,
        "delta": mixed_candidate - mixed_control,
    },
    "same_board": case_metrics,
    "primary_checks": primary_checks,
    "alternate_checks": alternate_checks,
    "primary_pass": primary_pass,
    "alternate_pass": alternate_pass,
    "cost": {
        "control_elapsed_sec": control_elapsed,
        "candidate_elapsed_sec": candidate_elapsed,
        "elapsed_overhead": elapsed_overhead,
        "control_peak_allocated_mb": control_alloc,
        "candidate_peak_allocated_mb": candidate_alloc,
        "peak_allocated_overhead": allocated_overhead,
        "inference_delta": 0,
    },
    "cost_checks": cost_checks,
    "accepted": accepted,
    "no_rescue": not accepted,
}
path = pathlib.Path(output_path)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
case_sections = "".join(
    hardest_case_html(label, *case_payloads[label]) for label in ranges
)
html_path = path.with_name("comparison.html")
html_path.write_text(
    f"""<!doctype html>
<html><head><meta charset="utf-8"><title>P-LOOP-002 matched comparison</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f8fafc;color:#111827}}
.boards{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0;padding:10px;background:white;border:1px solid #cbd5e1}}
figcaption{{font-size:12px;margin-bottom:6px}}.board{{display:grid;grid-template-columns:repeat(9,22px);border:2px solid #111827}}
.board span{{width:22px;height:22px;display:grid;place-items:center;border-right:1px solid #94a3b8;border-bottom:1px solid #94a3b8;font-size:12px;font-weight:700}}
.clue{{background:#e2e8f0}}.correct{{background:#bbf7d0;color:#14532d}}.wrong{{background:#fecaca;color:#7f1d1d}}
section{{margin:28px 0}}code{{overflow-wrap:anywhere}}
</style></head><body><h1>P-LOOP-002 FutureSeed opening-gradient projection</h1>
<p>Canonical and candidate use the same source, parent, boards and five-loop objective.</p>
<ul><li>hard macro delta: {hard_candidate-hard_control:+.6f}</li>
<li>mixed delta: {mixed_candidate-mixed_control:+.6f}</li>
<li>decision: {payload['decision']}</li></ul>{case_sections}</body></html>""",
    encoding="utf-8",
)
print(json.dumps({
    "decision": payload["decision"],
    "hard_macro_delta": hard_candidate - hard_control,
    "mixed_delta": mixed_candidate - mixed_control,
    "elapsed_overhead": elapsed_overhead,
    "peak_allocated_overhead": allocated_overhead,
}, sort_keys=True))
PY

PHASE=complete
printf 'P-LOOP-002 decision: %s\n' "$COMPARISON_DIR/decision.json"
