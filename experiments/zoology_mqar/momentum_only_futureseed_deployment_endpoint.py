from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.canonical_address_companion_endpoint import (
    transition_summary,
)
from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    evaluate,
    fixed_batch_hash,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.momentum_only_futureseed import (
    FULL_STATE_VALUES_PER_ROUTE,
    MOMENTUM_PAYLOAD_VALUES_PER_ROUTE,
    momentum_only_futureseed_diagnostics,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CONTROL_ARM = "future_seed_momentum_delta"
CANDIDATE_ARM = "future_seed_momentum_only"
EXPECTED_CHECKPOINT_SHA256 = (
    "13410aaad3be26fbdf07c7a12e1afa9fb38ced41a1939a519c02973be2b8508f"
)
EXPECTED_FORMAL_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FORMAL_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)
EXPECTED_COMPONENT_DIAGNOSTIC_SHA256 = (
    "75f7ee388aee78139c9a261238e102b4cb2670df1dbdc8f0acdf46d921091f07"
)
MAX_NATIVE_REPLAY_DISAGREEMENTS = 2
MAX_NATIVE_ACCURACY_DRIFT = 0.001000001
MAX_NATIVE_CE_DRIFT = 1e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_identity() -> dict[str, Any]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU, found {rows}")
    index, uuid, name = (item.strip() for item in rows[0].split(",", 2))
    expected = (
        "0",
        os.environ["EXPECTED_GPU_UUID"],
        os.environ["EXPECTED_GPU_NAME"],
    )
    if (index, uuid, name) != expected:
        raise RuntimeError(f"GPU mismatch: {(index, uuid, name)} != {expected}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P-FS2-017 requires exactly one CUDA device")
    return {"index": 0, "uuid": uuid, "name": name}


def _metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "future_accuracy": float(metrics["future"]["accuracy"]),
        "past_accuracy": float(metrics["past"]["accuracy"]),
        "joint_exact": float(metrics["joint_exact"]),
        "future_ce": float(metrics["future"]["ce"]),
        "past_ce": float(metrics["past"]["ce"]),
    }


def _prediction_disagreements(
    reference: list[dict[str, Any]],
    observed: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference) != len(observed):
        raise RuntimeError("Native replay changed the number of cases")
    disagreements = []
    event_fields = (
        "direction",
        "query_position",
        "write_position",
        "distance",
        "key",
        "target",
    )
    for expected_case, observed_case in zip(reference, observed, strict=True):
        case_fields = ("case_index", "case_id", "sequence_length")
        if any(
            expected_case[field] != observed_case[field] for field in case_fields
        ):
            raise RuntimeError("Native replay case identity drifted")
        expected_events = expected_case["events"]
        observed_events = observed_case["events"]
        if len(expected_events) != len(observed_events):
            raise RuntimeError("Native replay changed the number of events")
        for expected_event, observed_event in zip(
            expected_events, observed_events, strict=True
        ):
            if any(
                expected_event[field] != observed_event[field]
                for field in event_fields
            ):
                raise RuntimeError("Native replay event identity drifted")
            if expected_event["prediction"] != observed_event["prediction"]:
                disagreements.append(
                    {
                        "case_index": expected_case["case_index"],
                        "direction": expected_event["direction"],
                        "query_position": expected_event["query_position"],
                        "expected": expected_event["prediction"],
                        "observed": observed_event["prediction"],
                    }
                )
    return {"count": len(disagreements), "rows": disagreements}


@torch.inference_mode()
def _benchmark_forward(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, Any],
    *,
    warmup_steps: int = 8,
    measured_steps: int = 40,
) -> dict[str, float]:
    model.eval()
    inputs = batch[0].cuda()
    for _ in range(warmup_steps):
        model(inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measured_steps):
        model(inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_sec": elapsed,
        "measured_steps": float(measured_steps),
        "examples_per_sec": inputs.shape[0] * measured_steps / elapsed,
        "tokens_per_sec": inputs.numel() * measured_steps / elapsed,
        "peak_cuda_mem_bytes": float(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-score", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--component-diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.formal_score: EXPECTED_FORMAL_SCORE_SHA256,
        args.formal_cases: EXPECTED_FORMAL_CASES_SHA256,
        args.component_diagnostic: EXPECTED_COMPONENT_DIAGNOSTIC_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")

    gpu = _gpu_identity()
    frozen = json.loads(args.formal_score.read_text())
    formal_control = frozen["candidate"]
    formal_cases = json.loads(args.formal_cases.read_text())
    component_diagnostic = json.loads(args.component_diagnostic.read_text())
    frozen_momentum_only = component_diagnostic["variants"]["momentum_only"]
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    control_config = build_config(
        arm=CONTROL_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    candidate_config = build_config(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    train_loader, test_loader = prepare_data(control_config.data)
    test_digest = dataset_hash(test_loader)
    if test_digest != formal_control["data_hashes"]["test"]:
        raise RuntimeError("P059 test data hash drifted")
    fixed_batch = next(iter(train_loader))
    if fixed_batch_hash(fixed_batch) != formal_control["warmup_batch_hash"]:
        raise RuntimeError("P059 fixed benchmark batch drifted")

    control = make_model(control_config, CONTROL_ARM).cuda().eval()
    control.load_state_dict(state_dict, strict=True)
    expected_parameter_hash = formal_control["trained_parameter_hash"]
    if parameter_hash(control) != expected_parameter_hash:
        raise RuntimeError("Frozen P059 parameter hash drifted")

    control_metrics, control_cases = evaluate(
        control, test_loader, sequence_length=SEQUENCE_LENGTH
    )
    control_metric_row = _metric_row(control_metrics)
    formal_metric_row = _metric_row(formal_control["metrics"])
    replay_predictions = _prediction_disagreements(formal_cases, control_cases)
    accuracy_fields = (
        "balanced_accuracy",
        "future_accuracy",
        "past_accuracy",
        "joint_exact",
    )
    ce_fields = ("future_ce", "past_ce")
    replay_calibration = {
        "formal_metrics": formal_metric_row,
        "observed_metrics": control_metric_row,
        "prediction_disagreements": replay_predictions,
        "max_accuracy_drift": max(
            abs(control_metric_row[field] - formal_metric_row[field])
            for field in accuracy_fields
        ),
        "max_ce_drift": max(
            abs(control_metric_row[field] - formal_metric_row[field])
            for field in ce_fields
        ),
    }
    replay_calibration["passed"] = (
        replay_predictions["count"] <= MAX_NATIVE_REPLAY_DISAGREEMENTS
        and replay_calibration["max_accuracy_drift"] <= MAX_NATIVE_ACCURACY_DRIFT
        and replay_calibration["max_ce_drift"] <= MAX_NATIVE_CE_DRIFT
    )
    if not replay_calibration["passed"]:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "native_replay_audit.json").write_text(
            json.dumps(replay_calibration, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError("Native P059 replay exceeded the registered R3 bound")

    # Construct the deployment model only after the frozen native endpoint has
    # passed. This keeps candidate lifecycle effects outside the replay gate.
    candidate = make_model(candidate_config, CANDIDATE_ARM).cuda().eval()
    candidate.load_state_dict(state_dict, strict=True)
    if parameter_hash(candidate) != expected_parameter_hash:
        raise RuntimeError("Momentum-only model did not load P059 exactly")

    candidate_metrics, candidate_cases = evaluate(
        candidate, test_loader, sequence_length=SEQUENCE_LENGTH
    )
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    control_swaps = wrong_key_swap_summary(control_cases)
    diagnostics = momentum_only_futureseed_diagnostics(candidate)

    benchmark_rows: dict[str, list[dict[str, float]]] = {
        "control": [],
        "candidate": [],
    }
    for label, model in (
        ("control", control),
        ("candidate", candidate),
        ("candidate", candidate),
        ("control", control),
    ):
        benchmark_rows[label].append(_benchmark_forward(model, fixed_batch))
    benchmark = {
        label: {
            "elapsed_sec_median": statistics.median(
                row["elapsed_sec"] for row in rows
            ),
            "examples_per_sec_median": statistics.median(
                row["examples_per_sec"] for row in rows
            ),
            "tokens_per_sec_median": statistics.median(
                row["tokens_per_sec"] for row in rows
            ),
            "peak_cuda_mem_bytes_max": max(
                row["peak_cuda_mem_bytes"] for row in rows
            ),
            "rounds": rows,
        }
        for label, rows in benchmark_rows.items()
    }
    time_ratio = (
        benchmark["candidate"]["elapsed_sec_median"]
        / benchmark["control"]["elapsed_sec_median"]
    )
    allocation_ratio = (
        benchmark["candidate"]["peak_cuda_mem_bytes_max"]
        / benchmark["control"]["peak_cuda_mem_bytes_max"]
    )

    cm = _metric_row(control_metrics)
    mm = _metric_row(candidate_metrics)
    frozen_mm = frozen_momentum_only["metrics"]
    integrity_checks = {
        "frozen_artifacts_exact_and_native_replay_calibrated": (
            replay_calibration["passed"]
        ),
        "parameters_exact": parameter_hash(candidate) == parameter_hash(control),
        "transport_exactly_halved": (
            diagnostics["full_state_values_per_route"]
            == FULL_STATE_VALUES_PER_ROUTE
            and diagnostics["transported_values_per_route"]
            == MOMENTUM_PAYLOAD_VALUES_PER_ROUTE
            and diagnostics["transport_ratio"] == 0.5
        ),
        "state_parameters_and_scans_unchanged": (
            diagnostics["parameter_delta_vs_p059"] == 0
            and diagnostics["persistent_state_delta_vs_p059"] == 0
            and diagnostics["scan_delta_vs_p059"] == 0
        ),
    }
    route = diagnostics["per_route"][1]
    activation_checks = {
        "one_momentum_only_route": (
            diagnostics["active_layers"] == 2
            and diagnostics["active_futureseed_routes"] == 1
        ),
        "receiver_state_zero_and_momentum_finite": (
            route["seed_state_rms"] == 0.0
            and route["seed_momentum_rms"] is not None
            and math.isfinite(float(route["seed_momentum_rms"]))
            and float(route["seed_momentum_rms"]) >= 1e-4
        ),
    }
    quality_checks = {
        "balanced_within_0.005": (
            mm["balanced_accuracy"] >= cm["balanced_accuracy"] - 0.005
        ),
        "directions_within_0.005": (
            mm["future_accuracy"] >= cm["future_accuracy"] - 0.005
            and mm["past_accuracy"] >= cm["past_accuracy"] - 0.005
        ),
        "joint_within_0.01": mm["joint_exact"] >= cm["joint_exact"] - 0.01,
        "error_budget": candidate_swaps["errors"] <= 243,
        "wrong_key_swap_budget": (
            candidate_swaps["wrong_key_valid_value_swaps"] <= 171
        ),
        "frozen_momentum_only_accuracy_reproduced": all(
            abs(mm[field] - float(frozen_mm[field])) <= MAX_NATIVE_ACCURACY_DRIFT
            for field in accuracy_fields
        ),
        "frozen_momentum_only_ce_reproduced": all(
            abs(mm[field] - float(frozen_mm[field])) <= MAX_NATIVE_CE_DRIFT
            for field in ce_fields
        ),
    }
    cost_checks = {
        "inference_elapsed_below_1.05x": time_ratio < 1.05,
        "allocation_below_1.02x": allocation_ratio < 1.02,
    }
    passed = all(
        all(group.values())
        for group in (
            integrity_checks,
            activation_checks,
            quality_checks,
            cost_checks,
        )
    )
    result = {
        "status": "complete",
        "decision": "pareto_futureseed_deployment" if passed else "closed",
        "plan": "P-FS2-017",
        "claim_boundary": (
            "Train with native [S,M]; deploy the frozen model with M-only "
            "cross-layer FutureSeed. This is not an M-only training claim."
        ),
        "gpu": gpu,
        "protocol": {
            "training": False,
            "same_trained_weights": True,
            "checkpoint_sha256": _sha256(args.checkpoint),
            "formal_score_sha256": _sha256(args.formal_score),
            "formal_cases_sha256": _sha256(args.formal_cases),
            "component_diagnostic_sha256": _sha256(args.component_diagnostic),
            "test_data_sha256": test_digest,
            "benchmark_order": ["control", "candidate", "candidate", "control"],
        },
        "native_replay_calibration": replay_calibration,
        "control": {"metrics": cm, "swap_summary": control_swaps},
        "candidate": {
            "metrics": mm,
            "swap_summary": candidate_swaps,
            "momentum_only": diagnostics,
        },
        "paired_error_transitions": transition_summary(
            control_cases, candidate_cases
        ),
        "benchmark": benchmark,
        "cost_ratios": {
            "inference_elapsed": time_ratio,
            "peak_allocation": allocation_ratio,
        },
        "registered_gate": {
            "integrity_checks": integrity_checks,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
            "passed": passed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "control_cases.json").write_text(
        json.dumps(control_cases, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "cases.json").write_text(
        json.dumps(candidate_cases, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
