from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import (
    HISTORICAL_BALANCED,
    HISTORICAL_CASES_SHA256,
    HISTORICAL_FUTURE_ACCURACY,
    HISTORICAL_JOINT_EXACT,
    HISTORICAL_PAST_ACCURACY,
    HISTORICAL_SCORE_SHA256,
    HISTORICAL_SWAP_FRACTION,
    load_historical,
    wrong_key_swap_summary,
)
from experiments.zoology_mqar.length_scaling import dataset_hash, evaluate, run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MODEL_WIDTH = 128
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_KEY_DIM = 32
HEAD_VALUE_DIM = 32
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
CONTROL_ARM = "future_seed_gdn2"
CONTROL_OUTPUT_ARM = "future_seed_gdn2_runtime_control"
CANDIDATE_ARM = "future_seed_gdn2_surprise_regression"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(candidate: float, control: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(control) or control <= 0:
        raise ValueError("Matched cost values must be finite and positive")
    return candidate / control


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_at_least(value: Any, floor: float) -> bool:
    return _finite(value) and float(value) >= floor


def _diagnostics(candidate: dict[str, Any]) -> dict[str, Any]:
    diagnostics = candidate["surprise_regression"]
    rows = diagnostics["per_receiver"]
    if not isinstance(rows, list):
        raise TypeError("surprise_regression.per_receiver must be a list")
    return diagnostics


def _case_label_signatures(cases: list[dict[str, Any]]) -> dict[str, tuple[Any, ...]]:
    signatures = {
        case["case_id"]: tuple(
            sorted(
                (
                    event["direction"],
                    event["query_position"],
                    event["write_position"],
                    event["key"],
                    event["target"],
                )
                for event in case["events"]
            )
        )
        for case in cases
    }
    if len(signatures) != len(cases):
        raise RuntimeError("Directional MQAR case IDs are not unique")
    return signatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-reference-run", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError(
            "P-FS2-008 fixes one sequential native-control/candidate pair at "
            "10 epochs and batch32"
        )

    historical = load_historical(args.historical_reference_run.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control = run_arm(
        arm=CONTROL_ARM,
        output_arm_name=CONTROL_OUTPUT_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        model_width=MODEL_WIDTH,
        model_heads=MODEL_HEADS,
        gdn2_head_dim=HEAD_KEY_DIM,
        gdn2_expand_v=HEAD_VALUE_DIM / HEAD_KEY_DIM,
        save_checkpoint=True,
    )
    control_root = args.output_dir / "length_1024" / CONTROL_OUTPUT_ARM
    control_cases_path = control_root / "cases.json"
    control_cases = json.loads(control_cases_path.read_text())
    control_swaps = wrong_key_swap_summary(control_cases)
    control_admission_checks = {
        "fixed_native_control_protocol": (
            control["arm"] == CONTROL_OUTPUT_ARM
            and control["carrier_arm"] == CONTROL_ARM
            and control["sequence_length"] == SEQUENCE_LENGTH
            and control["num_kv_pairs"] == NUM_KV_PAIRS
            and control["model_width"] == MODEL_WIDTH
            and control["model_heads"] == MODEL_HEADS
            and control["gdn2_head_dim"] == HEAD_KEY_DIM
            and control["epochs"] == MAX_EPOCHS
        ),
        "historical_directional_data_exact": (
            control["data_hashes"] == historical["score"]["data_hashes"]
        ),
        "native_parameter_count_exact": control["parameters"] == 661_584,
        "native_state_budget_exact": (
            control["recurrent_state_values_per_layer"] == 4_096
        ),
        "native_futureseed_route_active": (
            control["future_seed"]["active_seed_routes"] == MODEL_LAYERS - 1
        ),
    }
    control_admission = {
        "passed": all(control_admission_checks.values()),
        "checks": control_admission_checks,
        "control": control,
        "control_swap_summary": control_swaps,
    }
    (args.output_dir / "control_admission.json").write_text(
        json.dumps(control_admission, indent=2, sort_keys=True) + "\n"
    )
    if not control_admission["passed"]:
        raise RuntimeError("Locked same-runtime native control admission failed")

    (args.output_dir / "candidate_started.json").write_text(
        json.dumps(
            {
                "plan": "P-FS2-008",
                "arm": CANDIDATE_ARM,
                "control_checkpoint_sha256": control["checkpoint_sha256"],
                "control_score_sha256": _sha256(control_root / "score.json"),
                "control_cases_sha256": _sha256(control_cases_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    candidate = run_arm(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        model_width=MODEL_WIDTH,
        model_heads=MODEL_HEADS,
        gdn2_head_dim=HEAD_KEY_DIM,
        gdn2_expand_v=HEAD_VALUE_DIM / HEAD_KEY_DIM,
        save_checkpoint=True,
    )
    candidate_root = args.output_dir / "length_1024" / CANDIDATE_ARM
    candidate_cases_path = candidate_root / "cases.json"
    candidate_cases = json.loads(candidate_cases_path.read_text())
    candidate_swaps = wrong_key_swap_summary(candidate_cases)
    diagnostics = _diagnostics(candidate)
    rows = diagnostics["per_receiver"]

    # Same trained weights, same test set, with only the candidate residual edge
    # disabled. This measures the actual labeled-query consequence of the edge.
    from experiments.zoology_mqar.length_scaling import build_config, make_model
    from zoology.data.utils import prepare_data
    import torch

    counterfactual_config = build_config(
        arm=CANDIDATE_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    _unused_train_loader, counterfactual_test_loader = prepare_data(
        counterfactual_config.data
    )
    counterfactual_test_hash = dataset_hash(counterfactual_test_loader)
    counterfactual_model = make_model(counterfactual_config, CANDIDATE_ARM).cuda()
    checkpoint_path = Path(candidate["checkpoint_path"])
    if _sha256(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise RuntimeError("Candidate checkpoint hash changed before counterfactual")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    if (
        checkpoint.get("arm") != CANDIDATE_ARM
        or checkpoint.get("carrier_arm") != CANDIDATE_ARM
    ):
        raise RuntimeError("Candidate checkpoint arm metadata is inconsistent")
    counterfactual_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    counterfactual_model.backbone.force_native_future_seed = True
    native_counterfactual_metrics, native_counterfactual_cases = evaluate(
        counterfactual_model,
        counterfactual_test_loader,
        sequence_length=SEQUENCE_LENGTH,
    )
    native_counterfactual_swaps = wrong_key_swap_summary(
        native_counterfactual_cases
    )
    candidate_case_labels = _case_label_signatures(candidate_cases)
    counterfactual_case_labels = _case_label_signatures(
        native_counterfactual_cases
    )
    del counterfactual_model
    torch.cuda.empty_cache()

    integrity_checks = {
        "fixed_candidate_protocol": (
            candidate["arm"] == CANDIDATE_ARM
            and candidate["carrier_arm"] == CANDIDATE_ARM
            and candidate["sequence_length"] == SEQUENCE_LENGTH
            and candidate["num_kv_pairs"] == NUM_KV_PAIRS
            and candidate["model_width"] == MODEL_WIDTH
            and candidate["model_heads"] == MODEL_HEADS
            and candidate["gdn2_head_dim"] == HEAD_KEY_DIM
            and candidate["epochs"] == MAX_EPOCHS
        ),
        "identical_directional_data": (
            candidate["data_hashes"]
            == control["data_hashes"]
            == historical["score"]["data_hashes"]
        ),
        "counterfactual_test_hash_exact": (
            counterfactual_test_hash == candidate["data_hashes"]["test"]
        ),
        "counterfactual_case_ids_and_labels_exact": (
            counterfactual_case_labels == candidate_case_labels
        ),
        "counterfactual_checkpoint_hash_and_arm_exact": (
            _sha256(checkpoint_path) == candidate["checkpoint_sha256"]
            and checkpoint["arm"] == CANDIDATE_ARM
            and checkpoint["carrier_arm"] == CANDIDATE_ARM
        ),
        "identical_warmup_and_first_epoch_anchor_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "identical_initial_model_state": (
            candidate["init_hash"] == control["init_hash"]
        ),
        "identical_initial_parameters": (
            candidate["init_parameter_hash"] == control["init_parameter_hash"]
        ),
        "identical_parameter_count": (
            candidate["parameters"] == control["parameters"] == 661_584
        ),
        "identical_persistent_recurrent_state_budget": (
            candidate["recurrent_state_values_per_layer"]
            == control["recurrent_state_values_per_layer"]
            == 4_096
        ),
        "historical_reference_locked": (
            historical["score"]["metrics"]["balanced_accuracy"]
            == HISTORICAL_BALANCED
            and historical["score"]["metrics"]["joint_exact"]
            == HISTORICAL_JOINT_EXACT
            and historical["score"]["metrics"]["future"]["accuracy"]
            == HISTORICAL_FUTURE_ACCURACY
            and historical["score"]["metrics"]["past"]["accuracy"]
            == HISTORICAL_PAST_ACCURACY
        ),
    }

    condition_numbers = [float(row["condition_max"]) for row in rows]
    solve_residuals = [float(row["solve_relative_residual"]) for row in rows]
    write_fit_mse_ratios = [
        _ratio(
            float(row["weighted_candidate_seed_write_fit_mse"]),
            float(row["weighted_native_seed_write_fit_mse"]),
        )
        for row in rows
    ]
    seed_rms = [float(row["seed_rms"]) for row in rows]
    seed_board_std = [float(row["seed_board_std"]) for row in rows]
    surprise_rms = [float(row["committed_edit_rms"]) for row in rows]
    surprise_token_std = [float(row["surprise_token_std"]) for row in rows]
    surprise_board_std = [float(row["surprise_board_std"]) for row in rows]
    weight_token_std = [float(row["weight_token_std"]) for row in rows]
    receiver_key_unit_errors = [
        float(row["receiver_key_unit_error"]) for row in rows
    ]
    receiver_payload_rms = [float(row["receiver_payload_rms"]) for row in rows]
    candidate_to_native_seed_rms = [
        float(row["candidate_to_native_seed_rms_ratio"]) for row in rows
    ]
    bounded_residual_to_native_head_rms = [
        float(row["bounded_residual_to_native_head_rms_max"])
        for row in rows
    ]
    candidate_to_native_head_rms = [
        float(row["candidate_to_native_head_rms_ratio_max"])
        for row in rows
    ]
    metrics = candidate["metrics"]
    control_metrics = control["metrics"]
    candidate_query_ce = sum(
        metrics[direction]["ce"] * metrics[direction]["queries"]
        for direction in ("future", "past")
    ) / sum(metrics[direction]["queries"] for direction in ("future", "past"))
    native_counterfactual_query_ce = sum(
        native_counterfactual_metrics[direction]["ce"]
        * native_counterfactual_metrics[direction]["queries"]
        for direction in ("future", "past")
    ) / sum(
        native_counterfactual_metrics[direction]["queries"]
        for direction in ("future", "past")
    )
    activation_checks = {
        "exactly_one_active_receiver_route": (
            diagnostics["receiver_routes"] == MODEL_LAYERS - 1
            and len(rows) == MODEL_LAYERS - 1
        ),
        "zero_new_parameters": diagnostics["new_parameters"] == 0,
        "zero_persistent_state_delta": diagnostics["persistent_state_delta"] == 0,
        "exactly_two_main_and_zero_additional_official_scans": (
            diagnostics["main_official_scans"] == MODEL_LAYERS
            and diagnostics["additional_official_scans"] == 0
        ),
        "all_tokens_used_without_sparse_selection": (
            diagnostics["admission"]
            == "all_tokens_weighted_by_stop_gradient_committed_edit_norm"
        ),
        "exact_committed_edit_surprise_active": (
            all(_finite_at_least(value, 1e-4) for value in surprise_rms)
            and all(_finite_at_least(value, 1e-4) for value in surprise_token_std)
            and all(_finite_at_least(value, 1e-6) for value in surprise_board_std)
            and all(_finite_at_least(value, 1e-4) for value in weight_token_std)
        ),
        "receiver_native_key_and_payload_active": (
            diagnostics["receiver_projection"]
            == "existing_causal_conv_k_v_and_pointwise_w"
            and all(
                _finite(value) and value <= 1e-3
                for value in receiver_key_unit_errors
            )
            and all(_finite_at_least(value, 1e-4) for value in receiver_payload_rms)
        ),
        "regression_seed_finite_nonzero_and_board_varying": (
            all(_finite_at_least(value, 1e-4) for value in seed_rms)
            and all(_finite_at_least(value, 1e-6) for value in seed_board_std)
        ),
        "cholesky_succeeds_without_fallback": (
            diagnostics["linear_system"] == "fp32_cholesky_no_fallback"
        ),
        "condition_number_at_most_1e4": (
            all(_finite(value) and value <= 1e4 for value in condition_numbers)
        ),
        "normal_equation_relative_residual_at_most_1e-4": (
            all(_finite(value) and value <= 1e-4 for value in solve_residuals)
        ),
        "weighted_receiver_write_fit_mse_down_at_least_25_percent": (
            all(_finite(value) and value <= 0.75 for value in write_fit_mse_ratios)
        ),
        "receiver_seed_rms_bounded_to_2x_native": (
            all(
                _finite(value) and value <= 2.0 + 1e-5
                for value in candidate_to_native_seed_rms
            )
        ),
        "residual_rms_bounded_to_native_per_board_head": (
            all(
                _finite(value) and value <= 1.0 + 1e-5
                for value in bounded_residual_to_native_head_rms
            )
        ),
        "candidate_seed_rms_bounded_to_2x_native_per_board_head": (
            all(
                _finite(value) and value <= 2.0 + 1e-5
                for value in candidate_to_native_head_rms
            )
        ),
        "actual_query_ce_improves_over_same_weight_native_counterfactual": (
            metrics["future"]["ce"] < native_counterfactual_metrics["future"]["ce"]
            and metrics["past"]["ce"] < native_counterfactual_metrics["past"]["ce"]
            and candidate_query_ce <= 0.98 * native_counterfactual_query_ce
        ),
    }

    swaps = {
        "historical_native_futureseed": historical["swap_summary"],
        "same_runtime_native_futureseed": control_swaps,
        "surprise_regression_seed": candidate_swaps,
        "same_weight_native_counterfactual": native_counterfactual_swaps,
    }
    quality_checks = {
        "balanced_accuracy_at_least_0.85": metrics["balanced_accuracy"] >= 0.85,
        "future_accuracy_at_least_0.83": metrics["future"]["accuracy"] >= 0.83,
        "past_accuracy_at_least_0.83": metrics["past"]["accuracy"] >= 0.83,
        "joint_exact_at_least_0.60": metrics["joint_exact"] >= 0.60,
        "balanced_gain_over_historical_at_least_0.10": (
            metrics["balanced_accuracy"] >= HISTORICAL_BALANCED + 0.10
        ),
        "balanced_gain_over_same_runtime_control_at_least_0.20": (
            metrics["balanced_accuracy"]
            >= control_metrics["balanced_accuracy"] + 0.20
        ),
        "wrong_key_swap_fraction_down_0.10_from_historical": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"]
            <= HISTORICAL_SWAP_FRACTION - 0.10
        ),
        "wrong_key_swap_fraction_down_0.10_from_same_runtime_control": (
            candidate_swaps["wrong_key_swap_fraction_of_errors"]
            <= control_swaps["wrong_key_swap_fraction_of_errors"] - 0.10
        ),
        "total_errors_lower_than_historical_and_same_runtime_control": (
            candidate_swaps["errors"] < historical["swap_summary"]["errors"]
            and candidate_swaps["errors"] < control_swaps["errors"]
        ),
    }

    cost = {
        "fit_elapsed_ratio": _ratio(
            candidate["elapsed_sec_including_validation"],
            control["elapsed_sec_including_validation"],
        ),
        "post_warm_wall_ratio": _ratio(
            candidate["post_warm_arm_wall_sec_through_checkpoint"],
            control["post_warm_arm_wall_sec_through_checkpoint"],
        ),
        "warmed_step_ratio": _ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            control["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_allocation_ratio": _ratio(
            candidate["peak_training_cuda_mem_bytes"],
            control["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "fit_elapsed_ratio_at_most_1.60": cost["fit_elapsed_ratio"] <= 1.60,
        "post_warm_wall_ratio_at_most_1.60": cost["post_warm_wall_ratio"] <= 1.60,
        "warmed_step_ratio_at_most_1.60": cost["warmed_step_ratio"] <= 1.60,
        "peak_allocation_ratio_at_most_1.25": (
            cost["peak_allocation_ratio"] <= 1.25
        ),
    }

    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    decision = {
        "status": "completed_passed" if passed else "completed_rejected",
        "plan": "P-FS2-008",
        "mechanism": "receiver-native exact-surprise weighted regression FutureSeed",
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "model": "D128/L2/H4/K32/V32 pinned official GDN2",
            "arms_in_fixed_order": [CONTROL_OUTPUT_ARM, CANDIDATE_ARM],
            "main_recurrence_changed": False,
            "new_parameters": 0,
            "new_persistent_state": 0,
            "additional_official_scans": 0,
            "all_tokens_used": True,
            "rescue_authorized": False,
        },
        "historical_reference": historical,
        "same_runtime_control": control,
        "candidate": candidate,
        "same_weight_native_counterfactual": {
            "metrics": native_counterfactual_metrics,
            "swap_summary": native_counterfactual_swaps,
            "test_hash": counterfactual_test_hash,
            "candidate_query_ce": candidate_query_ce,
            "native_query_ce": native_counterfactual_query_ce,
            "query_ce_ratio": _ratio(
                candidate_query_ce,
                native_counterfactual_query_ce,
            ),
        },
        "wrong_key_swaps": swaps,
        "integrity_checks": integrity_checks,
        "activation_checks": activation_checks,
        "write_fit_mse_ratio_candidate_to_native": write_fit_mse_ratios,
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "registered_gate": {"passed": passed},
        "artifact_sha256": {
            "historical_score": HISTORICAL_SCORE_SHA256,
            "historical_cases": HISTORICAL_CASES_SHA256,
            "control_config": _sha256(control_root / "config.json"),
            "control_score": _sha256(control_root / "score.json"),
            "control_cases": _sha256(control_cases_path),
            "control_checkpoint": control["checkpoint_sha256"],
            "candidate_config": _sha256(candidate_root / "config.json"),
            "candidate_score": _sha256(candidate_root / "score.json"),
            "candidate_cases": _sha256(candidate_cases_path),
            "candidate_checkpoint": candidate["checkpoint_sha256"],
        },
        "next_decision": (
            "Run one fixed uniform-weight attribution control before any hard-Sudoku transfer."
            if passed
            else "Close surprise-regression FutureSeed without lambda, weighting, projection, seed, optimizer, or scale rescue."
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
