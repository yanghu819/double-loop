from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
    run_arm,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
HISTORICAL_BALANCED = 0.7475


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _cases(root: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads((root / "length_1024" / arm / "cases.json").read_text())


def _write_matched_initialization(output_dir: Path) -> tuple[Path, str]:
    config = build_config(
        arm="future_seed_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    model = make_model(config, "future_seed_gdn2")
    path = output_dir / "matched_parent_initialization.pt"
    torch.save(model.state_dict(), path)
    digest = parameter_hash(model)
    del model
    return path, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-040 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched_init_path, matched_init_hash = _write_matched_initialization(
        args.output_dir
    )
    control_name = "future_seed_gdn2_control"
    candidate_name = "future_seed_linear_product_state_gdn2"
    control = run_arm(
        arm="future_seed_gdn2",
        output_arm_name=control_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=matched_init_path,
    )
    candidate = run_arm(
        arm=candidate_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
        matched_init_path=matched_init_path,
    )
    if control["init_parameter_hash"] != matched_init_hash:
        raise RuntimeError("Control did not use the matched initialization")
    if candidate["parent_init_parameter_hash"] != matched_init_hash:
        raise RuntimeError("Candidate parent initialization differs from control")
    for key in ("data_hashes", "warmup_batch_hash"):
        if candidate[key] != control[key]:
            raise RuntimeError(
                f"Matched arm mismatch for {key}: {control[key]} != {candidate[key]}"
            )
    if candidate["parameters"] != control["parameters"]:
        raise RuntimeError("Linear-product state must add zero parameters")
    if control["recurrent_state_values_per_layer"] != 4_096:
        raise RuntimeError("Unexpected control state size")
    if candidate["recurrent_state_values_per_layer"] != 12_288:
        raise RuntimeError("Unexpected linear-product state size")

    control_swaps = wrong_key_swap_summary(_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(_cases(args.output_dir, candidate_name))
    rows = candidate["linear_product_state"]["per_layer"]
    activation_checks = {
        "two_layers_one_official_scan": (
            candidate["linear_product_state"]["active_layers"] == 2
            and candidate["linear_product_state"]["official_scans_per_layer"] == 1
        ),
        "exact_k32_plus_k8xk8_state": (
            candidate["linear_product_state"]["linear_dim"] == 32
            and candidate["linear_product_state"]["factor_dim"] == 8
            and candidate["linear_product_state"]["product_dim"] == 64
            and candidate["linear_product_state"]["total_key_dim"] == 96
            and candidate["linear_product_state"]["state_values_per_layer"] == 12_288
            and candidate["linear_product_state"]["new_parameters"] == 0
        ),
        "direct_sum_algebra_stable": all(
            row["q_direct_sum_norm_max_error"] <= 0.02
            and row["k_direct_sum_norm_max_error"] <= 0.02
            and row["similarity_identity_max_error"] <= 0.02
            and row["decay_max"] <= 1e-6
            and row["erase_min"] >= 0.0
            and row["erase_max"] <= 1.0
            for row in rows
        ),
        "product_addresses_active_and_variable": all(
            row["q_product_rms"] >= 1e-3
            and row["k_product_rms"] >= 1e-3
            and row["q_product_token_std"] >= 1e-4
            and row["k_product_token_std"] >= 1e-4
            and row["q_product_board_std"] >= 1e-5
            and row["k_product_board_std"] >= 1e-5
            for row in rows
        ),
        "product_state_and_read_active": all(
            1e-4 <= row["product_native_state_rms_ratio"] <= 1e4
            and row["product_state_board_std"] >= 1e-5
            and 1e-4 <= row["product_native_read_rms_ratio"] <= 1e4
            and 1e-4 <= row["terminal_state_rms"] <= 1e4
            for row in rows
        ),
        "native_futureseed_active": candidate["future_seed"]["active_seed_routes"] == 1,
    }

    cm = control["metrics"]
    pm = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.85": pm["balanced_accuracy"] >= 0.85,
        "balanced_above_historical_by_0.10": (
            pm["balanced_accuracy"] - HISTORICAL_BALANCED >= 0.10
        ),
        "balanced_above_control_by_0.10": (
            pm["balanced_accuracy"] - cm["balanced_accuracy"] >= 0.10
        ),
        "future_at_least_0.85_and_gain_0.10": (
            pm["future"]["accuracy"] >= 0.85
            and pm["future"]["accuracy"] - cm["future"]["accuracy"] >= 0.10
        ),
        "past_at_least_0.85_and_gain_0.10": (
            pm["past"]["accuracy"] >= 0.85
            and pm["past"]["accuracy"] - cm["past"]["accuracy"] >= 0.10
        ),
        "joint_at_least_0.60_and_gain_0.10": (
            pm["joint_exact"] >= 0.60
            and pm["joint_exact"] - cm["joint_exact"] >= 0.10
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "swap_fraction_reduced_at_least_0.10": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.10
        ),
    }
    cost_ratios = {
        "elapsed": _ratio(
            candidate["elapsed_sec_including_validation"],
            control["elapsed_sec_including_validation"],
        ),
        "post_warm_wall": _ratio(
            candidate["post_warm_arm_wall_sec_through_checkpoint"],
            control["post_warm_arm_wall_sec_through_checkpoint"],
        ),
        "warmed_step": _ratio(
            candidate["warmed_step_benchmark"]["elapsed_sec"],
            control["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "peak_allocation": _ratio(
            candidate["peak_training_cuda_mem_bytes"],
            control["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "elapsed_below_3.25x": cost_ratios["elapsed"] < 3.25,
        "post_warm_wall_below_3.25x": cost_ratios["post_warm_wall"] < 3.25,
        "warmed_step_below_3.25x": cost_ratios["warmed_step"] < 3.25,
        "peak_allocation_below_2.50x": cost_ratios["peak_allocation"] < 2.50,
    }
    passed = (
        all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-040",
            "mechanism": "native linear plus exact product direct-sum GDN2 state",
            "run_order": [control_name, candidate_name],
            "model": "D128/L2/H4/K32 plus K8xK8/V32 GDN2 and native FutureSeed",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "new_parameters_vs_control": 0,
            "state_value_ratio": 3.0,
            "new_scans_per_layer": 0,
            "factor_dim": 8,
            "direct_sum_weight_sweep": None,
        },
        "matched_initialization": {
            "path": str(matched_init_path),
            "parameter_hash": matched_init_hash,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
        "decision": "admit_one_sudoku_transfer" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
