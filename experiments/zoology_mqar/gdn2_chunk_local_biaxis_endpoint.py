from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32


def ratio(value: float, reference: float) -> float:
    if reference <= 0.0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def load_cases(output_dir: Path, arm: str) -> list[dict[str, Any]]:
    return json.loads(
        (output_dir / "length_1024" / arm / "cases.json").read_text()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-035 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    control_name = "future_seed_gdn2_runtime_control"
    candidate_name = "future_seed_chunk_local_biaxis_gdn2"
    control = run_arm(
        arm="future_seed_gdn2",
        output_arm_name=control_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    candidate = run_arm(
        arm=candidate_name,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )

    if candidate["parent_init_parameter_hash"] != control["init_parameter_hash"]:
        raise RuntimeError("Bi-Axis candidate changed native parent initialization")
    if candidate["data_hashes"] != control["data_hashes"]:
        raise RuntimeError("Matched data hashes differ")
    if candidate["warmup_batch_hash"] != control["warmup_batch_hash"]:
        raise RuntimeError("Matched warmup batch differs")

    control_swaps = wrong_key_swap_summary(load_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(
        load_cases(args.output_dir, candidate_name)
    )
    biaxis = candidate["chunk_local_biaxis"]
    rows = biaxis["per_layer"]
    activation_checks = {
        "exactly_two_active_layers": biaxis["active_layers"] == 2,
        "parameter_delta_exactly_8192": biaxis["parameter_delta"] == 8_192,
        "anchor_size_exactly_64": biaxis["anchor_size"] == 64,
        "value_groups_exactly_8": biaxis["value_groups"] == 8,
        "exactly_16_physical_chunks_per_layer": all(
            row["chunks"] == 16 for row in rows
        ),
        "all_value_decay_paths_active": all(
            row["log_decay_abs"] >= 1e-4 and row["active_fraction"] >= 0.05
            for row in rows
        ),
        "all_value_decay_paths_vary": all(
            row["group_std"] > 0.0
            and row["batch_std"] > 0.0
            and row["token_std"] > 0.0
            for row in rows
        ),
        "all_local_frames_bounded": all(
            row["local_scale_min"] >= 0.249
            and row["local_scale_max"] <= 1.001
            and row["local_inverse_max"] <= 4.01
            for row in rows
        ),
        "all_terminal_states_finite": all(
            0.0 < row["terminal_state_rms"] < 1e4 for row in rows
        ),
    }

    control_metrics = control["metrics"]
    candidate_metrics = candidate["metrics"]
    quality_checks = {
        "balanced_at_least_0.35": candidate_metrics["balanced_accuracy"] >= 0.35,
        "balanced_gain_at_least_0.10": (
            candidate_metrics["balanced_accuracy"]
            - control_metrics["balanced_accuracy"]
            >= 0.10
        ),
        "future_gain_at_least_0.07": (
            candidate_metrics["future"]["accuracy"]
            - control_metrics["future"]["accuracy"]
            >= 0.07
        ),
        "past_gain_at_least_0.07": (
            candidate_metrics["past"]["accuracy"]
            - control_metrics["past"]["accuracy"]
            >= 0.07
        ),
        "joint_at_least_0.03": candidate_metrics["joint_exact"] >= 0.03,
        "joint_gain_at_least_0.03": (
            candidate_metrics["joint_exact"] - control_metrics["joint_exact"]
            >= 0.03
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "wrong_key_swap_fraction_not_worse_by_0.05": (
            float(candidate_swaps["wrong_key_swap_fraction_of_errors"])
            <= float(control_swaps["wrong_key_swap_fraction_of_errors"]) + 0.05
        ),
    }

    cost_ratios = {
        "elapsed": ratio(
            float(candidate["elapsed_sec_including_validation"]),
            float(control["elapsed_sec_including_validation"]),
        ),
        "post_warm_wall": ratio(
            float(candidate["post_warm_arm_wall_sec_through_checkpoint"]),
            float(control["post_warm_arm_wall_sec_through_checkpoint"]),
        ),
        "warmed_step": ratio(
            float(candidate["warmed_step_benchmark"]["elapsed_sec"]),
            float(control["warmed_step_benchmark"]["elapsed_sec"]),
        ),
        "peak_allocation": ratio(
            float(candidate["peak_training_cuda_mem_bytes"]),
            float(control["peak_training_cuda_mem_bytes"]),
        ),
    }
    cost_checks = {
        "prototype_elapsed_below_4x": cost_ratios["elapsed"] < 4.0,
        "prototype_post_warm_wall_below_4x": cost_ratios["post_warm_wall"] < 4.0,
        "prototype_warmed_step_below_4x": cost_ratios["warmed_step"] < 4.0,
        "peak_allocation_below_1.50x": cost_ratios["peak_allocation"] < 1.50,
    }
    passed = (
        all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-035",
            "mechanism": "bounded chunk-local persistent value-axis decay",
            "run_order": [control_name, candidate_name],
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 official GDN2 plus native FutureSeed",
            "new_parameters": 8_192,
            "new_recurrent_state": 0,
            "physical_anchor_size": 64,
            "value_groups": 8,
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
        "decision": "authorize_fused_biaxis_kernel" if passed else "closed",
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
