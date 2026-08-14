from __future__ import annotations

import argparse
import json
import math
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
from experiments.zoology_mqar.producer_readout_futureseed import (
    EXPECTED_PARAMETER_DELTA,
    READOUT_RANK,
    RESIDUAL_CAP,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = 673_872


def _ratio(value: float, reference: float) -> float:
    if reference <= 0:
        raise ValueError("Cost reference must be positive")
    return value / reference


def _cases(root: Path, arm: str, *, edge_off: bool = False) -> list[dict[str, Any]]:
    filename = "edge_off_cases.json" if edge_off else "cases.json"
    return json.loads((root / "length_1024" / arm / filename).read_text())


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
        raise ValueError("P-FS2-010 has one fixed 10-epoch/batch32 endpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matched_init_path, matched_init_hash = _write_matched_initialization(
        args.output_dir
    )
    control_name = "future_seed_gdn2_control"
    candidate_name = "future_seed_producer_readout_gdn2"
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

    integrity_checks = {
        "control_used_matched_initialization": (
            control["init_parameter_hash"] == matched_init_hash
        ),
        "candidate_parent_matches_control": (
            candidate["parent_init_parameter_hash"] == matched_init_hash
        ),
        "matched_data_hashes": candidate["data_hashes"] == control["data_hashes"],
        "matched_warmup_batch": (
            candidate["warmup_batch_hash"] == control["warmup_batch_hash"]
        ),
        "native_parameter_count_exact": (
            control["parameters"] == EXPECTED_NATIVE_PARAMETERS
        ),
        "candidate_parameter_count_exact": (
            candidate["parameters"] == EXPECTED_CANDIDATE_PARAMETERS
        ),
        "parameter_delta_exact": (
            candidate["parameters"] - control["parameters"]
            == EXPECTED_PARAMETER_DELTA
        ),
        "state_size_unchanged": (
            control["recurrent_state_values_per_layer"]
            == candidate["recurrent_state_values_per_layer"]
            == 4_096
        ),
    }

    diagnostics = candidate["producer_readout"]
    rows = diagnostics["per_edge"]
    activation_checks = {
        "one_adjacent_readout_edge": (
            diagnostics["active_edges"] == 1 and len(rows) == 1
        ),
        "fixed_topology_and_parameter_delta": (
            diagnostics["rank"] == READOUT_RANK
            and diagnostics["residual_cap"] == RESIDUAL_CAP
            and diagnostics["new_parameters"] == EXPECTED_PARAMETER_DELTA
            and diagnostics["new_persistent_state_values"] == 0
            and diagnostics["new_official_scans_per_layer"] == 0
        ),
        "producer_read_active_and_variable": all(
            row["read_rms"] >= 1e-3
            and row["read_board_std"] > 1e-6
            and row["read_token_std"] > 1e-6
            and row["feature_rms"] >= 1e-3
            for row in rows
        ),
        "fusion_trained_and_active": all(
            row["out_projection_rms"] >= 1e-5
            and row["residual_relative_rms"] >= 1e-3
            for row in rows
        ),
        "bounded_residual": all(
            row["residual_abs_max_over_hidden_rms"] <= 0.5001 for row in rows
        ),
        "transferred_state_finite_and_variable": all(
            math.isfinite(row["state_rms"])
            and 1e-4 <= row["state_rms"] <= 1e4
            and row["state_board_std"] > 1e-6
            for row in rows
        ),
        "native_futureseed_active": (
            candidate["future_seed"]["active_seed_routes"] == 1
        ),
    }

    control_swaps = wrong_key_swap_summary(_cases(args.output_dir, control_name))
    candidate_swaps = wrong_key_swap_summary(
        _cases(args.output_dir, candidate_name)
    )
    edge_off_swaps = wrong_key_swap_summary(
        _cases(args.output_dir, candidate_name, edge_off=True)
    )
    cm = control["metrics"]
    pm = candidate["metrics"]
    em = candidate["producer_readout_edge_off_metrics"]
    quality_checks = {
        "balanced_absolute_and_control_gain": (
            pm["balanced_accuracy"] >= max(0.55, cm["balanced_accuracy"] + 0.10)
        ),
        "future_absolute_and_control_gain": (
            pm["future"]["accuracy"]
            >= max(0.53, cm["future"]["accuracy"] + 0.08)
        ),
        "past_absolute_and_control_gain": (
            pm["past"]["accuracy"]
            >= max(0.53, cm["past"]["accuracy"] + 0.08)
        ),
        "joint_absolute_and_control_gain": (
            pm["joint_exact"] >= max(0.08, cm["joint_exact"] + 0.04)
        ),
        "total_errors_reduced": candidate_swaps["errors"] < control_swaps["errors"],
        "swap_fraction_reduced_at_least_0.05": (
            control_swaps["wrong_key_swap_fraction_of_errors"]
            - candidate_swaps["wrong_key_swap_fraction_of_errors"]
            >= 0.05
        ),
        "trained_edge_attribution_balanced_gain_0.03": (
            pm["balanced_accuracy"] - em["balanced_accuracy"] >= 0.03
        ),
        "trained_edge_attribution_fewer_errors": (
            candidate_swaps["errors"] < edge_off_swaps["errors"]
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
        "elapsed_below_1.50x": cost_ratios["elapsed"] < 1.50,
        "post_warm_wall_below_1.50x": cost_ratios["post_warm_wall"] < 1.50,
        "warmed_step_below_1.50x": cost_ratios["warmed_step"] < 1.50,
        "peak_allocation_below_1.35x": cost_ratios["peak_allocation"] < 1.35,
    }
    passed = (
        all(integrity_checks.values())
        and all(activation_checks.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-FS2-010",
            "mechanism": (
                "producer-native state decode plus bounded token-conditioned "
                "rank32 receiver fusion"
            ),
            "run_order": [control_name, candidate_name],
            "model": "D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "readout_rank": READOUT_RANK,
            "residual_cap": RESIDUAL_CAP,
            "new_parameters": EXPECTED_PARAMETER_DELTA,
            "new_recurrent_state": 0,
            "new_official_scans_per_layer": 0,
            "sweep": None,
        },
        "matched_initialization": {
            "path": str(matched_init_path),
            "parameter_hash": matched_init_hash,
        },
        "control": control,
        "control_swap_summary": control_swaps,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "candidate_edge_off_swap_summary": edge_off_swaps,
        "cost_ratios": cost_ratios,
        "registered_gate": {
            "passed": passed,
            "integrity_checks": integrity_checks,
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
