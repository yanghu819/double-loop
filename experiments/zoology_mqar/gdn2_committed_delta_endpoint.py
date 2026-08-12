from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
P020_SCORE_SHA256 = "a9c750e84fc918910c008b679016c3f24a5e821e3c20b84b1f0b07c1c64b9b3e"
P020_CASES_SHA256 = "127fb40e32a14a54f6f90c784dfe728c85dcf200535e0fb300e3e28a55c8751b"
P020_BALANCED = 0.48225
P020_SWAP_FRACTION = 0.9415652712566506
P020_PARENT_HASH = "3b0c133410eaed1135224cc0acb094705655cc7e97ea6beeba17b40beab1a368"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(candidate: float, reference: float) -> float:
    return candidate / reference


def finite_between(value: float, lower: float, upper: float) -> bool:
    return math.isfinite(value) and lower <= value <= upper


def activation_checks(score: dict[str, Any], *, learned: bool) -> dict[str, bool]:
    diagnostics = score["committed_delta"]
    rows = diagnostics["per_layer"]
    checks = {
        "exactly_two_active_layers": diagnostics["active_layers"] == 2,
        "expected_learned_mode": diagnostics["learned_subspace"] is learned,
        "main_state_exactly_4096": diagnostics["main_state_values_per_layer"] == 4096,
        "correction_state_exactly_2048": (
            diagnostics["correction_state_values_per_layer"] == 2048
        ),
        "total_state_exactly_6144": diagnostics["total_state_values_per_layer"] == 6144,
        "exactly_two_official_scans": diagnostics["official_scans_per_layer"] == 2,
        "exact_correction_parameter_count": diagnostics["correction_parameters"]
        == (3976 if learned else 8),
        "all_committed_edits_active": all(
            row["committed_edit_rms"] >= 1e-4
            and row["committed_edit_board_std"] > 0
            for row in rows
        ),
        "all_correction_reads_active": all(
            row["correction_read_abs_gate"] >= 1e-4
            and finite_between(row["correction_output_relative_rms"], 1e-4, 10.0)
            and row["correction_output_board_std"] > 0
            for row in rows
        ),
        "bounded_state_geometry": all(
            finite_between(row["correction_state_to_main_rms"], 1e-4, 10.0)
            and math.isfinite(row["correction_state_rms"])
            for row in rows
        ),
        "address_token_variation": all(
            row["correction_q_token_std"] > 0
            and row["correction_k_token_std"] > 0
            for row in rows
        ),
    }
    if learned:
        checks.update(
            {
                "learned_basis_moved": all(
                    row["basis_delta_fro_mean"] >= 1e-4 for row in rows
                ),
                "learned_basis_remains_semi_orthogonal": all(
                    row["row_orthogonality_max_error"] <= 1e-4 for row in rows
                ),
            }
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p020-score", type=Path, required=True)
    parser.add_argument("--p020-cases", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-025 has exactly two fixed 10-epoch batch32 arms")
    if sha256(args.p020_score) != P020_SCORE_SHA256:
        raise RuntimeError("P020 score provenance changed")
    if sha256(args.p020_cases) != P020_CASES_SHA256:
        raise RuntimeError("P020 cases provenance changed")

    p020 = json.loads(args.p020_score.read_text())
    p020_cases = json.loads(args.p020_cases.read_text())
    if not math.isclose(
        p020["metrics"]["balanced_accuracy"],
        P020_BALANCED,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("P020 reference metric changed")
    if p020["parent_init_parameter_hash"] != P020_PARENT_HASH:
        raise RuntimeError("P020 parent initialization changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared = run_arm(
        arm="future_seed_gdn2_shared_committed_delta",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    clustered = run_arm(
        arm="future_seed_gdn2_clustered_committed_delta",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    shared_dir = args.output_dir / "length_1024" / "future_seed_gdn2_shared_committed_delta"
    clustered_dir = (
        args.output_dir / "length_1024" / "future_seed_gdn2_clustered_committed_delta"
    )
    shared_cases = json.loads((shared_dir / "cases.json").read_text())
    clustered_cases = json.loads((clustered_dir / "cases.json").read_text())

    integrity_checks = {
        "all_data_hashes_match": (
            shared["data_hashes"] == clustered["data_hashes"] == p020["data_hashes"]
        ),
        "both_parent_initializations_match_p020": (
            shared["parent_init_parameter_hash"]
            == clustered["parent_init_parameter_hash"]
            == P020_PARENT_HASH
        ),
        "arms_have_identical_state_and_scan_budget": (
            shared["committed_delta"]["total_state_values_per_layer"]
            == clustered["committed_delta"]["total_state_values_per_layer"]
            == 6144
            and shared["committed_delta"]["official_scans_per_layer"]
            == clustered["committed_delta"]["official_scans_per_layer"]
            == 2
        ),
    }
    swaps = {
        "p020_log_spd": wrong_key_swap_summary(p020_cases),
        "shared_basis": wrong_key_swap_summary(shared_cases),
        "clustered_basis": wrong_key_swap_summary(clustered_cases),
    }
    shared_activation = activation_checks(shared, learned=False)
    clustered_activation = activation_checks(clustered, learned=True)
    metrics = clustered["metrics"]
    balanced_gain_vs_shared = (
        metrics["balanced_accuracy"] - shared["metrics"]["balanced_accuracy"]
    )
    joint_gain_vs_shared = metrics["joint_exact"] - shared["metrics"]["joint_exact"]
    quality_checks = {
        "clustered_balanced_at_least_0.70": metrics["balanced_accuracy"] >= 0.70,
        "clustered_joint_exact_at_least_0.25": metrics["joint_exact"] >= 0.25,
        "clustered_directions_each_at_least_0.68": min(
            metrics["past"]["accuracy"], metrics["future"]["accuracy"]
        )
        >= 0.68,
        "clustered_balanced_gain_over_p020_at_least_0.10": (
            metrics["balanced_accuracy"] >= P020_BALANCED + 0.10
        ),
        "clustered_beats_same_byte_shared_by_0.05": (
            balanced_gain_vs_shared >= 0.05 or joint_gain_vs_shared >= 0.05
        ),
        "wrong_key_swap_fraction_down_0.10_from_p020": (
            swaps["clustered_basis"]["wrong_key_swap_fraction_of_errors"]
            <= P020_SWAP_FRACTION - 0.10
        ),
    }
    cost = {
        "shared_fit_ratio_vs_p020": ratio(
            shared["elapsed_sec_including_validation"],
            p020["elapsed_sec_including_validation"],
        ),
        "clustered_fit_ratio_vs_p020": ratio(
            clustered["elapsed_sec_including_validation"],
            p020["elapsed_sec_including_validation"],
        ),
        "clustered_warmed_step_ratio_vs_p020": ratio(
            clustered["warmed_step_benchmark"]["elapsed_sec"],
            p020["warmed_step_benchmark"]["elapsed_sec"],
        ),
        "clustered_peak_ratio_vs_p020": ratio(
            clustered["peak_training_cuda_mem_bytes"],
            p020["peak_training_cuda_mem_bytes"],
        ),
        "clustered_fit_ratio_vs_shared": ratio(
            clustered["elapsed_sec_including_validation"],
            shared["elapsed_sec_including_validation"],
        ),
        "clustered_peak_ratio_vs_shared": ratio(
            clustered["peak_training_cuda_mem_bytes"],
            shared["peak_training_cuda_mem_bytes"],
        ),
    }
    cost_checks = {
        "both_fit_ratios_vs_p020_at_most_1.65": max(
            cost["shared_fit_ratio_vs_p020"], cost["clustered_fit_ratio_vs_p020"]
        )
        <= 1.65,
        "clustered_warmed_step_ratio_vs_p020_at_most_1.60": (
            cost["clustered_warmed_step_ratio_vs_p020"] <= 1.60
        ),
        "clustered_peak_ratio_vs_p020_at_most_1.60": (
            cost["clustered_peak_ratio_vs_p020"] <= 1.60
        ),
        "learned_vs_shared_system_delta_at_most_1.10": max(
            cost["clustered_fit_ratio_vs_shared"], cost["clustered_peak_ratio_vs_shared"]
        )
        <= 1.10,
    }
    passed = (
        all(integrity_checks.values())
        and all(shared_activation.values())
        and all(clustered_activation.values())
        and all(quality_checks.values())
        and all(cost_checks.values())
    )
    decision = {
        "status": "complete",
        "plan": "P-GDN3-025",
        "mechanism": (
            "exact official committed-delta correction state; fixed K16 address "
            "versus same-byte learned semi-orthogonal K32-to-K16 address"
        ),
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 official GDN2 plus native FutureSeed",
            "committed_edit_gradient_policy": "detached from parent recurrence",
            "arms": ["shared_basis", "clustered_basis"],
        },
        "reference": p020,
        "shared_basis": shared,
        "clustered_basis": clustered,
        "wrong_key_swaps": swaps,
        "integrity_checks": integrity_checks,
        "activation_checks": {
            "shared_basis": shared_activation,
            "clustered_basis": clustered_activation,
        },
        "quality_checks": quality_checks,
        "cost": cost,
        "cost_checks": cost_checks,
        "registered_gate": {
            "passed": passed,
            "balanced_gain_vs_p020": metrics["balanced_accuracy"] - P020_BALANCED,
            "balanced_gain_vs_shared": balanced_gain_vs_shared,
            "joint_gain_vs_shared": joint_gain_vs_shared,
        },
        "artifact_sha256": {
            "p020_score": sha256(args.p020_score),
            "p020_cases": sha256(args.p020_cases),
            "shared_score": sha256(shared_dir / "score.json"),
            "shared_cases": sha256(shared_dir / "cases.json"),
            "shared_checkpoint": shared["checkpoint_sha256"],
            "clustered_score": sha256(clustered_dir / "score.json"),
            "clustered_cases": sha256(clustered_dir / "cases.json"),
            "clustered_checkpoint": clustered["checkpoint_sha256"],
        },
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
