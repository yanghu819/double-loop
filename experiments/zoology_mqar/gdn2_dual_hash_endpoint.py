from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import run_arm


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
P020_BALANCED = 0.48225
P020_JOINT = 0.044
P020_SWAP_FRACTION = 0.9415652712566506
CONTROL_FIT_SEC = 146.97
CONTROL_PEAK_BYTES = 1_051_000_000


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-024 has one fixed D128/L2/H4/K32/10-epoch arm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score = run_arm(
        arm="future_seed_gdn2_dual_hash",
        output_arm_name="future_seed_gdn2_dual_hash",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    cases_path = (
        args.output_dir
        / f"length_{SEQUENCE_LENGTH}"
        / "future_seed_gdn2_dual_hash"
        / "cases.json"
    )
    swaps = wrong_key_swap_summary(load_cases(cases_path))
    metrics = score["metrics"]
    diagnostics = score["dual_hash"]
    fit_ratio = score["elapsed_sec_including_validation"] / CONTROL_FIT_SEC
    allocation_ratio = score["peak_training_cuda_mem_bytes"] / CONTROL_PEAK_BYTES
    checks = {
        "balanced_at_least_0.60": metrics["balanced_accuracy"] >= 0.60,
        "balanced_gain_over_p020_at_least_0.10": (
            metrics["balanced_accuracy"] >= P020_BALANCED + 0.10
        ),
        "joint_exact_at_least_0.15": metrics["joint_exact"] >= 0.15,
        "past_and_future_each_at_least_0.58": min(
            metrics["past"]["accuracy"], metrics["future"]["accuracy"]
        ) >= 0.58,
        "swap_fraction_reduced_at_least_0.10_from_p020": (
            swaps["wrong_key_swap_fraction_of_errors"]
            <= P020_SWAP_FRACTION - 0.10
        ),
        "exact_zero_parameter_same_state_one_scan_contract": (
            diagnostics["parameter_delta"] == 0
            and diagnostics["state_values_per_layer"] == 4096
            and diagnostics["official_scans_per_layer"] == 1
        ),
        "both_hashes_active_and_balanced": all(
            row["q_hash_token_std"] > 0
            and row["k_hash_token_std"] > 0
            and row["q_hash_norm_imbalance_max"] <= 1e-3
            and row["k_hash_norm_imbalance_max"] <= 1e-3
            for row in diagnostics["per_layer"]
        ),
        "fit_ratio_at_most_1.25": fit_ratio <= 1.25,
        "allocation_ratio_at_most_1.10": allocation_ratio <= 1.10,
    }
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-024",
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seed": 123,
            "model": "D128/L2/H4/K32/V32 official GDN2 plus native FutureSeed",
            "mechanism": "zero-parameter compact bilinear binding of two normalized K16 factors inside every K32 head",
        },
        "p020_reference": {
            "balanced_accuracy": P020_BALANCED,
            "joint_exact": P020_JOINT,
            "wrong_key_swap_fraction_of_errors": P020_SWAP_FRACTION,
            "fit_elapsed_sec": CONTROL_FIT_SEC,
            "peak_training_cuda_mem_bytes": CONTROL_PEAK_BYTES,
        },
        "candidate": score,
        "wrong_key_swaps": swaps,
        "registered_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "balanced_gain_over_p020": metrics["balanced_accuracy"] - P020_BALANCED,
            "joint_gain_over_p020": metrics["joint_exact"] - P020_JOINT,
            "fit_ratio": fit_ratio,
            "allocation_ratio": allocation_ratio,
        },
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
