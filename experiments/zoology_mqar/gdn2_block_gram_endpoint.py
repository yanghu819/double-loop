from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_block_gram import frozen_prefix_gram_diagnostic
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
P020_SCORE_SHA256 = "a9c750e84fc918910c008b679016c3f24a5e821e3c20b84b1f0b07c1c64b9b3e"
P020_CASES_SHA256 = "127fb40e32a14a54f6f90c784dfe728c85dcf200535e0fb300e3e28a55c8751b"
P020_CHECKPOINT_SHA256 = "e12633faf53c615040fe685fc0cd1a2acd24cbc27a023a7e7bdfcf8d55473e54"
P020_BALANCED = 0.48225
P020_JOINT = 0.044
P020_SWAP = 0.9415741187831965


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_p020(reference_dir: Path) -> dict[str, Any]:
    score_path = reference_dir / "score.json"
    cases_path = reference_dir / "cases.json"
    checkpoint_path = reference_dir / "model_state.pt"
    expected = {
        score_path: P020_SCORE_SHA256,
        cases_path: P020_CASES_SHA256,
        checkpoint_path: P020_CHECKPOINT_SHA256,
    }
    for path, digest in expected.items():
        if _sha(path) != digest:
            raise RuntimeError(f"Frozen P020 artifact changed: {path}")
    score = json.loads(score_path.read_text())
    cases = json.loads(cases_path.read_text())
    swaps = wrong_key_swap_summary(cases)
    if abs(float(score["metrics"]["balanced_accuracy"]) - P020_BALANCED) > 1e-12:
        raise RuntimeError("Frozen P020 balanced accuracy changed")
    if abs(float(score["metrics"]["joint_exact"]) - P020_JOINT) > 1e-12:
        raise RuntimeError("Frozen P020 joint exact changed")
    if abs(float(swaps["wrong_key_swap_fraction_of_errors"]) - P020_SWAP) > 1e-12:
        raise RuntimeError("Frozen P020 wrong-key swap evidence changed")
    score["checkpoint_path"] = str(checkpoint_path)
    return {"score": score, "swap_summary": swaps, "cases_path": str(cases_path)}


def load_frozen_p020_model(reference: dict[str, Any]) -> tuple[torch.nn.Module, Any]:
    config = build_config(
        arm="future_seed_gdn2_log_spd",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(123)
    model = make_model(config, "future_seed_gdn2_log_spd")
    payload = torch.load(reference["score"]["checkpoint_path"], map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if parameter_hash(model) != reference["score"]["trained_parameter_hash"]:
        raise RuntimeError("Frozen P020 model state does not match its score")
    _train_loader, test_loader = prepare_data(config.data)
    return model.cuda(), test_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p020-reference", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.max_epochs != MAX_EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("P-GDN3-026 has one fixed 10-epoch/batch32 arm")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_p020(args.p020_reference)
    model, test_loader = load_frozen_p020_model(reference)
    diagnostic = frozen_prefix_gram_diagnostic(model, test_loader)
    del model
    torch.cuda.empty_cache()
    diagnostic_checks = {
        "transparent_zero_quality_intervention": not diagnostic["intervention_changes_logits"],
        "fixed_128_examples": diagnostic["examples"] == 128,
        "rank_gain_median_at_least_0.01": (
            diagnostic["effective_rank_fraction_gain"]["median"] >= 0.01
        ),
        "rank_improved_fraction_at_least_0.60": (
            diagnostic["effective_rank_improved_fraction"] >= 0.60
        ),
        "anisotropy_ratio_median_at_most_0.90": (
            diagnostic["anisotropy_ratio"]["median"] <= 0.90
        ),
        "anisotropy_improved_fraction_at_least_0.60": (
            diagnostic["anisotropy_improved_fraction"] >= 0.60
        ),
        "binding_gain_median_positive": diagnostic["binding_margin_gain"]["median"] > 0.0,
        "binding_improved_fraction_at_least_0.55": (
            diagnostic["binding_improved_fraction"] >= 0.55
        ),
        "future_binding_improved_fraction_at_least_0.55": (
            diagnostic["binding"]["future"]["improved_fraction"] >= 0.55
        ),
        "past_binding_improved_fraction_at_least_0.55": (
            diagnostic["binding"]["past"]["improved_fraction"] >= 0.55
        ),
    }
    admission = {
        "passed": all(diagnostic_checks.values()),
        "checks": diagnostic_checks,
        "diagnostic": diagnostic,
        "decision": "train_single_candidate" if all(diagnostic_checks.values()) else "close",
    }
    (args.output_dir / "diagnostic_admission.json").write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n"
    )
    if not admission["passed"]:
        print(json.dumps(admission, indent=2, sort_keys=True))
        raise SystemExit(3)

    (args.output_dir / "candidate_started.json").write_text(
        json.dumps(
            {
                "plan": "P-GDN3-026",
                "arm": "future_seed_gdn2_log_spd_block_gram",
                "p020_checkpoint_sha256": P020_CHECKPOINT_SHA256,
                "diagnostic_sample_sha256": diagnostic["sample_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    candidate = run_arm(
        arm="future_seed_gdn2_log_spd_block_gram",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        output_dir=args.output_dir,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        save_checkpoint=True,
    )
    if candidate["parent_init_parameter_hash"] != reference["score"]["init_parameter_hash"]:
        raise RuntimeError("Candidate did not preserve the frozen P020 initialization")
    if candidate["data_hashes"] != reference["score"]["data_hashes"]:
        raise RuntimeError("Directional MQAR data changed")
    cases_path = (
        args.output_dir
        / "length_1024"
        / "future_seed_gdn2_log_spd_block_gram"
        / "cases.json"
    )
    candidate_swaps = wrong_key_swap_summary(json.loads(cases_path.read_text()))
    metrics = candidate["metrics"]
    activation_checks = {
        "exactly_two_active_layers": candidate["block_gram"]["active_layers"] == 2,
        "exactly_eight_new_parameters": (
            candidate["parameters"] - reference["score"]["parameters"] == 8
        ),
        "all_head_strengths_active": all(
            row["strength_abs_min"] >= 1e-4 for row in candidate["block_gram"]["per_layer"]
        ),
        "all_factors_nontrivial": all(
            row["factor_delta_fro_mean"] >= 1e-4
            for row in candidate["block_gram"]["per_layer"]
        ),
        "all_factors_bounded": all(
            row["factor_eigenvalue_min"] >= 0.75 - 1e-5
            and row["factor_eigenvalue_max"] <= 1.25 + 1e-5
            and row["factor_condition_max"] < 1.67
            for row in candidate["block_gram"]["per_layer"]
        ),
    }
    quality_checks = {
        "balanced_accuracy_at_least_0.65": metrics["balanced_accuracy"] >= 0.65,
        "balanced_gain_over_p020_at_least_0.10": (
            metrics["balanced_accuracy"] - P020_BALANCED >= 0.10
        ),
        "joint_exact_at_least_0.15": metrics["joint_exact"] >= 0.15,
        "past_accuracy_at_least_0.62": metrics["past"]["accuracy"] >= 0.62,
        "future_accuracy_at_least_0.62": metrics["future"]["accuracy"] >= 0.62,
        "wrong_key_swap_fraction_reduced_at_least_0.10": (
            P020_SWAP - float(candidate_swaps["wrong_key_swap_fraction_of_errors"]) >= 0.10
        ),
    }
    control = reference["score"]
    cost_checks = {
        "fit_elapsed_overhead_below_35_percent": (
            candidate["elapsed_sec_including_validation"]
            < 1.35 * control["elapsed_sec_including_validation"]
        ),
        "post_warm_wall_overhead_below_35_percent": (
            candidate["post_warm_arm_wall_sec_through_checkpoint"]
            < 1.35 * control["post_warm_arm_wall_sec_through_checkpoint"]
        ),
        "warmed_step_overhead_below_35_percent": (
            candidate["warmed_step_benchmark"]["elapsed_sec"]
            < 1.35 * control["warmed_step_benchmark"]["elapsed_sec"]
        ),
        "peak_allocation_overhead_below_20_percent": (
            candidate["peak_training_cuda_mem_bytes"]
            < 1.20 * control["peak_training_cuda_mem_bytes"]
        ),
    }
    comparison = {
        "status": "complete",
        "protocol": {
            "plan": "P-GDN3-026",
            "mechanism": "static Log-SPD plus block-causal prefix-Gram query conditioner",
            "sequence_length": 1024,
            "block_size": 64,
            "epochs": 10,
            "batch_size": 32,
            "seed": 123,
            "new_parameters_over_p020": 8,
            "new_payload_state": 0,
            "new_official_scans": 0,
        },
        "frozen_p020": reference,
        "diagnostic_admission": admission,
        "candidate": candidate,
        "candidate_swap_summary": candidate_swaps,
        "registered_gate": {
            "passed": all(activation_checks.values())
            and all(quality_checks.values())
            and all(cost_checks.values()),
            "activation_checks": activation_checks,
            "quality_checks": quality_checks,
            "cost_checks": cost_checks,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
