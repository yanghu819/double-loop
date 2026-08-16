from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
)
from experiments.zoology_mqar.momentum_address_diagnostic import (
    ARM,
    BATCH_SIZE,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_FORMAL_CASES_SHA256,
    EXPECTED_FORMAL_SCORE_SHA256,
    MAX_EPOCHS,
    NUM_KV_PAIRS,
    SEED,
    SEQUENCE_LENGTH,
    _gpu_identity,
    _sha256,
)


MODES = {
    "baseline": frozenset(),
    "layer0_own_key": frozenset({0}),
    "layer1_own_key": frozenset({1}),
    "both_layers_own_key": frozenset({0, 1}),
}


class QueryOwnerIntervention:
    def __init__(self, operation: Any) -> None:
        self.operation = operation
        self.target_layers: frozenset[int] = frozenset()
        self.events: list[list[dict[str, Any]]] = []
        self.call_index = 0

    def begin(
        self,
        target_layers: frozenset[int],
        events: list[list[dict[str, Any]]],
    ) -> None:
        self.target_layers = target_layers
        self.events = events
        self.call_index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        layer_index = self.call_index
        self.call_index += 1
        if layer_index not in self.target_layers:
            return self.operation(*args, **kwargs)

        q = kwargs["q"] if "q" in kwargs else args[0]
        k = kwargs["k"] if "k" in kwargs else args[1]
        q_modified = q.clone()
        for row_index, row_events in enumerate(self.events):
            for event in row_events:
                q_modified[row_index, event["query_position"]] = k[
                    row_index, event["write_position"]
                ]

        if "q" in kwargs:
            modified_kwargs = dict(kwargs)
            modified_kwargs["q"] = q_modified
            return self.operation(*args, **modified_kwargs)
        modified_args = list(args)
        modified_args[0] = q_modified
        return self.operation(*modified_args, **kwargs)


def _baseline_category(
    event_index: int,
    events: list[dict[str, Any]],
) -> str:
    event = events[event_index]
    if event["correct"]:
        return "correct"
    if any(
        index != event_index and candidate["target"] == event["prediction"]
        for index, candidate in enumerate(events)
    ):
        return "wrong_key_swap"
    return "other_error"


def _mode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    baseline_correct = [row for row in rows if row["baseline_category"] == "correct"]
    swaps = [row for row in rows if row["baseline_category"] == "wrong_key_swap"]
    other_errors = [
        row for row in rows if row["baseline_category"] == "other_error"
    ]

    def accuracy(selected: list[dict[str, Any]]) -> float:
        if not selected:
            return 0.0
        return sum(row["prediction"] == row["target"] for row in selected) / len(
            selected
        )

    directions = {}
    for direction in ("future", "past"):
        selected = [row for row in rows if row["direction"] == direction]
        directions[direction] = {
            "events": len(selected),
            "accuracy": accuracy(selected),
        }
    return {
        "events": total,
        "accuracy": accuracy(rows),
        "errors": sum(row["prediction"] != row["target"] for row in rows),
        "directions": directions,
        "baseline_correct": {
            "events": len(baseline_correct),
            "retained_fraction": accuracy(baseline_correct),
            "new_errors": sum(
                row["prediction"] != row["target"] for row in baseline_correct
            ),
        },
        "wrong_key_swap": {
            "events": len(swaps),
            "repaired_fraction": accuracy(swaps),
            "still_original_prediction_fraction": (
                sum(row["prediction"] == row["baseline_prediction"] for row in swaps)
                / len(swaps)
                if swaps
                else 0.0
            ),
        },
        "other_error": {
            "events": len(other_errors),
            "repaired_fraction": accuracy(other_errors),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-score", type=Path, required=True)
    parser.add_argument("--formal-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.formal_score: EXPECTED_FORMAL_SCORE_SHA256,
        args.formal_cases: EXPECTED_FORMAL_CASES_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")

    gpu = _gpu_identity()
    formal = json.loads(args.formal_score.read_text())
    formal_cases = json.loads(args.formal_cases.read_text())
    frozen = formal["candidate"]
    if frozen["arm"] != ARM:
        raise RuntimeError("Frozen score is not P-GDN3-059")
    cases_by_index = {case["case_index"]: case for case in formal_cases}
    if len(cases_by_index) != 1_000:
        raise RuntimeError("Frozen case bank is incomplete")

    config = build_config(
        arm=ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    _train_loader, test_loader = prepare_data(config.data)
    test_digest = dataset_hash(test_loader)
    if test_digest != frozen["data_hashes"]["test"]:
        raise RuntimeError("P059 test data hash drifted")

    model = make_model(config, ARM).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained_hash = parameter_hash(model)
    if trained_hash != frozen["trained_parameter_hash"]:
        raise RuntimeError("P059 checkpoint parameter hash drifted")

    layer_module = importlib.import_module("fla.layers.momentum_deltanet")
    original_operation = layer_module.chunk_mode_rule
    intervention = QueryOwnerIntervention(original_operation)
    layer_module.chunk_mode_rule = intervention
    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    replayed_cases = 0
    try:
        with torch.no_grad():
            for inputs, _targets, _slices in test_loader:
                batch_cases = []
                for row_index in range(inputs.shape[0]):
                    case_index = replayed_cases + row_index
                    case = cases_by_index[case_index]
                    case_id = hashlib.sha256(
                        inputs[row_index].contiguous().numpy().tobytes()
                    ).hexdigest()[:16]
                    if case["case_id"] != case_id:
                        raise RuntimeError(f"Case identity drifted at {case_index}")
                    batch_cases.append(case)

                for mode, target_layers in MODES.items():
                    intervention.begin(
                        target_layers,
                        [case["events"] for case in batch_cases],
                    )
                    predictions = model(inputs.cuda()).argmax(dim=-1).cpu()
                    if intervention.call_index != 2:
                        raise RuntimeError(
                            f"Expected two Momentum scans, got {intervention.call_index}"
                        )
                    for row_index, case in enumerate(batch_cases):
                        for event_index, event in enumerate(case["events"]):
                            prediction = int(
                                predictions[row_index, event["query_position"]]
                            )
                            if mode == "baseline" and prediction != event["prediction"]:
                                raise RuntimeError(
                                    f"Prediction replay drifted at case {case['case_index']}"
                                )
                            mode_rows[mode].append(
                                {
                                    "case_index": case["case_index"],
                                    "event_index": event_index,
                                    "direction": event["direction"],
                                    "target": event["target"],
                                    "baseline_prediction": event["prediction"],
                                    "baseline_category": _baseline_category(
                                        event_index,
                                        case["events"],
                                    ),
                                    "prediction": prediction,
                                }
                            )
                replayed_cases += inputs.shape[0]
    finally:
        layer_module.chunk_mode_rule = original_operation

    if replayed_cases != 1_000:
        raise RuntimeError(f"Expected 1000 replayed cases, got {replayed_cases}")
    summaries = {mode: _mode_summary(rows) for mode, rows in mode_rows.items()}
    baseline_accuracy = float(summaries["baseline"]["accuracy"])
    layer1 = summaries["layer1_own_key"]
    layer1_gain = float(layer1["accuracy"]) - baseline_accuracy
    layer1_swap_repair = float(layer1["wrong_key_swap"]["repaired_fraction"])
    layer1_correct_retention = float(
        layer1["baseline_correct"]["retained_fraction"]
    )
    max_swap_repair = max(
        float(summary["wrong_key_swap"]["repaired_fraction"])
        for mode, summary in summaries.items()
        if mode != "baseline"
    )
    if (
        layer1_swap_repair >= 0.50
        and layer1_correct_retention >= 0.95
        and layer1_gain >= 0.015
    ):
        diagnosis = "query_address_is_causal_bottleneck"
        next_branch = "distinct_owner_address_organization"
    elif max_swap_repair <= 0.10:
        diagnosis = "exact_owner_query_does_not_recover_state_content"
        next_branch = "committed_state_edit_transition"
    else:
        diagnosis = "address_state_interaction_without_clean_owner_intervention"
        next_branch = "no_architecture_until_more_specific_counterfactual"

    result = {
        "status": "complete",
        "plan": "P-DIAG-MOMQCF-001",
        "gpu": gpu,
        "protocol": {
            "training": False,
            "formal_quality_claim": False,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "intervention": (
                "replace corrected Momentum query at each MQAR query token with "
                "the corrected key at its own write token"
            ),
            "selection_thresholds": {
                "layer1_swap_repair": 0.50,
                "layer1_correct_retention": 0.95,
                "layer1_accuracy_gain": 0.015,
                "state_max_swap_repair": 0.10,
            },
        },
        "trained_parameter_hash": trained_hash,
        "replayed_cases": replayed_cases,
        "modes": summaries,
        "decision_statistics": {
            "layer1_accuracy_gain": layer1_gain,
            "layer1_swap_repair_fraction": layer1_swap_repair,
            "layer1_correct_retention_fraction": layer1_correct_retention,
            "max_nonbaseline_swap_repair_fraction": max_swap_repair,
        },
        "diagnosis": diagnosis,
        "next_branch": next_branch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
