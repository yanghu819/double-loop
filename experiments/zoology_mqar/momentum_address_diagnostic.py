from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from fla.modules.l2norm import l2norm_fwd
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.gdn2_diagnostics.committed_edit import key_gram_statistics
from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
MAX_EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
ARM = "future_seed_momentum_delta"
EXPECTED_CHECKPOINT_SHA256 = (
    "13410aaad3be26fbdf07c7a12e1afa9fb38ced41a1939a519c02973be2b8508f"
)
EXPECTED_FORMAL_SCORE_SHA256 = (
    "7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a"
)
EXPECTED_FORMAL_CASES_SHA256 = (
    "f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    expected = ("0", os.environ["EXPECTED_GPU_UUID"], os.environ["EXPECTED_GPU_NAME"])
    if (index, uuid, name) != expected:
        raise RuntimeError(f"GPU mismatch: {(index, uuid, name)} != {expected}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The diagnostic requires exactly one CUDA device")
    return {"index": 0, "uuid": uuid, "name": name}


class MomentumAddressRecorder:
    def __init__(self, operation: Any) -> None:
        self.operation = operation
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset(self) -> None:
        self.records.clear()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        q = kwargs["q"] if "q" in kwargs else args[0]
        k = kwargs["k"] if "k" in kwargs else args[1]
        self.records.append((q.detach().clone(), k.detach().clone()))
        return self.operation(*args, **kwargs)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "negative_fraction": float((tensor < 0).double().mean()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    categories = ("correct", "wrong_key_swap", "other_error")
    for category in categories:
        selected = [row for row in rows if row["category"] == category]
        output[category] = {
            "events": len(selected),
            "own_minus_best_other": _summary(
                [row["own_minus_best_other"] for row in selected]
            ),
            "own_head_top1_fraction": _summary(
                [row["own_head_top1_fraction"] for row in selected]
            ),
            "own_minus_prediction_owner": _summary(
                [
                    row["own_minus_prediction_owner"]
                    for row in selected
                    if row["own_minus_prediction_owner"] is not None
                ]
            ),
        }
    swaps = [row for row in rows if row["category"] == "wrong_key_swap"]
    output["wrong_key_swap"]["best_other_is_prediction_owner_fraction"] = (
        sum(row["best_other_is_prediction_owner"] for row in swaps) / len(swaps)
        if swaps
        else 0.0
    )
    output["wrong_key_swap"]["best_other_is_adjacent_owner_fraction"] = (
        sum(row["best_other_is_adjacent_owner"] for row in swaps) / len(swaps)
        if swaps
        else 0.0
    )
    return output


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
    recorder = MomentumAddressRecorder(original_operation)
    layer_module.chunk_mode_rule = recorder
    per_layer_rows: list[list[dict[str, Any]]] = [[], []]
    gram_rows: list[list[dict[str, float]]] = [[], []]
    replayed_cases = 0
    try:
        with torch.no_grad():
            for inputs, _targets, _slices in test_loader:
                recorder.reset()
                logits = model(inputs.cuda())
                predictions = logits.argmax(dim=-1).cpu()
                if len(recorder.records) != 2:
                    raise RuntimeError(
                        f"Expected two Momentum scans, got {len(recorder.records)}"
                    )
                normalized_records = []
                for q, k in recorder.records:
                    q_normalized, _ = l2norm_fwd(q)
                    k_normalized, _ = l2norm_fwd(k)
                    normalized_records.append((q_normalized, k_normalized))
                    gram = key_gram_statistics(k_normalized)
                    gram_rows[len(normalized_records) - 1].append(
                        {
                            "effective_rank_fraction": float(
                                gram["effective_rank_fraction"].mean()
                            ),
                            "anisotropy": float(gram["anisotropy"].mean()),
                            "condition": float(gram["condition"].mean()),
                        }
                    )

                for row_index in range(inputs.shape[0]):
                    case_index = replayed_cases + row_index
                    case = cases_by_index[case_index]
                    case_id = hashlib.sha256(
                        inputs[row_index].contiguous().numpy().tobytes()
                    ).hexdigest()[:16]
                    if case["case_id"] != case_id:
                        raise RuntimeError(f"Case identity drifted at {case_index}")
                    events = case["events"]
                    write_order = sorted(
                        range(len(events)),
                        key=lambda index: events[index]["write_position"],
                    )
                    write_ranks = {
                        event_index: rank
                        for rank, event_index in enumerate(write_order)
                    }
                    for event_index, event in enumerate(events):
                        if int(predictions[row_index, event["query_position"]]) != event["prediction"]:
                            raise RuntimeError(
                                f"Prediction replay drifted at case {case_index}"
                            )
                        prediction_owner = next(
                            (
                                index
                                for index, candidate_event in enumerate(events)
                                if index != event_index
                                and candidate_event["target"] == event["prediction"]
                            ),
                            None,
                        )
                        category = (
                            "correct"
                            if event["correct"]
                            else "wrong_key_swap"
                            if prediction_owner is not None
                            else "other_error"
                        )
                        for layer_index, (q, k) in enumerate(normalized_records):
                            query = q[row_index, event["query_position"]]
                            owner_keys = torch.stack(
                                [
                                    k[row_index, candidate_event["write_position"]]
                                    for candidate_event in events
                                ]
                            )
                            similarities = torch.einsum(
                                "hd,nhd->nh",
                                query,
                                owner_keys,
                            ).float()
                            mean_similarities = similarities.mean(dim=-1)
                            other_indices = [
                                index
                                for index in range(len(events))
                                if index != event_index
                            ]
                            best_other = max(
                                other_indices,
                                key=lambda index: float(mean_similarities[index]),
                            )
                            own = float(mean_similarities[event_index])
                            prediction_margin = (
                                None
                                if prediction_owner is None
                                else own - float(mean_similarities[prediction_owner])
                            )
                            per_layer_rows[layer_index].append(
                                {
                                    "case_index": case_index,
                                    "event_index": event_index,
                                    "category": category,
                                    "direction": event["direction"],
                                    "own_minus_best_other": (
                                        own - float(mean_similarities[best_other])
                                    ),
                                    "own_minus_prediction_owner": prediction_margin,
                                    "own_head_top1_fraction": float(
                                        (
                                            similarities.argmax(dim=0)
                                            == event_index
                                        ).float().mean()
                                    ),
                                    "best_other_is_prediction_owner": (
                                        prediction_owner is not None
                                        and best_other == prediction_owner
                                    ),
                                    "best_other_is_adjacent_owner": (
                                        abs(
                                            write_ranks[best_other]
                                            - write_ranks[event_index]
                                        )
                                        == 1
                                    ),
                                }
                            )
                replayed_cases += inputs.shape[0]
    finally:
        layer_module.chunk_mode_rule = original_operation

    if replayed_cases != 1_000:
        raise RuntimeError(f"Expected 1000 replayed cases, got {replayed_cases}")
    summaries = [_aggregate(rows) for rows in per_layer_rows]
    layer1 = summaries[1]
    swap_margin = layer1["wrong_key_swap"]["own_minus_prediction_owner"]
    correct_margin = layer1["correct"]["own_minus_best_other"]
    negative_swap_fraction = float(swap_margin["negative_fraction"])
    margin_separation = float(correct_margin["median"]) - float(
        swap_margin["median"]
    )
    prediction_owner_alignment = float(
        layer1["wrong_key_swap"]["best_other_is_prediction_owner_fraction"]
    )
    if (
        negative_swap_fraction >= 0.60
        and margin_separation >= 0.02
        and prediction_owner_alignment >= 0.50
    ):
        diagnosis = "address_geometry_points_to_wrong_owner"
        next_branch = "distinct_owner_address_organization"
    elif negative_swap_fraction <= 0.30:
        diagnosis = "address_prefers_correct_owner_but_state_returns_wrong_value"
        next_branch = "committed_state_edit_transition"
    else:
        diagnosis = "mixed_address_and_state_failure"
        next_branch = "no_architecture_until_more_specific_counterfactual"

    gram_summary = []
    for layer_rows in gram_rows:
        gram_summary.append(
            {
                key: _summary([row[key] for row in layer_rows])
                for key in (
                    "effective_rank_fraction",
                    "anisotropy",
                    "condition",
                )
            }
        )
    result = {
        "status": "complete",
        "plan": "P-DIAG-MOMADDR-001",
        "gpu": gpu,
        "protocol": {
            "training": False,
            "logits_modified": False,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "records": 4_000,
            "selection_thresholds": {
                "address_negative_swap_fraction": 0.60,
                "address_correct_vs_swap_median_separation": 0.02,
                "address_prediction_owner_alignment": 0.50,
                "state_negative_swap_fraction_max": 0.30,
            },
        },
        "trained_parameter_hash": trained_hash,
        "replayed_cases": replayed_cases,
        "per_layer": summaries,
        "key_gram": gram_summary,
        "decision_statistics": {
            "layer1_negative_prediction_owner_margin_fraction": negative_swap_fraction,
            "layer1_correct_vs_swap_median_margin_separation": margin_separation,
            "layer1_best_other_prediction_owner_fraction": prediction_owner_alignment,
        },
        "diagnosis": diagnosis,
        "next_branch": next_branch,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
