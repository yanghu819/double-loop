from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.length_scaling import (
    MODEL_HEADS,
    build_config,
    make_model,
    parameter_hash,
)


SEQUENCE_LENGTH = 1024
KV_PAIRS = 4
EPOCHS = 10
BATCH_SIZE = 32
SEED = 123
EXPECTED_PARAMETERS = 661_584
EXPECTED_PARAMETER_HASH = (
    "fa28ed30664a92d3536ed52132fb70887550fec5f2176f9e11711e20c04898b3"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e35c61928583f5630c45c8477bbcfa0d38cdc1441eb0b08407020787707c2adf"
)
EXPECTED_CASES_SHA256 = (
    "2322ddc61dac4eb1819c6cac160b3a280c17c2bb0dfdde1fae96a6858b9b32bd"
)
EXPECTED_BASELINE = {
    "balanced_accuracy": 0.494,
    "future_accuracy": 0.454,
    "past_accuracy": 0.534,
    "joint_exact": 0.041,
    "errors": 2024,
    "wrong_key_swaps": 1546,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(inputs: torch.Tensor) -> str:
    return hashlib.sha256(inputs.contiguous().numpy().tobytes()).hexdigest()[:16]


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError("Unexpected frozen checkpoint container")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Frozen checkpoint model_state_dict is invalid")
    return state


def _frozen_predictions(path: Path) -> dict[tuple[str, int], int]:
    rows = json.loads(path.read_text())
    predictions: dict[tuple[str, int], int] = {}
    for case in rows:
        case_id = str(case["case_id"])
        for event in case["events"]:
            key = (case_id, int(event["query_position"]))
            predictions[key] = int(event["prediction"])
    return predictions


def _mixers(model: torch.nn.Module) -> list[GatedDeltaNet2]:
    mixers: list[GatedDeltaNet2] = []
    for block in model.backbone.layers:
        layer = block.sequence_mixer.layer
        if type(layer) is not GatedDeltaNet2:
            raise TypeError("Diagnostic requires exact official GatedDeltaNet2 layers")
        if layer.num_heads != MODEL_HEADS or layer.head_v_dim != 32:
            raise RuntimeError("Frozen D128/H4/V32 geometry changed")
        mixers.append(layer)
    if len(mixers) != 2:
        raise RuntimeError("Diagnostic requires exactly two GDN2 layers")
    return mixers


@contextmanager
def _head_mask_hooks(
    mixers: list[GatedDeltaNet2],
    masks: dict[int, torch.Tensor],
) -> Iterator[None]:
    handles = []
    for layer_index, mixer in enumerate(mixers):
        mask = masks[layer_index]
        if tuple(mask.shape) != (MODEL_HEADS,):
            raise ValueError("Each head mask must contain four entries")

        def hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            *,
            layer_mask: torch.Tensor = mask,
        ) -> tuple[torch.Tensor, ...]:
            value = inputs[0]
            expanded = layer_mask.to(device=value.device, dtype=value.dtype)
            expanded = expanded.repeat_interleave(value.shape[-1] // MODEL_HEADS)
            return (value * expanded.view(1, 1, -1), *inputs[1:])

        handles.append(mixer.o_proj.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _variant_masks() -> dict[str, dict[int, torch.Tensor]]:
    variants = {
        "full": {
            0: torch.ones(MODEL_HEADS),
            1: torch.ones(MODEL_HEADS),
        }
    }
    for layer in range(2):
        for head in range(MODEL_HEADS):
            drop = torch.ones(MODEL_HEADS)
            drop[head] = 0.0
            only = torch.zeros(MODEL_HEADS)
            only[head] = 1.0
            variants[f"drop_l{layer}_h{head}"] = {
                0: drop.clone() if layer == 0 else torch.ones(MODEL_HEADS),
                1: drop.clone() if layer == 1 else torch.ones(MODEL_HEADS),
            }
            variants[f"only_l{layer}_h{head}"] = {
                0: only.clone() if layer == 0 else torch.ones(MODEL_HEADS),
                1: only.clone() if layer == 1 else torch.ones(MODEL_HEADS),
            }
    return variants


@torch.no_grad()
def _evaluate_variant(
    model: torch.nn.Module,
    dataloader: Any,
    mixers: list[GatedDeltaNet2],
    masks: dict[int, torch.Tensor] | None,
) -> dict[str, Any]:
    model.eval()
    quarter = SEQUENCE_LENGTH // 4
    predictions: list[int] = []
    targets: list[int] = []
    directions: list[str] = []
    case_indices: list[int] = []
    case_ids: list[str] = []
    query_positions: list[int] = []
    top1_margins: list[float] = []
    valid_values: list[list[int]] = []
    case_index = 0
    torch.cuda.synchronize()
    started = time.perf_counter()

    hook_context = (
        nullcontext() if masks is None else _head_mask_hooks(mixers, masks)
    )
    with hook_context:
        for inputs, labels, _slices in dataloader:
            logits = model(inputs.cuda())
            labels_gpu = labels.cuda()
            query_mask = labels_gpu != -100
            top2 = logits.topk(k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            pred = logits.argmax(dim=-1)

            for row in range(inputs.shape[0]):
                row_inputs = inputs[row].cpu()
                row_labels = labels[row].cpu()
                row_case_id = _case_id(row_inputs)
                row_values = sorted(
                    int(value.item()) for value in row_labels[row_labels != -100]
                )
                for position in torch.nonzero(row_labels != -100).flatten().tolist():
                    predictions.append(int(pred[row, position].item()))
                    targets.append(int(row_labels[position].item()))
                    directions.append("future" if position < quarter else "past")
                    case_indices.append(case_index)
                    case_ids.append(row_case_id)
                    query_positions.append(int(position))
                    top1_margins.append(float(margin[row, position].item()))
                    valid_values.append(row_values)
                case_index += 1

    torch.cuda.synchronize()
    return {
        "predictions": predictions,
        "targets": targets,
        "directions": directions,
        "case_indices": case_indices,
        "case_ids": case_ids,
        "query_positions": query_positions,
        "top1_margins": top1_margins,
        "valid_values": valid_values,
        "elapsed_sec": time.perf_counter() - started,
    }


def _summary(predictions: list[int], reference: dict[str, Any]) -> dict[str, Any]:
    targets = reference["targets"]
    directions = reference["directions"]
    case_indices = reference["case_indices"]
    valid_values = reference["valid_values"]
    by_direction: dict[str, dict[str, float | int]] = {}
    errors = 0
    swaps = 0
    correct_by_case: dict[int, bool] = {}
    for prediction, target, direction, case_index, values in zip(
        predictions,
        targets,
        directions,
        case_indices,
        valid_values,
    ):
        bucket = by_direction.setdefault(direction, {"correct": 0, "queries": 0})
        is_correct = prediction == target
        bucket["correct"] += int(is_correct)
        bucket["queries"] += 1
        correct_by_case[case_index] = (
            correct_by_case.get(case_index, True) and is_correct
        )
        if not is_correct:
            errors += 1
            swaps += int(prediction in values)
    for bucket in by_direction.values():
        bucket["accuracy"] = bucket["correct"] / bucket["queries"]
    return {
        "balanced_accuracy": 0.5
        * (
            float(by_direction["future"]["accuracy"])
            + float(by_direction["past"]["accuracy"])
        ),
        "future_accuracy": by_direction["future"]["accuracy"],
        "past_accuracy": by_direction["past"]["accuracy"],
        "joint_exact": sum(correct_by_case.values()) / len(correct_by_case),
        "errors": errors,
        "wrong_key_swaps": swaps,
        "swap_fraction_of_errors": swaps / max(errors, 1),
    }


def _majority(
    names: list[str],
    variants: dict[str, dict[str, Any]],
    baseline: list[int],
) -> list[int]:
    output = []
    for index, fallback in enumerate(baseline):
        counts = Counter(variants[name]["predictions"][index] for name in names)
        winner, count = counts.most_common(1)[0]
        ties = [value for value, value_count in counts.items() if value_count == count]
        output.append(fallback if len(ties) > 1 else winner)
    return output


def _max_margin(
    names: list[str],
    variants: dict[str, dict[str, Any]],
) -> list[int]:
    output = []
    for index in range(len(variants["full"]["targets"])):
        selected = max(names, key=lambda name: variants[name]["top1_margins"][index])
        output.append(variants[selected]["predictions"][index])
    return output


def _oracle(
    names: list[str],
    variants: dict[str, dict[str, Any]],
) -> list[int]:
    targets = variants["full"]["targets"]
    baseline = variants["full"]["predictions"]
    output = []
    for index, target in enumerate(targets):
        output.append(
            target
            if any(variants[name]["predictions"][index] == target for name in names)
            else baseline[index]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if _sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen replay-B checkpoint hash changed")
    if _sha256(args.frozen_cases) != EXPECTED_CASES_SHA256:
        raise RuntimeError("Frozen replay-B cases hash changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P-DIAG-OWN-002 requires exactly one visible CUDA GPU")

    config = build_config(
        arm="future_seed_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=KV_PAIRS,
        max_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    set_determinism(SEED)
    model = make_model(config, "future_seed_gdn2")
    model.load_state_dict(_checkpoint_state(args.checkpoint), strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen parameter count changed")
    if parameter_hash(model) != EXPECTED_PARAMETER_HASH:
        raise RuntimeError("Frozen trained parameter hash changed")
    model.cuda().eval()
    mixers = _mixers(model)
    _train_loader, test_loader = prepare_data(config.data)

    masks_by_variant = _variant_masks()
    variants: dict[str, dict[str, Any]] = {}
    torch.cuda.reset_peak_memory_stats()
    variants["full"] = _evaluate_variant(model, test_loader, mixers, None)
    print(
        f"variant=full elapsed={variants['full']['elapsed_sec']:.3f}s",
        flush=True,
    )

    baseline = variants["full"]
    frozen = _frozen_predictions(args.frozen_cases)
    replay_matches = sum(
        int(frozen[(case_id, position)] == prediction)
        for case_id, position, prediction in zip(
            baseline["case_ids"],
            baseline["query_positions"],
            baseline["predictions"],
        )
    )
    print(
        f"full_query_exact_replay={replay_matches}/{len(baseline['predictions'])}",
        flush=True,
    )
    if replay_matches != len(baseline["predictions"]):
        raise RuntimeError("Frozen baseline replay is not query-exact")
    baseline_summary = _summary(baseline["predictions"], baseline)
    if any(
        baseline_summary[key] != expected
        for key, expected in EXPECTED_BASELINE.items()
    ):
        raise RuntimeError(f"Frozen baseline metrics changed: {baseline_summary}")

    for name, masks in masks_by_variant.items():
        if name == "full":
            continue
        variants[name] = _evaluate_variant(model, test_loader, mixers, masks)
        elapsed = variants[name]["elapsed_sec"]
        print(f"variant={name} elapsed={elapsed:.3f}s", flush=True)

    variant_summaries = {
        name: _summary(row["predictions"], baseline)
        for name, row in variants.items()
    }

    drop_names = [name for name in variants if name.startswith("drop_")]
    only_names = [name for name in variants if name.startswith("only_")]
    final_drop = [name for name in drop_names if name.startswith("drop_l1")]
    final_only = [name for name in only_names if name.startswith("only_l1")]
    all_names = list(variants)
    derived_predictions = {
        "majority_final_drop": _majority(
            final_drop, variants, baseline["predictions"]
        ),
        "majority_final_only": _majority(
            final_only, variants, baseline["predictions"]
        ),
        "max_margin_all": _max_margin(all_names, variants),
        "max_margin_final_only": _max_margin(["full", *final_only], variants),
        "max_margin_drop": _max_margin(["full", *drop_names], variants),
        "oracle_all": _oracle(all_names, variants),
        "oracle_drop": _oracle(["full", *drop_names], variants),
        "oracle_only": _oracle(["full", *only_names], variants),
        "oracle_final_only": _oracle(["full", *final_only], variants),
    }
    derived_summaries = {
        name: _summary(predictions, baseline)
        for name, predictions in derived_predictions.items()
    }

    swap_indices = [
        index
        for index, (prediction, target, values) in enumerate(
            zip(
                baseline["predictions"],
                baseline["targets"],
                baseline["valid_values"],
            )
        )
        if prediction != target and prediction in values
    ]
    corrected_by_drop = sum(
        any(
            variants[name]["predictions"][index]
            == baseline["targets"][index]
            for name in drop_names
        )
        for index in swap_indices
    )
    corrected_by_only = sum(
        any(
            variants[name]["predictions"][index]
            == baseline["targets"][index]
            for name in only_names
        )
        for index in swap_indices
    )
    corrected_by_final_only = sum(
        any(
            variants[name]["predictions"][index]
            == baseline["targets"][index]
            for name in final_only
        )
        for index in swap_indices
    )
    unanimous_wrong_final_only = sum(
        all(
            variants[name]["predictions"][index]
            == baseline["predictions"][index]
            for name in final_only
        )
        for index in swap_indices
    )
    margin_summary = derived_summaries["max_margin_final_only"]
    oracle_swap_repair = corrected_by_final_only / len(swap_indices)
    readout_route_open = (
        oracle_swap_repair >= 0.25
        and margin_summary["balanced_accuracy"]
        >= baseline_summary["balanced_accuracy"] + 0.02
        and margin_summary["wrong_key_swaps"]
        <= 0.90 * baseline_summary["wrong_key_swaps"]
    )

    output = {
        "status": "complete",
        "plan": "P-DIAG-OWN-002",
        "decision": (
            "open_query_dependent_head_confidence"
            if readout_route_open
            else "close_head_fusion_and_target_within_head_transition"
        ),
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "frozen_cases": str(args.frozen_cases),
            "frozen_cases_sha256": EXPECTED_CASES_SHA256,
            "trained_parameter_hash": EXPECTED_PARAMETER_HASH,
            "query_exact_replay": replay_matches,
            "queries": len(baseline["predictions"]),
        },
        "baseline": baseline_summary,
        "variant_summaries": variant_summaries,
        "derived_summaries": derived_summaries,
        "swap_topology": {
            "baseline_wrong_key_swaps": len(swap_indices),
            "corrected_by_any_drop": corrected_by_drop,
            "corrected_by_any_drop_fraction": corrected_by_drop / len(swap_indices),
            "corrected_by_any_only": corrected_by_only,
            "corrected_by_any_only_fraction": corrected_by_only
            / len(swap_indices),
            "corrected_by_final_only": corrected_by_final_only,
            "corrected_by_final_only_fraction": oracle_swap_repair,
            "unanimous_baseline_wrong_owner_across_final_only": (
                unanimous_wrong_final_only
            ),
            "unanimous_baseline_wrong_owner_fraction": (
                unanimous_wrong_final_only / len(swap_indices)
            ),
        },
        "registered_gate": {
            "final_only_oracle_swap_repair_at_least_0.25": (
                oracle_swap_repair >= 0.25
            ),
            "final_only_max_margin_balanced_gain_at_least_0.02": (
                margin_summary["balanced_accuracy"]
                >= baseline_summary["balanced_accuracy"] + 0.02
            ),
            "final_only_max_margin_swaps_reduce_at_least_0.10": (
                margin_summary["wrong_key_swaps"]
                <= 0.90 * baseline_summary["wrong_key_swaps"]
            ),
            "readout_route_open": readout_route_open,
        },
        "systems": {
            "variant_count": len(variants),
            "total_inference_sec": sum(row["elapsed_sec"] for row in variants.values()),
            "peak_cuda_mem_bytes": int(torch.cuda.max_memory_allocated()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
