from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.head_ownership_diagnostic import (
    BATCH_SIZE,
    EPOCHS,
    EXPECTED_BASELINE,
    EXPECTED_CASES_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_PARAMETER_HASH,
    KV_PAIRS,
    MAX_ACCURACY_DRIFT,
    MAX_COUNT_DRIFT,
    MIN_FROZEN_PREDICTION_AGREEMENT,
    SEED,
    SEQUENCE_LENGTH,
    _case_id,
    _checkpoint_state,
    _frozen_predictions,
    _mixers,
    _sha256,
    _summary,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
)


COMPONENTS = ("erase", "write")
SCOPES = {
    "l0": (0,),
    "l1": (1,),
    "both": (0, 1),
}
MIN_ERASE_SWAP_REPAIR = 0.20
MIN_ERASE_REPAIR_WITH_OWNER_RETAINED = 0.15
MIN_ERASE_OWNER_RETENTION = 0.80
MIN_WRITE_SWAP_REPAIR = 0.25
MIN_WRITE_REPAIR_ADVANTAGE = 0.05
MAX_WRITE_OWNER_RETENTION_FOR_COLLISION = 0.60


def _frozen_binding_map(path: Path) -> dict[str, list[dict[str, int]]]:
    result: dict[str, list[dict[str, int]]] = {}
    for case in json.loads(path.read_text()):
        case_id = str(case["case_id"])
        rows = [
            {
                "query_position": int(event["query_position"]),
                "write_position": int(event["write_position"]),
                "value_position": int(event["write_position"]) + 1,
                "key": int(event["key"]),
                "target": int(event["target"]),
            }
            for event in case["events"]
        ]
        rows.sort(key=lambda row: row["write_position"])
        if case_id in result or len(rows) != KV_PAIRS:
            raise RuntimeError("Frozen cases do not contain one four-binding row")
        result[case_id] = rows
    return result


def _bindings(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    frozen_rows: list[dict[str, int]],
) -> list[dict[str, int]]:
    rows = [dict(row) for row in frozen_rows]
    if len(rows) != KV_PAIRS:
        raise RuntimeError(f"Expected {KV_PAIRS} recovered bindings")
    if len({row["target"] for row in rows}) != KV_PAIRS:
        raise RuntimeError("Diagnostic requires four distinct MQAR values")
    if len({row["value_position"] for row in rows}) != KV_PAIRS:
        raise RuntimeError("Recovered value positions are not unique")
    query_positions = set(torch.nonzero(labels != -100).flatten().tolist())
    if query_positions != {row["query_position"] for row in rows}:
        raise RuntimeError("Frozen query positions do not match replay labels")
    for row in rows:
        if int(inputs[row["write_position"]].item()) != row["key"]:
            raise RuntimeError("Frozen write key does not match replay input")
        if int(inputs[row["value_position"]].item()) != row["target"]:
            raise RuntimeError("Frozen write value does not match replay input")
        if int(inputs[row["query_position"]].item()) != row["key"]:
            raise RuntimeError("Frozen query key does not match replay input")
        if int(labels[row["query_position"]].item()) != row["target"]:
            raise RuntimeError("Frozen query target does not match replay label")
    return rows


class _GateMaskController:
    def __init__(self) -> None:
        self.position_mask: torch.Tensor | None = None

    def set_mask(self, mask: torch.Tensor) -> None:
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("Gate intervention mask must be boolean [B,T]")
        self.position_mask = mask

    def hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.position_mask is None:
            raise RuntimeError("Gate intervention mask was not set")
        mask = self.position_mask.to(device=output.device)
        if tuple(mask.shape) != tuple(output.shape[:2]):
            raise RuntimeError("Gate intervention mask shape changed")
        return output.masked_fill(mask.unsqueeze(-1), float("-inf"))


@contextmanager
def _gate_mask_hooks(
    mixers: list[GatedDeltaNet2],
    component: str,
    layers: tuple[int, ...],
) -> Iterator[_GateMaskController]:
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")
    controller = _GateMaskController()
    handles = []
    for layer_index in layers:
        mixer = mixers[layer_index]
        projection = mixer.b_proj if component == "erase" else mixer.w_proj
        handles.append(projection.register_forward_hook(controller.hook))
    try:
        yield controller
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _evaluate_variant(
    model: torch.nn.Module,
    dataloader: Any,
    mixers: list[GatedDeltaNet2],
    binding_map: dict[str, list[dict[str, int]]],
    *,
    component: str | None,
    layers: tuple[int, ...] = (),
    owner_rank: int | None = None,
) -> dict[str, Any]:
    model.eval()
    quarter = SEQUENCE_LENGTH // 4
    predictions: list[int] = []
    targets: list[int] = []
    directions: list[str] = []
    case_indices: list[int] = []
    case_ids: list[str] = []
    query_positions: list[int] = []
    valid_values: list[list[int]] = []
    owner_values: list[list[int]] = []
    target_owner_ranks: list[int] = []
    case_index = 0
    torch.cuda.synchronize()
    started = time.perf_counter()

    hook_context = (
        nullcontext(None)
        if component is None
        else _gate_mask_hooks(mixers, component, layers)
    )
    with hook_context as controller:
        for inputs, labels, _slices in dataloader:
            batch_case_ids = [
                _case_id(inputs[row].cpu()) for row in range(inputs.shape[0])
            ]
            batch_bindings = [
                _bindings(inputs[row], labels[row], binding_map[batch_case_ids[row]])
                for row in range(inputs.shape[0])
            ]
            if controller is not None:
                if owner_rank is None or not 0 <= owner_rank < KV_PAIRS:
                    raise ValueError("Intervention requires one fixed owner rank")
                mask = torch.zeros(
                    inputs.shape[0], inputs.shape[1], dtype=torch.bool
                )
                for row, bindings in enumerate(batch_bindings):
                    mask[row, bindings[owner_rank]["value_position"]] = True
                controller.set_mask(mask)

            logits = model(inputs.cuda())
            pred = logits.argmax(dim=-1)
            for row, bindings in enumerate(batch_bindings):
                row_labels = labels[row].cpu()
                row_case_id = batch_case_ids[row]
                values_by_owner = [binding["target"] for binding in bindings]
                owner_by_query = {
                    binding["query_position"]: rank
                    for rank, binding in enumerate(bindings)
                }
                for position in torch.nonzero(row_labels != -100).flatten().tolist():
                    predictions.append(int(pred[row, position].item()))
                    targets.append(int(row_labels[position].item()))
                    directions.append("future" if position < quarter else "past")
                    case_indices.append(case_index)
                    case_ids.append(row_case_id)
                    query_positions.append(int(position))
                    valid_values.append(sorted(values_by_owner))
                    owner_values.append(values_by_owner)
                    target_owner_ranks.append(owner_by_query[int(position)])
                case_index += 1

    torch.cuda.synchronize()
    return {
        "predictions": predictions,
        "targets": targets,
        "directions": directions,
        "case_indices": case_indices,
        "case_ids": case_ids,
        "query_positions": query_positions,
        "valid_values": valid_values,
        "owner_values": owner_values,
        "target_owner_ranks": target_owner_ranks,
        "elapsed_sec": time.perf_counter() - started,
    }


def _family_analysis(
    component: str,
    scope: str,
    variants: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    predictions = baseline["predictions"]
    targets = baseline["targets"]
    values = baseline["valid_values"]
    owner_values = baseline["owner_values"]
    target_owner_ranks = baseline["target_owner_ranks"]
    case_indices = baseline["case_indices"]
    event_by_case_owner: dict[tuple[int, int], int] = {}
    for index, (case_index, owner_rank) in enumerate(
        zip(case_indices, target_owner_ranks)
    ):
        key = (case_index, owner_rank)
        if key in event_by_case_owner:
            raise RuntimeError("Each case/owner pair must have exactly one query")
        event_by_case_owner[key] = index

    swap_indices = [
        index
        for index, (prediction, target, row_values) in enumerate(
            zip(predictions, targets, values)
        )
        if prediction != target and prediction in row_values
    ]
    repairs = 0
    owner_query_eligible = 0
    owner_query_retained = 0
    repairs_with_owner_retained = 0
    swap_prediction_changes = 0
    selected_predictions = list(predictions)
    for index in swap_indices:
        predicted_owner = owner_values[index].index(predictions[index])
        name = f"{component}_off_{scope}_owner{predicted_owner}"
        intervened_prediction = variants[name]["predictions"][index]
        selected_predictions[index] = intervened_prediction
        repaired = intervened_prediction == targets[index]
        repairs += int(repaired)
        swap_prediction_changes += int(intervened_prediction != predictions[index])

        owner_index = event_by_case_owner[(case_indices[index], predicted_owner)]
        owner_was_correct = predictions[owner_index] == targets[owner_index]
        if owner_was_correct:
            owner_query_eligible += 1
            owner_retained = (
                variants[name]["predictions"][owner_index] == targets[owner_index]
            )
            owner_query_retained += int(owner_retained)
            repairs_with_owner_retained += int(repaired and owner_retained)

    self_owner_correct = 0
    self_owner_retained = 0
    for index, (prediction, target, owner_rank) in enumerate(
        zip(predictions, targets, target_owner_ranks)
    ):
        if prediction != target:
            continue
        self_owner_correct += 1
        name = f"{component}_off_{scope}_owner{owner_rank}"
        self_owner_retained += int(
            variants[name]["predictions"][index] == target
        )

    swap_count = len(swap_indices)
    return {
        "component": component,
        "scope": scope,
        "baseline_swaps": swap_count,
        "swap_repairs": repairs,
        "swap_repair_fraction": repairs / max(swap_count, 1),
        "swap_prediction_changes": swap_prediction_changes,
        "swap_prediction_change_fraction": swap_prediction_changes
        / max(swap_count, 1),
        "owner_query_eligible": owner_query_eligible,
        "owner_query_retained": owner_query_retained,
        "owner_query_retention_fraction": owner_query_retained
        / max(owner_query_eligible, 1),
        "repairs_with_owner_retained": repairs_with_owner_retained,
        "repair_with_owner_retained_fraction": repairs_with_owner_retained
        / max(swap_count, 1),
        "self_owner_baseline_correct": self_owner_correct,
        "self_owner_correct_retained": self_owner_retained,
        "self_owner_correct_retention_fraction": self_owner_retained
        / max(self_owner_correct, 1),
        "label_mediated_selected_summary": _summary(
            selected_predictions, baseline
        ),
    }


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
        raise RuntimeError("P-DIAG-EDIT-001 requires exactly one visible CUDA GPU")

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
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Frozen parameter count changed")
    if parameter_hash(model) != EXPECTED_PARAMETER_HASH:
        raise RuntimeError("Frozen trained parameter hash changed")
    model.cuda().eval()
    mixers = _mixers(model)
    binding_map = _frozen_binding_map(args.frozen_cases)
    _train_loader, test_loader = prepare_data(config.data)

    torch.cuda.reset_peak_memory_stats()
    variants: dict[str, dict[str, Any]] = {}
    variants["full"] = _evaluate_variant(
        model,
        test_loader,
        mixers,
        binding_map,
        component=None,
    )
    baseline = variants["full"]
    print(f"variant=full elapsed={baseline['elapsed_sec']:.3f}s", flush=True)

    frozen = _frozen_predictions(args.frozen_cases)
    replay_matches = sum(
        int(frozen[(case_id, position)] == prediction)
        for case_id, position, prediction in zip(
            baseline["case_ids"],
            baseline["query_positions"],
            baseline["predictions"],
        )
    )
    baseline_summary = _summary(baseline["predictions"], baseline)
    replay_fraction = replay_matches / len(baseline["predictions"])
    accuracy_stable = all(
        abs(baseline_summary[key] - EXPECTED_BASELINE[key])
        <= MAX_ACCURACY_DRIFT
        for key in (
            "balanced_accuracy",
            "future_accuracy",
            "past_accuracy",
            "joint_exact",
        )
    )
    count_stable = all(
        abs(baseline_summary[key] - EXPECTED_BASELINE[key]) <= MAX_COUNT_DRIFT
        for key in ("errors", "wrong_key_swaps")
    )
    if (
        replay_fraction < MIN_FROZEN_PREDICTION_AGREEMENT
        or not accuracy_stable
        or not count_stable
    ):
        raise RuntimeError("Frozen fresh-process baseline exceeds drift bounds")
    print(
        f"full_frozen_replay={replay_matches}/{len(baseline['predictions'])} "
        f"fraction={replay_fraction:.6f} metrics={baseline_summary}",
        flush=True,
    )

    for component in COMPONENTS:
        for scope, layers in SCOPES.items():
            for owner_rank in range(KV_PAIRS):
                name = f"{component}_off_{scope}_owner{owner_rank}"
                variants[name] = _evaluate_variant(
                    model,
                    test_loader,
                    mixers,
                    binding_map,
                    component=component,
                    layers=layers,
                    owner_rank=owner_rank,
                )
                print(
                    f"variant={name} elapsed={variants[name]['elapsed_sec']:.3f}s",
                    flush=True,
                )

    variant_summaries = {
        name: _summary(row["predictions"], baseline)
        for name, row in variants.items()
    }
    family_analyses = {
        f"{component}_{scope}": _family_analysis(
            component, scope, variants, baseline
        )
        for component in COMPONENTS
        for scope in SCOPES
    }
    erase = family_analyses["erase_both"]
    write = family_analyses["write_both"]
    erase_route_open = (
        erase["swap_repair_fraction"] >= MIN_ERASE_SWAP_REPAIR
        and erase["repair_with_owner_retained_fraction"]
        >= MIN_ERASE_REPAIR_WITH_OWNER_RETAINED
        and erase["owner_query_retention_fraction"] >= MIN_ERASE_OWNER_RETENTION
        and erase["self_owner_correct_retention_fraction"]
        >= MIN_ERASE_OWNER_RETENTION
    )
    write_collision_open = (
        write["swap_repair_fraction"] >= MIN_WRITE_SWAP_REPAIR
        and write["swap_repair_fraction"]
        >= erase["swap_repair_fraction"] + MIN_WRITE_REPAIR_ADVANTAGE
        and write["owner_query_retention_fraction"]
        <= MAX_WRITE_OWNER_RETENTION_FOR_COLLISION
    )
    if erase_route_open:
        decision = "open_coherent_erase_ownership_transition"
    elif write_collision_open:
        decision = "open_separable_state_topology_not_write_selector"
    else:
        decision = "close_local_gate_edits_as_distributed_binding_entanglement"

    output = {
        "status": "complete",
        "plan": "P-DIAG-EDIT-001",
        "decision": decision,
        "diagnostic_only": True,
        "deployable_selector": False,
        "intervention": {
            "position": "recovered MQAR value token at write_position+1",
            "components": list(COMPONENTS),
            "scopes": {name: list(layers) for name, layers in SCOPES.items()},
            "owner_rank": "ascending recovered write position, fixed per replay",
            "implementation": "projection-logit hook sets selected gate to exact sigmoid zero before official GDN2",
            "parameter_delta": 0,
            "state_delta": 0,
            "training_steps": 0,
        },
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "frozen_cases": str(args.frozen_cases),
            "frozen_cases_sha256": EXPECTED_CASES_SHA256,
            "trained_parameter_hash": EXPECTED_PARAMETER_HASH,
            "frozen_prediction_matches": replay_matches,
            "frozen_prediction_agreement": replay_fraction,
            "queries": len(baseline["predictions"]),
            "fresh_process_accuracy_stable": accuracy_stable,
            "fresh_process_count_stable": count_stable,
        },
        "baseline": baseline_summary,
        "variant_summaries": variant_summaries,
        "family_analyses": family_analyses,
        "registered_gate": {
            "erase": {
                "swap_repair_at_least_0.20": (
                    erase["swap_repair_fraction"] >= MIN_ERASE_SWAP_REPAIR
                ),
                "repair_with_owner_retained_at_least_0.15": (
                    erase["repair_with_owner_retained_fraction"]
                    >= MIN_ERASE_REPAIR_WITH_OWNER_RETAINED
                ),
                "owner_query_retention_at_least_0.80": (
                    erase["owner_query_retention_fraction"]
                    >= MIN_ERASE_OWNER_RETENTION
                ),
                "self_owner_correct_retention_at_least_0.80": (
                    erase["self_owner_correct_retention_fraction"]
                    >= MIN_ERASE_OWNER_RETENTION
                ),
                "route_open": erase_route_open,
            },
            "write_collision": {
                "swap_repair_at_least_0.25": (
                    write["swap_repair_fraction"] >= MIN_WRITE_SWAP_REPAIR
                ),
                "repair_advantage_over_erase_at_least_0.05": (
                    write["swap_repair_fraction"]
                    >= erase["swap_repair_fraction"] + MIN_WRITE_REPAIR_ADVANTAGE
                ),
                "owner_query_retention_at_most_0.60": (
                    write["owner_query_retention_fraction"]
                    <= MAX_WRITE_OWNER_RETENTION_FOR_COLLISION
                ),
                "route_open": write_collision_open,
            },
        },
        "systems": {
            "variant_count": len(variants),
            "total_inference_sec": sum(
                row["elapsed_sec"] for row in variants.values()
            ),
            "peak_cuda_bytes_allocated": torch.cuda.max_memory_allocated(),
            "peak_cuda_bytes_reserved": torch.cuda.max_memory_reserved(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"decision={decision} erase={erase['swap_repair_fraction']:.6f} "
        f"write={write['swap_repair_fraction']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
