from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import fla.layers.gdn2 as gdn2_layer_module
from fla.layers.gdn2 import GatedDeltaNet2
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.edit_component_diagnostic import (
    _bindings,
    _frozen_binding_map,
)
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


TIED_PREDICTION_AGREEMENT_FLOOR = 0.98
TIED_BALANCED_DRIFT_CEILING = 0.02
DUAL_BALANCED_GAIN_FLOOR = 0.15
DUAL_ERROR_REDUCTION_FLOOR = 0.20
DUAL_SWAP_REPAIR_FLOOR = 0.50
DUAL_CORRECT_RETENTION_FLOOR = 0.80
PAIRING_OFFDIAGONAL_CEILING = 2e-4
PAIRING_DIAGONAL_FLOOR = 1e-3
GRAM_CONDITION_CEILING = 1e6


class _DualAddressController:
    def __init__(
        self,
        original_chunk: Any,
        *,
        dualize_layer0: bool,
    ) -> None:
        self.original_chunk = original_chunk
        self.dualize_layer0 = bool(dualize_layer0)
        self.query_positions: torch.Tensor | None = None
        self.write_positions: torch.Tensor | None = None
        self.call_index = 0
        self._diagnostics: dict[str, list[torch.Tensor]] = {
            "gram_condition_max": [],
            "gram_eigenvalue_min": [],
            "pairing_offdiagonal_max": [],
            "pairing_diagonal_min": [],
            "native_owner_pairing_diagonal_mean": [],
            "write_address_relative_rms": [],
        }

    def begin_batch(
        self,
        query_positions: torch.Tensor,
        write_positions: torch.Tensor,
    ) -> None:
        if self.query_positions is not None or self.write_positions is not None:
            raise RuntimeError("Previous diagnostic batch did not finish")
        if query_positions.shape != write_positions.shape:
            raise ValueError("Query/write owner tables must have identical shape")
        if query_positions.ndim != 2 or query_positions.shape[1] != KV_PAIRS:
            raise ValueError("Owner tables must be [B,4]")
        self.query_positions = query_positions
        self.write_positions = write_positions
        self.call_index = 0

    def finish_batch(self) -> None:
        if self.call_index != 2:
            raise RuntimeError(
                f"Expected exactly two official layer calls, got {self.call_index}"
            )
        self.query_positions = None
        self.write_positions = None

    @staticmethod
    def _gather_owner_rows(
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, _length, heads, width = tensor.shape
        index = positions.to(device=tensor.device).view(batch, KV_PAIRS, 1, 1)
        index = index.expand(batch, KV_PAIRS, heads, width)
        return tensor.gather(1, index).permute(0, 2, 1, 3)

    @staticmethod
    def _replace_owner_rows(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        result = tensor.clone()
        rows = positions.to(device=tensor.device)
        for owner in range(KV_PAIRS):
            result[
                torch.arange(tensor.shape[0], device=tensor.device),
                rows[:, owner],
            ] = values[:, :, owner]
        return result

    def _dual_write_address(
        self,
        q_unit: torch.Tensor,
        k_unit: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_positions is None or self.write_positions is None:
            raise RuntimeError("Owner positions were not registered")
        with torch.autocast(device_type=q_unit.device.type, enabled=False):
            owner_q = self._gather_owner_rows(
                q_unit, self.query_positions
            ).float()
            native_write = self._gather_owner_rows(
                k_unit, self.write_positions
            ).float()
            gram = torch.einsum("bhik,bhjk->bhij", owner_q, owner_q)
            eigenvalues = torch.linalg.eigvalsh(gram)
            if not torch.isfinite(eigenvalues).all() or (eigenvalues <= 0).any():
                raise RuntimeError("Owner query Gram is not finite positive definite")
            condition = eigenvalues[..., -1] / eigenvalues[..., 0]
            if float(condition.max().item()) > GRAM_CONDITION_CEILING:
                raise RuntimeError(
                    "Owner query Gram exceeds the registered condition cap"
                )

            dual = torch.linalg.solve(gram, owner_q)
            dual_unit = F.normalize(dual, dim=-1, eps=1e-6)
            pairing = torch.einsum("bhik,bhjk->bhij", owner_q, dual_unit)
            diagonal = pairing.diagonal(dim1=-2, dim2=-1)
            offdiagonal = pairing - torch.diag_embed(diagonal)
            native_diagonal = torch.einsum(
                "bhik,bhik->bhi", owner_q, native_write
            )
            relative_change = (
                (dual_unit - native_write).square().mean().sqrt()
                / native_write.square().mean().sqrt().clamp_min(1e-8)
            )
        self._diagnostics["gram_condition_max"].append(condition.max().detach())
        self._diagnostics["gram_eigenvalue_min"].append(
            eigenvalues[..., 0].min().detach()
        )
        self._diagnostics["pairing_offdiagonal_max"].append(
            offdiagonal.abs().max().detach()
        )
        self._diagnostics["pairing_diagonal_min"].append(
            diagonal.abs().min().detach()
        )
        self._diagnostics["native_owner_pairing_diagonal_mean"].append(
            native_diagonal.mean().detach()
        )
        self._diagnostics["write_address_relative_rms"].append(
            relative_change.detach()
        )
        return self._replace_owner_rows(
            k_unit,
            self.write_positions,
            dual_unit.to(dtype=k_unit.dtype),
        )

    def diagnostics(self) -> dict[str, float | int]:
        if not self._diagnostics["gram_condition_max"]:
            return {"active_layers": 0, "batches": 0}
        return {
            "active_layers": 1 if self.dualize_layer0 else 0,
            "batches": len(self._diagnostics["gram_condition_max"]),
            "gram_condition_max": float(
                torch.stack(self._diagnostics["gram_condition_max"]).max().item()
            ),
            "gram_eigenvalue_min": float(
                torch.stack(self._diagnostics["gram_eigenvalue_min"]).min().item()
            ),
            "pairing_offdiagonal_max": float(
                torch.stack(self._diagnostics["pairing_offdiagonal_max"]).max().item()
            ),
            "pairing_diagonal_min": float(
                torch.stack(self._diagnostics["pairing_diagonal_min"]).min().item()
            ),
            "native_owner_pairing_diagonal_mean": float(
                torch.stack(
                    self._diagnostics["native_owner_pairing_diagonal_mean"]
                ).mean().item()
            ),
            "write_address_relative_rms": float(
                torch.stack(
                    self._diagnostics["write_address_relative_rms"]
                ).mean().item()
            ),
        }

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layer_index = self.call_index
        self.call_index += 1
        if layer_index != 0:
            return self.original_chunk(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                use_gate_in_kernel=use_gate_in_kernel,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        if not use_qk_l2norm_in_kernel or use_gate_in_kernel or cu_seqlens is not None:
            raise RuntimeError("Diagnostic requires normalized, unpacked official GDN2")
        q_unit = F.normalize(q.float(), dim=-1, eps=1e-6).to(dtype=q.dtype)
        k_erase = F.normalize(k.float(), dim=-1, eps=1e-6).to(dtype=k.dtype)
        k_write = (
            self._dual_write_address(q_unit, k_erase)
            if self.dualize_layer0
            else k_erase
        )
        alpha = torch.exp(g.float()) * b.float() * k_erase.float()
        beta = -k_erase.float()
        return chunk_dplr_delta_rule(
            q=q_unit,
            k=k_write,
            v=w * v,
            a=alpha.to(dtype=k.dtype),
            b=beta.to(dtype=k.dtype),
            gk=g,
            scale=(q.shape[-1] ** -0.5 if scale is None else scale),
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=None,
            safe_gate=False,
            chunk_size=16,
        )


@contextmanager
def _replace_layer0_transition(
    *,
    dualize_layer0: bool,
) -> Iterator[_DualAddressController]:
    original = gdn2_layer_module.chunk_gdn2
    controller = _DualAddressController(
        original,
        dualize_layer0=dualize_layer0,
    )
    gdn2_layer_module.chunk_gdn2 = controller
    try:
        yield controller
    finally:
        gdn2_layer_module.chunk_gdn2 = original


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    binding_map: dict[str, list[dict[str, int]]],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"native", "tied_dplr_l0", "oracle_dual_l0"}:
        raise ValueError(f"Unknown diagnostic mode: {mode}")
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

    transition_context = (
        nullcontext(None)
        if mode == "native"
        else _replace_layer0_transition(dualize_layer0=mode == "oracle_dual_l0")
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with transition_context as controller:
        for inputs, labels, _slices in dataloader:
            batch_case_ids = [
                _case_id(inputs[row].cpu()) for row in range(inputs.shape[0])
            ]
            batch_bindings = [
                _bindings(inputs[row], labels[row], binding_map[batch_case_ids[row]])
                for row in range(inputs.shape[0])
            ]
            owner_queries = torch.tensor(
                [
                    [binding["query_position"] for binding in bindings]
                    for bindings in batch_bindings
                ],
                dtype=torch.long,
            )
            owner_writes = torch.tensor(
                [
                    [binding["value_position"] for binding in bindings]
                    for bindings in batch_bindings
                ],
                dtype=torch.long,
            )
            if controller is not None:
                controller.begin_batch(owner_queries, owner_writes)
            logits = model(inputs.cuda())
            if controller is not None:
                controller.finish_batch()
            pred = logits.argmax(dim=-1)

            for row, bindings in enumerate(batch_bindings):
                row_labels = labels[row].cpu()
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
                    case_ids.append(batch_case_ids[row])
                    query_positions.append(int(position))
                    valid_values.append(sorted(values_by_owner))
                    owner_values.append(values_by_owner)
                    target_owner_ranks.append(owner_by_query[int(position)])
                case_index += 1
    torch.cuda.synchronize()
    result = {
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
        "transition": {} if controller is None else controller.diagnostics(),
    }
    return result


def _transition_analysis(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base_pred = baseline["predictions"]
    cand_pred = candidate["predictions"]
    targets = baseline["targets"]
    values = baseline["valid_values"]
    baseline_correct = [i for i, (p, y) in enumerate(zip(base_pred, targets)) if p == y]
    baseline_swaps = [
        i
        for i, (p, y, valid) in enumerate(zip(base_pred, targets, values))
        if p != y and p in valid
    ]
    candidate_errors = sum(p != y for p, y in zip(cand_pred, targets))
    baseline_errors = sum(p != y for p, y in zip(base_pred, targets))
    repairs = sum(cand_pred[i] == targets[i] for i in baseline_swaps)
    correct_retained = sum(cand_pred[i] == targets[i] for i in baseline_correct)
    transitions: dict[str, int] = {}
    for before, after, target, valid in zip(base_pred, cand_pred, targets, values):
        before_class = (
            "correct" if before == target else "wrong_key" if before in valid else "other_wrong"
        )
        after_class = (
            "correct" if after == target else "wrong_key" if after in valid else "other_wrong"
        )
        key = f"{before_class}_to_{after_class}"
        transitions[key] = transitions.get(key, 0) + 1
    return {
        "baseline_errors": baseline_errors,
        "candidate_errors": candidate_errors,
        "error_reduction_fraction": (baseline_errors - candidate_errors)
        / max(baseline_errors, 1),
        "baseline_wrong_key_swaps": len(baseline_swaps),
        "wrong_key_swap_repairs": repairs,
        "wrong_key_swap_repair_fraction": repairs / max(len(baseline_swaps), 1),
        "baseline_correct": len(baseline_correct),
        "baseline_correct_retained": correct_retained,
        "baseline_correct_retention_fraction": correct_retained
        / max(len(baseline_correct), 1),
        "paired_transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen checkpoint hash changed")
    if _sha256(args.frozen_cases) != EXPECTED_CASES_SHA256:
        raise RuntimeError("Frozen cases hash changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P-DIAG-DUAL-001 requires exactly one visible CUDA GPU")

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
    if any(type(mixer) is not GatedDeltaNet2 for mixer in mixers):
        raise RuntimeError("Diagnostic requires exact official GDN2 layers")
    binding_map = _frozen_binding_map(args.frozen_cases)
    _train_loader, test_loader = prepare_data(config.data)

    torch.cuda.reset_peak_memory_stats()
    native = _evaluate(model, test_loader, binding_map, mode="native")
    tied = _evaluate(model, test_loader, binding_map, mode="tied_dplr_l0")
    dual = _evaluate(model, test_loader, binding_map, mode="oracle_dual_l0")
    native_summary = _summary(native["predictions"], native)
    tied_summary = _summary(tied["predictions"], native)
    dual_summary = _summary(dual["predictions"], native)

    frozen = _frozen_predictions(args.frozen_cases)
    replay_matches = sum(
        frozen[(case_id, position)] == prediction
        for case_id, position, prediction in zip(
            native["case_ids"], native["query_positions"], native["predictions"]
        )
    )
    replay_fraction = replay_matches / len(native["predictions"])
    accuracy_stable = all(
        abs(native_summary[key] - EXPECTED_BASELINE[key]) <= MAX_ACCURACY_DRIFT
        for key in ("balanced_accuracy", "future_accuracy", "past_accuracy", "joint_exact")
    )
    count_stable = all(
        abs(native_summary[key] - EXPECTED_BASELINE[key]) <= MAX_COUNT_DRIFT
        for key in ("errors", "wrong_key_swaps")
    )
    if (
        replay_fraction < MIN_FROZEN_PREDICTION_AGREEMENT
        or not accuracy_stable
        or not count_stable
    ):
        raise RuntimeError("Frozen native baseline exceeds drift bounds")

    tied_agreement = sum(
        left == right for left, right in zip(native["predictions"], tied["predictions"])
    ) / len(native["predictions"])
    tied_control_ok = (
        tied_agreement >= TIED_PREDICTION_AGREEMENT_FLOOR
        and abs(
            tied_summary["balanced_accuracy"] - native_summary["balanced_accuracy"]
        )
        <= TIED_BALANCED_DRIFT_CEILING
    )
    transition = _transition_analysis(native, dual)
    dual_diag = dual["transition"]
    geometry_ok = (
        dual_diag["active_layers"] == 1
        and dual_diag["pairing_offdiagonal_max"] <= PAIRING_OFFDIAGONAL_CEILING
        and dual_diag["pairing_diagonal_min"] >= PAIRING_DIAGONAL_FLOOR
        and dual_diag["gram_condition_max"] <= GRAM_CONDITION_CEILING
    )
    quality_ceiling_open = (
        dual_summary["balanced_accuracy"] - native_summary["balanced_accuracy"]
        >= DUAL_BALANCED_GAIN_FLOOR
        and transition["error_reduction_fraction"] >= DUAL_ERROR_REDUCTION_FLOOR
        and transition["wrong_key_swap_repair_fraction"] >= DUAL_SWAP_REPAIR_FLOOR
        and transition["baseline_correct_retention_fraction"]
        >= DUAL_CORRECT_RETENTION_FLOOR
    )
    decision = (
        "open_causal_bounded_dual_state_transition"
        if tied_control_ok and geometry_ok and quality_ceiling_open
        else "close_direct_dual_address_route"
    )
    output = {
        "status": "complete",
        "plan": "P-DIAG-DUAL-001",
        "diagnostic_only": True,
        "deployable_model": False,
        "uses_frozen_owner_positions": True,
        "decision": decision,
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "frozen_cases": str(args.frozen_cases),
            "frozen_cases_sha256": EXPECTED_CASES_SHA256,
            "trained_parameter_hash": EXPECTED_PARAMETER_HASH,
            "frozen_prediction_matches": replay_matches,
            "frozen_prediction_agreement": replay_fraction,
            "queries": len(native["predictions"]),
        },
        "intervention": {
            "layer": 0,
            "owner_count": KV_PAIRS,
            "erase_address": "unchanged native normalized key",
            "write_address": "normalized dual of the four frozen owner-query directions",
            "query_address": "unchanged native normalized query",
            "implementation": "one official DPLR scan replacing only layer0 official GDN2 for the diagnostic",
            "new_parameters": 0,
            "new_persistent_state": 0,
            "training_steps": 0,
            "causal_or_deployable": False,
        },
        "native": {"summary": native_summary, "elapsed_sec": native["elapsed_sec"]},
        "tied_dplr_layer0": {
            "summary": tied_summary,
            "elapsed_sec": tied["elapsed_sec"],
            "prediction_agreement_with_native": tied_agreement,
            "control_ok": tied_control_ok,
        },
        "oracle_dual_layer0": {
            "summary": dual_summary,
            "elapsed_sec": dual["elapsed_sec"],
            "transition": transition,
            "geometry": dual_diag,
        },
        "registered_gate": {
            "geometry_ok": geometry_ok,
            "quality_ceiling_open": quality_ceiling_open,
            "all_passed": tied_control_ok and geometry_ok and quality_ceiling_open,
            "thresholds": {
                "tied_prediction_agreement_floor": TIED_PREDICTION_AGREEMENT_FLOOR,
                "tied_balanced_drift_ceiling": TIED_BALANCED_DRIFT_CEILING,
                "dual_balanced_gain_floor": DUAL_BALANCED_GAIN_FLOOR,
                "dual_error_reduction_floor": DUAL_ERROR_REDUCTION_FLOOR,
                "dual_swap_repair_floor": DUAL_SWAP_REPAIR_FLOOR,
                "dual_correct_retention_floor": DUAL_CORRECT_RETENTION_FLOOR,
                "pairing_offdiagonal_ceiling": PAIRING_OFFDIAGONAL_CEILING,
                "pairing_diagonal_floor": PAIRING_DIAGONAL_FLOOR,
                "gram_condition_ceiling": GRAM_CONDITION_CEILING,
            },
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
