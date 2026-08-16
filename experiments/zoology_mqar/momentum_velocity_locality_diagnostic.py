from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Optional

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


DIAGNOSTIC_BATCH_SIZE = 128
REPLAY_RELATIVE_RMS_MAX = 0.05


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((delta / scale).item())


def _exact_l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return value / (value.square().sum(dim=-1, keepdim=True).sqrt() + 1e-6)


def _rms(tensor: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    return tensor.float().square().mean(dim=dimensions).sqrt()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    numerator = (left_flat * right_flat).sum(dim=-1)
    denominator = (
        left_flat.square().sum(dim=-1).sqrt()
        * right_flat.square().sum(dim=-1).sqrt()
    ).clamp_min(1e-8)
    return numerator / denominator


def _event_category(
    event_index: int,
    events: list[dict[str, Any]],
) -> tuple[str, Optional[int]]:
    event = events[event_index]
    if event["correct"]:
        return "correct", None
    prediction_owner = next(
        (
            index
            for index, candidate in enumerate(events)
            if index != event_index and candidate["target"] == event["prediction"]
        ),
        None,
    )
    if prediction_owner is not None:
        return "wrong_key_swap", prediction_owner
    return "other_error", None


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
        "finite_fraction": float(torch.isfinite(tensor).double().mean()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "old_velocity_rms",
        "write_update_rms",
        "old_velocity_to_update_ratio",
        "old_velocity_update_cosine",
        "native_post_commit_residual_rms",
        "erase_post_commit_residual_rms",
        "erase_to_native_residual_ratio",
        "erase_state_perturbation_relative_rms",
        "manual_state_relative_rms",
        "own_prediction_owner_key_cosine",
    )
    result: dict[str, Any] = {}
    for layer in (0, 1):
        layer_rows = [row for row in rows if row["layer"] == layer]
        result[f"layer{layer}"] = {}
        for direction in ("future", "past"):
            direction_rows = [
                row for row in layer_rows if row["direction"] == direction
            ]
            result[f"layer{layer}"][direction] = {}
            for category in ("correct", "wrong_key_swap", "other_error"):
                selected = [
                    row
                    for row in direction_rows
                    if row["category"] == category
                ]
                result[f"layer{layer}"][direction][category] = {
                    "events": len(selected),
                    **{
                        metric: _summary(
                            [
                                row[metric]
                                for row in selected
                                if row[metric] is not None
                            ]
                        )
                        for metric in metrics
                    },
                }
    causal_rows = [row for row in rows if row["causal_layer"]]
    result["causal_layer"] = {}
    for category in ("correct", "wrong_key_swap", "other_error"):
        selected = [row for row in causal_rows if row["category"] == category]
        result["causal_layer"][category] = {
            "events": len(selected),
            **{
                metric: _summary(
                    [row[metric] for row in selected if row[metric] is not None]
                )
                for metric in metrics
            },
        }
    return result


def _operation_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    index: int,
    default: Any = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


class MomentumVelocityLocalityRecorder:
    def __init__(self, operation: Any, recurrent_operation: Any) -> None:
        self.operation = operation
        self.recurrent_operation = recurrent_operation
        self.batch_cases: list[dict[str, Any]] = []
        self.call_index = 0
        self.rows: list[dict[str, Any]] = []
        self.replay_checks: list[dict[str, float | int]] = []

    def begin(self, batch_cases: list[dict[str, Any]]) -> None:
        self.batch_cases = batch_cases
        self.call_index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        layer_index = self.call_index
        self.call_index += 1
        if layer_index not in (0, 1):
            raise RuntimeError(f"Unexpected Momentum layer call {layer_index}")

        parent_output, parent_final_state = self.operation(*args, **kwargs)
        q = _operation_argument(args, kwargs, "q", 0)
        key = _operation_argument(args, kwargs, "k", 1)
        value = _operation_argument(args, kwargs, "v", 2)
        log_alpha = _operation_argument(args, kwargs, "log_alpha", 3)
        log_mu = _operation_argument(args, kwargs, "log_mu", 4)
        auxiliary_key = _operation_argument(args, kwargs, "p", 5)
        beta = _operation_argument(args, kwargs, "beta", 6)
        eta = _operation_argument(args, kwargs, "eta", 7)
        scale = _operation_argument(args, kwargs, "scale", 8)
        initial_state = _operation_argument(args, kwargs, "initial_state", 9)
        output_final_state = bool(
            _operation_argument(args, kwargs, "output_final_state", 10, False)
        )
        cu_seqlens = _operation_argument(args, kwargs, "cu_seqlens", 11)
        normalize_in_kernel = bool(
            _operation_argument(
                args,
                kwargs,
                "use_qk_l2norm_in_kernel",
                12,
                True,
            )
        )
        p_times_alpha = bool(
            _operation_argument(
                args,
                kwargs,
                "use_p_times_alpha",
                13,
                True,
            )
        )
        if (
            auxiliary_key is not None
            or cu_seqlens is not None
            or not output_final_state
            or not normalize_in_kernel
            or not p_times_alpha
        ):
            raise RuntimeError("P059 Momentum recurrence contract drifted")
        if beta is None or eta is None or beta.ndim != 3 or eta.ndim != 3:
            raise RuntimeError("P059 requires scalar per-token/head beta and eta")
        if q.shape[0] != len(self.batch_cases) or q.shape[1] != SEQUENCE_LENGTH:
            raise RuntimeError("Diagnostic batch/case alignment drifted")

        event_positions: dict[int, list[tuple[int, int]]] = {}
        write_ranks: list[dict[int, int]] = []
        for row_index, case in enumerate(self.batch_cases):
            events = case["events"]
            order = sorted(
                range(len(events)),
                key=lambda event_index: events[event_index]["write_position"],
            )
            write_ranks.append(
                {event_index: rank for rank, event_index in enumerate(order)}
            )
            for event_index, event in enumerate(events):
                event_positions.setdefault(event["write_position"], []).append(
                    (row_index, event_index)
                )

        if initial_state is None:
            state = torch.zeros(
                2,
                q.shape[0],
                q.shape[2],
                key.shape[-1],
                value.shape[-1],
                device=q.device,
                dtype=torch.float32,
            )
        else:
            state = initial_state.detach().float().clone()
        sequential_outputs = []
        for token_index in range(q.shape[1]):
            token_slice = slice(token_index, token_index + 1)
            previous_state = state
            token_output, state = self.recurrent_operation(
                q=q[:, token_slice],
                k=key[:, token_slice],
                v=value[:, token_slice],
                log_alpha=log_alpha[:, token_slice],
                log_mu=log_mu[:, token_slice],
                beta=beta[:, token_slice],
                eta=eta[:, token_slice],
                scale=scale,
                initial_state=previous_state,
                output_final_state=True,
                cu_seqlens=None,
                use_qk_l2norm_in_kernel=True,
                use_p_times_alpha=True,
            )
            sequential_outputs.append(token_output)
            entries = event_positions.get(token_index)
            if not entries:
                continue

            row_indices = torch.tensor(
                [entry[0] for entry in entries],
                device=q.device,
                dtype=torch.long,
            )
            key_now = _exact_l2_normalize(key[row_indices, token_index])
            value_now = value[row_indices, token_index].float()
            alpha_now = log_alpha[row_indices, token_index].float().exp()
            mu_now = log_mu[row_indices, token_index].float().exp()
            beta_now = beta[row_indices, token_index].float()
            eta_now = eta[row_indices, token_index].float()
            state_before = previous_state[:, row_indices].float()
            state_after = state[:, row_indices].float()
            state_before_s, state_before_m = state_before[0], state_before[1]

            decayed_m = mu_now[..., None, None] * state_before_m
            committed_residual = value_now - alpha_now[..., None] * torch.einsum(
                "rhk,rhkv->rhv",
                key_now,
                state_before_s,
            )
            write_update = eta_now[..., None] * committed_residual
            old_velocity = torch.einsum(
                "rhk,rhkv->rhv",
                key_now,
                decayed_m,
            )
            native_m = decayed_m - (
                key_now.unsqueeze(-1) * write_update.unsqueeze(-2)
            )
            native_s = (
                alpha_now[..., None, None] * state_before_s
                - beta_now[..., None, None] * native_m
            )
            cleaned_m = decayed_m - (
                key_now.unsqueeze(-1) * old_velocity.unsqueeze(-2)
            )
            erase_m = cleaned_m - (
                key_now.unsqueeze(-1) * write_update.unsqueeze(-2)
            )
            erase_s = (
                alpha_now[..., None, None] * state_before_s
                - beta_now[..., None, None] * erase_m
            )
            native_post_residual = value_now - torch.einsum(
                "rhk,rhkv->rhv",
                key_now,
                native_s,
            )
            erase_post_residual = value_now - torch.einsum(
                "rhk,rhkv->rhv",
                key_now,
                erase_s,
            )

            old_velocity_rms = _rms(old_velocity, (1, 2))
            write_update_rms = _rms(write_update, (1, 2))
            native_residual_rms = _rms(native_post_residual, (1, 2))
            erase_residual_rms = _rms(erase_post_residual, (1, 2))
            erase_perturbation = _rms(erase_s - native_s, (1, 2, 3)) / _rms(
                native_s,
                (1, 2, 3),
            ).clamp_min(1e-8)
            manual_state = torch.stack((native_s, native_m), dim=0)

            for local_index, (row_index, event_index) in enumerate(entries):
                case = self.batch_cases[row_index]
                events = case["events"]
                event = events[event_index]
                category, prediction_owner = _event_category(event_index, events)
                owner_cosine = None
                owner_rank_distance = None
                if prediction_owner is not None:
                    owner_event = events[prediction_owner]
                    owner_key = _exact_l2_normalize(
                        key[
                            row_index,
                            owner_event["write_position"],
                        ]
                    )
                    owner_cosine = float(
                        (key_now[local_index] * owner_key)
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    owner_rank_distance = abs(
                        write_ranks[row_index][event_index]
                        - write_ranks[row_index][prediction_owner]
                    )
                manual_error = _relative_rms(
                    manual_state[:, local_index],
                    state_after[:, local_index],
                )
                self.rows.append(
                    {
                        "case_index": case["case_index"],
                        "event_index": event_index,
                        "layer": layer_index,
                        "direction": event["direction"],
                        "category": category,
                        "causal_layer": (
                            event["direction"] == "future" and layer_index == 0
                        )
                        or (event["direction"] == "past" and layer_index == 1),
                        "write_position": token_index,
                        "query_position": event["query_position"],
                        "owner_rank_distance": owner_rank_distance,
                        "old_velocity_rms": float(
                            old_velocity_rms[local_index].item()
                        ),
                        "write_update_rms": float(
                            write_update_rms[local_index].item()
                        ),
                        "old_velocity_to_update_ratio": float(
                            (
                                old_velocity_rms[local_index]
                                / write_update_rms[local_index].clamp_min(1e-8)
                            ).item()
                        ),
                        "old_velocity_update_cosine": float(
                            _cosine(old_velocity, write_update)[local_index].item()
                        ),
                        "native_post_commit_residual_rms": float(
                            native_residual_rms[local_index].item()
                        ),
                        "erase_post_commit_residual_rms": float(
                            erase_residual_rms[local_index].item()
                        ),
                        "erase_to_native_residual_ratio": float(
                            (
                                erase_residual_rms[local_index]
                                / native_residual_rms[local_index].clamp_min(1e-8)
                            ).item()
                        ),
                        "erase_state_perturbation_relative_rms": float(
                            erase_perturbation[local_index].item()
                        ),
                        "manual_state_relative_rms": manual_error,
                        "own_prediction_owner_key_cosine": owner_cosine,
                        "alpha": float(alpha_now[local_index].mean().item()),
                        "mu": float(mu_now[local_index].mean().item()),
                        "beta": float(beta_now[local_index].mean().item()),
                        "eta": float(eta_now[local_index].mean().item()),
                    }
                )

        sequential_output = torch.cat(sequential_outputs, dim=1)
        output_error = _relative_rms(sequential_output, parent_output)
        state_error = _relative_rms(state, parent_final_state)
        self.replay_checks.append(
            {
                "layer": layer_index,
                "batch": len(self.replay_checks) // 2,
                "output_relative_rms": output_error,
                "state_relative_rms": state_error,
            }
        )
        if (
            output_error > REPLAY_RELATIVE_RMS_MAX
            or state_error > REPLAY_RELATIVE_RMS_MAX
        ):
            raise RuntimeError(
                f"Chunk/recurrent replay drifted: output={output_error}, "
                f"state={state_error}"
            )
        return parent_output, parent_final_state


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

    swap_case_indices = sorted(
        case_index
        for case_index, case in cases_by_index.items()
        if any(
            _event_category(event_index, case["events"])[0] == "wrong_key_swap"
            for event_index in range(len(case["events"]))
        )
    )
    all_correct_indices = sorted(
        (
            case_index
            for case_index, case in cases_by_index.items()
            if all(event["correct"] for event in case["events"])
        ),
        key=lambda case_index: hashlib.sha256(
            cases_by_index[case_index]["case_id"].encode()
        ).hexdigest(),
    )
    if not swap_case_indices or len(all_correct_indices) < len(swap_case_indices):
        raise RuntimeError("Frozen cases cannot provide the registered matched sample")
    correct_case_indices = all_correct_indices[: len(swap_case_indices)]
    selected_indices = set(swap_case_indices) | set(correct_case_indices)

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

    selected_inputs: list[torch.Tensor] = []
    selected_cases: list[dict[str, Any]] = []
    case_cursor = 0
    for inputs, _targets, _slices in test_loader:
        for row_index in range(inputs.shape[0]):
            case_index = case_cursor + row_index
            if case_index not in selected_indices:
                continue
            case = cases_by_index[case_index]
            case_id = hashlib.sha256(
                inputs[row_index].contiguous().numpy().tobytes()
            ).hexdigest()[:16]
            if case["case_id"] != case_id:
                raise RuntimeError(f"Case identity drifted at {case_index}")
            selected_inputs.append(inputs[row_index].clone())
            selected_cases.append(case)
        case_cursor += inputs.shape[0]
    if case_cursor != 1_000 or len(selected_inputs) != len(selected_indices):
        raise RuntimeError("Selected case materialization was incomplete")

    model = make_model(config, ARM).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trained_hash = parameter_hash(model)
    if trained_hash != frozen["trained_parameter_hash"]:
        raise RuntimeError("P059 checkpoint parameter hash drifted")

    layer_module = importlib.import_module("fla.layers.momentum_deltanet")
    operation_module = importlib.import_module("fla.ops.momentum_delta_rule")
    original_operation = layer_module.chunk_mode_rule
    recorder = MomentumVelocityLocalityRecorder(
        original_operation,
        operation_module.fused_recurrent_mode_rule,
    )
    layer_module.chunk_mode_rule = recorder
    replayed_cases = 0
    try:
        with torch.no_grad():
            for start in range(0, len(selected_inputs), DIAGNOSTIC_BATCH_SIZE):
                stop = min(start + DIAGNOSTIC_BATCH_SIZE, len(selected_inputs))
                batch_inputs = torch.stack(selected_inputs[start:stop]).cuda()
                batch_cases = selected_cases[start:stop]
                recorder.begin(batch_cases)
                predictions = model(batch_inputs).argmax(dim=-1).cpu()
                if recorder.call_index != 2:
                    raise RuntimeError(
                        f"Expected two Momentum scans, got {recorder.call_index}"
                    )
                for row_index, case in enumerate(batch_cases):
                    for event in case["events"]:
                        prediction = int(
                            predictions[row_index, event["query_position"]]
                        )
                        if prediction != event["prediction"]:
                            raise RuntimeError(
                                f"Prediction replay drifted at case {case['case_index']}"
                            )
                replayed_cases += len(batch_cases)
    finally:
        layer_module.chunk_mode_rule = original_operation

    if replayed_cases != len(selected_indices):
        raise RuntimeError("Diagnostic replay count drifted")
    if not recorder.rows or not all(
        math.isfinite(value)
        for row in recorder.rows
        for value in row.values()
        if isinstance(value, float)
    ):
        raise RuntimeError("Diagnostic emitted empty or nonfinite metrics")
    maximum_manual_state_error = max(
        row["manual_state_relative_rms"] for row in recorder.rows
    )
    if maximum_manual_state_error > REPLAY_RELATIVE_RMS_MAX:
        raise RuntimeError(
            "Manual Momentum equation drifted from the external recurrent op: "
            f"{maximum_manual_state_error}"
        )

    aggregates = _aggregate(recorder.rows)
    causal = aggregates["causal_layer"]
    swaps = causal["wrong_key_swap"]
    correct = causal["correct"]
    swap_old_ratio = float(swaps["old_velocity_to_update_ratio"]["median"])
    correct_old_ratio = float(correct["old_velocity_to_update_ratio"]["median"])
    swap_erase_ratio = float(swaps["erase_to_native_residual_ratio"]["median"])
    correct_erase_ratio = float(
        correct["erase_to_native_residual_ratio"]["median"]
    )
    old_velocity_selectivity = swap_old_ratio / max(correct_old_ratio, 1e-8)
    erase_selectivity = correct_erase_ratio - swap_erase_ratio
    decision_checks = {
        "swap_old_velocity_to_update_ratio": swap_old_ratio >= 0.25,
        "swap_vs_correct_old_velocity_ratio": old_velocity_selectivity >= 1.25,
        "swap_erase_residual_ratio": swap_erase_ratio <= 0.85,
        "correct_erase_residual_ratio": correct_erase_ratio <= 1.05,
        "erase_selectivity": erase_selectivity >= 0.10,
    }
    open_successor = all(decision_checks.values())
    result = {
        "status": "complete",
        "plan": "P-DIAG-MOMVEL-001",
        "gpu": gpu,
        "protocol": {
            "training": False,
            "logits_modified": False,
            "formal_quality_claim": False,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "formal_score_sha256": EXPECTED_FORMAL_SCORE_SHA256,
            "formal_cases_sha256": EXPECTED_FORMAL_CASES_SHA256,
            "test_data_sha256": test_digest,
            "swap_cases": len(swap_case_indices),
            "matched_all_correct_cases": len(correct_case_indices),
            "diagnostic_batch_size": DIAGNOSTIC_BATCH_SIZE,
            "replay_relative_rms_max": REPLAY_RELATIVE_RMS_MAX,
            "selection_thresholds": {
                "swap_old_velocity_to_update_ratio": 0.25,
                "swap_vs_correct_old_velocity_ratio": 1.25,
                "swap_erase_residual_ratio": 0.85,
                "correct_erase_residual_ratio": 1.05,
                "erase_selectivity": 0.10,
            },
        },
        "trained_parameter_hash": trained_hash,
        "replayed_cases": replayed_cases,
        "replay_checks": recorder.replay_checks,
        "maximum_manual_state_relative_rms": maximum_manual_state_error,
        "aggregates": aggregates,
        "decision_statistics": {
            "causal_swap_old_velocity_to_update_median": swap_old_ratio,
            "causal_correct_old_velocity_to_update_median": correct_old_ratio,
            "swap_vs_correct_old_velocity_ratio": old_velocity_selectivity,
            "causal_swap_erase_to_native_residual_median": swap_erase_ratio,
            "causal_correct_erase_to_native_residual_median": correct_erase_ratio,
            "correct_minus_swap_erase_ratio": erase_selectivity,
        },
        "decision_checks": decision_checks,
        "diagnosis": (
            "selective_stale_key_local_momentum"
            if open_successor
            else "key_local_momentum_erase_not_selectively_supported"
        ),
        "next_branch": (
            "clean_room_key_local_momentum_erase_transition"
            if open_successor
            else "distinct_momentum_lifetime_or_state_organization"
        ),
        "raw_event_rows": recorder.rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
