#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
import diagnose_loop_gradient_conflict_cuda as base
from typing import Any

torch = base.torch


EXPECTED_RECEIVING_STATES = 5 * 3 * 11
EXPECTED_OFFICIAL_BACKWARDS = 5 * 3 * 12
OPENING_COEFFICIENT = 4.0 / 5.0


def parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denom = float(left.norm().item() * right.norm().item())
    if denom <= 1e-30:
        return None
    return float(torch.dot(left, right).item() / denom)


def pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    return tensor_cosine(left, right)


def finite_float(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


class InjectionCapture:
    def __init__(self, model: base.study.FutureSeedLoopSudoku) -> None:
        self.records: list[dict[str, Any]] = []
        self.invocations = [0 for _ in model.reasoner.blocks]
        self.layer_zero_none_count = 0
        self.handles = []
        for layer_idx, block in enumerate(model.reasoner.blocks):
            self.handles.append(
                block.register_forward_pre_hook(
                    self._make_hook(layer_idx),
                    with_kwargs=True,
                )
            )

    def _make_hook(self, layer_idx: int):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            invocation = self.invocations[layer_idx]
            self.invocations[layer_idx] += 1
            initial_state = kwargs.get("initial_state")
            if layer_idx == 0:
                if initial_state is not None:
                    raise RuntimeError("layer zero unexpectedly received FutureSeed state")
                self.layer_zero_none_count += 1
                return
            if initial_state is None:
                raise RuntimeError(
                    f"receiving layer {layer_idx} invocation {invocation} has no state"
                )
            if not initial_state.requires_grad:
                raise RuntimeError("captured FutureSeed state does not require gradient")
            self.records.append(
                {
                    "layer": layer_idx,
                    "edge": layer_idx - 1,
                    "invocation": invocation,
                    "macro_loop": invocation // 3,
                    "depth_call": invocation % 3,
                    "state": initial_state,
                }
            )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def summarize_phase_credit(
    q_by_loss: torch.Tensor,
    records: list[dict[str, Any]],
    continuation_grads: list[torch.Tensor],
    gates: list[torch.Tensor],
    wrong_by_loop: torch.Tensor,
) -> dict[str, Any]:
    if tuple(q_by_loss.shape[:2]) != (5, EXPECTED_RECEIVING_STATES):
        raise RuntimeError(f"unexpected q tensor shape: {tuple(q_by_loss.shape)}")
    batch_size = int(q_by_loss.shape[2])
    heads = int(q_by_loss.shape[3])
    edges = 11

    equal_loss_q = q_by_loss.mean(dim=0)
    opening = torch.zeros(batch_size, edges, heads, dtype=torch.float32)
    continuation_sum = torch.zeros_like(opening)
    for record_idx, record in enumerate(records):
        target = opening if int(record["macro_loop"]) == 0 else continuation_sum
        target[:, int(record["edge"]), :] += equal_loss_q[record_idx]

    continuation = continuation_sum / 4.0
    shared = opening + 4.0 * continuation
    contrast = OPENING_COEFFICIENT * (opening - continuation)
    opening_parameter = opening.sum(dim=0)
    continuation_parameter = continuation.sum(dim=0)
    shared_parameter = shared.sum(dim=0)
    contrast_parameter = contrast.sum(dim=0)

    overlap = torch.minimum(opening.abs(), continuation.abs())
    overlap_total = float(overlap.sum().item())
    local_conflict_fraction = (
        float(overlap[opening * continuation < 0.0].sum().item() / overlap_total)
        if overlap_total > 1e-30
        else None
    )
    overlap_ratio = 2.0 * overlap_total / max(
        float((opening.abs() + continuation.abs()).sum().item()),
        1e-30,
    )

    contrast_norm = float(contrast_parameter.norm().item())
    shared_norm = float(shared_parameter.norm().item())
    phase_leverage = contrast_norm / max(shared_norm, 1e-30)
    board_contrast_norms = contrast.flatten(1).norm(dim=1)
    board_aggregation = contrast_norm / max(
        float(board_contrast_norms.sum().item()), 1e-30
    )
    phase_absolute_energy_ratio = contrast_norm / max(
        OPENING_COEFFICIENT
        * float(
            (
                opening.flatten(1).norm(dim=1)
                + continuation.flatten(1).norm(dim=1)
            ).sum().item()
        ),
        1e-30,
    )
    max_board_energy_share = float(
        board_contrast_norms.square().max().item()
        / board_contrast_norms.square().sum().clamp_min(1e-30).item()
    )
    leave_one_out_cosines = []
    for board_idx in range(batch_size):
        leave_one_out = contrast_parameter - contrast[board_idx]
        value = tensor_cosine(leave_one_out, contrast_parameter)
        if value is not None:
            leave_one_out_cosines.append(value)
    leave_one_out_min_cosine = (
        min(leave_one_out_cosines) if leave_one_out_cosines else None
    )

    edge_rows: list[dict[str, Any]] = []
    active_edges = 0
    total_phase_energy = float(contrast.square().sum().item())
    for edge_idx in range(edges):
        edge_shared = shared_parameter[edge_idx]
        edge_contrast = contrast_parameter[edge_idx]
        edge_shared_norm = float(edge_shared.norm().item())
        edge_contrast_norm = float(edge_contrast.norm().item())
        edge_leverage = edge_contrast_norm / max(edge_shared_norm, 1e-30)
        edge_overlap = torch.minimum(
            opening[:, edge_idx].abs(), continuation[:, edge_idx].abs()
        )
        edge_overlap_total = float(edge_overlap.sum().item())
        edge_conflict = (
            float(
                edge_overlap[
                    opening[:, edge_idx] * continuation[:, edge_idx] < 0.0
                ].sum().item()
                / edge_overlap_total
            )
            if edge_overlap_total > 1e-30
            else None
        )
        edge_energy_share = float(
            contrast[:, edge_idx].square().sum().item()
            / max(total_phase_energy, 1e-30)
        )
        active = bool(
            edge_energy_share >= 0.02
            and finite_float(edge_conflict)
            and edge_conflict >= 0.50
            and edge_leverage >= 0.15
        )
        active_edges += int(active)
        edge_rows.append(
            {
                "receiver_layer": edge_idx + 1,
                "opening_continuation_cosine": tensor_cosine(
                    opening_parameter[edge_idx],
                    continuation_parameter[edge_idx],
                ),
                "shared_gradient_norm": edge_shared_norm,
                "contrast_gradient_norm": edge_contrast_norm,
                "contrast_shared_norm_ratio": edge_leverage,
                "weighted_sign_conflict_fraction": edge_conflict,
                "phase_energy_share": edge_energy_share,
                "active": active,
            }
        )

    orthogonal_values: list[torch.Tensor] = []
    orthogonal_by_board = torch.zeros(batch_size, dtype=torch.float32)
    orthogonal_board_counts = torch.zeros(batch_size, dtype=torch.float32)
    for record, gradient in zip(records, continuation_grads):
        state = record["state"].detach().float().cpu()
        gate = gates[int(record["edge"])].detach().float().cpu().view(
            1, heads, 1, 1
        )
        radial_axis = state / gate.clamp_min(1e-8)
        grad = gradient.float()
        grad_sq = grad.square().sum(dim=(-1, -2))
        axis_sq = radial_axis.square().sum(dim=(-1, -2)).clamp_min(1e-30)
        projection_sq = (
            (grad * radial_axis).sum(dim=(-1, -2)).square() / axis_sq
        )
        orthogonal = (
            (grad_sq - projection_sq).clamp_min(0.0)
            / grad_sq.clamp_min(1e-30)
        ).sqrt()
        valid = grad_sq > 1e-24
        if bool(valid.any()):
            orthogonal_values.append(orthogonal[valid])
            orthogonal_by_board += (orthogonal * valid).sum(dim=1)
            orthogonal_board_counts += valid.sum(dim=1)

    if orthogonal_values:
        orthogonal_all = torch.cat(orthogonal_values)
        orthogonal_mean = float(orthogonal_all.mean().item())
        orthogonal_median = float(orthogonal_all.median().item())
    else:
        orthogonal_mean = None
        orthogonal_median = None
    orthogonal_by_board = orthogonal_by_board / orthogonal_board_counts.clamp_min(1.0)

    board_correction = wrong_by_loop[0] - wrong_by_loop[-1]
    board_phase_magnitude = contrast.abs().mean(dim=(1, 2))
    return {
        "opening_continuation_cosine": tensor_cosine(
            opening_parameter, continuation_parameter
        ),
        "weighted_sign_conflict_fraction": local_conflict_fraction,
        "opening_continuation_overlap_ratio": overlap_ratio,
        "shared_gradient_norm": shared_norm,
        "contrast_gradient_norm": contrast_norm,
        "contrast_shared_norm_ratio": phase_leverage,
        "phase_absolute_energy_ratio": phase_absolute_energy_ratio,
        "cross_board_aggregation_ratio": board_aggregation,
        "max_board_phase_energy_share": max_board_energy_share,
        "leave_one_board_out_min_cosine": leave_one_out_min_cosine,
        "active_edge_count": active_edges,
        "active_edges": [
            row["receiver_layer"] for row in edge_rows if row["active"]
        ],
        "edges": edge_rows,
        "aggregate_phase_gradient": contrast_parameter.reshape(-1).tolist(),
        "continuation_cotangent_orthogonal_fraction_mean": orthogonal_mean,
        "continuation_cotangent_orthogonal_fraction_median": orthogonal_median,
        "board_phase_magnitude_vs_wrong_cell_correction_pearson": pearson(
            board_phase_magnitude, board_correction
        ),
        "board_orthogonal_fraction_vs_wrong_cell_correction_pearson": pearson(
            orthogonal_by_board, board_correction
        ),
        "board_phase_magnitude": board_phase_magnitude.tolist(),
        "board_continuation_orthogonal_fraction": orthogonal_by_board.tolist(),
        "board_wrong_cell_correction": board_correction.tolist(),
    }


def bucket_qualified(phase: dict[str, Any], late_correction: bool) -> bool:
    cosine = phase["opening_continuation_cosine"]
    conflict = phase["weighted_sign_conflict_fraction"]
    leave_one_out = phase["leave_one_board_out_min_cosine"]
    return bool(
        late_correction
        and finite_float(cosine)
        and cosine <= -0.25
        and finite_float(conflict)
        and conflict >= 0.60
        and phase["opening_continuation_overlap_ratio"] >= 0.20
        and phase["contrast_shared_norm_ratio"] >= 0.25
        and phase["phase_absolute_energy_ratio"] >= 0.15
        and phase["cross_board_aggregation_ratio"] >= 0.50
        and phase["max_board_phase_energy_share"] <= 0.30
        and finite_float(leave_one_out)
        and leave_one_out >= 0.75
        and phase["active_edge_count"] >= 8
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-parameter receiver-edge phase-credit audit for native FutureSeed"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--parent-source-sha", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=52051)
    parser.add_argument(
        "--blank-ranges",
        type=base.parse_range,
        nargs="+",
        default=[(51, 55), (56, 60), (61, 64)],
    )
    parser.add_argument("--blank-weight", type=float, default=8.0)
    args = parser.parse_args()

    started = time.perf_counter()
    gpu = base.validate_gpu(args.expected_gpu_uuid)
    if base.study.GAIN_BUDGET_FLA_SHA != base.EXPECTED_FLA_SHA:
        raise RuntimeError("pinned FLA constant drift")
    checkpoint_hash = base.sha256_file(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError("checkpoint SHA mismatch")

    checkpoint_run = args.checkpoint.resolve().parents[1].name
    persistent_root = Path("/huyang2/double-loop")
    source_head_path = persistent_root / "runs" / checkpoint_run / "source_HEAD.txt"
    source_patch_path = persistent_root / "runs" / checkpoint_run / "source.patch"
    if source_head_path.read_text(encoding="utf-8").strip() != args.parent_source_sha:
        raise RuntimeError("parent source provenance mismatch")
    if not source_patch_path.is_file() or source_patch_path.stat().st_size != 0:
        raise RuntimeError("parent source patch is missing or nonempty")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved_args = dict(checkpoint["args"])
    base.validate_parent(saved_args, int(checkpoint.get("saved_at_step", -1)))
    base.study.configure_sudoku(int(saved_args["size"]), 0, 0)

    device = torch.device("cuda", 0)
    model = base.build_model(saved_args).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint is not exact: missing={missing}, unexpected={unexpected}"
        )
    del checkpoint
    model.train()
    runtime = base.study.strict_fla_runtime_summary(model, "gdn2")
    if runtime["fla_source_sha"] != base.EXPECTED_FLA_SHA:
        raise RuntimeError("runtime FLA SHA drift")
    if len(runtime["layers"]) != 12:
        raise RuntimeError("expected exactly 12 official GDN2 layers")

    gate_parameters = [
        model.reasoner.blocks[layer_idx].future_seed_logit
        for layer_idx in range(1, 12)
    ]
    gates = [torch.sigmoid(parameter.float()) for parameter in gate_parameters]
    parameter_hash_before = parameter_sha256(model)
    dataset = base.study.OfficialSudokuDataset(args.data_dir, "test")
    bucket_rows: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)

    for bucket_idx, (holes_min, holes_max) in enumerate(args.blank_ranges):
        bucket_name = f"{holes_min}-{holes_max}"
        inputs, labels, clue_mask = dataset.fixed_batch_by_blank_range(
            args.batch_size,
            args.eval_seed + bucket_idx * 1000,
            holes_min=holes_min,
            holes_max=holes_max,
            device=device,
        )
        bucket_started = time.perf_counter()
        capture = InjectionCapture(model)
        with base.study.forward_autocast("bfloat16", device):
            loop_logits, _trace = model.forward_trace(
                inputs,
                loops=5,
                noise_scale=0.0,
            )
        capture.close()
        if capture.invocations != [15] * 12:
            raise RuntimeError(f"unexpected block invocation counts: {capture.invocations}")
        if capture.layer_zero_none_count != 15:
            raise RuntimeError("layer-zero call count mismatch")
        if len(capture.records) != EXPECTED_RECEIVING_STATES:
            raise RuntimeError(
                f"captured {len(capture.records)} states, expected {EXPECTED_RECEIVING_STATES}"
            )

        graph = base.graph_names(loop_logits[-1])
        backward_count = graph.count("ChunkGDN2FunctionBackward")
        if backward_count != EXPECTED_OFFICIAL_BACKWARDS:
            raise RuntimeError(
                f"official backward count {backward_count} != {EXPECTED_OFFICIAL_BACKWARDS}"
            )
        losses = [
            base.study.loss_from_logits(
                logits,
                labels,
                clue_mask,
                blank_weight=args.blank_weight,
            )
            for logits in loop_logits
        ]

        loop_metrics: list[dict[str, Any]] = []
        wrong_rows: list[torch.Tensor] = []
        for loop_idx, (logits, loss) in enumerate(zip(loop_logits, losses), start=1):
            metrics, prediction = base.study.metrics_from_logits(
                logits,
                labels,
                clue_mask,
            )
            wrong = ((prediction != labels) & ~clue_mask).sum(dim=1).float()
            wrong_rows.append(wrong.cpu())
            loop_metrics.append(
                {
                    "loop": loop_idx,
                    "ce": float(loss.detach().item()),
                    **asdict(metrics),
                    "wrong_blank_cells_mean": float(wrong.mean().item()),
                    "wrong_blank_cells_max": int(wrong.max().item()),
                }
            )

        state_tensors = [record["state"] for record in capture.records]
        q_by_loss = torch.zeros(
            5,
            len(state_tensors),
            args.batch_size,
            8,
            dtype=torch.float32,
        )
        continuation_grads = [
            torch.zeros_like(state, dtype=torch.float32, device="cpu")
            for state in state_tensors
        ]
        reconstruction_rows: list[dict[str, Any]] = []
        for target_loop, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss,
                state_tensors + gate_parameters,
                retain_graph=target_loop < 4,
                allow_unused=True,
            )
            state_grads = grads[: len(state_tensors)]
            direct_gate_grads = grads[len(state_tensors) :]
            active_state_gradient_count = sum(
                gradient is not None for gradient in state_grads
            )
            expected_active_state_gradients = 33 * (target_loop + 1)
            if active_state_gradient_count != expected_active_state_gradients:
                raise RuntimeError(
                    "causal receiver-state ancestry mismatch: "
                    f"loop={target_loop + 1} active={active_state_gradient_count} "
                    f"expected={expected_active_state_gradients}"
                )
            for record_idx, (record, state, gradient) in enumerate(
                zip(capture.records, state_tensors, state_grads)
            ):
                if gradient is None:
                    continue
                gate = gates[int(record["edge"])].view(1, 8)
                q = (
                    (gradient.float() * state.float()).sum(dim=(-1, -2))
                    * (1.0 - gate)
                )
                q_by_loss[target_loop, record_idx] = q.detach().cpu()
                if target_loop >= 1:
                    continuation_grads[record_idx].add_(
                        gradient.detach().float().cpu(),
                        alpha=0.25,
                    )

            max_abs_error = 0.0
            max_relative_error = 0.0
            direct_flattened = []
            reconstructed_flattened = []
            for edge_idx, direct_gradient in enumerate(direct_gate_grads):
                if direct_gradient is None:
                    raise RuntimeError(
                        f"FutureSeed gate edge {edge_idx} has no direct gradient"
                    )
                record_indices = [
                    idx
                    for idx, record in enumerate(capture.records)
                    if int(record["edge"]) == edge_idx
                ]
                reconstructed = q_by_loss[
                    target_loop, record_indices
                ].sum(dim=(0, 1))
                direct = direct_gradient.detach().float().cpu().reshape(-1)
                reconstructed_flattened.append(reconstructed)
                direct_flattened.append(direct)
                absolute = float((reconstructed - direct).abs().max().item())
                relative = absolute / max(float(direct.abs().max().item()), 1e-12)
                max_abs_error = max(max_abs_error, absolute)
                max_relative_error = max(max_relative_error, relative)
            reconstruction_rows.append(
                {
                    "target_loop": target_loop + 1,
                    "active_state_gradient_count": active_state_gradient_count,
                    "max_abs_error": max_abs_error,
                    "max_relative_error": max_relative_error,
                    "cosine": tensor_cosine(
                        torch.cat(reconstructed_flattened),
                        torch.cat(direct_flattened),
                    ),
                }
            )

        wrong_by_loop = torch.stack(wrong_rows)
        phase = summarize_phase_credit(
            q_by_loss,
            capture.records,
            continuation_grads,
            gates,
            wrong_by_loop,
        )
        late_correction = bool(
            loop_metrics[-1]["blank_acc"] - loop_metrics[0]["blank_acc"] >= 0.02
            and loop_metrics[-1]["wrong_blank_cells_mean"]
            < loop_metrics[0]["wrong_blank_cells_mean"]
        )
        qualified = bucket_qualified(phase, late_correction)
        max_reconstruction_relative_error = max(
            row["max_relative_error"] for row in reconstruction_rows
        )
        min_reconstruction_cosine = min(
            row["cosine"]
            for row in reconstruction_rows
            if row["cosine"] is not None
        )
        if (
            max_reconstruction_relative_error > 5e-3
            or min_reconstruction_cosine < 0.9999
        ):
            raise RuntimeError(
                "cotangent replay did not reconstruct FutureSeed gate gradient: "
                f"relative={max_reconstruction_relative_error} "
                f"cosine={min_reconstruction_cosine}"
            )
        torch.cuda.synchronize(device)
        bucket_rows[bucket_name] = {
            "blank_range": [holes_min, holes_max],
            "batch_size": int(inputs.shape[0]),
            "captured_receiving_states": len(capture.records),
            "layer_zero_none_count": capture.layer_zero_none_count,
            "official_backward_count": backward_count,
            "loop_metrics": loop_metrics,
            "phase_credit": phase,
            "gate_gradient_reconstruction": reconstruction_rows,
            "late_correction": late_correction,
            "qualified": qualified,
            "elapsed_sec": time.perf_counter() - bucket_started,
        }
        del loop_logits, losses, grads, state_grads, direct_gate_grads
        del state_tensors, continuation_grads, q_by_loss, inputs, labels, clue_mask
        torch.cuda.empty_cache()

    parameter_hash_after = parameter_sha256(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("zero-parameter diagnostic mutated model parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("zero-parameter diagnostic populated parameter .grad")

    qualified_bucket_count = sum(row["qualified"] for row in bucket_rows.values())
    qualified_names = [
        name for name, row in bucket_rows.items() if row["qualified"]
    ]
    qualified_phase_rows = []
    for name in qualified_names:
        phase = bucket_rows[name]["phase_credit"]
        qualified_phase_rows.append(
            {
                "name": name,
                "active_edges": set(phase["active_edges"]),
                "gradient": torch.tensor(
                    phase["aggregate_phase_gradient"], dtype=torch.float32
                ),
            }
        )
    cross_bucket_pairs = []
    for left_idx, left in enumerate(qualified_phase_rows):
        for right in qualified_phase_rows[left_idx + 1 :]:
            cross_bucket_pairs.append(
                {
                    "buckets": [left["name"], right["name"]],
                    "shared_active_edge_count": len(
                        left["active_edges"].intersection(right["active_edges"])
                    ),
                    "phase_gradient_cosine": tensor_cosine(
                        left["gradient"], right["gradient"]
                    ),
                }
            )
    admitted_pairs = [
        row
        for row in cross_bucket_pairs
        if row["shared_active_edge_count"] >= 6
        and finite_float(row["phase_gradient_cosine"])
        and row["phase_gradient_cosine"] >= 0.25
    ]
    cross_bucket_consistency = bool(admitted_pairs)
    candidate_admitted = bool(
        qualified_bucket_count >= 2 and cross_bucket_consistency
    )
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    source_clean = (
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        == ""
    )
    detached_head = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0
    payload = {
        "plan_id": "P-LOOP-003",
        "kind": "zero_parameter_receiver_edge_phase_credit_diagnostic",
        "source_sha": current_head,
        "source_clean": source_clean,
        "detached_head": detached_head,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "parent_source_sha": args.parent_source_sha,
        "data_dir": str(args.data_dir),
        "gpu": gpu,
        "torch_version": torch.__version__,
        "runtime": runtime,
        "parameter_sha256_before": parameter_hash_before,
        "parameter_sha256_after": parameter_hash_after,
        "buckets": bucket_rows,
        "decision": {
            "candidate_admitted": candidate_admitted,
            "qualified_bucket_count": qualified_bucket_count,
            "qualified_buckets": qualified_names,
            "required_qualified_buckets": 2,
            "cross_bucket_pairs": cross_bucket_pairs,
            "admitted_cross_bucket_pairs": admitted_pairs,
            "cross_bucket_consistency": cross_bucket_consistency,
            "candidate_if_admitted": (
                "P-LOOP-004 opening/continuation FutureSeed phase contrast"
            ),
            "if_rejected": (
                "close scalar FutureSeed loop-credit coordination and move to a "
                "new recurrent-transition GDN3 mechanism"
            ),
            "thresholds": {
                "opening_continuation_cosine_max": -0.25,
                "weighted_sign_conflict_fraction_min": 0.60,
                "opening_continuation_overlap_ratio_min": 0.20,
                "contrast_shared_norm_ratio_min": 0.25,
                "phase_absolute_energy_ratio_min": 0.15,
                "cross_board_aggregation_ratio_min": 0.50,
                "max_board_phase_energy_share_max": 0.30,
                "leave_one_board_out_min_cosine_min": 0.75,
                "active_edge_count_min": 8,
                "shared_active_edge_count_min": 6,
                "cross_bucket_phase_cosine_min": 0.25,
                "loop5_minus_loop1_blank_min": 0.02,
                "wrong_cells_must_decrease": True,
            },
        },
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "elapsed_sec": time.perf_counter() - started,
    }
    if not payload["source_clean"] or not payload["detached_head"]:
        raise RuntimeError("diagnostic source provenance is not clean detached")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "decision": payload["decision"],
                "elapsed_sec": payload["elapsed_sec"],
                "peak_allocated_mib": payload["peak_allocated_mib"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
