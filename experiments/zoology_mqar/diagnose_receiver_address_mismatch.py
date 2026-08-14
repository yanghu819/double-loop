from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from fla.modules.l2norm import l2norm_fwd
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.gdn2_diagnostics.committed_edit import (
    _selected_survival,
    binary_auroc,
    key_gram_statistics,
)
from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_log_spd import capture_chunk_addresses
from experiments.zoology_mqar.gdn2_log_spd_endpoint import wrong_key_swap_summary
from experiments.zoology_mqar.length_scaling import (
    _query_event,
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)


SEQUENCE_LENGTH = 1024
NUM_KV_PAIRS = 4
BATCH_SIZE = 32
MODEL_ARM = "future_seed_gdn2_surprise_replay"
EVENT_TAPE_SIZE = 16
STATE_DIAGNOSTIC_EXAMPLES = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return _mean(finite)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return float("nan")
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denominator == 0:
        return 0.0
    return float((x * y).sum().div(denominator).item())


def _rank_and_margin(query: torch.Tensor, candidates: torch.Tensor, target: int) -> dict[str, float | int]:
    per_head = torch.einsum("hk,nhk->nh", query.float(), candidates.float())
    target_score = per_head[target]
    other = torch.cat((per_head[:target], per_head[target + 1 :]), dim=0)
    ranks = 1 + (other > target_score.unsqueeze(0)).sum(dim=0)
    aggregate = per_head.mean(dim=-1)
    aggregate_target = aggregate[target]
    aggregate_other = torch.cat((aggregate[:target], aggregate[target + 1 :]))
    return {
        "aggregate_rank": 1 + int((aggregate_other > aggregate_target).sum().item()),
        "head_top1": float((ranks == 1).float().mean().item()),
        "head_mrr": float(ranks.float().reciprocal().mean().item()),
        "head_margin": float((target_score - other.max(dim=0).values).mean().item()),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "producer_basis_top1": _mean([float(row["producer_basis_head_top1"]) for row in rows]),
        "receiver_native_top1": _mean([float(row["receiver_native_head_top1"]) for row in rows]),
        "top1_gain_receiver_minus_producer": _mean(
            [float(row["receiver_native_head_top1"]) - float(row["producer_basis_head_top1"]) for row in rows]
        ),
        "producer_basis_mrr": _mean([float(row["producer_basis_head_mrr"]) for row in rows]),
        "receiver_native_mrr": _mean([float(row["receiver_native_head_mrr"]) for row in rows]),
        "mrr_gain_receiver_minus_producer": _mean(
            [float(row["receiver_native_head_mrr"]) - float(row["producer_basis_head_mrr"]) for row in rows]
        ),
        "producer_margin_mean": _mean([float(row["producer_basis_margin"]) for row in rows]),
        "receiver_margin_mean": _mean([float(row["receiver_native_margin"]) for row in rows]),
        "target_write_surprise_mean": _mean([float(row["target_write_surprise"]) for row in rows]),
        "target_write_surprise_percentile_mean": _mean(
            [float(row["target_write_surprise_percentile"]) for row in rows]
        ),
        "target_write_selected_top16_fraction": _mean(
            [float(row["target_write_selected_top16"]) for row in rows]
        ),
        "producer_receiver_target_key_cosine_mean": _mean(
            [float(row["producer_receiver_target_key_cosine"]) for row in rows]
        ),
        "target_write_survival_mean": _finite_mean(
            [float(row["target_write_survival"]) for row in rows]
        ),
    }


def _geometry_summary(records: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    if not records:
        raise RuntimeError("No key-geometry records")
    output: dict[str, float] = {}
    for metric in ("anisotropy", "effective_rank_fraction", "condition", "coherence"):
        values = torch.cat([record[metric].reshape(-1).cpu() for record in records]).float()
        output[f"{metric}_mean"] = float(values.mean().item())
        output[f"{metric}_median"] = float(values.median().item())
        output[f"{metric}_p90"] = float(torch.quantile(values, 0.90).item())
    return output


def _branch_decision(
    groups: dict[str, dict[str, Any]],
    swaps: dict[str, Any],
    accuracy: float,
    geometry: dict[str, dict[str, float]],
    survival: dict[str, Any],
) -> dict[str, Any]:
    all_rows = groups["all"]
    error_rows = groups["error"]
    swap_rows = groups["wrong_key_swap"]
    high = groups["target_write_top16_surprise"]
    low = groups["target_write_not_top16_surprise"]
    mismatch = (
        swaps["wrong_key_swap_fraction_of_errors"] >= 0.70
        and (
            all_rows["top1_gain_receiver_minus_producer"] >= 0.10
            or all_rows["mrr_gain_receiver_minus_producer"] >= 0.10
        )
        and (
            swap_rows.get("top1_gain_receiver_minus_producer", -1.0) >= 0.10
            or swap_rows.get("mrr_gain_receiver_minus_producer", -1.0) >= 0.10
        )
    )
    surprise_cache = (
        error_rows.get("target_write_selected_top16_fraction", 0.0)
        >= groups["correct"].get("target_write_selected_top16_fraction", 1.0) + 0.10
        and high.get("mrr_gain_receiver_minus_producer", -1.0)
        >= low.get("mrr_gain_receiver_minus_producer", 1.0) + 0.05
    )
    write_interference = (
        survival["target_survival_error_auroc"] >= 0.70
        and survival["target_survival_correct_minus_error"] >= 0.10
    )
    anisotropic = (
        geometry["receiver"]["anisotropy_median"] >= 4.0
        and geometry["receiver"]["effective_rank_fraction_median"] <= 0.60
    )
    convergence = (
        all_rows["receiver_native_top1"] >= 0.80
        and error_rows.get("receiver_native_top1", 0.0) >= 0.60
        and accuracy < 0.85
    )
    if mismatch and surprise_cache:
        branch = "receiver_native_surprise_cache"
    elif mismatch:
        branch = "receiver_native_address_binding"
    elif write_interference and anisotropic:
        branch = "live_state_address_interference"
    elif convergence:
        branch = "loop_or_readout_convergence"
    else:
        branch = "kill_address_and_cache_line"
    return {
        "branch": branch,
        "receiver_query_write_mismatch_supported": mismatch,
        "surprise_localizes_actionable_mismatch": surprise_cache,
        "write_survival_predicts_errors": write_interference,
        "receiver_key_geometry_anisotropic": anisotropic,
        "receiver_address_already_good_but_logits_wrong": convergence,
        "rules": {
            "mismatch": "swap/error>=0.70; overall and swap-only receiver-native top1 or MRR gain >=0.10",
            "surprise_cache": "top16-surprise target writes are >=0.10 more common in errors and their receiver-native MRR gain exceeds low-surprise by >=0.05",
            "live_state_address_interference": "-target survival error AUROC>=0.70, correct-minus-error survival>=0.10, receiver anisotropy median>=4, and effective-rank fraction median<=0.60",
            "convergence": "receiver-native top1>=0.80 overall and >=0.60 on errors while token accuracy<0.85",
            "otherwise": "stop receiver-address/cache mechanisms and inspect loop/readout or another bottleneck",
        },
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    args = parser.parse_args()
    if args.max_examples != 1000:
        raise ValueError("The diagnostic is fixed to the full 1000-case validation set")
    if _sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA256 mismatch")

    config = build_config(
        arm=MODEL_ARM,
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=NUM_KV_PAIRS,
        max_epochs=10,
        batch_size=BATCH_SIZE,
    )
    set_determinism(config.seed)
    model = make_model(config, MODEL_ARM).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("carrier_arm") != MODEL_ARM:
        raise RuntimeError(f"Unexpected checkpoint carrier: {checkpoint.get('carrier_arm')}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_sha_before = parameter_hash(model)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Diagnostic started with model gradients")

    _train_loader, test_loader = prepare_data(config.data)
    test_sha256 = dataset_hash(test_loader)
    first_inputs, _first_targets, _first_slices = next(iter(test_loader))
    model(first_inputs.cuda())
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    baseline_first_logits = None
    baseline_started = time.perf_counter()
    baseline_examples = 0
    for inputs, _targets, _slices in test_loader:
        logits = model(inputs.cuda())
        if baseline_first_logits is None:
            baseline_first_logits = logits.detach().clone()
        baseline_examples += inputs.shape[0]
    torch.cuda.synchronize()
    baseline_elapsed = time.perf_counter() - baseline_started
    baseline_peak = int(torch.cuda.max_memory_allocated())
    if baseline_examples != args.max_examples or baseline_first_logits is None:
        raise RuntimeError("Validation cardinality changed")

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    producer_geometry_records: list[dict[str, torch.Tensor]] = []
    receiver_geometry_records: list[dict[str, torch.Tensor]] = []
    surprise_survival_boards: list[float] = []
    recency_survival_boards: list[float] = []
    logits_bitwise_equal = True
    replay_wiring_exact = True
    replay_state_active = True
    predictions_correct = 0
    query_count = 0
    torch.cuda.reset_peak_memory_stats()
    diagnostic_started = time.perf_counter()
    for batch_index, (inputs, targets, _slices) in enumerate(test_loader):
        inputs_gpu = inputs.cuda()
        with capture_chunk_addresses() as addresses, capture_committed_edit() as edits:
            logits = model(inputs_gpu)
        lengths = [int(q.shape[1]) for q, _k in addresses.records]
        edit_lengths = [int(edit.shape[1]) for edit in edits]
        if (
            len(addresses.records) != 2
            or len(addresses.transition_records) != 2
            or len(edits) != 2
            or lengths != [SEQUENCE_LENGTH, SEQUENCE_LENGTH]
            or edit_lengths != lengths
        ):
            raise RuntimeError(
                "Expected the two L1024 chunk scans; the K16 replay uses the official "
                "short-sequence recurrent path. Got "
                f"{len(addresses.records)} addresses, {len(addresses.transition_records)} transitions, "
                f"and {len(edits)} edits with lengths addresses={lengths}, edits={edit_lengths}"
            )
        if batch_index == 0:
            logits_bitwise_equal = torch.equal(baseline_first_logits, logits)

        producer_q, producer_k = addresses.records[0]
        receiver_q, receiver_k = addresses.records[1]
        with torch.autocast(device_type="cuda", enabled=False):
            producer_q, _ = l2norm_fwd(producer_q)
            producer_k, _ = l2norm_fwd(producer_k)
            receiver_q, _ = l2norm_fwd(receiver_q)
            receiver_k, _ = l2norm_fwd(receiver_k)
        del producer_q
        surprise_gpu = edits[0].float().square().sum(dim=(-1, -2)).sqrt()
        expected_selected_gpu = torch.argsort(
            surprise_gpu,
            dim=1,
            descending=True,
            stable=True,
        )[:, :EVENT_TAPE_SIZE].sort(dim=1).values
        producer_mixer = model.backbone.layers[0].sequence_mixer
        receiver_mixer = model.backbone.layers[1].sequence_mixer
        production_selected = producer_mixer.last_selected_indices
        selected_evidence = producer_mixer.last_selected_evidence
        replay_input = receiver_mixer.last_replay_input
        replay_diagnostics = receiver_mixer.last_replay_diagnostics
        batch_replay_wiring_exact = (
            production_selected is not None
            and selected_evidence is not None
            and replay_input is not None
            and production_selected.shape == expected_selected_gpu.shape
            and torch.equal(production_selected, expected_selected_gpu)
            and selected_evidence.shape[:2] == (inputs.shape[0], EVENT_TAPE_SIZE)
            and replay_input is selected_evidence
            and torch.equal(replay_input, selected_evidence)
        )
        if not batch_replay_wiring_exact:
            raise RuntimeError("Production top16 selection or exact producer-to-receiver replay wiring changed")
        required_replay_metrics = (
            "replay_terminal_rms",
            "replay_terminal_board_std",
            "replay_seed_rms",
            "replay_seed_abs_max",
        )
        batch_replay_state_active = (
            all(name in replay_diagnostics for name in required_replay_metrics)
            and all(torch.isfinite(replay_diagnostics[name]).all() for name in required_replay_metrics)
            and float(replay_diagnostics["replay_terminal_rms"].item()) >= 1e-4
            and float(replay_diagnostics["replay_terminal_board_std"].item()) > 0.0
            and float(replay_diagnostics["replay_seed_rms"].item()) >= 1e-4
        )
        if not batch_replay_state_active:
            raise RuntimeError(f"Receiver K16 replay state is inactive: {replay_diagnostics}")
        replay_wiring_exact = replay_wiring_exact and batch_replay_wiring_exact
        replay_state_active = replay_state_active and batch_replay_state_active
        selected_gpu = production_selected
        predictions = logits.argmax(dim=-1).cpu()

        batch_events: list[list[dict[str, Any]]] = []
        for row_index in range(inputs.shape[0]):
            query_positions = torch.nonzero(targets[row_index] != -100).flatten().tolist()
            batch_events.append(
                [
                    _query_event(
                        inputs[row_index],
                        targets[row_index],
                        predictions[row_index],
                        position,
                        SEQUENCE_LENGTH // 4,
                    )
                    for position in query_positions
                ]
            )

        target_survival = torch.full(
            (inputs.shape[0], NUM_KV_PAIRS),
            float("nan"),
            dtype=torch.float32,
        )
        if batch_index * BATCH_SIZE < STATE_DIAGNOSTIC_EXAMPLES:
            producer_transition = addresses.transition_records[0]
            receiver_transition = addresses.transition_records[1]
            if not producer_transition["use_qk_l2norm_in_kernel"] or not receiver_transition["use_qk_l2norm_in_kernel"]:
                raise RuntimeError("The fixed checkpoint no longer uses normalized Q/K in both full scans")
            producer_geometry_records.append(
                {key: value.cpu() for key, value in key_gram_statistics(producer_k).items()}
            )
            receiver_geometry_records.append(
                {key: value.cpu() for key, value in key_gram_statistics(receiver_k).items()}
            )
            target_positions_gpu = torch.tensor(
                [
                    [int(event["write_position"]) + 1 for event in events]
                    for events in batch_events
                ],
                device=producer_k.device,
                dtype=torch.long,
            )
            recency_gpu = torch.arange(
                SEQUENCE_LENGTH - EVENT_TAPE_SIZE,
                SEQUENCE_LENGTH,
                device=producer_k.device,
            ).unsqueeze(0).expand(inputs.shape[0], -1)
            survival = _selected_survival(
                producer_k.float(),
                producer_transition["g"].float(),
                producer_transition["b"].float(),
                torch.cat((target_positions_gpu, selected_gpu, recency_gpu), dim=1),
            )
            target_survival = survival[:, :, :NUM_KV_PAIRS].mean(dim=1).cpu()
            surprise_survival_boards.extend(
                survival[:, :, NUM_KV_PAIRS : NUM_KV_PAIRS + EVENT_TAPE_SIZE]
                .mean(dim=(1, 2))
                .cpu()
                .tolist()
            )
            recency_survival_boards.extend(
                survival[:, :, NUM_KV_PAIRS + EVENT_TAPE_SIZE :]
                .mean(dim=(1, 2))
                .cpu()
                .tolist()
            )

        producer_k_cpu = producer_k.float().cpu()
        receiver_q_cpu = receiver_q.float().cpu()
        receiver_k_cpu = receiver_k.float().cpu()
        surprise = surprise_gpu.cpu()
        selected = selected_gpu.cpu()

        for row_index, events in enumerate(batch_events):
            committed_write_positions = [int(event["write_position"]) + 1 for event in events]
            target_values = {int(event["target"]) for event in events}
            for event_index, event in enumerate(events):
                query_position = int(event["query_position"])
                committed_write_position = int(event["write_position"]) + 1
                producer_geometry = _rank_and_margin(
                    receiver_q_cpu[row_index, query_position],
                    producer_k_cpu[row_index, committed_write_positions],
                    event_index,
                )
                receiver_geometry = _rank_and_margin(
                    receiver_q_cpu[row_index, query_position],
                    receiver_k_cpu[row_index, committed_write_positions],
                    event_index,
                )
                producer_target = producer_k_cpu[row_index, committed_write_position]
                receiver_target = receiver_k_cpu[row_index, committed_write_position]
                target_cosine = torch.nn.functional.cosine_similarity(
                    producer_target,
                    receiver_target,
                    dim=-1,
                ).mean()
                target_surprise = surprise[row_index, committed_write_position]
                record = {
                    **event,
                    "case_index": batch_index * BATCH_SIZE + row_index,
                    "committed_write_position": committed_write_position,
                    "wrong_key_swap": bool((not event["correct"]) and int(event["prediction"]) in target_values),
                    "producer_basis_rank": producer_geometry["aggregate_rank"],
                    "receiver_native_rank": receiver_geometry["aggregate_rank"],
                    "producer_basis_head_top1": producer_geometry["head_top1"],
                    "receiver_native_head_top1": receiver_geometry["head_top1"],
                    "producer_basis_head_mrr": producer_geometry["head_mrr"],
                    "receiver_native_head_mrr": receiver_geometry["head_mrr"],
                    "producer_basis_margin": producer_geometry["head_margin"],
                    "receiver_native_margin": receiver_geometry["head_margin"],
                    "target_write_surprise": float(target_surprise.item()),
                    "target_write_surprise_percentile": float((surprise[row_index] <= target_surprise).float().mean().item()),
                    "target_write_selected_top16": bool((selected[row_index] == committed_write_position).any().item()),
                    "producer_receiver_target_key_cosine": float(target_cosine.item()),
                    "target_write_survival": float(target_survival[row_index, event_index].item()),
                }
                rows.append(record)
                predictions_correct += int(event["correct"])
                query_count += 1
            cases.append({"case_index": batch_index * BATCH_SIZE + row_index, "events": events})

    torch.cuda.synchronize()
    diagnostic_elapsed = time.perf_counter() - diagnostic_started
    diagnostic_peak = int(torch.cuda.max_memory_allocated())
    parameter_sha_after = parameter_hash(model)
    no_gradients = all(parameter.grad is None for parameter in model.parameters())

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows["all"].append(row)
        grouped_rows[row["direction"]].append(row)
        grouped_rows["correct" if row["correct"] else "error"].append(row)
        if row["wrong_key_swap"]:
            grouped_rows["wrong_key_swap"].append(row)
        grouped_rows[
            "target_write_top16_surprise"
            if row["target_write_selected_top16"]
            else "target_write_not_top16_surprise"
        ].append(row)
    group_names = (
        "all",
        "future",
        "past",
        "correct",
        "error",
        "wrong_key_swap",
        "target_write_top16_surprise",
        "target_write_not_top16_surprise",
    )
    groups = {name: _group_summary(grouped_rows[name]) for name in group_names}
    swaps = wrong_key_swap_summary(cases)
    accuracy = predictions_correct / query_count
    geometry = {
        "producer": _geometry_summary(producer_geometry_records),
        "receiver": _geometry_summary(receiver_geometry_records),
    }
    survival_rows = [row for row in rows if math.isfinite(float(row["target_write_survival"]))]
    survival_scores = torch.tensor(
        [-float(row["target_write_survival"]) for row in survival_rows],
        dtype=torch.float32,
    )
    survival_errors = torch.tensor(
        [not bool(row["correct"]) for row in survival_rows],
        dtype=torch.bool,
    )
    correct_survival = _finite_mean(
        [float(row["target_write_survival"]) for row in survival_rows if row["correct"]]
    )
    error_survival = _finite_mean(
        [float(row["target_write_survival"]) for row in survival_rows if not row["correct"]]
    )
    survival = {
        "examples": STATE_DIAGNOSTIC_EXAMPLES,
        "queries": len(survival_rows),
        "target_survival_correct_mean": correct_survival,
        "target_survival_error_mean": error_survival,
        "target_survival_correct_minus_error": correct_survival - error_survival,
        "target_survival_error_auroc": binary_auroc(survival_scores, survival_errors),
        "surprise_top16_survival_mean": _mean(surprise_survival_boards),
        "recency_top16_survival_mean": _mean(recency_survival_boards),
        "surprise_minus_recency_survival": _mean(surprise_survival_boards)
        - _mean(recency_survival_boards),
    }
    cost = {
        "baseline_eval_seconds": baseline_elapsed,
        "diagnostic_eval_seconds": diagnostic_elapsed,
        "elapsed_ratio": diagnostic_elapsed / baseline_elapsed,
        "baseline_peak_allocated_bytes": baseline_peak,
        "diagnostic_peak_allocated_bytes": diagnostic_peak,
        "peak_allocation_ratio": diagnostic_peak / baseline_peak,
    }
    integrity = {
        "zero_new_parameters": parameter_count == 661_584,
        "checkpoint_sha256_exact": _sha256(args.checkpoint) == args.expected_checkpoint_sha256,
        "first_batch_logits_bitwise_equal": logits_bitwise_equal,
        "parameter_sha_unchanged": parameter_sha_before == parameter_sha_after,
        "all_parameter_gradients_none": no_gradients,
        "exactly_two_instrumented_l1024_chunk_scans_per_forward": True,
        "production_top16_and_receiver_replay_wiring_exact": replay_wiring_exact,
        "receiver_k16_replay_state_active": replay_state_active,
        "full_validation_1000_cases": len(cases) == 1000 and len(rows) == 4000,
    }
    cost_checks = {
        "elapsed_ratio_at_most_3.00": cost["elapsed_ratio"] <= 3.00,
        "peak_allocation_ratio_at_most_1.50": cost["peak_allocation_ratio"] <= 1.50,
    }
    if not all(integrity.values()):
        raise RuntimeError(f"Read-only integrity contract failed: {integrity}")
    result = {
        "status": "completed",
        "plan": "P-DIAG-ADDR-001",
        "diagnostic_only": True,
        "model_smoke": False,
        "model_arm": MODEL_ARM,
        "protocol": {
            "sequence_length": SEQUENCE_LENGTH,
            "num_kv_pairs": NUM_KV_PAIRS,
            "examples": len(cases),
            "queries": len(rows),
            "state_geometry_and_survival_examples": STATE_DIAGNOSTIC_EXAMPLES,
            "seed": config.seed,
            "producer_surprise": "exact committed edit Frobenius norm per token",
            "committed_write_token": "value token at recovered write_position+1",
            "producer_basis_score": "per-head receiver normalized Q dot producer normalized committed-write K",
            "receiver_native_score": "per-head receiver normalized Q dot receiver normalized committed-write K",
            "capture_contract": (
                "two instrumented L1024 chunk scans; the official K16 short-sequence replay is "
                "validated by exact production selection/wiring identity and active receiver state"
            ),
            "logits_modified": False,
            "parameters_added": 0,
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
        "test_sha256": test_sha256,
        "parameter_sha256": parameter_sha_after,
        "integrity": integrity,
        "accuracy": accuracy,
        "wrong_key_swaps": swaps,
        "groups": groups,
        "key_geometry": geometry,
        "write_survival": survival,
        "correlations": {
            "surprise_vs_error": _pearson(
                [float(row["target_write_surprise"]) for row in rows],
                [float(not row["correct"]) for row in rows],
            ),
            "surprise_percentile_vs_error": _pearson(
                [float(row["target_write_surprise_percentile"]) for row in rows],
                [float(not row["correct"]) for row in rows],
            ),
            "receiver_minus_producer_rank_gain_vs_error": _pearson(
                [float(row["receiver_native_head_mrr"] - row["producer_basis_head_mrr"]) for row in rows],
                [float(not row["correct"]) for row in rows],
            ),
            "negative_target_survival_vs_error": _pearson(
                [-float(row["target_write_survival"]) for row in survival_rows],
                [float(not row["correct"]) for row in survival_rows],
            ),
        },
        "cost": cost,
        "cost_checks": cost_checks,
        "branch_decision": _branch_decision(groups, swaps, accuracy, geometry, survival),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.with_name("events.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all(cost_checks.values()):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
