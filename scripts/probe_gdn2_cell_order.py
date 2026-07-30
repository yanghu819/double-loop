#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO_ROOT / "experiments" / "rwkv_fs_sudoku"
sys.path.insert(0, str(RUNNER_DIR))

import study_rwkv_futureseed_loop as runner  # noqa: E402


def saved_value(saved_args: Dict[str, Any], name: str, default: Any) -> Any:
    value = saved_args.get(name, default)
    return default if value is None else value


def build_model(saved_args: Dict[str, Any]) -> runner.FutureSeedLoopSudoku:
    return runner.FutureSeedLoopSudoku(
        d_model=int(saved_args["d_model"]),
        layers=int(saved_args["layers"]),
        heads=int(saved_args["heads"]),
        head_dim=int(saved_args["head_dim"]),
        channel_mult=int(saved_args["channel_mult"]),
        l_cycles=int(saved_args["l_cycles"]),
        max_loops=int(saved_args["max_loops"]),
        lambda_=float(saved_args["lambda_"]),
        loop_update_mode=str(saved_value(saved_args, "loop_update_mode", "fixed")),
        loop_update_gate_init=float(
            saved_value(saved_args, "loop_update_gate_init", 0.95)
        ),
        future_seed_scale=float(saved_value(saved_args, "future_seed_scale", 1.0)),
        future_seed_decay=float(saved_value(saved_args, "future_seed_decay", 0.0)),
        future_seed_update=str(
            saved_value(saved_args, "future_seed_update", "fixed")
        ),
        future_seed_norm_mode=str(
            saved_value(saved_args, "future_seed_norm_mode", "unit")
        ),
        future_seed_gate_mode=str(
            saved_value(saved_args, "future_seed_gate_mode", "head")
        ),
        future_seed_scope=str(saved_value(saved_args, "future_seed_scope", "layer")),
        future_seed_readout_hop=int(
            saved_value(saved_args, "future_seed_readout_hop", 0)
        ),
        loop_feedback_scale=float(
            saved_value(saved_args, "loop_feedback_scale", 0.0)
        ),
        loop_feedback_detach=bool(
            saved_value(saved_args, "loop_feedback_detach", False)
        ),
        loop_feedback_corrupt_prob=float(
            saved_value(saved_args, "loop_feedback_corrupt_prob", 0.0)
        ),
        loop_feedback_corrupt_mix=float(
            saved_value(saved_args, "loop_feedback_corrupt_mix", 0.0)
        ),
        loop_feedback_corrupt_mode=str(
            saved_value(saved_args, "loop_feedback_corrupt_mode", "random_token")
        ),
        loop_time_scale=float(saved_value(saved_args, "loop_time_scale", 0.0)),
        scratch_mode=str(saved_value(saved_args, "scratch_mode", "none")),
        scratch_scale=float(saved_value(saved_args, "scratch_scale", 1.0)),
        scratch_noise_scale=float(
            saved_value(saved_args, "scratch_noise_scale", 0.0)
        ),
        scratch_gauss_projections=int(
            saved_value(saved_args, "scratch_gauss_projections", 0)
        ),
        scratch_gate_bias=float(
            saved_value(saved_args, "scratch_gate_bias", -2.0)
        ),
        scratch_decay_bias=float(
            saved_value(saved_args, "scratch_decay_bias", 2.0)
        ),
        hidden_agg_noise_scale=float(
            saved_value(saved_args, "hidden_agg_noise_scale", 0.0)
        ),
        hidden_agg_noise_temp=float(
            saved_value(saved_args, "hidden_agg_noise_temp", 1.0)
        ),
        hidden_agg_noise_detach=bool(
            saved_value(saved_args, "hidden_agg_noise_detach", True)
        ),
        hidden_agg_noise_mode=str(
            saved_value(saved_args, "hidden_agg_noise_mode", "gumbel")
        ),
        hidden_agg_noise_topk=int(
            saved_value(saved_args, "hidden_agg_noise_topk", 8)
        ),
        hidden_agg_noise_max_norm=float(
            saved_value(saved_args, "hidden_agg_noise_max_norm", 0.0)
        ),
        activation_checkpoint=False,
        rwkv_kernel=str(saved_value(saved_args, "rwkv_kernel", "auto")),
        backbone=str(saved_args["backbone"]),
        gdn_mode=str(saved_args["gdn_mode"]),
        gdn_expand_v=float(saved_args["gdn_expand_v"]),
        gdn_progressive_base_expand_v=float(
            saved_value(saved_args, "gdn_progressive_base_expand_v", 0.0)
        ),
        gdn_use_short_conv=bool(saved_args["gdn_use_short_conv"]),
        gdn_conv_size=int(saved_value(saved_args, "gdn_conv_size", 4)),
        gdn_allow_neg_eigval=bool(
            saved_value(saved_args, "gdn_allow_neg_eigval", False)
        ),
        gdn2_gain_budget_mode="none",
        gdn2_fast_slow_decay_mode="none",
        gdn2_address_mode=str(
            saved_value(saved_args, "gdn2_address_mode", "none")
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_orders(size: int, seed: int) -> Dict[str, torch.Tensor]:
    cells = size * size
    row_major = list(range(cells))
    column_major = [row * size + col for col in range(size) for row in range(size)]
    box_size = int(size**0.5)
    if box_size * box_size != size:
        raise ValueError("The diagnostic currently requires square Sudoku boxes")
    box_major = [
        row * size + col
        for box_row in range(box_size)
        for box_col in range(box_size)
        for row in range(box_row * box_size, (box_row + 1) * box_size)
        for col in range(box_col * box_size, (box_col + 1) * box_size)
    ]
    random_order = list(row_major)
    random.Random(seed).shuffle(random_order)
    orders = {
        "row_major": row_major,
        "reverse": list(reversed(row_major)),
        "column_major": column_major,
        "box_major": box_major,
        "random_fixed": random_order,
    }
    expected = set(row_major)
    for name, order in orders.items():
        if len(order) != cells or set(order) != expected:
            raise AssertionError(f"{name} is not a permutation")
    return {name: torch.tensor(order, dtype=torch.long) for name, order in orders.items()}


def canonicalize(sequence_tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if sequence_tensor.shape[1] != permutation.numel():
        raise ValueError("Sequence length and permutation disagree")
    canonical = torch.empty_like(sequence_tensor)
    canonical[:, permutation] = sequence_tensor
    return canonical


@torch.no_grad()
def evaluate_order(
    model: runner.FutureSeedLoopSudoku,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    permutation: torch.Tensor,
    *,
    loops: int,
    forward_dtype: str,
) -> Dict[str, Any]:
    device = inputs.device
    permutation = permutation.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with runner.forward_autocast(forward_dtype, device):
        sequence_logits, trace = model.forward_trace(
            inputs,
            loops=loops,
            noise_scale=0.0,
            cell_order=permutation,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    canonical_logits = [logits.detach().float() for logits in sequence_logits]
    loop_rows: Dict[str, Any] = {}
    predictions = []
    for loop_idx, logits in enumerate(canonical_logits, start=1):
        metrics, prediction = runner.metrics_from_logits(logits, labels, clue_mask)
        loop_rows[f"loop{loop_idx}"] = asdict(metrics)
        predictions.append(prediction.detach().cpu())

    encoded_sequence = (
        model.embed(inputs[:, permutation])
        + model.position(permutation).unsqueeze(0)
    )
    canonical_encoding = canonicalize(encoded_sequence, permutation)
    reference_encoding = model.embed(inputs) + model.position(
        torch.arange(inputs.shape[1], device=device)
    ).unsqueeze(0)
    encoding_error = float(
        (canonical_encoding.float() - reference_encoding.float()).abs().max().cpu()
    )
    return {
        "loops": loop_rows,
        "predictions": predictions,
        "trace": [
            runner.fs_metrics_from_trace(loop_trace) for loop_trace in trace
        ],
        "wall_sec": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "paired_encoding_roundtrip_max_abs": encoding_error,
    }


def digit(value: int) -> str:
    return str(value + 1)


def render_board(
    inputs: list[int],
    labels: list[int],
    prediction: list[int],
    *,
    size: int,
) -> str:
    cells = []
    blank = size
    for index, value in enumerate(prediction):
        is_clue = inputs[index] != blank
        correct = value == labels[index]
        classes = ["cell"]
        classes.append("clue" if is_clue else ("correct" if correct else "wrong"))
        title = (
            f"R{index // size + 1}C{index % size + 1}: "
            f"target {digit(labels[index])}, prediction {digit(value)}"
        )
        cells.append(
            f'<div class="{" ".join(classes)}" title="{html.escape(title)}">'
            f"{digit(value)}</div>"
        )
    return (
        f'<div class="board" style="grid-template-columns:repeat({size},30px)">'
        + "".join(cells)
        + "</div>"
    )


def build_html(payload: Dict[str, Any], cases: list[Dict[str, Any]]) -> str:
    orders = list(payload["orders"])
    loops = payload["loops"]
    runtime = payload["runtime"]
    runtime_label = runtime.get("implementation")
    if runtime_label is None:
        layers = runtime.get("layers", [])
        runtime_label = (
            f"{layers[0].get('class', 'unknown')}/{layers[0].get('execution_path', 'unknown')}"
            if layers
            else "unknown"
        )
    metric_rows = []
    for order in orders:
        row = payload["orders"][order]
        for loop in loops:
            metrics = row["loops"][f"loop{loop}"]
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(order)}</td><td>{loop}</td>"
                f"<td>{metrics['label_exact']:.4f}</td>"
                f"<td>{metrics['blank_acc']:.4f}</td>"
                f"<td>{metrics['valid_sudoku']:.4f}</td>"
                f"<td>{row['blank_prediction_change_vs_row'][f'loop{loop}']:.4f}</td>"
                f"<td>{row['wall_sec']:.2f}</td>"
                "</tr>"
            )
    case_sections = []
    for case in cases:
        panels = []
        for order in orders:
            order_case = case["orders"][order]
            boards = []
            for loop in (1, loops[-1]):
                boards.append(
                    "<div>"
                    f"<h4>{html.escape(order)} loop{loop}</h4>"
                    + render_board(
                        case["input"],
                        case["label"],
                        order_case[f"loop{loop}"],
                        size=payload["board_size"],
                    )
                    + f"<p>wrong blanks: {order_case[f'loop{loop}_wrong_blanks']}</p>"
                    + "</div>"
                )
            panels.append('<div class="order-case">' + "".join(boards) + "</div>")
        case_sections.append(
            f"<section><h2>Case {case['batch_index']}</h2>"
            '<div class="case-grid">'
            + "".join(panels)
            + "</div></section>"
        )
    decision = payload["decision"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GDN2 cell-order counterfactual</title>
<style>
body {{ margin:24px; font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#111827; background:#f8fafc; }}
h1,h2,h3,h4 {{ letter-spacing:0; }} p {{ line-height:1.45; }}
.decision {{ border-left:5px solid {'#16a34a' if decision['address_hypothesis_supported'] else '#dc2626'}; background:#fff; padding:14px; margin:16px 0; }}
table {{ width:100%; border-collapse:collapse; background:#fff; }}
th,td {{ border:1px solid #cbd5e1; padding:7px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
.case-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(610px,1fr)); gap:12px; }}
.order-case {{ display:flex; gap:10px; padding:10px; background:#fff; border:1px solid #d1d5db; }}
.board {{ display:grid; width:max-content; border-left:2px solid #334155; border-top:2px solid #334155; }}
.cell {{ width:30px; height:30px; display:flex; align-items:center; justify-content:center; box-sizing:border-box; border-right:1px solid #94a3b8; border-bottom:1px solid #94a3b8; font-weight:700; }}
.clue {{ background:#e2e8f0; }} .correct {{ background:#bbf7d0; }} .wrong {{ background:#fecaca; color:#991b1b; }}
code {{ background:#e2e8f0; padding:2px 4px; }}
</style></head><body>
<h1>Does GDN2 bind cells to positions, or memorize row-major order?</h1>
<p>Every traversal sees the same board. Each content token keeps its correct absolute position embedding, and logits are restored to canonical cell order before scoring. Only recurrent traversal order changes.</p>
<div class="decision"><strong>{html.escape(decision['label'])}</strong><br>{html.escape(decision['reason'])}</div>
<h2>Metrics</h2>
<table><thead><tr><th>Traversal</th><th>Loop</th><th>Exact</th><th>Blank accuracy</th><th>Valid</th><th>Blank predictions changed vs row</th><th>Wall sec</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table>
<h2>Integrity</h2>
<p>Checkpoint <code>{html.escape(payload['checkpoint_sha256'])}</code>; source <code>{html.escape(payload['source_sha'])}</code>; official runtime <code>{html.escape(str(runtime_label))}</code>. Paired encoding roundtrip max error is <code>{payload['max_paired_encoding_roundtrip_abs']:.3e}</code>.</p>
{''.join(case_sections)}
</body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-only paired cell-order counterfactual for GDN2+FutureSeed"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-n", type=int, default=256)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--order-seed", type=int, default=52081)
    parser.add_argument("--eval-seed", type=int, default=52551)
    parser.add_argument("--case-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0; this probe is GPU1-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; CPU model fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.checkpoint_sha256:
        raise RuntimeError(
            f"Checkpoint hash mismatch: expected {args.checkpoint_sha256}, got {checkpoint_hash}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    saved_args = checkpoint["args"]
    if saved_args.get("backbone") != "gdn2":
        raise RuntimeError("The position-binding probe is restricted to GDN2")
    runner.configure_sudoku(int(saved_args["size"]), 0, 0)
    device = torch.device("cuda", 0)
    model = build_model(saved_args).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is not exact for this source: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    runtime = runner.strict_fla_runtime_summary(model, "gdn2")
    dataset = runner.OfficialSudokuDataset(args.data_dir, "test")
    inputs, labels, clue_mask = dataset.fixed_batch_by_blank_range(
        args.eval_n,
        args.eval_seed,
        holes_min=51,
        holes_max=64,
        device=device,
    )
    orders = cell_orders(int(saved_args["size"]), args.order_seed)
    raw_results: Dict[str, Any] = {}
    for name, permutation in orders.items():
        result = evaluate_order(
            model,
            inputs,
            labels,
            clue_mask,
            permutation,
            loops=args.loops,
            forward_dtype=args.forward_dtype,
        )
        raw_results[name] = result
        final = result["loops"][f"loop{args.loops}"]
        print(
            f"{name} loop{args.loops} exact={final['label_exact']:.4f} "
            f"blank={final['blank_acc']:.4f} wall={result['wall_sec']:.2f}s",
            flush=True,
        )

    row_predictions = raw_results["row_major"]["predictions"]
    blank_mask_cpu = (~clue_mask).detach().cpu()
    for name, result in raw_results.items():
        changes = {}
        for loop_idx, prediction in enumerate(result["predictions"], start=1):
            changed = prediction.ne(row_predictions[loop_idx - 1]) & blank_mask_cpu
            changes[f"loop{loop_idx}"] = float(
                changed.sum().float() / blank_mask_cpu.sum().clamp_min(1)
            )
        result["blank_prediction_change_vs_row"] = changes

    row_final = raw_results["row_major"]["loops"][f"loop{args.loops}"]
    nonrow = [name for name in orders if name != "row_major"]
    worst_exact_delta = min(
        raw_results[name]["loops"][f"loop{args.loops}"]["label_exact"]
        - row_final["label_exact"]
        for name in nonrow
    )
    worst_blank_delta = min(
        raw_results[name]["loops"][f"loop{args.loops}"]["blank_acc"]
        - row_final["blank_acc"]
        for name in nonrow
    )
    hypothesis_supported = worst_exact_delta < -0.03 or worst_blank_delta < -0.02
    decision = {
        "address_hypothesis_supported": hypothesis_supported,
        "worst_nonrow_exact_delta": worst_exact_delta,
        "worst_nonrow_blank_delta": worst_blank_delta,
        "label": (
            "Position-binding hypothesis survives this falsifier."
            if hypothesis_supported
            else "Position-binding hypothesis is rejected at this gate."
        ),
        "reason": (
            "At least one paired traversal loses more than 3 exact points or 2 blank-accuracy points. "
            "The model sees identical token-position pairs but depends materially on serialization order."
            if hypothesis_supported
            else "All paired traversals remain within the preregistered tolerance. A fixed memory address is unlikely to be the current bottleneck."
        ),
    }

    random_final = raw_results["random_fixed"]["predictions"][-1]
    row_final_predictions = row_predictions[-1]
    per_case_disagreement = (
        row_final_predictions.ne(random_final) & blank_mask_cpu
    ).sum(dim=1)
    case_count = min(args.case_count, inputs.shape[0])
    selected = torch.topk(per_case_disagreement, k=case_count).indices.tolist()
    inputs_cpu = inputs.detach().cpu()
    labels_cpu = labels.detach().cpu()
    cases = []
    for batch_index in selected:
        case: Dict[str, Any] = {
            "batch_index": int(batch_index),
            "input": inputs_cpu[batch_index].tolist(),
            "label": labels_cpu[batch_index].tolist(),
            "orders": {},
        }
        blank_case = blank_mask_cpu[batch_index]
        for name, result in raw_results.items():
            order_case: Dict[str, Any] = {}
            for loop_idx, prediction in enumerate(result["predictions"], start=1):
                pred = prediction[batch_index]
                order_case[f"loop{loop_idx}"] = pred.tolist()
                order_case[f"loop{loop_idx}_wrong_blanks"] = int(
                    (pred.ne(labels_cpu[batch_index]) & blank_case).sum()
                )
            case["orders"][name] = order_case
        cases.append(case)

    serializable_orders = {}
    for name, result in raw_results.items():
        serializable_orders[name] = {
            key: value
            for key, value in result.items()
            if key not in {"predictions"}
        }
    payload = {
        "schema_version": "gdn2_cell_order_counterfactual.v1",
        "hypothesis": (
            "If GDN2 does not bind content to canonical positions, changing traversal "
            "while preserving every token-position pair will sharply reduce hard Sudoku quality."
        ),
        "intervention": (
            "Permute complete cell representations, preserve absolute position IDs, "
            "and restore logits to canonical order before scoring."
        ),
        "kill_criteria": {
            "max_exact_drop_to_reject_address_direction": 0.03,
            "max_blank_accuracy_drop_to_reject_address_direction": 0.02,
        },
        "decision": decision,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": int(checkpoint.get("saved_at_step", -1)),
        "source_sha": str(saved_args.get("source_sha", checkpoint.get("source_sha", ""))),
        "board_size": int(saved_args["size"]),
        "blank_range": [51, 64],
        "eval_n": int(inputs.shape[0]),
        "loops": list(range(1, args.loops + 1)),
        "runtime": runtime,
        "gpu": torch.cuda.get_device_name(device),
        "orders": serializable_orders,
        "max_paired_encoding_roundtrip_abs": max(
            result["paired_encoding_roundtrip_max_abs"]
            for result in raw_results.values()
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "score.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "index.html").write_text(
        build_html(payload, cases),
        encoding="utf-8",
    )
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
