#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "rwkv_fs_sudoku"))

from study_rwkv_futureseed_loop import (  # noqa: E402
    FutureSeedRWKV,
    forward_autocast,
    strict_fla_runtime_summary,
)


PAST_QUERY = 1
FUTURE_QUERY = 2


@dataclass
class MQARBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_kind: torch.Tensor

    def to(self, device: torch.device) -> "MQARBatch":
        return MQARBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            query_kind=self.query_kind.to(device),
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def load_zoology_generator(zoology_root: Path, expected_sha: str):
    root = zoology_root.resolve()
    if not (root / ".git").is_dir():
        raise RuntimeError(f"Zoology checkout is missing: {root}")
    actual_sha = git_head(root)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Zoology SHA mismatch: {actual_sha} != {expected_sha}")
    sys.path.insert(0, str(root))
    module = importlib.import_module("zoology.data.multiquery_ar")
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"Imported MQAR generator outside pinned checkout: {source}")
    return module.multiquery_ar, {
        "repo": "https://github.com/HazyResearch/zoology",
        "root": str(root),
        "git_sha": actual_sha,
        "generator_file": str(source.relative_to(root)),
        "generator_sha256": file_sha256(source),
        "transform": "balanced_half_batch_context_query_block_swap_v1",
    }


def generate_balanced_batch(
    *,
    generator_fn,
    batch_size: int,
    seq_len: int,
    pairs: int,
    vocab_size: int,
    seed: int,
) -> MQARBatch:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be a positive even integer")
    context_size = 2 * pairs
    if context_size >= seq_len:
        raise ValueError("MQAR context must leave room for queries")

    # The upstream generator seeds NumPy internally but uses the global Torch RNG
    # for distractors. Forking the RNG makes the complete batch deterministic.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        segment = generator_fn(
            vocab_size=vocab_size,
            num_examples=batch_size,
            input_seq_len=seq_len,
            seed=seed,
            power_a=0.01,
            num_kv_pairs=pairs,
            num_passes=1,
            random_non_queries=True,
            include_slices=True,
        )

    input_ids = segment.inputs.clone().to(torch.long)
    labels = segment.labels.clone().to(torch.long)
    query_kind = torch.zeros_like(labels)
    midpoint = batch_size // 2
    past_mask = labels[:midpoint] >= 0
    query_kind[:midpoint] = torch.where(
        past_mask,
        torch.full_like(query_kind[:midpoint], PAST_QUERY),
        query_kind[:midpoint],
    )

    # This is the only task transformation: preserve the exact upstream MQAR
    # tokens and targets, but move the query block before its K/V context.
    future_inputs = input_ids[midpoint:].clone()
    future_labels = labels[midpoint:].clone()
    input_ids[midpoint:] = torch.cat(
        [future_inputs[:, context_size:], future_inputs[:, :context_size]],
        dim=1,
    )
    labels[midpoint:] = torch.cat(
        [future_labels[:, context_size:], future_labels[:, :context_size]],
        dim=1,
    )
    future_mask = labels[midpoint:] >= 0
    query_kind[midpoint:] = torch.where(
        future_mask,
        torch.full_like(query_kind[midpoint:], FUTURE_QUERY),
        query_kind[midpoint:],
    )

    batch = MQARBatch(input_ids=input_ids, labels=labels, query_kind=query_kind)
    validate_batch(batch, pairs=pairs, vocab_size=vocab_size, context_size=context_size)
    return batch


def validate_batch(
    batch: MQARBatch,
    *,
    pairs: int,
    vocab_size: int,
    context_size: int,
) -> None:
    query_mask = batch.labels >= 0
    if not torch.all(query_mask.sum(dim=1) == pairs):
        raise RuntimeError("Every MQAR sequence must contain exactly num_kv_pairs queries")
    midpoint = batch.input_ids.shape[0] // 2
    if not torch.all((batch.query_kind[:midpoint] == PAST_QUERY).sum(dim=1) == pairs):
        raise RuntimeError("Past half has an invalid query-kind mask")
    if not torch.all((batch.query_kind[midpoint:] == FUTURE_QUERY).sum(dim=1) == pairs):
        raise RuntimeError("Future half has an invalid query-kind mask")
    if not torch.equal(query_mask, batch.query_kind > 0):
        raise RuntimeError("Labels and query-kind masks disagree")

    key_limit = vocab_size // 2
    for row in range(batch.input_ids.shape[0]):
        if row < midpoint:
            context = batch.input_ids[row, :context_size]
        else:
            context = batch.input_ids[row, -context_size:]
        keys = context[0::2]
        values = context[1::2]
        if not torch.all((keys > 0) & (keys < key_limit)):
            raise RuntimeError("Upstream MQAR key vocabulary contract failed")
        if not torch.all((values >= key_limit) & (values < vocab_size)):
            raise RuntimeError("Upstream MQAR value vocabulary contract failed")
        for pos in torch.nonzero(query_mask[row], as_tuple=False).flatten().tolist():
            query_key = batch.input_ids[row, pos]
            matches = torch.nonzero(keys == query_key, as_tuple=False).flatten()
            if matches.numel() != 1:
                raise RuntimeError("Every MQAR query must match exactly one K/V pair")
            expected = values[int(matches.item())]
            if batch.labels[row, pos] != expected:
                raise RuntimeError("MQAR target does not match its K/V pair")


class FutureSeedMQARModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        future_seed_scale: float,
        loop_lambda: float,
        gdn_mode: str,
        gdn_expand_v: float,
        gdn_conv_size: int,
    ) -> None:
        super().__init__()
        self.loop_lambda = float(loop_lambda)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.hidden_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.reasoner = FutureSeedRWKV(
            d_model,
            layers,
            heads,
            head_dim,
            channel_mult,
            future_seed_scale=future_seed_scale,
            future_seed_decay=0.0,
            future_seed_update="fixed",
            future_seed_norm_mode="unit",
            future_seed_gate_mode="head",
            future_seed_scope="layer",
            activation_checkpoint=False,
            backbone="gdn2",
            gdn_mode=gdn_mode,
            gdn_expand_v=gdn_expand_v,
            gdn_use_short_conv=True,
            gdn_conv_size=gdn_conv_size,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward_trace(
        self,
        batch: MQARBatch,
        *,
        loops: int,
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        base = self.token_embed(batch.input_ids)
        hidden = self.hidden_init.expand(base.shape[0], base.shape[1], -1)
        query_mask = batch.labels >= 0
        logits_by_loop: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        for _ in range(int(loops)):
            proposed, diag, _seed_memory = self.reasoner(hidden + base)
            hidden = hidden + self.loop_lambda * (proposed - hidden)
            selected = self.out_norm(hidden)[query_mask]
            logits_by_loop.append(F.linear(selected, self.token_embed.weight))
            diagnostics.append(diag)
        return logits_by_loop, diagnostics


def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_sha256(batch: MQARBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (batch.input_ids, batch.labels, batch.query_kind):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def query_loss(logits_by_loop: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    selected_labels = labels[labels >= 0]
    losses = [F.cross_entropy(logits.float(), selected_labels) for logits in logits_by_loop]
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: FutureSeedMQARModel,
    batch: MQARBatch,
    *,
    eval_batch: int,
    loops: int,
    device: torch.device,
    forward_dtype: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    model.eval()
    totals = {
        loop: {
            "correct": 0.0,
            "count": 0.0,
            "past_correct": 0.0,
            "past_count": 0.0,
            "future_correct": 0.0,
            "future_count": 0.0,
            "exact": 0.0,
            "past_exact": 0.0,
            "future_exact": 0.0,
            "cases": 0.0,
            "past_cases": 0.0,
            "future_cases": 0.0,
            "ce_sum": 0.0,
        }
        for loop in range(1, loops + 1)
    }
    case_rows: list[dict[str, Any]] = []
    size = batch.input_ids.shape[0]
    for start in range(0, size, eval_batch):
        stop = min(start + eval_batch, size)
        cpu_slice = MQARBatch(
            input_ids=batch.input_ids[start:stop],
            labels=batch.labels[start:stop],
            query_kind=batch.query_kind[start:stop],
        )
        gpu_batch = cpu_slice.to(device)
        with forward_autocast(forward_dtype, device):
            logits_by_loop, _diagnostics = model.forward_trace(gpu_batch, loops=loops)
        query_mask = gpu_batch.labels >= 0
        selected_labels = gpu_batch.labels[query_mask]
        selected_kind = gpu_batch.query_kind[query_mask]
        past_rows = (gpu_batch.query_kind == PAST_QUERY).any(dim=1)
        future_rows = (gpu_batch.query_kind == FUTURE_QUERY).any(dim=1)
        payloads = [
            {
                "case_index": start + idx,
                "input_ids": cpu_slice.input_ids[idx].tolist(),
                "labels": cpu_slice.labels[idx].tolist(),
                "query_kind": cpu_slice.query_kind[idx].tolist(),
                "predictions": {},
            }
            for idx in range(stop - start)
        ]
        for loop_idx, logits in enumerate(logits_by_loop, start=1):
            selected_pred = logits.argmax(dim=-1)
            selected_correct = selected_pred == selected_labels
            pred_matrix = torch.full_like(gpu_batch.labels, -1)
            pred_matrix[query_mask] = selected_pred
            row_correct = (pred_matrix == gpu_batch.labels) | ~query_mask
            row = totals[loop_idx]
            row["correct"] += float(selected_correct.sum())
            row["count"] += float(selected_correct.numel())
            row["past_correct"] += float(selected_correct[selected_kind == PAST_QUERY].sum())
            row["past_count"] += float((selected_kind == PAST_QUERY).sum())
            row["future_correct"] += float(selected_correct[selected_kind == FUTURE_QUERY].sum())
            row["future_count"] += float((selected_kind == FUTURE_QUERY).sum())
            row["exact"] += float(row_correct.all(dim=1).sum())
            row["past_exact"] += float(row_correct.all(dim=1)[past_rows].sum())
            row["future_exact"] += float(row_correct.all(dim=1)[future_rows].sum())
            row["cases"] += float(stop - start)
            row["past_cases"] += float(past_rows.sum())
            row["future_cases"] += float(future_rows.sum())
            row["ce_sum"] += float(F.cross_entropy(logits.float(), selected_labels, reduction="sum"))
            pred_cpu = pred_matrix.detach().cpu()
            for idx, payload in enumerate(payloads):
                payload["predictions"][f"loop{loop_idx}"] = pred_cpu[idx].tolist()
        case_rows.extend(payloads)

    metrics: dict[str, dict[str, float]] = {}
    for loop_idx, row in totals.items():
        metrics[f"loop{loop_idx}"] = {
            "query_acc": row["correct"] / row["count"],
            "past_acc": row["past_correct"] / row["past_count"],
            "future_acc": row["future_correct"] / row["future_count"],
            "query_exact": row["exact"] / row["cases"],
            "past_exact": row["past_exact"] / row["past_cases"],
            "future_exact": row["future_exact"] / row["future_cases"],
            "query_ce": row["ce_sum"] / row["count"],
        }
    return metrics, case_rows


def select_hard_cases(cases: list[dict[str, Any]], *, loops: int, limit: int) -> list[dict[str, Any]]:
    def errors(case: dict[str, Any]) -> tuple[int, int]:
        pred = case["predictions"][f"loop{loops}"]
        count = sum(
            int(label >= 0 and pred[idx] != label)
            for idx, label in enumerate(case["labels"])
        )
        is_future = int(any(kind == FUTURE_QUERY for kind in case["query_kind"]))
        return count, is_future

    return sorted(cases, key=lambda case: (-errors(case)[0], -errors(case)[1], case["case_index"]))[:limit]


def render_visualization(cases: list[dict[str, Any]], *, loops: int, path: Path, title: str) -> None:
    visual_loops = sorted({1, 2, loops})
    sections: list[str] = []
    for case in cases:
        direction = "FUTURE" if any(kind == FUTURE_QUERY for kind in case["query_kind"]) else "PAST"
        rows: list[str] = []
        for pos, label in enumerate(case["labels"]):
            if label < 0:
                continue
            predictions = []
            for loop in visual_loops:
                pred = case["predictions"][f"loop{loop}"][pos]
                cls = "ok" if pred == label else "bad"
                predictions.append(f'<td class="{cls}">{pred}</td>')
            rows.append(
                f"<tr><td>{pos}</td><td>{case['input_ids'][pos]}</td><td>{label}</td>{''.join(predictions)}</tr>"
            )
        headers = "".join(f"<th>Loop {loop}</th>" for loop in visual_loops)
        sections.append(
            f"<section><h2>Case {case['case_index']} / {direction}</h2>"
            f"<table><thead><tr><th>Query position</th><th>Key token</th><th>Target value</th>{headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#f6f7f8;color:#181b20;font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
header{{padding:24px 28px;background:#fff;border-bottom:1px solid #c8cdd3}} h1{{margin:0 0 8px;font-size:24px;letter-spacing:0}}
p{{margin:0;line-height:1.5}} main{{padding:0 28px 36px}} section{{padding:22px 0;border-bottom:1px solid #c8cdd3}}
h2{{font-size:18px;margin:0 0 12px}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;background:#fff}}
th,td{{padding:8px 10px;border:1px solid #d5d9de;text-align:left}} th{{background:#eceff2}} .ok{{background:#dff3e4;color:#145c2b;font-weight:700}}
.bad{{background:#f8dfda;color:#8a2417;font-weight:700}} @media(max-width:760px){{header,main{{padding-left:12px;padding-right:12px}} table{{font-size:12px}}}}
</style></head><body><header><h1>{html.escape(title)}</h1><p>PAST is untouched upstream MQAR. FUTURE moves the unchanged query block before its K/V context. Green is correct and red is wrong.</p></header><main>{''.join(sections)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition", default="")
    parser.add_argument("--zoology-root", type=Path, required=True)
    parser.add_argument("--zoology-sha", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=1024)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--channel-mult", type=int, default=4)
    parser.add_argument("--train-loops", type=int, default=2)
    parser.add_argument("--eval-loops", type=int, default=4)
    parser.add_argument("--loop-lambda", type=float, default=0.95)
    parser.add_argument("--future-seed-scale", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument("--gdn-mode", choices=("chunk",), default="chunk")
    parser.add_argument("--gdn-expand-v", type=float, default=1.0)
    parser.add_argument("--gdn-conv-size", type=int, default=4)
    parser.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--viz-cases", type=int, default=8)
    parser.add_argument("--carrier-kill-step", type=int, default=500)
    parser.add_argument("--carrier-min-past-acc", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This probe requires exactly one visible CUDA GPU")
    if args.d_model != args.heads * args.head_dim:
        raise ValueError("d_model must equal heads * head_dim")
    if args.eval_n % 2 or args.batch % 2:
        raise ValueError("Training and evaluation batch sizes must be even")

    generator_fn, zoology_provenance = load_zoology_generator(args.zoology_root, args.zoology_sha)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    model = FutureSeedMQARModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        channel_mult=args.channel_mult,
        future_seed_scale=args.future_seed_scale,
        loop_lambda=args.loop_lambda,
        gdn_mode=args.gdn_mode,
        gdn_expand_v=args.gdn_expand_v,
        gdn_conv_size=args.gdn_conv_size,
    ).to(device)
    init_hash = model_sha256(model)
    fla_runtime = strict_fla_runtime_summary(model, "gdn2")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=0.0)
    eval_batch = generate_balanced_batch(
        generator_fn=generator_fn,
        batch_size=args.eval_n,
        seq_len=args.seq_len,
        pairs=args.pairs,
        vocab_size=args.vocab_size,
        seed=args.seed + 10_000,
    )
    eval_hash = batch_sha256(eval_batch)
    history: list[dict[str, Any]] = []
    aborted = False
    abort: dict[str, Any] | None = None
    t0 = time.time()
    last_cases: list[dict[str, Any]] = []
    last_metrics: dict[str, dict[str, float]] = {}
    completed_steps = 0

    for step in range(1, args.steps + 1):
        model.train()
        batch = generate_balanced_batch(
            generator_fn=generator_fn,
            batch_size=args.batch,
            seq_len=args.seq_len,
            pairs=args.pairs,
            vocab_size=args.vocab_size,
            seed=args.seed + step,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with forward_autocast(args.forward_dtype, device):
            logits_by_loop, _diagnostics = model.forward_trace(batch, loops=args.train_loops)
            loss = query_loss(logits_by_loop, batch.labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        completed_steps = step

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            last_metrics, last_cases = evaluate(
                model,
                eval_batch,
                eval_batch=args.eval_batch,
                loops=args.eval_loops,
                device=device,
                forward_dtype=args.forward_dtype,
            )
            loop1 = last_metrics["loop1"]
            final = last_metrics[f"loop{args.eval_loops}"]
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": time.time() - t0,
                "loop1": loop1,
                "final": final,
                "future_loop_gain": final["future_acc"] - loop1["future_acc"],
                "past_loop_gain": final["past_acc"] - loop1["past_acc"],
            }
            history.append(row)
            print(
                f"[zoology_mqar] step={step:04d} loss={row['loss']:.4f} "
                f"past={loop1['past_acc']:.4f}->{final['past_acc']:.4f} "
                f"future={loop1['future_acc']:.4f}->{final['future_acc']:.4f} "
                f"exact={loop1['query_exact']:.4f}->{final['query_exact']:.4f} "
                f"ce={final['query_ce']:.4f}",
                flush=True,
            )
            if (
                args.carrier_kill_step > 0
                and step >= args.carrier_kill_step
                and final["past_acc"] < args.carrier_min_past_acc
            ):
                aborted = True
                abort = {
                    "reason": "upstream_mqar_past_carrier_failed",
                    "step": step,
                    "past_acc": final["past_acc"],
                    "threshold": args.carrier_min_past_acc,
                    "scientific_failure": False,
                }
                break

    elapsed = time.time() - t0
    final_key = f"loop{args.eval_loops}"
    final = last_metrics[final_key]
    hard_cases = select_hard_cases(last_cases, loops=args.eval_loops, limit=args.viz_cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.out_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    (visual_dir / "cases.json").write_text(json.dumps(hard_cases, indent=2) + "\n", encoding="utf-8")
    render_visualization(
        hard_cases,
        loops=args.eval_loops,
        path=visual_dir / "index.html",
        title=f"GDN2 on upstream Zoology MQAR: {args.condition or args.run_name}",
    )
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "task": "zoology_mqar_balanced_directionality",
        "config": vars(args) | {
            "out_dir": str(args.out_dir),
            "zoology_root": str(args.zoology_root),
        },
        "zoology_provenance": zoology_provenance,
        "init_sha256": init_hash,
        "eval_batch_sha256": eval_hash,
        "parameter_count": parameter_count,
        "fla_runtime": fla_runtime,
        "history": history,
        "metrics": last_metrics,
        "score": final["future_acc"],
        "score_key": f"metrics.{final_key}.future_acc",
        "loop_gain": final["future_acc"] - last_metrics["loop1"]["future_acc"],
        "elapsed_sec": elapsed,
        "tokens_per_sec": args.batch * args.seq_len * completed_steps / max(elapsed, 1e-9),
        "peak_cuda_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "aborted": aborted,
        "abort": abort,
        "decision": "invalid_carrier" if aborted else "completed",
    }
    result_path = args.out_dir / "gdn2_zoology_mqar_probe.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    run_dir = args.out_dir.parent
    source_head_path = run_dir / "source_HEAD.txt"
    git_sha = source_head_path.read_text().strip() if source_head_path.is_file() else "unknown"
    (run_dir / "score.json").write_text(
        json.dumps(
            {
                "score": payload["score"],
                "score_key": payload["score_key"],
                "git_sha": git_sha,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if abort is not None:
        (run_dir / "abort.json").write_text(json.dumps(abort, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_name": args.run_name, "score": payload["score"], "aborted": aborted}, indent=2))


if __name__ == "__main__":
    main()
