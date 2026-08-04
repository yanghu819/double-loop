#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
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


FILLER = 0
WRITE = 1
QUERY = 2
PAST_QUERY = 1
FUTURE_QUERY = 2


@dataclass
class RetrievalBatch:
    token_type: torch.Tensor
    key_id: torch.Tensor
    value_id: torch.Tensor
    labels: torch.Tensor
    query_kind: torch.Tensor

    def to(self, device: torch.device) -> "RetrievalBatch":
        return RetrievalBatch(
            token_type=self.token_type.to(device),
            key_id=self.key_id.to(device),
            value_id=self.value_id.to(device),
            labels=self.labels.to(device),
            query_kind=self.query_kind.to(device),
        )


def generate_batch(
    *,
    batch_size: int,
    seq_len: int,
    pairs: int,
    key_vocab: int,
    value_vocab: int,
    generator: torch.Generator,
) -> RetrievalBatch:
    if pairs < 2 or pairs % 2:
        raise ValueError("pairs must be an even integer >= 2")
    if seq_len < 2 * pairs:
        raise ValueError("seq_len must leave two positions per association")
    if key_vocab < pairs or value_vocab < pairs:
        raise ValueError("key/value vocab must support unique items within a sequence")

    token_type = torch.full((batch_size, seq_len), FILLER, dtype=torch.long)
    key_id = torch.zeros((batch_size, seq_len), dtype=torch.long)
    value_id = torch.zeros((batch_size, seq_len), dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    query_kind = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for batch_idx in range(batch_size):
        positions = torch.randperm(seq_len, generator=generator)[: 2 * pairs]
        keys = torch.randperm(key_vocab, generator=generator)[:pairs] + 1
        values = torch.randperm(value_vocab, generator=generator)[:pairs]
        for pair_idx in range(pairs):
            first, second = positions[2 * pair_idx : 2 * pair_idx + 2].sort().values.tolist()
            is_future = pair_idx < pairs // 2
            query_pos, write_pos = (first, second) if is_future else (second, first)
            key = int(keys[pair_idx])
            value = int(values[pair_idx])

            token_type[batch_idx, write_pos] = WRITE
            key_id[batch_idx, write_pos] = key
            value_id[batch_idx, write_pos] = value + 1

            token_type[batch_idx, query_pos] = QUERY
            key_id[batch_idx, query_pos] = key
            labels[batch_idx, query_pos] = value
            query_kind[batch_idx, query_pos] = FUTURE_QUERY if is_future else PAST_QUERY

    validate_batch(
        RetrievalBatch(token_type, key_id, value_id, labels, query_kind),
        pairs=pairs,
    )
    return RetrievalBatch(token_type, key_id, value_id, labels, query_kind)


def validate_batch(batch: RetrievalBatch, *, pairs: int) -> None:
    query_mask = batch.token_type == QUERY
    write_mask = batch.token_type == WRITE
    if not torch.all(query_mask.sum(dim=1) == pairs):
        raise RuntimeError("retrieval batch has the wrong query count")
    if not torch.all(write_mask.sum(dim=1) == pairs):
        raise RuntimeError("retrieval batch has the wrong write count")
    if not torch.all((batch.query_kind == PAST_QUERY).sum(dim=1) == pairs // 2):
        raise RuntimeError("retrieval batch has the wrong past-query count")
    if not torch.all((batch.query_kind == FUTURE_QUERY).sum(dim=1) == pairs // 2):
        raise RuntimeError("retrieval batch has the wrong future-query count")
    if not torch.equal(query_mask, batch.labels >= 0):
        raise RuntimeError("labels must exist exactly at query positions")

    for row in range(batch.token_type.shape[0]):
        for pos in torch.nonzero(query_mask[row], as_tuple=False).flatten().tolist():
            key = int(batch.key_id[row, pos])
            matches = torch.nonzero(
                write_mask[row] & (batch.key_id[row] == key),
                as_tuple=False,
            ).flatten()
            if matches.numel() != 1:
                raise RuntimeError("every query must have one matching write")
            write_pos = int(matches.item())
            expected = int(batch.value_id[row, write_pos]) - 1
            if int(batch.labels[row, pos]) != expected:
                raise RuntimeError("query target does not match its write value")
            is_future = int(batch.query_kind[row, pos]) == FUTURE_QUERY
            if is_future != (pos < write_pos):
                raise RuntimeError("query-kind ordering contract is broken")


class FutureRetrievalModel(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        key_vocab: int,
        value_vocab: int,
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
        self.seq_len = int(seq_len)
        self.loop_lambda = float(loop_lambda)
        self.type_embed = nn.Embedding(3, d_model)
        self.key_embed = nn.Embedding(key_vocab + 1, d_model, padding_idx=0)
        self.value_embed = nn.Embedding(value_vocab + 1, d_model, padding_idx=0)
        self.position = nn.Embedding(seq_len, d_model)
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
        self.head = nn.Linear(d_model, value_vocab, bias=False)

    def embed(self, batch: RetrievalBatch) -> torch.Tensor:
        positions = torch.arange(self.seq_len, device=batch.token_type.device)
        return (
            self.type_embed(batch.token_type)
            + self.key_embed(batch.key_id)
            + self.value_embed(batch.value_id)
            + self.position(positions).unsqueeze(0)
        )

    def forward_trace(self, batch: RetrievalBatch, *, loops: int) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        base = self.embed(batch)
        hidden = self.hidden_init.expand(base.shape[0], base.shape[1], -1)
        logits_by_loop: list[torch.Tensor] = []
        diagnostics: list[dict[str, torch.Tensor]] = []
        for _ in range(int(loops)):
            proposed, diag, _seed_memory = self.reasoner(hidden + base)
            hidden = hidden + self.loop_lambda * (proposed - hidden)
            logits_by_loop.append(self.head(self.out_norm(hidden)))
            diagnostics.append(diag)
        return logits_by_loop, diagnostics


def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def batch_sha256(batch: RetrievalBatch) -> str:
    digest = hashlib.sha256()
    for tensor in (batch.token_type, batch.key_id, batch.value_id, batch.labels, batch.query_kind):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def query_loss(logits_by_loop: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    mask = labels >= 0
    losses = [F.cross_entropy(logits.float()[mask], labels[mask]) for logits in logits_by_loop]
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: FutureRetrievalModel,
    batch: RetrievalBatch,
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
            "ce_sum": 0.0,
        }
        for loop in range(1, loops + 1)
    }
    case_rows: list[dict[str, Any]] = []
    size = batch.token_type.shape[0]
    for start in range(0, size, eval_batch):
        stop = min(start + eval_batch, size)
        cpu_slice = RetrievalBatch(
            *(tensor[start:stop] for tensor in (
                batch.token_type,
                batch.key_id,
                batch.value_id,
                batch.labels,
                batch.query_kind,
            ))
        )
        gpu_batch = cpu_slice.to(device)
        with forward_autocast(forward_dtype, device):
            logits_by_loop, _diagnostics = model.forward_trace(gpu_batch, loops=loops)
        query_mask = gpu_batch.labels >= 0
        past_mask = gpu_batch.query_kind == PAST_QUERY
        future_mask = gpu_batch.query_kind == FUTURE_QUERY
        case_payload = [
            {
                "case_index": start + idx,
                "token_type": cpu_slice.token_type[idx].tolist(),
                "key_id": cpu_slice.key_id[idx].tolist(),
                "value_id": cpu_slice.value_id[idx].tolist(),
                "labels": cpu_slice.labels[idx].tolist(),
                "query_kind": cpu_slice.query_kind[idx].tolist(),
                "predictions": {},
            }
            for idx in range(stop - start)
        ]
        for loop_idx, logits in enumerate(logits_by_loop, start=1):
            pred = logits.argmax(dim=-1)
            correct = pred == gpu_batch.labels
            selected_logits = logits.float()[query_mask]
            selected_labels = gpu_batch.labels[query_mask]
            ce_sum = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
            row = totals[loop_idx]
            row["correct"] += float(correct[query_mask].sum())
            row["count"] += float(query_mask.sum())
            row["past_correct"] += float(correct[past_mask].sum())
            row["past_count"] += float(past_mask.sum())
            row["future_correct"] += float(correct[future_mask].sum())
            row["future_count"] += float(future_mask.sum())
            row["exact"] += float((correct | ~query_mask).all(dim=1).sum())
            row["past_exact"] += float((correct | ~past_mask).all(dim=1).sum())
            row["future_exact"] += float((correct | ~future_mask).all(dim=1).sum())
            row["cases"] += float(stop - start)
            row["ce_sum"] += float(ce_sum)
            pred_cpu = pred.detach().cpu()
            for idx, payload in enumerate(case_payload):
                payload["predictions"][f"loop{loop_idx}"] = pred_cpu[idx].tolist()
        case_rows.extend(case_payload)

    metrics: dict[str, dict[str, float]] = {}
    for loop_idx, row in totals.items():
        metrics[f"loop{loop_idx}"] = {
            "query_acc": row["correct"] / row["count"],
            "past_acc": row["past_correct"] / row["past_count"],
            "future_acc": row["future_correct"] / row["future_count"],
            "query_exact": row["exact"] / row["cases"],
            "past_exact": row["past_exact"] / row["cases"],
            "future_exact": row["future_exact"] / row["cases"],
            "query_ce": row["ce_sum"] / row["count"],
        }
    return metrics, case_rows


def select_hard_cases(cases: list[dict[str, Any]], *, loops: int, limit: int) -> list[dict[str, Any]]:
    def future_errors(case: dict[str, Any]) -> int:
        pred = case["predictions"][f"loop{loops}"]
        return sum(
            1
            for idx, kind in enumerate(case["query_kind"])
            if kind == FUTURE_QUERY and pred[idx] != case["labels"][idx]
        )

    return sorted(cases, key=lambda case: (-future_errors(case), case["case_index"]))[:limit]


def render_visualization(cases: list[dict[str, Any]], *, loops: int, path: Path, title: str) -> None:
    visual_loops = sorted({1, 2, loops})
    sections: list[str] = []
    for case in cases:
        events = []
        for pos, token_type in enumerate(case["token_type"]):
            if token_type == FILLER:
                continue
            key = case["key_id"][pos]
            if token_type == WRITE:
                events.append(f"<tr><td>{pos}</td><td>WRITE</td><td>K{key}</td><td>V{case['value_id'][pos]-1}</td>" + "<td></td>" * len(visual_loops) + "</tr>")
                continue
            kind = "FUTURE" if case["query_kind"][pos] == FUTURE_QUERY else "PAST"
            target = case["labels"][pos]
            predictions = []
            for loop in visual_loops:
                pred = case["predictions"][f"loop{loop}"][pos]
                cls = "ok" if pred == target else "bad"
                predictions.append(f'<td class="{cls}">V{pred}</td>')
            events.append(
                f"<tr><td>{pos}</td><td>QUERY {kind}</td><td>K{key}</td><td>target V{target}</td>{''.join(predictions)}</tr>"
            )
        loop_headers = "".join(f"<th>Loop {loop}</th>" for loop in visual_loops)
        sections.append(
            f"<section><h2>Case {case['case_index']}</h2><table><thead><tr><th>Pos</th><th>Event</th><th>Key</th><th>Value</th>{loop_headers}</tr></thead><tbody>{''.join(events)}</tbody></table></section>"
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
</style></head><body><header><h1>{html.escape(title)}</h1><p>PAST queries follow their write; FUTURE queries precede it. Green is correct and red is wrong.</p></header><main>{''.join(sections)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition", default="")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-n", type=int, default=1024)
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--key-vocab", type=int, default=64)
    parser.add_argument("--value-vocab", type=int, default=32)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--viz-cases", type=int, default=8)
    parser.add_argument("--carrier-kill-step", type=int, default=200)
    parser.add_argument("--carrier-min-past-acc", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This probe requires exactly one visible CUDA GPU")
    if args.d_model != args.heads * args.head_dim:
        raise ValueError("d_model must equal heads * head_dim")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    model = FutureRetrievalModel(
        seq_len=args.seq_len,
        key_vocab=args.key_vocab,
        value_vocab=args.value_vocab,
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
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    eval_generator = torch.Generator().manual_seed(args.seed + 10_000)
    eval_batch = generate_batch(
        batch_size=args.eval_n,
        seq_len=args.seq_len,
        pairs=args.pairs,
        key_vocab=args.key_vocab,
        value_vocab=args.value_vocab,
        generator=eval_generator,
    )
    eval_hash = batch_sha256(eval_batch)
    history: list[dict[str, Any]] = []
    aborted = False
    abort: dict[str, Any] | None = None
    t0 = time.time()
    last_cases: list[dict[str, Any]] = []
    last_metrics: dict[str, dict[str, float]] = {}

    for step in range(1, args.steps + 1):
        model.train()
        batch = generate_batch(
            batch_size=args.batch,
            seq_len=args.seq_len,
            pairs=args.pairs,
            key_vocab=args.key_vocab,
            value_vocab=args.value_vocab,
            generator=train_generator,
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with forward_autocast(args.forward_dtype, device):
            logits_by_loop, _diagnostics = model.forward_trace(batch, loops=args.train_loops)
            loss = query_loss(logits_by_loop, batch.labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

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
                "elapsed_sec": time.time() - t0,
                "loop1": loop1,
                "final": final,
                "future_loop_gain": final["future_acc"] - loop1["future_acc"],
                "past_loop_gain": final["past_acc"] - loop1["past_acc"],
            }
            history.append(row)
            print(
                f"[future_retrieval] step={step:04d} loss={row['loss']:.4f} "
                f"past={loop1['past_acc']:.4f}->{final['past_acc']:.4f} "
                f"future={loop1['future_acc']:.4f}->{final['future_acc']:.4f} "
                f"future_exact={loop1['future_exact']:.4f}->{final['future_exact']:.4f} "
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
                    "reason": "past_query_carrier_failed",
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
        title=f"GDN2 associative retrieval: {args.condition or args.run_name}",
    )
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "task": "balanced_past_future_associative_retrieval",
        "config": vars(args) | {"out_dir": str(args.out_dir)},
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
        "peak_cuda_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "aborted": aborted,
        "abort": abort,
        "decision": "invalid_carrier" if aborted else "completed",
    }
    result_path = args.out_dir / "gdn2_future_retrieval_probe.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    run_dir = args.out_dir.parent
    (run_dir / "score.json").write_text(
        json.dumps(
            {
                "score": payload["score"],
                "score_key": payload["score_key"],
                "git_sha": run_dir.joinpath("source_HEAD.txt").read_text().strip(),
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
