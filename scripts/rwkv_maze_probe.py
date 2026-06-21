#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PAD_ID = 0
WALL_ID = 1
OPEN_ID = 2
START_ID = 3
GOAL_ID = 4
PATH_ID = 5
VOCAB_SIZE = 6
TOKEN_CLASS = {
    PAD_ID: "pad",
    WALL_ID: "wall",
    OPEN_ID: "open",
    START_ID: "start",
    GOAL_ID: "goal",
    PATH_ID: "path",
}
TOKEN_GLYPH = {
    PAD_ID: "",
    WALL_ID: "",
    OPEN_ID: "",
    START_ID: "S",
    GOAL_ID: "G",
    PATH_ID: ".",
}


def import_rwkv(repo_root: Path):
    sys.path.insert(0, str(repo_root / "experiments" / "rwkv_fs_sudoku"))
    from study_rwkv_futureseed_loop import FutureSeedRWKV, forward_autocast, statepassing_available

    return FutureSeedRWKV, forward_autocast, statepassing_available


class FutureSeedLoopMaze(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        l_cycles: int,
        lambda_: float,
        future_seed_scale: float,
        future_seed_decay: float,
        future_seed_update: str,
        activation_checkpoint: bool,
        rwkv_kernel: str,
        rwkv_cls: type[nn.Module],
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.l_cycles = int(l_cycles)
        self.lambda_ = float(lambda_)
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.position = nn.Embedding(self.seq_len, d_model)
        self.reasoner = rwkv_cls(
            d_model,
            layers,
            heads,
            head_dim,
            channel_mult,
            future_seed_scale=future_seed_scale,
            future_seed_decay=future_seed_decay,
            future_seed_update=future_seed_update,
            activation_checkpoint=activation_checkpoint,
            rwkv_kernel=rwkv_kernel,
        )
        self.h_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.l_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)

    def input_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.seq_len, dtype=torch.long, device=inputs.device)
        return self.embed(inputs) + self.position(positions).unsqueeze(0)

    def _depth_update(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        seed_memory: List[torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor] | None]:
        updated, diag, next_seed_memory = self.reasoner(hidden + injection, seed_memory=seed_memory)
        return hidden + self.lambda_ * (updated - hidden), diag, next_seed_memory

    def forward_trace(self, inputs: torch.Tensor, *, loops: int) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        x = self.input_sequence(inputs)
        batch_size, seq_len, _channels = x.shape
        z_h = self.h_init.expand(batch_size, seq_len, -1)
        z_l = self.l_init.expand(batch_size, seq_len, -1)
        h_seed_memory: List[torch.Tensor] | None = None
        l_seed_memory: List[torch.Tensor] | None = None
        logits_by_loop: List[torch.Tensor] = []
        traces: List[Dict[str, torch.Tensor]] = []
        zero = x.new_zeros(())
        for _loop_idx in range(int(loops)):
            l_gates: List[torch.Tensor] = []
            for _ in range(self.l_cycles):
                z_l, l_diag, l_seed_memory = self._depth_update(z_l, z_h + x, l_seed_memory)
                if "future_seed_gate" in l_diag:
                    l_gates.append(l_diag["future_seed_gate"])
            z_h, h_diag, h_seed_memory = self._depth_update(z_h, z_l, h_seed_memory)
            logits_by_loop.append(self.head(self.out_norm(z_h)))
            trace = dict(h_diag)
            if l_gates:
                trace["future_seed_gate_l"] = torch.stack(l_gates).mean()
            else:
                trace["future_seed_gate_l"] = zero
            traces.append(trace)
        return logits_by_loop, traces


@dataclass
class PathMetrics:
    token_acc: float
    exact: float
    path_precision: float
    path_recall: float
    path_f1: float
    pred_path_frac: float
    true_path_frac: float
    path_tp: float
    path_fp: float
    path_fn: float


def load_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    inputs = np.load(data_dir / split / "all__inputs.npy").astype(np.int64)
    labels = np.load(data_dir / split / "all__labels.npy").astype(np.int64)
    if inputs.shape != labels.shape:
        raise ValueError(f"inputs/labels shape mismatch for {split}: {inputs.shape} vs {labels.shape}")
    return inputs, labels


def sample_batch(inputs: np.ndarray, labels: np.ndarray, batch: int, rng: np.random.Generator, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = rng.integers(0, inputs.shape[0], size=int(batch))
    x = torch.as_tensor(inputs[idx], dtype=torch.long, device=device)
    y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
    return x, y


def weighted_loop_loss(logits_by_loop: List[torch.Tensor], labels: torch.Tensor, path_weight: float, loop_loss: str) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    label_flat = labels.reshape(-1)
    weights = torch.where(
        label_flat == PATH_ID,
        torch.full_like(label_flat, float(path_weight), dtype=torch.float32),
        torch.ones_like(label_flat, dtype=torch.float32),
    )
    denom = weights.sum().clamp_min(1.0)
    selected = logits_by_loop if loop_loss == "all" else [logits_by_loop[-1]]
    for logits in selected:
        loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), label_flat, reduction="none")
        losses.append((loss * weights).sum() / denom)
    return torch.stack(losses).mean()


def metrics_from_pred(pred: torch.Tensor, labels: torch.Tensor) -> PathMetrics:
    valid = labels != PAD_ID
    token_acc = (((pred == labels) & valid).sum(dim=1).float() / valid.sum(dim=1).clamp_min(1).float()).mean()
    exact = (((pred == labels) | ~valid).all(dim=1)).float().mean()
    true_path = (labels == PATH_ID) & valid
    pred_path = (pred == PATH_ID) & valid
    tp = (true_path & pred_path).sum(dim=1).float()
    fp = (~true_path & pred_path & valid).sum(dim=1).float()
    fn = (true_path & ~pred_path).sum(dim=1).float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    pred_frac = pred_path.float().sum(dim=1) / valid.sum(dim=1).clamp_min(1).float()
    true_frac = true_path.float().sum(dim=1) / valid.sum(dim=1).clamp_min(1).float()
    return PathMetrics(
        token_acc=float(token_acc.mean().detach().cpu()),
        exact=float(exact.mean().detach().cpu()),
        path_precision=float(precision.mean().detach().cpu()),
        path_recall=float(recall.mean().detach().cpu()),
        path_f1=float(f1.mean().detach().cpu()),
        pred_path_frac=float(pred_frac.mean().detach().cpu()),
        true_path_frac=float(true_frac.mean().detach().cpu()),
        path_tp=float(tp.mean().detach().cpu()),
        path_fp=float(fp.mean().detach().cpu()),
        path_fn=float(fn.mean().detach().cpu()),
    )


@torch.no_grad()
def evaluate(
    model: FutureSeedLoopMaze,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    eval_n: int,
    batch: int,
    loops: int,
    device: torch.device,
    forward_dtype: str,
) -> Dict[str, PathMetrics]:
    model.eval()
    n = min(int(eval_n), inputs.shape[0])
    all_preds: Dict[int, List[torch.Tensor]] = {idx: [] for idx in range(1, loops + 1)}
    all_labels: List[torch.Tensor] = []
    for start in range(0, n, batch):
        end = min(start + batch, n)
        x = torch.as_tensor(inputs[start:end], dtype=torch.long, device=device)
        y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
        with forward_autocast(forward_dtype, device):
            logits_by_loop, _trace = model.forward_trace(x, loops=loops)
        for loop_idx, logits in enumerate(logits_by_loop, start=1):
            all_preds[loop_idx].append(logits.argmax(dim=-1).detach().cpu())
        all_labels.append(y.detach().cpu())
    labels_cat = torch.cat(all_labels, dim=0)
    metrics: Dict[str, PathMetrics] = {}
    for loop_idx, chunks in all_preds.items():
        pred_cat = torch.cat(chunks, dim=0)
        metrics[f"loop{loop_idx}"] = metrics_from_pred(pred_cat, labels_cat)
    return metrics


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def render_board(tokens: Iterable[int], labels: Iterable[int] | None, title: str) -> str:
    vals = [int(v) for v in tokens]
    labs = [int(v) for v in labels] if labels is not None else None
    n = int(math.sqrt(len(vals)))
    rows = [f"<h3>{html.escape(title)}</h3>", "<table class='maze'>"]
    for y in range(n):
        rows.append("<tr>")
        for x in range(n):
            idx = y * n + x
            token = vals[idx]
            cls = TOKEN_CLASS.get(token, "unk")
            extra = ""
            if labs is not None:
                target_path = labs[idx] == PATH_ID
                pred_path = token == PATH_ID
                if pred_path and target_path:
                    extra = " tp"
                elif pred_path and not target_path:
                    extra = " fp"
                elif target_path and not pred_path:
                    extra = " fn"
            rows.append(f"<td class='{cls}{extra}'>{TOKEN_GLYPH.get(token, '')}</td>")
        rows.append("</tr>")
    rows.append("</table>")
    return "\n".join(rows)


@torch.no_grad()
def write_visuals(
    model: FutureSeedLoopMaze,
    inputs: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    loops: int,
    cases: int,
    device: torch.device,
    forward_dtype: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(cases), inputs.shape[0])
    x = torch.as_tensor(inputs[:n], dtype=torch.long, device=device)
    y = torch.as_tensor(labels[:n], dtype=torch.long, device=device)
    model.eval()
    with forward_autocast(forward_dtype, device):
        logits_by_loop, _trace = model.forward_trace(x, loops=loops)
    pred1 = logits_by_loop[0].argmax(dim=-1).detach().cpu()
    pred_last = logits_by_loop[-1].argmax(dim=-1).detach().cpu()
    labs = y.detach().cpu()
    m1 = [asdict(metrics_from_pred(pred1[i : i + 1], labs[i : i + 1])) for i in range(n)]
    ml = [asdict(metrics_from_pred(pred_last[i : i + 1], labs[i : i + 1])) for i in range(n)]
    order = sorted(range(n), key=lambda i: (ml[i]["path_fp"] + ml[i]["path_fn"], 1.0 - ml[i]["path_f1"]), reverse=True)
    chosen = order[: min(8, n)]
    case_payload = []
    html_cases = []
    for idx in chosen:
        case_payload.append(
            {
                "case_index": int(idx),
                "loop1": m1[idx],
                f"loop{loops}": ml[idx],
                "input": x[idx].detach().cpu().tolist(),
                "label": labs[idx].tolist(),
                "pred_loop1": pred1[idx].tolist(),
                f"pred_loop{loops}": pred_last[idx].tolist(),
            }
        )
        html_cases.append("<section class='case'>")
        html_cases.append(
            f"<h2>Case {idx}: loop1 F1 {m1[idx]['path_f1']:.3f} -> loop{loops} F1 {ml[idx]['path_f1']:.3f}, "
            f"FP {m1[idx]['path_fp']:.0f}->{ml[idx]['path_fp']:.0f}, FN {m1[idx]['path_fn']:.0f}->{ml[idx]['path_fn']:.0f}</h2>"
        )
        html_cases.append("<div class='boards'>")
        html_cases.append(render_board(x[idx].detach().cpu().tolist(), None, "Input"))
        html_cases.append(render_board(labs[idx].tolist(), None, "Target"))
        html_cases.append(render_board(pred1[idx].tolist(), labs[idx].tolist(), "Loop 1"))
        html_cases.append(render_board(pred_last[idx].tolist(), labs[idx].tolist(), f"Loop {loops}"))
        html_cases.append("</div></section>")
    (out_dir / "cases.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#202124}
main{max-width:1500px;margin:0 auto;padding:24px}.case{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px;margin:16px 0}
.boards{display:grid;grid-template-columns:repeat(4,max-content);gap:14px;align-items:start}
table.maze{border-collapse:collapse}.maze td{width:7px;height:7px;min-width:7px;max-width:7px;padding:0;text-align:center;font-size:5px;line-height:7px}
.wall{background:#111827}.open{background:#f8fafc}.start{background:#16a34a;color:#fff}.goal{background:#dc2626;color:#fff}.path{background:#2563eb;color:#fff}
.tp{box-shadow:inset 0 0 0 1px #22c55e}.fp{background:#f97316!important;color:#111}.fn{background:#fee2e2!important;box-shadow:inset 0 0 0 1px #dc2626}
h1{font-size:24px}h2{font-size:15px}h3{font-size:13px;margin:8px 0}
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><main><h1>RWKV Maze hard cases</h1>{''.join(html_cases)}</main></body></html>"
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--condition", default="")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-n", type=int, default=512)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--channel-mult", type=int, default=4)
    p.add_argument("--l-cycles", type=int, default=2)
    p.add_argument("--train-loops", type=int, default=4)
    p.add_argument("--eval-loops", type=int, default=8)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    p.add_argument("--future-seed-scale", type=float, default=1.0)
    p.add_argument("--future-seed-decay", type=float, default=0.0)
    p.add_argument("--future-seed-update", choices=("fixed", "learned", "loop_residual"), default="fixed")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--rwkv-kernel", choices=("auto", "torch", "cuda", "statepassing", "wind"), default="statepassing")
    p.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--path-weight", type=float, default=8.0)
    p.add_argument("--loop-loss", choices=("final", "all"), default="all")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--viz-cases", type=int, default=64)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("rwkv_maze_probe is CUDA-only; CPU smoke/training is intentionally disabled")
    device = torch.device("cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    FutureSeedRWKV, forward_autocast, statepassing_available = import_rwkv(args.repo_root.resolve())
    globals()["forward_autocast"] = forward_autocast
    if args.rwkv_kernel in {"cuda", "statepassing"}:
        ok, reason = statepassing_available(args.head_dim)
        if not ok:
            raise RuntimeError(f"RWKV statepassing unavailable: {reason}")

    train_x, train_y = load_split(args.data_dir, "train")
    test_x, test_y = load_split(args.data_dir, "test")
    seq_len = int(train_x.shape[1])
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed + 17)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = FutureSeedLoopMaze(
        seq_len=seq_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        channel_mult=args.channel_mult,
        l_cycles=args.l_cycles,
        lambda_=args.lambda_,
        future_seed_scale=args.future_seed_scale,
        future_seed_decay=args.future_seed_decay,
        future_seed_update=args.future_seed_update,
        activation_checkpoint=args.activation_checkpoint,
        rwkv_kernel=args.rwkv_kernel,
        rwkv_cls=FutureSeedRWKV,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        xb, yb = sample_batch(train_x, train_y, args.batch, np_rng, device)
        opt.zero_grad(set_to_none=True)
        with forward_autocast(args.forward_dtype, device):
            logits_by_loop, _trace = model.forward_trace(xb, loops=args.train_loops)
            loss = weighted_loop_loss(logits_by_loop, yb, args.path_weight, args.loop_loss)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            eval_metrics = evaluate(
                model,
                test_x,
                test_y,
                eval_n=args.eval_n,
                batch=args.eval_batch,
                loops=args.eval_loops,
                device=device,
                forward_dtype=args.forward_dtype,
            )
            loop1 = eval_metrics["loop1"]
            final = eval_metrics[f"loop{args.eval_loops}"]
            row = {
                "step": step,
                "elapsed_sec": time.time() - t0,
                "loss": float(loss.detach().cpu()),
                "loop1_path_f1": loop1.path_f1,
                "loop_last_path_f1": final.path_f1,
                "loop_gain": final.path_f1 - loop1.path_f1,
                "loop_last_precision": final.path_precision,
                "loop_last_recall": final.path_recall,
                "loop_last_pred_path_frac": final.pred_path_frac,
                "loop_last_fp": final.path_fp,
                "loop_last_fn": final.path_fn,
            }
            history.append(row)
            print(
                "[rwkv_maze] "
                f"step={step:04d} loss={row['loss']:.4f} "
                f"loop1={row['loop1_path_f1']:.4f} loop{args.eval_loops}={row['loop_last_path_f1']:.4f} "
                f"gain={row['loop_gain']:+.4f} pred={row['loop_last_pred_path_frac']:.4f} "
                f"fp={row['loop_last_fp']:.1f} fn={row['loop_last_fn']:.1f}",
                flush=True,
            )
    final_metrics = evaluate(
        model,
        test_x,
        test_y,
        eval_n=args.eval_n,
        batch=args.eval_batch,
        loops=args.eval_loops,
        device=device,
        forward_dtype=args.forward_dtype,
    )
    write_visuals(
        model,
        test_x,
        test_y,
        args.out_dir / "visualizations",
        loops=args.eval_loops,
        cases=args.viz_cases,
        device=device,
        forward_dtype=args.forward_dtype,
    )
    final = final_metrics[f"loop{args.eval_loops}"]
    loop1 = final_metrics["loop1"]
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "task": "official_maze30_rwkv_path",
        "data_dir": args.data_dir,
        "config": vars(args),
        "train_history": history,
        "final_metrics": {key: asdict(value) for key, value in final_metrics.items()},
        "score": final.path_f1,
        "score_key": f"final_metrics.loop{args.eval_loops}.path_f1",
        "loop_gain": final.path_f1 - loop1.path_f1,
        "peak_cuda_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "elapsed_sec": time.time() - t0,
        "decision": "FutureSeed-positive" if args.future_seed_scale > 0 and final.path_f1 >= 0.03 else "record-only",
    }
    (args.out_dir / "rwkv_maze_probe.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    readme = [
        f"# {args.run_name}",
        "",
        "Official Maze30 path recovery with a causal RWKV7 state-passing backbone.",
        "",
        f"- condition: `{args.condition}`",
        f"- future_seed_scale: `{args.future_seed_scale}`",
        f"- loop1 path F1: `{loop1.path_f1:.4f}`",
        f"- loop{args.eval_loops} path F1: `{final.path_f1:.4f}`",
        f"- loop gain: `{final.path_f1 - loop1.path_f1:+.4f}`",
        f"- precision/recall: `{final.path_precision:.4f}` / `{final.path_recall:.4f}`",
        f"- pred PATH frac: `{final.pred_path_frac:.4f}`",
        f"- FP/FN per case: `{final.path_fp:.1f}` / `{final.path_fn:.1f}`",
        "",
        "No selector, search, repair, or maze-specific postprocessing is used.",
        "",
        "- `rwkv_maze_probe.json`: metrics and config",
        "- `visualizations/index.html`: hard-case loop visualization",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({"run_name": args.run_name, "score": final.path_f1, "loop_gain": final.path_f1 - loop1.path_f1}, indent=2))


if __name__ == "__main__":
    main()
