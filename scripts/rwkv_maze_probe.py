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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        feedback_mode: str,
        feedback_scale: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.l_cycles = int(l_cycles)
        self.lambda_ = float(lambda_)
        self.feedback_mode = feedback_mode
        self.feedback_scale = float(feedback_scale)
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.position = nn.Embedding(self.seq_len, d_model)
        self.feedback_proj = nn.Linear(VOCAB_SIZE, d_model, bias=False)
        self.null_feedback = nn.Parameter(torch.zeros(1, 1, d_model))
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
        self.path_budget_head = nn.Linear(d_model, 1)

    def input_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.seq_len, dtype=torch.long, device=inputs.device)
        return self.embed(inputs) + self.position(positions).unsqueeze(0)

    def _feedback_from_probs(self, probs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.feedback_proj(probs.to(dtype=dtype)) * self.feedback_scale

    def _depth_update(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        seed_memory: List[torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor] | None]:
        updated, diag, next_seed_memory = self.reasoner(hidden + injection, seed_memory=seed_memory)
        return hidden + self.lambda_ * (updated - hidden), diag, next_seed_memory

    def forward_trace(
        self,
        inputs: torch.Tensor,
        *,
        loops: int,
        initial_feedback_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        base_x = self.input_sequence(inputs)
        batch_size, seq_len, _channels = base_x.shape
        z_h = self.h_init.expand(batch_size, seq_len, -1)
        z_l = self.l_init.expand(batch_size, seq_len, -1)
        feedback = base_x.new_zeros(batch_size, seq_len, base_x.shape[-1])
        if self.feedback_mode != "none":
            if initial_feedback_probs is None:
                feedback = self.null_feedback.expand(batch_size, seq_len, -1) * self.feedback_scale
            else:
                feedback = self._feedback_from_probs(initial_feedback_probs, base_x.dtype)
        h_seed_memory: List[torch.Tensor] | None = None
        l_seed_memory: List[torch.Tensor] | None = None
        logits_by_loop: List[torch.Tensor] = []
        traces: List[Dict[str, torch.Tensor]] = []
        zero = base_x.new_zeros(())
        for _loop_idx in range(int(loops)):
            x = base_x + feedback
            l_gates: List[torch.Tensor] = []
            for _ in range(self.l_cycles):
                z_l, l_diag, l_seed_memory = self._depth_update(z_l, z_h + x, l_seed_memory)
                if "future_seed_gate" in l_diag:
                    l_gates.append(l_diag["future_seed_gate"])
            z_h, h_diag, h_seed_memory = self._depth_update(z_h, z_l, h_seed_memory)
            normed_h = self.out_norm(z_h)
            logits_by_loop.append(self.head(normed_h))
            trace = dict(h_diag)
            trace["path_budget_logit"] = self.path_budget_head(normed_h.mean(dim=1)).squeeze(-1)
            if l_gates:
                trace["future_seed_gate_l"] = torch.stack(l_gates).mean()
            else:
                trace["future_seed_gate_l"] = zero
            traces.append(trace)
            if self.feedback_mode != "none":
                feedback_probs = logits_by_loop[-1].float().softmax(dim=-1).detach()
                feedback = self._feedback_from_probs(feedback_probs, base_x.dtype)
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


def path_margin_objective_loss(
    logits_by_loop: List[torch.Tensor],
    labels: torch.Tensor,
    loop_loss: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Generic foreground decision-boundary loss for sparse PATH labels.

    PATH is treated as a foreground class against the strongest non-PATH class.
    This keeps the objective tied to the hard argmax boundary instead of only
    increasing PATH softmax confidence.
    """
    selected = logits_by_loop if loop_loss == "all" else [logits_by_loop[-1]]
    valid = labels != PAD_ID
    true_path = (labels == PATH_ID) & valid
    non_path = (~true_path) & valid
    valid_f = valid.to(torch.float32)
    true_f = true_path.to(torch.float32)
    pos_count = true_f.sum().clamp_min(1.0)
    neg_count = non_path.to(torch.float32).sum().clamp_min(1.0)
    valid_count_per_case = valid_f.sum(dim=1).clamp_min(1.0)
    true_frac = true_f.sum(dim=1) / valid_count_per_case

    binary_losses: List[torch.Tensor] = []
    budget_losses: List[torch.Tensor] = []
    pos_margins: List[torch.Tensor] = []
    neg_margins: List[torch.Tensor] = []
    prob_fracs: List[torch.Tensor] = []
    for logits in selected:
        logits_f = logits.float()
        non_path_logits = torch.cat([logits_f[..., :PATH_ID], logits_f[..., PATH_ID + 1 :]], dim=-1)
        margin = logits_f[..., PATH_ID] - non_path_logits.max(dim=-1).values
        bce = F.binary_cross_entropy_with_logits(margin, true_f, reduction="none")
        pos_loss = (bce * true_f).sum() / pos_count
        neg_loss = (bce * non_path.to(torch.float32)).sum() / neg_count
        binary_losses.append(0.5 * (pos_loss + neg_loss))
        path_prob = torch.sigmoid(margin) * valid_f
        prob_frac = path_prob.sum(dim=1) / valid_count_per_case
        budget_losses.append((prob_frac - true_frac).abs().mean())
        if true_path.any():
            pos_margins.append(margin[true_path].mean())
        if non_path.any():
            neg_margins.append(margin[non_path].mean())
        prob_fracs.append(prob_frac.mean())

    zero = labels.new_zeros((), dtype=torch.float32)
    diag = {
        "path_margin_pos": torch.stack(pos_margins).mean() if pos_margins else zero,
        "path_margin_neg": torch.stack(neg_margins).mean() if neg_margins else zero,
        "path_prob_frac": torch.stack(prob_fracs).mean() if prob_fracs else zero,
        "path_true_frac": true_frac.mean(),
    }
    return torch.stack(binary_losses).mean(), torch.stack(budget_losses).mean(), diag


def path_count_loss(
    traces: List[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    loop_loss: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    selected = traces if loop_loss == "all" else [traces[-1]]
    valid = labels != PAD_ID
    true_path = (labels == PATH_ID) & valid
    valid_count = valid.to(torch.float32).sum(dim=1).clamp_min(1.0)
    target_frac = true_path.to(torch.float32).sum(dim=1) / valid_count
    losses: List[torch.Tensor] = []
    pred_fracs: List[torch.Tensor] = []
    abs_errs: List[torch.Tensor] = []
    for trace in selected:
        logits = trace["path_budget_logit"].float()
        loss = F.binary_cross_entropy_with_logits(logits, target_frac, reduction="mean")
        pred_frac = torch.sigmoid(logits)
        losses.append(loss)
        pred_fracs.append(pred_frac.mean())
        abs_errs.append((pred_frac - target_frac).abs().mean())
    zero = labels.new_zeros((), dtype=torch.float32)
    diag = {
        "path_count_pred_frac": torch.stack(pred_fracs).mean() if pred_fracs else zero,
        "path_count_true_frac": target_frac.mean(),
        "path_count_abs_err": torch.stack(abs_errs).mean() if abs_errs else zero,
    }
    return torch.stack(losses).mean(), diag


def token_probs(tokens: torch.Tensor, *, smooth: float) -> torch.Tensor:
    probs = F.one_hot(tokens.clamp(0, VOCAB_SIZE - 1), num_classes=VOCAB_SIZE).to(torch.float32)
    if smooth > 0:
        probs = probs * (1.0 - float(smooth)) + float(smooth) / VOCAB_SIZE
    return probs


def make_corrupted_token_probs(
    labels: torch.Tensor,
    *,
    logits: Optional[torch.Tensor],
    token_corrupt_prob: float,
    add_path_prob: float,
    delete_path_prob: float,
    model_error_prob: float,
    smooth: float,
) -> torch.Tensor:
    """Build a generic noisy output-state distribution for attractor training.

    This corrupts token states only. It does not inspect maze topology, compute
    paths, or apply any repair/search rule.
    """
    corrupt = labels.clone()
    valid = labels != PAD_ID
    if token_corrupt_prob > 0:
        random_tokens = torch.randint(1, VOCAB_SIZE, labels.shape, device=labels.device)
        mask = (torch.rand(labels.shape, device=labels.device) < float(token_corrupt_prob)) & valid
        corrupt = torch.where(mask, random_tokens, corrupt)
    if add_path_prob > 0:
        mask = (torch.rand(labels.shape, device=labels.device) < float(add_path_prob)) & valid & (labels != PATH_ID)
        corrupt = torch.where(mask, torch.full_like(corrupt, PATH_ID), corrupt)
    if delete_path_prob > 0:
        random_non_path = torch.randint(1, max(PATH_ID, 2), labels.shape, device=labels.device)
        mask = (torch.rand(labels.shape, device=labels.device) < float(delete_path_prob)) & valid & (labels == PATH_ID)
        corrupt = torch.where(mask, random_non_path, corrupt)
    if model_error_prob > 0 and logits is not None:
        pred = logits.detach().argmax(dim=-1)
        wrong = (pred != labels) & valid
        mask = (torch.rand(labels.shape, device=labels.device) < float(model_error_prob)) & wrong
        corrupt = torch.where(mask, pred, corrupt)
    return token_probs(corrupt, smooth=smooth)


def weighted_ce_per_case(logits: torch.Tensor, labels: torch.Tensor, path_weight: float) -> torch.Tensor:
    batch_size = labels.shape[0]
    per_token = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), labels.reshape(-1), reduction="none").view(
        batch_size, -1
    )
    weights = torch.where(
        labels == PATH_ID,
        torch.full_like(labels, float(path_weight), dtype=torch.float32),
        torch.ones_like(labels, dtype=torch.float32),
    )
    return (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def weighted_ce_mean(logits: torch.Tensor, labels: torch.Tensor, path_weight: float) -> torch.Tensor:
    return weighted_ce_per_case(logits, labels, path_weight).mean()


def denoising_attractor_loss(
    model: FutureSeedLoopMaze,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    normal_logits: torch.Tensor,
    *,
    dat_loops: int,
    path_weight: float,
    token_corrupt_prob: float,
    add_path_prob: float,
    delete_path_prob: float,
    model_error_prob: float,
    smooth: float,
    stability_weight: float,
    improvement_weight: float,
    improvement_margin: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    corrupt_probs = make_corrupted_token_probs(
        labels,
        logits=normal_logits,
        token_corrupt_prob=token_corrupt_prob,
        add_path_prob=add_path_prob,
        delete_path_prob=delete_path_prob,
        model_error_prob=model_error_prob,
        smooth=smooth,
    )
    clean_probs = token_probs(labels, smooth=smooth)
    dat_inputs = torch.cat([inputs, inputs], dim=0)
    dat_labels = torch.cat([labels, labels], dim=0)
    init_probs = torch.cat([corrupt_probs, clean_probs], dim=0)
    dat_logits_by_loop, _traces = model.forward_trace(dat_inputs, loops=dat_loops, initial_feedback_probs=init_probs)
    batch_size = labels.shape[0]
    final_logits = dat_logits_by_loop[-1]
    denoise_ce = weighted_ce_mean(final_logits[:batch_size], labels, path_weight)
    stable_ce = weighted_ce_mean(final_logits[batch_size:], labels, path_weight)
    stable_probs = final_logits[batch_size:].float().softmax(dim=-1)
    stable_mse = F.mse_loss(stable_probs, clean_probs)
    if len(dat_logits_by_loop) > 1:
        first_risk = weighted_ce_per_case(dat_logits_by_loop[0], dat_labels, path_weight).detach()
        final_risk = weighted_ce_per_case(final_logits, dat_labels, path_weight)
        improve = F.relu(final_risk - first_risk + float(improvement_margin)).mean()
    else:
        improve = final_logits.new_zeros(())
    total = denoise_ce + float(stability_weight) * (stable_ce + stable_mse) + float(improvement_weight) * improve
    diag = {
        "dat_denoise_ce": denoise_ce,
        "dat_stable_ce": stable_ce,
        "dat_stable_mse": stable_mse,
        "dat_improve": improve,
        "dat_corrupt_path_frac": corrupt_probs[..., PATH_ID].mean(),
        "dat_clean_path_frac": clean_probs[..., PATH_ID].mean(),
    }
    return total, diag


def budget_decode_pred(logits: torch.Tensor, budget_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Decode PATH as a learned-size foreground set from generic class margins."""
    logits_f = logits.float()
    valid = labels != PAD_ID
    non_path_logits = torch.cat([logits_f[..., :PATH_ID], logits_f[..., PATH_ID + 1 :]], dim=-1)
    non_path_pred = non_path_logits.argmax(dim=-1)
    margin = logits_f[..., PATH_ID] - non_path_logits.max(dim=-1).values
    pred = non_path_pred.clone()
    valid_count = valid.sum(dim=1)
    budget_frac = torch.sigmoid(budget_logit.float()).clamp(0.0, 1.0)
    budget_count = torch.round(budget_frac * valid_count.to(torch.float32)).to(torch.long)
    budget_count = torch.minimum(torch.maximum(budget_count, torch.zeros_like(budget_count)), valid_count)
    masked_margin = margin.masked_fill(~valid, -1e9)
    for batch_idx in range(logits.shape[0]):
        k = int(budget_count[batch_idx].item())
        if k <= 0:
            continue
        top_idx = torch.topk(masked_margin[batch_idx], k=k, largest=True).indices
        pred[batch_idx, top_idx] = PATH_ID
    return pred


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
    budget_decoder: bool,
) -> Dict[str, PathMetrics]:
    model.eval()
    n = min(int(eval_n), inputs.shape[0])
    all_preds: Dict[int, List[torch.Tensor]] = {idx: [] for idx in range(1, loops + 1)}
    all_budget_preds: Dict[int, List[torch.Tensor]] = {idx: [] for idx in range(1, loops + 1)}
    all_labels: List[torch.Tensor] = []
    for start in range(0, n, batch):
        end = min(start + batch, n)
        x = torch.as_tensor(inputs[start:end], dtype=torch.long, device=device)
        y = torch.as_tensor(labels[start:end], dtype=torch.long, device=device)
        with forward_autocast(forward_dtype, device):
            logits_by_loop, traces = model.forward_trace(x, loops=loops)
        for loop_idx, (logits, trace) in enumerate(zip(logits_by_loop, traces), start=1):
            all_preds[loop_idx].append(logits.argmax(dim=-1).detach().cpu())
            if budget_decoder:
                budget_pred = budget_decode_pred(logits, trace["path_budget_logit"], y)
                all_budget_preds[loop_idx].append(budget_pred.detach().cpu())
        all_labels.append(y.detach().cpu())
    labels_cat = torch.cat(all_labels, dim=0)
    metrics: Dict[str, PathMetrics] = {}
    for loop_idx, chunks in all_preds.items():
        pred_cat = torch.cat(chunks, dim=0)
        metrics[f"loop{loop_idx}"] = metrics_from_pred(pred_cat, labels_cat)
    if budget_decoder:
        for loop_idx, chunks in all_budget_preds.items():
            pred_cat = torch.cat(chunks, dim=0)
            metrics[f"budget_loop{loop_idx}"] = metrics_from_pred(pred_cat, labels_cat)
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
    budget_decoder: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(cases), inputs.shape[0])
    x = torch.as_tensor(inputs[:n], dtype=torch.long, device=device)
    y = torch.as_tensor(labels[:n], dtype=torch.long, device=device)
    model.eval()
    with forward_autocast(forward_dtype, device):
        logits_by_loop, traces = model.forward_trace(x, loops=loops)
    visual_steps = sorted({step for step in (1, 4, 8, loops) if 1 <= step <= loops})
    preds_by_step = {step: logits_by_loop[step - 1].argmax(dim=-1).detach().cpu() for step in visual_steps}
    if budget_decoder:
        budget_preds_by_step = {
            step: budget_decode_pred(logits_by_loop[step - 1], traces[step - 1]["path_budget_logit"], y).detach().cpu()
            for step in visual_steps
        }
    else:
        budget_preds_by_step = {}
    labs = y.detach().cpu()
    metrics_by_step = {
        step: [asdict(metrics_from_pred(preds_by_step[step][i : i + 1], labs[i : i + 1])) for i in range(n)]
        for step in visual_steps
    }
    final_step = visual_steps[-1]
    if budget_decoder and budget_preds_by_step:
        budget_metrics_by_step = {
            step: [asdict(metrics_from_pred(budget_preds_by_step[step][i : i + 1], labs[i : i + 1])) for i in range(n)]
            for step in visual_steps
        }
        final_budget_metrics = budget_metrics_by_step[final_step]
        order = sorted(
            range(n),
            key=lambda i: (final_budget_metrics[i]["path_fp"] + final_budget_metrics[i]["path_fn"], 1.0 - final_budget_metrics[i]["path_f1"]),
            reverse=True,
        )
    else:
        budget_metrics_by_step = {}
        final_metrics = metrics_by_step[final_step]
        order = sorted(
            range(n),
            key=lambda i: (final_metrics[i]["path_fp"] + final_metrics[i]["path_fn"], 1.0 - final_metrics[i]["path_f1"]),
            reverse=True,
        )
    chosen = order[: min(8, n)]
    case_payload = []
    html_cases = []
    for idx in chosen:
        metrics_payload = {f"loop{step}": metrics_by_step[step][idx] for step in visual_steps}
        pred_payload = {f"pred_loop{step}": preds_by_step[step][idx].tolist() for step in visual_steps}
        budget_payload = {}
        if budget_decoder and budget_preds_by_step:
            budget_payload = {
                **{f"budget_loop{step}": budget_metrics_by_step[step][idx] for step in visual_steps},
                **{f"budget_pred_loop{step}": budget_preds_by_step[step][idx].tolist() for step in visual_steps},
            }
        case_payload.append(
            {
                "case_index": int(idx),
                "visual_steps": visual_steps,
                "input": x[idx].detach().cpu().tolist(),
                "label": labs[idx].tolist(),
                **metrics_payload,
                **pred_payload,
                **budget_payload,
            }
        )
        html_cases.append("<section class='case'>")
        first = metrics_by_step[visual_steps[0]][idx]
        last = metrics_by_step[final_step][idx]
        if budget_decoder and budget_metrics_by_step:
            budget_last = budget_metrics_by_step[final_step][idx]
            suffix = (
                f", budget loop{final_step} F1 {budget_last['path_f1']:.3f}, "
                f"budget FP/FN {budget_last['path_fp']:.0f}/{budget_last['path_fn']:.0f}"
            )
        else:
            suffix = ""
        html_cases.append(
            f"<h2>Case {idx}: loop{visual_steps[0]} F1 {first['path_f1']:.3f} -> loop{final_step} F1 {last['path_f1']:.3f}, "
            f"FP {first['path_fp']:.0f}->{last['path_fp']:.0f}, FN {first['path_fn']:.0f}->{last['path_fn']:.0f}{suffix}</h2>"
        )
        html_cases.append("<div class='boards'>")
        html_cases.append(render_board(x[idx].detach().cpu().tolist(), None, "Input"))
        html_cases.append(render_board(labs[idx].tolist(), None, "Target"))
        for step in visual_steps:
            m = metrics_by_step[step][idx]
            html_cases.append(
                render_board(
                    preds_by_step[step][idx].tolist(),
                    labs[idx].tolist(),
                    f"Loop {step} F1 {m['path_f1']:.2f} FP {m['path_fp']:.0f} FN {m['path_fn']:.0f}",
                )
            )
        if budget_decoder and budget_preds_by_step:
            budget_last = budget_metrics_by_step[final_step][idx]
            html_cases.append(
                render_board(
                    budget_preds_by_step[final_step][idx].tolist(),
                    labs[idx].tolist(),
                    f"Budget loop {final_step} F1 {budget_last['path_f1']:.2f}",
                )
            )
        html_cases.append("</div></section>")
    (out_dir / "cases.json").write_text(json.dumps(case_payload, indent=2), encoding="utf-8")
    css = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#202124}
main{max-width:1500px;margin:0 auto;padding:24px}.case{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px;margin:16px 0}
.boards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,max-content));gap:14px;align-items:start}
table.maze{border-collapse:collapse}.maze td{width:7px;height:7px;min-width:7px;max-width:7px;padding:0;text-align:center;font-size:5px;line-height:7px}
.wall{background:#111827}.open{background:#f8fafc}.start{background:#16a34a;color:#fff}.goal{background:#dc2626;color:#fff}.path{background:#2563eb;color:#fff}
.tp{box-shadow:inset 0 0 0 1px #22c55e}.fp{background:#f97316!important;color:#111}.fn{background:#fee2e2!important;box-shadow:inset 0 0 0 1px #dc2626}
h1{font-size:24px}h2{font-size:15px}h3{font-size:13px;margin:8px 0}
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><main><h1>RWKV Maze hard cases</h1>{''.join(html_cases)}</main></body></html>"
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")


def write_probe_summary(
    *,
    model: FutureSeedLoopMaze,
    inputs: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    history: List[Dict[str, Any]],
    final_metrics: Dict[str, PathMetrics],
    t0: float,
    device: torch.device,
    aborted: bool,
    abort_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    write_visuals(
        model,
        inputs,
        labels,
        args.out_dir / "visualizations",
        loops=args.eval_loops,
        cases=args.viz_cases,
        device=device,
        forward_dtype=args.forward_dtype,
        budget_decoder=args.budget_decoder,
    )
    final = final_metrics[f"loop{args.eval_loops}"]
    loop1 = final_metrics["loop1"]
    score_metric = final_metrics[f"budget_loop{args.eval_loops}"] if args.budget_decoder else final
    score_loop1 = final_metrics["budget_loop1"] if args.budget_decoder else loop1
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "task": "official_maze30_rwkv_path",
        "data_dir": args.data_dir,
        "config": vars(args),
        "train_history": history,
        "final_metrics": {key: asdict(value) for key, value in final_metrics.items()},
        "score": score_metric.path_f1,
        "score_key": f"final_metrics.{'budget_' if args.budget_decoder else ''}loop{args.eval_loops}.path_f1",
        "loop_gain": score_metric.path_f1 - score_loop1.path_f1,
        "peak_cuda_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "elapsed_sec": time.time() - t0,
        "aborted": bool(aborted),
        "abort": abort_info or None,
        "decision": "aborted" if aborted else (
            "FutureSeed-positive" if args.future_seed_scale > 0 and score_metric.path_f1 >= 0.03 else "record-only"
        ),
    }
    (args.out_dir / "rwkv_maze_probe.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    if aborted and abort_info is not None:
        (args.out_dir / "abort.json").write_text(json.dumps(jsonable(abort_info), indent=2), encoding="utf-8")

    readme = [
        f"# {args.run_name}",
        "",
        "Official Maze30 path recovery with a causal RWKV7 state-passing backbone.",
        "",
        f"- status: `{'aborted' if aborted else 'completed'}`",
        f"- condition: `{args.condition}`",
        f"- future_seed_scale: `{args.future_seed_scale}`",
        f"- loop1 path F1: `{loop1.path_f1:.4f}`",
        f"- loop{args.eval_loops} path F1: `{final.path_f1:.4f}`",
        f"- loop gain: `{final.path_f1 - loop1.path_f1:+.4f}`",
        f"- precision/recall: `{loop1.path_precision:.4f}` / `{loop1.path_recall:.4f}` -> `{final.path_precision:.4f}` / `{final.path_recall:.4f}`",
        f"- pred PATH frac: `{loop1.pred_path_frac:.4f}` -> `{final.pred_path_frac:.4f}`",
        f"- FP/FN per case: `{loop1.path_fp:.1f}` / `{loop1.path_fn:.1f}` -> `{final.path_fp:.1f}` / `{final.path_fn:.1f}`",
        f"- feedback mode: `{args.feedback_mode}`",
        f"- DAT weight: `{args.dat_weight}`",
        f"- budget decoder: `{args.budget_decoder}`",
    ]
    if args.budget_decoder:
        readme.extend(
            [
                f"- budget loop1 path F1: `{score_loop1.path_f1:.4f}`",
                f"- budget loop{args.eval_loops} path F1: `{score_metric.path_f1:.4f}`",
                f"- budget loop gain: `{score_metric.path_f1 - score_loop1.path_f1:+.4f}`",
                f"- budget precision/recall: `{score_metric.path_precision:.4f}` / `{score_metric.path_recall:.4f}`",
                f"- budget pred PATH frac: `{score_metric.pred_path_frac:.4f}`",
                f"- budget FP/FN per case: `{score_metric.path_fp:.1f}` / `{score_metric.path_fn:.1f}`",
            ]
        )
    if abort_info is not None:
        readme.extend(
            [
                "",
                "## Abort",
                "",
                f"- reason: `{abort_info.get('reason', '')}`",
                f"- step: `{abort_info.get('step', '')}`",
            ]
        )
    readme.extend(
        [
            "",
            "No selector, search, repair, or maze-specific postprocessing is used.",
            "",
            "- `rwkv_maze_probe.json`: metrics and config",
            "- `visualizations/index.html`: hard-case loop visualization",
        ]
    )
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return payload


def broad_mask_abort_info(args: argparse.Namespace, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if int(args.broad_mask_kill_step) <= 0 or int(row["step"]) < int(args.broad_mask_kill_step):
        return None
    prefix = "budget_" if args.broad_mask_target == "budget" and args.budget_decoder else ""
    precision_key = f"{prefix}loop_last_precision"
    final_fp_key = f"{prefix}loop_last_fp"
    loop1_fp_key = f"{prefix}loop1_fp"
    pred_frac_key = f"{prefix}loop_last_pred_path_frac"
    if precision_key not in row or final_fp_key not in row or loop1_fp_key not in row:
        return None
    precision = float(row[precision_key])
    final_fp = float(row[final_fp_key])
    loop1_fp = float(row[loop1_fp_key])
    fp_drop = loop1_fp - final_fp
    pred_frac = float(row.get(pred_frac_key, 0.0))
    if precision < float(args.broad_mask_min_precision) and fp_drop <= float(args.broad_mask_min_fp_drop):
        return {
            "reason": "broad_mask_low_precision_no_fp_drop",
            "step": int(row["step"]),
            "target": args.broad_mask_target,
            "min_precision": float(args.broad_mask_min_precision),
            "min_fp_drop": float(args.broad_mask_min_fp_drop),
            "precision": precision,
            "pred_path_frac": pred_frac,
            "loop1_fp": loop1_fp,
            "loop_last_fp": final_fp,
            "fp_drop": fp_drop,
            "row": row,
        }
    return None


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
    p.add_argument("--feedback-mode", choices=("none", "pred"), default="none")
    p.add_argument("--feedback-scale", type=float, default=1.0)
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--rwkv-kernel", choices=("auto", "torch", "cuda", "statepassing", "wind"), default="statepassing")
    p.add_argument("--forward-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--path-weight", type=float, default=8.0)
    p.add_argument("--path-binary-weight", type=float, default=0.0)
    p.add_argument("--path-budget-weight", type=float, default=0.0)
    p.add_argument("--path-count-weight", type=float, default=0.0)
    p.add_argument("--budget-decoder", action="store_true")
    p.add_argument("--dat-weight", type=float, default=0.0)
    p.add_argument("--dat-loops", type=int, default=2)
    p.add_argument("--dat-token-corrupt-prob", type=float, default=0.03)
    p.add_argument("--dat-add-path-prob", type=float, default=0.20)
    p.add_argument("--dat-delete-path-prob", type=float, default=0.15)
    p.add_argument("--dat-model-error-prob", type=float, default=0.30)
    p.add_argument("--dat-smooth", type=float, default=0.02)
    p.add_argument("--dat-stability-weight", type=float, default=0.50)
    p.add_argument("--dat-improvement-weight", type=float, default=0.25)
    p.add_argument("--dat-improvement-margin", type=float, default=0.01)
    p.add_argument("--loop-loss", choices=("final", "all"), default="all")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--viz-cases", type=int, default=64)
    p.add_argument("--broad-mask-kill-step", type=int, default=0)
    p.add_argument("--broad-mask-min-precision", type=float, default=0.35)
    p.add_argument("--broad-mask-min-fp-drop", type=float, default=0.0)
    p.add_argument("--broad-mask-target", choices=("raw", "budget"), default="raw")
    args = p.parse_args()

    if args.dat_weight > 0 and args.feedback_mode == "none":
        raise ValueError("--dat-weight > 0 requires --feedback-mode pred")
    if args.dat_loops < 1:
        raise ValueError("--dat-loops must be >= 1")
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
        feedback_mode=args.feedback_mode,
        feedback_scale=args.feedback_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        xb, yb = sample_batch(train_x, train_y, args.batch, np_rng, device)
        opt.zero_grad(set_to_none=True)
        with forward_autocast(args.forward_dtype, device):
            logits_by_loop, traces = model.forward_trace(xb, loops=args.train_loops)
            ce_loss = weighted_loop_loss(logits_by_loop, yb, args.path_weight, args.loop_loss)
            path_binary_loss, path_budget_loss, path_diag = path_margin_objective_loss(logits_by_loop, yb, args.loop_loss)
            count_loss, count_diag = path_count_loss(traces, yb, args.loop_loss)
            dat_loss = ce_loss.new_zeros(())
            dat_diag = {
                "dat_denoise_ce": ce_loss.new_zeros(()),
                "dat_stable_ce": ce_loss.new_zeros(()),
                "dat_stable_mse": ce_loss.new_zeros(()),
                "dat_improve": ce_loss.new_zeros(()),
                "dat_corrupt_path_frac": ce_loss.new_zeros(()),
                "dat_clean_path_frac": ce_loss.new_zeros(()),
            }
            if args.dat_weight > 0:
                dat_loss, dat_diag = denoising_attractor_loss(
                    model,
                    xb,
                    yb,
                    logits_by_loop[-1],
                    dat_loops=args.dat_loops,
                    path_weight=args.path_weight,
                    token_corrupt_prob=args.dat_token_corrupt_prob,
                    add_path_prob=args.dat_add_path_prob,
                    delete_path_prob=args.dat_delete_path_prob,
                    model_error_prob=args.dat_model_error_prob,
                    smooth=args.dat_smooth,
                    stability_weight=args.dat_stability_weight,
                    improvement_weight=args.dat_improvement_weight,
                    improvement_margin=args.dat_improvement_margin,
                )
            loss = (
                ce_loss
                + float(args.path_binary_weight) * path_binary_loss
                + float(args.path_budget_weight) * path_budget_loss
                + float(args.path_count_weight) * count_loss
                + float(args.dat_weight) * dat_loss
            )
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
                budget_decoder=args.budget_decoder,
            )
            loop1 = eval_metrics["loop1"]
            final = eval_metrics[f"loop{args.eval_loops}"]
            budget_loop1 = eval_metrics.get("budget_loop1")
            budget_final = eval_metrics.get(f"budget_loop{args.eval_loops}")
            row = {
                "step": step,
                "elapsed_sec": time.time() - t0,
                "loss": float(loss.detach().cpu()),
                "ce_loss": float(ce_loss.detach().cpu()),
                "path_binary_loss": float(path_binary_loss.detach().cpu()),
                "path_budget_loss": float(path_budget_loss.detach().cpu()),
                "path_count_loss": float(count_loss.detach().cpu()),
                "dat_loss": float(dat_loss.detach().cpu()),
                "dat_denoise_ce": float(dat_diag["dat_denoise_ce"].detach().cpu()),
                "dat_stable_ce": float(dat_diag["dat_stable_ce"].detach().cpu()),
                "dat_stable_mse": float(dat_diag["dat_stable_mse"].detach().cpu()),
                "dat_improve": float(dat_diag["dat_improve"].detach().cpu()),
                "dat_corrupt_path_frac": float(dat_diag["dat_corrupt_path_frac"].detach().cpu()),
                "dat_clean_path_frac": float(dat_diag["dat_clean_path_frac"].detach().cpu()),
                "path_margin_pos": float(path_diag["path_margin_pos"].detach().cpu()),
                "path_margin_neg": float(path_diag["path_margin_neg"].detach().cpu()),
                "path_prob_frac": float(path_diag["path_prob_frac"].detach().cpu()),
                "path_true_frac": float(path_diag["path_true_frac"].detach().cpu()),
                "path_count_pred_frac": float(count_diag["path_count_pred_frac"].detach().cpu()),
                "path_count_true_frac": float(count_diag["path_count_true_frac"].detach().cpu()),
                "path_count_abs_err": float(count_diag["path_count_abs_err"].detach().cpu()),
                "loop1_path_f1": loop1.path_f1,
                "loop1_precision": loop1.path_precision,
                "loop1_recall": loop1.path_recall,
                "loop1_pred_path_frac": loop1.pred_path_frac,
                "loop1_fp": loop1.path_fp,
                "loop1_fn": loop1.path_fn,
                "loop_last_path_f1": final.path_f1,
                "loop_gain": final.path_f1 - loop1.path_f1,
                "loop_last_precision": final.path_precision,
                "loop_last_recall": final.path_recall,
                "loop_last_pred_path_frac": final.pred_path_frac,
                "loop_last_fp": final.path_fp,
                "loop_last_fn": final.path_fn,
            }
            if budget_loop1 is not None and budget_final is not None:
                row.update(
                    {
                        "budget_loop1_path_f1": budget_loop1.path_f1,
                        "budget_loop1_precision": budget_loop1.path_precision,
                        "budget_loop1_recall": budget_loop1.path_recall,
                        "budget_loop1_pred_path_frac": budget_loop1.pred_path_frac,
                        "budget_loop1_fp": budget_loop1.path_fp,
                        "budget_loop1_fn": budget_loop1.path_fn,
                        "budget_loop_last_path_f1": budget_final.path_f1,
                        "budget_loop_gain": budget_final.path_f1 - budget_loop1.path_f1,
                        "budget_loop_last_precision": budget_final.path_precision,
                        "budget_loop_last_recall": budget_final.path_recall,
                        "budget_loop_last_pred_path_frac": budget_final.pred_path_frac,
                        "budget_loop_last_fp": budget_final.path_fp,
                        "budget_loop_last_fn": budget_final.path_fn,
                    }
                )
            history.append(row)
            budget_msg = ""
            if budget_final is not None:
                budget_msg = (
                    f" budget_loop{args.eval_loops}={budget_final.path_f1:.4f}"
                    f" bpred={budget_final.pred_path_frac:.4f}"
                    f" bfp={budget_final.path_fp:.1f} bfn={budget_final.path_fn:.1f}"
                )
            print(
                "[rwkv_maze] "
                f"step={step:04d} loss={row['loss']:.4f} ce={row['ce_loss']:.4f} "
                f"bin={row['path_binary_loss']:.4f} budget={row['path_budget_loss']:.4f} count={row['path_count_loss']:.4f} "
                f"dat={row['dat_loss']:.4f} improve={row['dat_improve']:.4f} "
                f"loop1={row['loop1_path_f1']:.4f} loop{args.eval_loops}={row['loop_last_path_f1']:.4f} "
                f"gain={row['loop_gain']:+.4f} pred={row['loop1_pred_path_frac']:.4f}->{row['loop_last_pred_path_frac']:.4f} "
                f"fp={row['loop1_fp']:.1f}->{row['loop_last_fp']:.1f} fn={row['loop1_fn']:.1f}->{row['loop_last_fn']:.1f} "
                f"count_pred={row['path_count_pred_frac']:.4f}{budget_msg}",
                flush=True,
            )
            abort_info = broad_mask_abort_info(args, row)
            if abort_info is not None:
                payload = write_probe_summary(
                    model=model,
                    inputs=test_x,
                    labels=test_y,
                    args=args,
                    history=history,
                    final_metrics=eval_metrics,
                    t0=t0,
                    device=device,
                    aborted=True,
                    abort_info=abort_info,
                )
                print(
                    json.dumps(
                        {
                            "run_name": args.run_name,
                            "aborted": True,
                            "reason": abort_info["reason"],
                            "score": payload["score"],
                            "loop_gain": payload["loop_gain"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                return
    final_metrics = evaluate(
        model,
        test_x,
        test_y,
        eval_n=args.eval_n,
        batch=args.eval_batch,
        loops=args.eval_loops,
        device=device,
        forward_dtype=args.forward_dtype,
        budget_decoder=args.budget_decoder,
    )
    payload = write_probe_summary(
        model=model,
        inputs=test_x,
        labels=test_y,
        args=args,
        history=history,
        final_metrics=final_metrics,
        t0=t0,
        device=device,
        aborted=False,
    )
    print(json.dumps({"run_name": args.run_name, "score": payload["score"], "loop_gain": payload["loop_gain"]}, indent=2))


if __name__ == "__main__":
    main()
