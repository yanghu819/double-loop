from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _argument(
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


def _safe_float(value: Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _effective_rank(eigenvalues: Tensor) -> Tensor:
    values = eigenvalues.clamp_min(0)
    probabilities = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.exp()


def key_gram_statistics(k: Tensor) -> dict[str, Tensor]:
    """Return per-board/head geometry for normalized keys [B,T,H,K]."""
    gram = torch.einsum("bthk,bthl->bhkl", k, k) / float(k.shape[1])
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    mean_eigenvalue = eigenvalues.mean(dim=-1).clamp_min(1e-12)
    largest = eigenvalues[..., -1]
    positive_floor = mean_eigenvalue * 1e-6
    smallest = eigenvalues[..., 0].clamp_min(positive_floor)
    normalized_rank = _effective_rank(eigenvalues) / float(k.shape[-1])

    coherence_keys = k
    if k.shape[1] > 256:
        positions = torch.linspace(
            0,
            k.shape[1] - 1,
            256,
            device=k.device,
        ).round().long()
        coherence_keys = k.index_select(1, positions)
    similarity = torch.einsum("bthk,bshk->bhts", coherence_keys, coherence_keys)
    eye = torch.eye(coherence_keys.shape[1], device=k.device, dtype=torch.bool)
    off_diagonal = similarity.masked_fill(eye[None, None], 0).abs()
    coherence = off_diagonal.amax(dim=(-1, -2))
    return {
        "anisotropy": largest / mean_eigenvalue,
        "effective_rank_fraction": normalized_rank,
        "condition": largest / smallest,
        "coherence": coherence,
    }


def _selected_survival(
    k: Tensor,
    g: Tensor,
    b: Tensor,
    selected: Tensor,
) -> Tensor:
    """Track selected write-address vectors through later left transitions."""
    batch, tokens, heads, key_dim = k.shape
    slots = selected.shape[1]
    vectors = torch.zeros(
        batch,
        heads,
        slots,
        key_dim,
        dtype=torch.float32,
        device=k.device,
    )
    initial_norm = torch.ones(
        batch,
        heads,
        slots,
        dtype=torch.float32,
        device=k.device,
    )
    for token in range(tokens):
        decay = g[:, token].exp().unsqueeze(2)
        decayed = vectors * decay
        erase_key = (b[:, token] * k[:, token]).unsqueeze(2)
        projection = (erase_key * decayed).sum(dim=-1, keepdim=True)
        transitioned = decayed - k[:, token].unsqueeze(2) * projection
        vectors = transitioned

        activate = selected.eq(token).unsqueeze(1).unsqueeze(-1)
        token_key = k[:, token].unsqueeze(2).expand(-1, -1, slots, -1)
        vectors = torch.where(activate, token_key, vectors)
        selected_norm = token_key.square().sum(dim=-1).sqrt().clamp_min(1e-12)
        initial_norm = torch.where(activate.squeeze(-1), selected_norm, initial_norm)
    return vectors.square().sum(dim=-1).sqrt() / initial_norm


@torch.no_grad()
def committed_edit_diagnostics(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    initial_state: Optional[Tensor],
    official_final_state: Optional[Tensor],
    use_qk_l2norm: bool,
    top_k: int,
) -> dict[str, Any]:
    """Replay the exact GDN2 edit in FP32 without changing model outputs."""
    del q
    qk_k = F.normalize(k.float(), dim=-1) if use_qk_l2norm else k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = b.float()
    w_f = w.float()
    batch, tokens, heads, key_dim = qk_k.shape
    value_dim = v_f.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=k.device,
        )
    else:
        state = initial_state.float().clone()

    surprise = torch.empty(batch, tokens, dtype=torch.float32, device=k.device)
    for token in range(tokens):
        state = state * g_f[:, token].exp().unsqueeze(-1)
        erase = (
            (b_f[:, token] * qk_k[:, token]).unsqueeze(-1) * state
        ).sum(dim=-2)
        committed = w_f[:, token] * v_f[:, token] - erase
        update = qk_k[:, token].unsqueeze(-1) * committed.unsqueeze(-2)
        state = state + update
        surprise[:, token] = update.square().sum(dim=(-1, -2, -3)).sqrt()

    replay_error = torch.zeros((), device=k.device)
    replay_relative_error = torch.zeros((), device=k.device)
    if official_final_state is not None:
        official = official_final_state.float()
        replay_error = (state - official).abs().amax()
        replay_relative_error = (
            (state - official).square().mean().sqrt()
            / official.square().mean().sqrt().clamp_min(1e-8)
        )

    selected_count = min(max(int(top_k), 1), tokens)
    surprise_indices = surprise.topk(selected_count, dim=-1).indices
    recency_indices = torch.arange(
        tokens - selected_count,
        tokens,
        device=k.device,
    ).unsqueeze(0).expand(batch, -1)
    surprise_survival = _selected_survival(qk_k, g_f, b_f, surprise_indices)
    recency_survival = _selected_survival(qk_k, g_f, b_f, recency_indices)
    concentration = (
        surprise.gather(1, surprise_indices).sum(dim=-1)
        / surprise.sum(dim=-1).clamp_min(1e-12)
    )
    geometry = key_gram_statistics(qk_k)
    return {
        "surprise": surprise.cpu(),
        "surprise_indices": surprise_indices.cpu(),
        "surprise_topk_concentration": concentration.cpu(),
        "surprise_survival": surprise_survival.mean(dim=(1, 2)).cpu(),
        "recency_survival": recency_survival.mean(dim=(1, 2)).cpu(),
        "surprise_survival_by_slot": surprise_survival.mean(dim=1).cpu(),
        "recency_survival_by_slot": recency_survival.mean(dim=1).cpu(),
        "geometry": {key: value.cpu() for key, value in geometry.items()},
        "replay_max_abs_error": _safe_float(replay_error),
        "replay_relative_rms_error": _safe_float(replay_relative_error),
    }


@dataclass
class CaptureContext:
    selected: bool
    metadata: dict[str, Any]


class ChunkGDN2Recorder:
    """Transparent wrapper around the pinned official chunk_gdn2 function."""

    def __init__(
        self,
        operation: Callable[..., Any],
        *,
        context: Callable[[int], CaptureContext],
        top_k: int = 16,
    ) -> None:
        self.operation = operation
        self.context = context
        self.top_k = int(top_k)
        self.enabled = False
        self.call_index = 0
        self.records: list[dict[str, Any]] = []

    def reset(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.call_index = 0
        self.records = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call_index = self.call_index
        self.call_index += 1
        output = self.operation(*args, **kwargs)
        capture_context = self.context(call_index)
        if not self.enabled or not capture_context.selected:
            return output

        q = _argument(args, kwargs, "q", 0)
        k = _argument(args, kwargs, "k", 1)
        v = _argument(args, kwargs, "v", 2)
        g = _argument(args, kwargs, "g", 3)
        b = _argument(args, kwargs, "b", 4)
        w = _argument(args, kwargs, "w", 5)
        initial_state = _argument(args, kwargs, "initial_state", 7)
        use_qk_l2norm = bool(
            _argument(args, kwargs, "use_qk_l2norm_in_kernel", 9, False)
        )
        final_state = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        diagnostics = committed_edit_diagnostics(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=initial_state,
            official_final_state=final_state,
            use_qk_l2norm=use_qk_l2norm,
            top_k=self.top_k,
        )
        self.records.append({**capture_context.metadata, **diagnostics})
        return output


def pearson_binary(scores: Tensor, labels: Tensor) -> float:
    scores_f = scores.float().reshape(-1)
    labels_f = labels.float().reshape(-1)
    if scores_f.numel() == 0 or labels_f.min() == labels_f.max():
        return float("nan")
    scores_f = scores_f - scores_f.mean()
    labels_f = labels_f - labels_f.mean()
    denominator = scores_f.square().sum().sqrt() * labels_f.square().sum().sqrt()
    return float((scores_f * labels_f).sum().div(denominator.clamp_min(1e-12)).item())


def binary_auroc(scores: Tensor, labels: Tensor) -> float:
    scores_f = scores.float().reshape(-1)
    labels_b = labels.bool().reshape(-1)
    positives = int(labels_b.sum().item())
    negatives = int((~labels_b).sum().item())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores_f.argsort(stable=True)
    ranks = torch.empty_like(scores_f)
    ranks[order] = torch.arange(
        1,
        scores_f.numel() + 1,
        device=scores_f.device,
        dtype=scores_f.dtype,
    )
    positive_rank_sum = ranks[labels_b].sum()
    auc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)
    return float(auc.item())


def top_quantile_lift(scores: Tensor, labels: Tensor, fraction: float = 0.25) -> float:
    scores_f = scores.float().reshape(-1)
    labels_f = labels.float().reshape(-1)
    if scores_f.numel() == 0 or labels_f.mean() <= 0:
        return float("nan")
    count = max(1, int(math.ceil(scores_f.numel() * float(fraction))))
    selected = scores_f.topk(count).indices
    return float((labels_f[selected].mean() / labels_f.mean()).item())


def summarize_geometry(records: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in ("anisotropy", "effective_rank_fraction", "condition", "coherence"):
        values = torch.cat(
            [record["geometry"][metric].reshape(-1) for record in records]
        ).float()
        output[f"{metric}_mean"] = float(values.mean().item())
        output[f"{metric}_median"] = float(values.median().item())
        output[f"{metric}_p90"] = float(torch.quantile(values, 0.90).item())
    output["replay_relative_rms_error_max"] = max(
        float(record["replay_relative_rms_error"]) for record in records
    )
    output["replay_max_abs_error_max"] = max(
        float(record["replay_max_abs_error"]) for record in records
    )
    return output
