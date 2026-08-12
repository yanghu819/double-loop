from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from experiments.zoology_mqar.gdn2_log_spd import (
    LogSPDGatedDeltaNet2,
    ZoologyLogSPDGDN2FutureSeedMixer,
    _write_events,
    capture_chunk_addresses,
)
from experiments.gdn2_diagnostics.committed_edit import key_gram_statistics
from fla.modules.l2norm import l2norm_fwd


class BlockCausalGramConditioner(nn.Module):
    """Bounded query conditioner built only from completed key blocks."""

    def __init__(self, num_heads: int, head_dim: int, block_size: int = 64) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.raw_strength = nn.Parameter(torch.zeros(num_heads))
        self.last_factors: torch.Tensor | None = None

    def transform_queries(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("Block-Gram expects matched [B,T,H,K] Q/K tensors")
        if q.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(f"Unexpected Q/K shape: {tuple(q.shape)}")
        if q.shape[1] % self.block_size != 0:
            raise ValueError("P-GDN3-026 requires sequence length divisible by block size")

        batch, length, heads, head_dim = k.shape
        blocks = length // self.block_size
        unit_k = F.normalize(k.float(), dim=-1).reshape(
            batch, blocks, self.block_size, heads, head_dim
        )
        block_grams = torch.einsum("bcthk,bcthj->bchkj", unit_k, unit_k)
        prefix_grams = torch.cat(
            (torch.zeros_like(block_grams[:, :1]), block_grams.cumsum(dim=1)[:, :-1]),
            dim=1,
        )
        counts = (
            torch.arange(blocks, device=k.device, dtype=torch.float32)
            * self.block_size
        ).clamp_min(1.0)
        gram = prefix_grams / counts[None, :, None, None, None]
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        identity = torch.eye(head_dim, device=k.device, dtype=torch.float32)
        normalized = head_dim * gram / trace.clamp_min(1e-6).unsqueeze(-1)
        centered = normalized - identity
        frobenius = centered.square().sum(dim=(-1, -2), keepdim=True).sqrt()
        bounded = centered / (1.0 + frobenius)
        alpha = (torch.tanh(self.raw_strength.float()) / 4.0)[None, None, :, None, None]
        active = (torch.arange(blocks, device=k.device) > 0).float()[None, :, None, None, None]
        factors = identity - active * alpha * bounded
        q_blocks = q.reshape(batch, blocks, self.block_size, heads, head_dim)
        delta = (factors - identity).to(dtype=q.dtype)
        conditioned = q_blocks + torch.einsum("bcthk,bchkj->bcthj", q_blocks, delta)
        self.last_factors = factors.detach().transpose(0, 1)
        return conditioned.reshape_as(q)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float | int]:
        if self.last_factors is None:
            raise RuntimeError("Block-Gram diagnostics require one completed forward")
        factors = self.last_factors.float()
        identity = torch.eye(
            self.head_dim,
            device=factors.device,
            dtype=factors.dtype,
        )
        eigenvalues = torch.linalg.eigvalsh(factors)
        return {
            "blocks": int(factors.shape[0]),
            "strength_abs_mean": float(torch.tanh(self.raw_strength).abs().mean().item()),
            "strength_abs_min": float(torch.tanh(self.raw_strength).abs().amin().item()),
            "factor_delta_fro_mean": float(
                (factors - identity).square().sum(dim=(-1, -2)).sqrt().mean().item()
            ),
            "factor_eigenvalue_min": float(eigenvalues.amin().item()),
            "factor_eigenvalue_max": float(eigenvalues.amax().item()),
            "factor_condition_max": float(
                (eigenvalues[..., -1] / eigenvalues[..., 0]).amax().item()
            ),
        }


class BlockGramLogSPDGatedDeltaNet2(LogSPDGatedDeltaNet2):
    """Static Log-SPD plus a block-causal query-side full-Gram metric."""

    def __init__(self, base: nn.Module, block_size: int = 64) -> None:
        super().__init__(base)
        self.block_gram = BlockCausalGramConditioner(
            num_heads=self.num_heads,
            head_dim=self.head_k_dim,
            block_size=block_size,
        )

    def transform_addresses(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.address_metric.transform_pair(q, k)
        return self.block_gram.transform_queries(q, k), k


class ZoologyBlockGramGDN2FutureSeedMixer(ZoologyLogSPDGDN2FutureSeedMixer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        base = self.layer.base
        self.layer = BlockGramLogSPDGatedDeltaNet2(base)


def block_gram_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyBlockGramGDN2FutureSeedMixer)
    ]
    if not mixers:
        raise RuntimeError("No Block-Gram GDN2 mixers found")
    rows = []
    for mixer in mixers:
        rows.append(mixer.layer.block_gram.diagnostics())
    return {
        "active_layers": len(mixers),
        "parameter_delta_over_log_spd": sum(
            mixer.layer.block_gram.raw_strength.numel() for mixer in mixers
        ),
        "block_size": mixers[0].layer.block_gram.block_size,
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    tensors = []
    for name, parameter in model.named_parameters():
        if name.endswith("layer.block_gram.raw_strength"):
            continue
        parent_name = name
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
    }


@torch.no_grad()
def frozen_prefix_gram_diagnostic(
    model: nn.Module,
    dataloader: Any,
    *,
    sequence_length: int = 1024,
    max_examples: int = 128,
    block_size: int = 64,
) -> dict[str, Any]:
    """Score a fixed full-strength conditioner without changing model logits."""
    model.eval()
    layer_count = len(model.backbone.layers)
    conditioner = BlockCausalGramConditioner(4, 32, block_size=block_size).cuda()
    conditioner.raw_strength.fill_(10.0)
    rank_gains = []
    anisotropy_ratios = []
    binding_gains: dict[str, list[torch.Tensor]] = {"future": [], "past": []}
    binding_before: dict[str, list[torch.Tensor]] = {"future": [], "past": []}
    sample_digest = hashlib.sha256()
    examples = 0
    with capture_chunk_addresses() as recorder:
        for inputs, targets, _slices in dataloader:
            if examples >= max_examples:
                break
            take = min(inputs.shape[0], max_examples - examples)
            inputs = inputs[:take]
            targets = targets[:take]
            sample_digest.update(inputs.contiguous().numpy().tobytes())
            sample_digest.update(targets.contiguous().numpy().tobytes())
            recorder.reset()
            model(inputs.cuda())
            if len(recorder.records) != layer_count:
                raise RuntimeError("Prefix-Gram recorder missed an official GDN2 call")
            for q, k in recorder.records:
                conditioned_q = conditioner.transform_queries(q[:take], k[:take])
                with torch.autocast(device_type="cuda", enabled=False):
                    q_norm, _ = l2norm_fwd(q[:take])
                    conditioned_q_norm, _ = l2norm_fwd(conditioned_q)
                    k_norm, _ = l2norm_fwd(k[:take])
                for start in range(block_size, sequence_length, block_size):
                    block = slice(start, start + block_size)
                    before = key_gram_statistics(q_norm[:, block])
                    after = key_gram_statistics(conditioned_q_norm[:, block])
                    rank_gains.append(
                        (
                            after["effective_rank_fraction"]
                            - before["effective_rank_fraction"]
                        ).flatten().cpu()
                    )
                    anisotropy_ratios.append(
                        (
                            after["anisotropy"]
                            / before["anisotropy"].clamp_min(1e-6)
                        ).flatten().cpu()
                    )
                for row in range(take):
                    events = _write_events(inputs[row], targets[row], sequence_length)
                    write_positions = [event[1] for event in events]
                    for event_index, (query_position, write_position, direction) in enumerate(events):
                        other_positions = [
                            position
                            for index, position in enumerate(write_positions)
                            if index != event_index
                        ]
                        before_own = (
                            q_norm[row, query_position] * k_norm[row, write_position]
                        ).sum(dim=-1)
                        after_own = (
                            conditioned_q_norm[row, query_position]
                            * k_norm[row, write_position]
                        ).sum(dim=-1)
                        before_other = torch.einsum(
                            "hk,nhk->nh",
                            q_norm[row, query_position],
                            k_norm[row, other_positions],
                        ).amax(dim=0)
                        after_other = torch.einsum(
                            "hk,nhk->nh",
                            conditioned_q_norm[row, query_position],
                            k_norm[row, other_positions],
                        ).amax(dim=0)
                        before_margin = before_own - before_other
                        after_margin = after_own - after_other
                        binding_before[direction].append(before_margin.cpu())
                        binding_gains[direction].append(
                            (after_margin - before_margin).cpu()
                        )
            examples += take

    rank_gain = torch.cat(rank_gains)
    anisotropy_ratio = torch.cat(anisotropy_ratios)
    direction_rows = {}
    all_binding_gains = []
    for direction in ("future", "past"):
        gains = torch.cat(binding_gains[direction])
        before = torch.cat(binding_before[direction])
        all_binding_gains.append(gains)
        direction_rows[direction] = {
            "before_margin": _summary(before),
            "margin_gain": _summary(gains),
            "improved_fraction": float((gains > 0).float().mean().item()),
        }
    all_gains = torch.cat(all_binding_gains)
    return {
        "intervention_changes_logits": False,
        "examples": examples,
        "sample_sha256": sample_digest.hexdigest(),
        "layers": layer_count,
        "block_size": block_size,
        "active_blocks_per_sequence": sequence_length // block_size - 1,
        "fixed_tanh_strength": float(torch.tanh(conditioner.raw_strength).mean().item()),
        "effective_rank_fraction_gain": _summary(rank_gain),
        "effective_rank_improved_fraction": float((rank_gain > 0).float().mean().item()),
        "anisotropy_ratio": _summary(anisotropy_ratio),
        "anisotropy_improved_fraction": float((anisotropy_ratio < 1).float().mean().item()),
        "binding": direction_rows,
        "binding_margin_gain": _summary(all_gains),
        "binding_improved_fraction": float((all_gains > 0).float().mean().item()),
        "factor": conditioner.diagnostics(),
    }
