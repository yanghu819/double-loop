from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from typing import Any

import fla.layers.gdn2 as fla_gdn2_layer
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2

from experiments.gdn2_diagnostics.committed_edit import key_gram_statistics
from experiments.zoology_mqar.gdn2_futureseed import ZoologyGDN2FutureSeedMixer


class BoundedLogSPDAddressMetric(nn.Module):
    """A volume-preserving, bounded SPD metric over each GDN2 key space."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        indices = torch.triu_indices(head_dim, head_dim)
        keep = ~((indices[0] == head_dim - 1) & (indices[1] == head_dim - 1))
        self.register_buffer("row_indices", indices[0, keep], persistent=False)
        self.register_buffer("col_indices", indices[1, keep], persistent=False)
        self.raw = nn.Parameter(torch.zeros(num_heads, int(keep.sum().item())))

    @property
    def parameters_per_head(self) -> int:
        return self.head_dim * (self.head_dim + 1) // 2 - 1

    def generator(self) -> torch.Tensor:
        raw = self.raw.float()
        matrix = raw.new_zeros(self.num_heads, self.head_dim, self.head_dim)
        matrix[:, self.row_indices, self.col_indices] = raw
        off_diagonal = self.row_indices != self.col_indices
        matrix[
            :, self.col_indices[off_diagonal], self.row_indices[off_diagonal]
        ] = raw[:, off_diagonal]
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        diagonal[:, -1] = -diagonal[:, :-1].sum(dim=-1)
        frobenius = matrix.square().sum(dim=(-1, -2), keepdim=True).sqrt()
        return matrix / (1.0 + frobenius)

    def matrix(self) -> torch.Tensor:
        with torch.autocast(device_type=self.raw.device.type, enabled=False):
            return torch.matrix_exp(0.5 * math.log(2.0) * self.generator())

    def applied_matrix(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the factor represented by the residual path at its actual dtype."""
        matrix = self.matrix()
        identity = torch.eye(
            self.head_dim,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        applied_delta = (matrix - identity).to(dtype=dtype)
        applied_identity = torch.eye(
            self.head_dim,
            device=matrix.device,
            dtype=dtype,
        )
        return (applied_identity + applied_delta).float()

    def _check_shape(self, tensor: torch.Tensor) -> None:
        if tensor.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"Expected (..., {self.num_heads}, {self.head_dim}), got {tuple(tensor.shape)}"
            )

    def _transform(self, tensor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        self._check_shape(tensor)
        delta = delta.to(dtype=tensor.dtype)
        return tensor + torch.einsum("...hk,hkj->...hj", tensor, delta)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        delta = (self.matrix() - torch.eye(
            self.head_dim,
            device=tensor.device,
            dtype=torch.float32,
        ))
        return self._transform(tensor, delta)

    def transform_pair(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.matrix() - torch.eye(
            self.head_dim,
            device=q.device,
            dtype=torch.float32,
        )
        return self._transform(q, delta), self._transform(k, delta)

    @torch.no_grad()
    def diagnostics(self, applied_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
        factor = self.matrix()
        fp32_metric = factor.transpose(-1, -2) @ factor
        applied_factor = self.applied_matrix(applied_dtype)
        applied_metric = applied_factor.transpose(-1, -2) @ applied_factor
        factor_eigenvalues = torch.linalg.eigvalsh(factor)
        fp32_metric_eigenvalues = torch.linalg.eigvalsh(fp32_metric)
        applied_metric_eigenvalues = torch.linalg.eigvalsh(applied_metric)
        identity = torch.eye(self.head_dim, device=factor.device, dtype=factor.dtype)
        return {
            "applied_dtype": str(applied_dtype),
            "factor_delta_fro_mean": float(
                (factor - identity).square().sum(dim=(-1, -2)).sqrt().mean().item()
            ),
            "factor_eigenvalue_min": float(factor_eigenvalues.amin().item()),
            "factor_eigenvalue_max": float(factor_eigenvalues.amax().item()),
            "factor_condition_max": float(
                (factor_eigenvalues[..., -1] / factor_eigenvalues[..., 0]).amax().item()
            ),
            "factor_logdet_abs_max": float(
                torch.linalg.slogdet(factor).logabsdet.abs().amax().item()
            ),
            "fp32_metric_eigenvalue_min": float(fp32_metric_eigenvalues.amin().item()),
            "fp32_metric_eigenvalue_max": float(fp32_metric_eigenvalues.amax().item()),
            "fp32_metric_condition_max": float(
                (
                    fp32_metric_eigenvalues[..., -1]
                    / fp32_metric_eigenvalues[..., 0]
                ).amax().item()
            ),
            "fp32_metric_logdet_abs_max": float(
                torch.linalg.slogdet(fp32_metric).logabsdet.abs().amax().item()
            ),
            "actual_metric_delta_fro_mean": float(
                (applied_metric - identity)
                .square()
                .sum(dim=(-1, -2))
                .sqrt()
                .mean()
                .item()
            ),
            "actual_metric_eigenvalue_min": float(
                applied_metric_eigenvalues.amin().item()
            ),
            "actual_metric_eigenvalue_max": float(
                applied_metric_eigenvalues.amax().item()
            ),
            "actual_metric_condition_max": float(
                (
                    applied_metric_eigenvalues[..., -1]
                    / applied_metric_eigenvalues[..., 0]
                ).amax().item()
            ),
            "actual_metric_logdet_abs_max": float(
                torch.linalg.slogdet(applied_metric).logabsdet.abs().amax().item()
            ),
            "generator_trace_abs_max": float(
                self.generator().diagonal(dim1=-2, dim2=-1).sum(dim=-1).abs().amax().item()
            ),
        }


class LogSPDGatedDeltaNet2(nn.Module):
    """Official GDN2 projections/kernel with one shared Q/K address metric."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_v_heads != base.num_heads:
            raise ValueError("The first Log-SPD test requires matched QK/V heads")
        self.base = base
        self.address_metric = BoundedLogSPDAddressMetric(
            num_heads=base.num_heads,
            head_dim=base.head_k_dim,
        )

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        del output_attentions
        if attention_mask is not None:
            raise ValueError("P-GDN3-020 does not use padded MQAR batches")

        layer = self.base
        q_len = hidden_states.shape[1]
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else layer.mode
        if self.training and mode != "chunk":
            raise RuntimeError("P-GDN3-020 training must use the official chunk kernel")

        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
        if layer.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = layer.q_conv1d(
                x=layer.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = layer.k_conv1d(
                x=layer.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = layer.v_conv1d(
                x=layer.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(layer.q_proj(hidden_states))
            k = F.silu(layer.k_proj(hidden_states))
            v = F.silu(layer.v_proj(hidden_states))
            conv_state_q, conv_state_k, conv_state_v = None, None, None

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        b = layer.b_proj(hidden_states).sigmoid()
        w = layer.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(x, "... (h d) -> ... h d", d=layer.head_k_dim)
            for x in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=layer.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=layer.head_v_dim)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g

        q, k = self.address_metric.transform_pair(q, k)
        if layer.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        operation = chunk_gdn2 if mode == "chunk" else fused_recurrent_gdn2
        output, recurrent_state = operation(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if layer.use_short_conv else None,
            offset=q_len,
        )
        output = layer.o_norm(
            output,
            rearrange(layer.g_proj(hidden_states), "... (h d) -> ... h d", d=layer.head_v_dim),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologyLogSPDGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed with a bounded Log-SPD metric inside each GDN2 address path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = LogSPDGatedDeltaNet2(self.layer)


def log_spd_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyLogSPDGDN2FutureSeedMixer)
    ]
    if not mixers:
        raise RuntimeError("No Log-SPD GDN2 mixers found")
    return {
        "active_layers": len(mixers),
        "parameter_delta": sum(mixer.layer.address_metric.raw.numel() for mixer in mixers),
        "per_layer": [mixer.layer.address_metric.diagnostics() for mixer in mixers],
    }


class ChunkAddressRecorder:
    """Transparent recorder for the actual Q/K tensors committed to GDN2."""

    def __init__(self, operation: Any) -> None:
        self.operation = operation
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset(self) -> None:
        self.records.clear()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        q = kwargs["q"] if "q" in kwargs else args[0]
        k = kwargs["k"] if "k" in kwargs else args[1]
        self.records.append((q.detach(), k.detach()))
        return self.operation(*args, **kwargs)


@contextmanager
def capture_chunk_addresses():
    global chunk_gdn2

    local_operation = chunk_gdn2
    layer_operation = fla_gdn2_layer.chunk_gdn2
    if local_operation is not layer_operation:
        raise RuntimeError("Candidate and official layer use different GDN2 operations")
    recorder = ChunkAddressRecorder(local_operation)
    chunk_gdn2 = recorder
    fla_gdn2_layer.chunk_gdn2 = recorder
    try:
        yield recorder
    finally:
        chunk_gdn2 = local_operation
        fla_gdn2_layer.chunk_gdn2 = layer_operation


def _write_events(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    sequence_length: int,
) -> list[tuple[int, int, str]]:
    quarter = sequence_length // 4
    events = []
    for query_position in torch.nonzero(targets != -100).flatten().tolist():
        key = int(inputs[query_position].item())
        target = int(targets[query_position].item())
        write_position = None
        for candidate in torch.nonzero(inputs == key).flatten().tolist():
            if candidate == query_position or candidate + 1 >= sequence_length:
                continue
            if int(inputs[candidate + 1].item()) == target:
                write_position = candidate
                break
        if write_position is None:
            raise RuntimeError("Could not recover an MQAR write position")
        direction = "future" if query_position < quarter else "past"
        events.append((query_position, write_position, direction))
    return events


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
    }


@torch.no_grad()
def address_geometry_diagnostics(
    model: nn.Module,
    dataloader: Any,
    *,
    sequence_length: int,
    max_examples: int = 128,
) -> dict[str, Any]:
    """Capture zero-parameter address geometry on a fixed validation prefix."""
    model.eval()
    layer_count = len(model.backbone.layers)
    per_layer = [
        {"rank": [], "anisotropy": [], "condition": [], "future": [], "past": []}
        for _ in range(layer_count)
    ]
    examples = 0
    with capture_chunk_addresses() as recorder:
        for inputs, targets, _slices in dataloader:
            if examples >= max_examples:
                break
            take = min(inputs.shape[0], max_examples - examples)
            inputs = inputs[:take]
            targets = targets[:take]
            recorder.reset()
            model(inputs.cuda())
            if len(recorder.records) != layer_count:
                raise RuntimeError(
                    f"Expected {layer_count} GDN2 calls, got {len(recorder.records)}"
                )

            for layer_index, (q, k) in enumerate(recorder.records):
                with torch.autocast(device_type="cuda", enabled=False):
                    q = F.normalize(q[:take].float(), dim=-1)
                    k = F.normalize(k[:take].float(), dim=-1)
                    gram = key_gram_statistics(k)
                per_layer[layer_index]["rank"].append(
                    gram["effective_rank_fraction"].flatten().cpu()
                )
                per_layer[layer_index]["anisotropy"].append(
                    gram["anisotropy"].flatten().cpu()
                )
                per_layer[layer_index]["condition"].append(
                    gram["condition"].flatten().cpu()
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
                        query = q[row, query_position]
                        own = (query * k[row, write_position]).sum(dim=-1)
                        other = torch.einsum(
                            "hk,nhk->nh",
                            query,
                            k[row, other_positions],
                        ).amax(dim=0)
                        per_layer[layer_index][direction].append((own - other).cpu())
            examples += take

    rows = []
    all_rank = []
    all_anisotropy = []
    for layer_index, values in enumerate(per_layer):
        rank = torch.cat(values["rank"])
        anisotropy = torch.cat(values["anisotropy"])
        condition = torch.cat(values["condition"])
        affected = (rank <= 0.50) | (anisotropy >= 4.0)
        rows.append(
            {
                "layer_index": layer_index,
                "effective_rank_fraction": _summary(rank),
                "anisotropy": _summary(anisotropy),
                "condition": _summary(condition),
                "affected_fraction": float(affected.float().mean().item()),
                "binding_contrast": {
                    direction: _summary(torch.cat(values[direction]))
                    for direction in ("future", "past")
                },
            }
        )
        all_rank.append(rank)
        all_anisotropy.append(anisotropy)

    rank = torch.cat(all_rank)
    anisotropy = torch.cat(all_anisotropy)
    affected = (rank <= 0.50) | (anisotropy >= 4.0)
    return {
        "examples": examples,
        "records": int(rank.numel()),
        "effective_rank_fraction": _summary(rank),
        "anisotropy": _summary(anisotropy),
        "affected_fraction": float(affected.float().mean().item()),
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    """Hash candidate parent parameters under their original GDN2 names."""
    tensors = []
    for name, parameter in model.named_parameters():
        if name.endswith("layer.address_metric.raw"):
            continue
        parent_name = name.replace(".sequence_mixer.layer.base.", ".sequence_mixer.layer.")
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
