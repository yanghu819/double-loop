from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import fla.ops.gdn2.chunk_fwd as gdn2_chunk_fwd
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


@contextmanager
def capture_committed_edit() -> Iterator[list[torch.Tensor]]:
    """Capture the exact v_new already produced by the official chunk forward."""

    original = gdn2_chunk_fwd.chunk_gated_delta_rule_fwd_h
    captured: list[torch.Tensor] = []

    def recorder(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        captured.append(result[1])
        return result

    gdn2_chunk_fwd.chunk_gated_delta_rule_fwd_h = recorder
    try:
        yield captured
    finally:
        gdn2_chunk_fwd.chunk_gated_delta_rule_fwd_h = original


class LearnedCorrectionSubspace(nn.Module):
    """A stable learned K32 -> K16 semi-orthogonal address basis."""

    def __init__(self, num_heads: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim != 2 * output_dim:
            raise ValueError("The first correction subspace requires K32 -> K16")
        self.num_heads = int(num_heads)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        row, col = torch.triu_indices(input_dim, input_dim, offset=1)
        self.register_buffer("row_indices", row, persistent=False)
        self.register_buffer("col_indices", col, persistent=False)
        self.raw = nn.Parameter(torch.zeros(num_heads, row.numel()))

        base = torch.zeros(output_dim, input_dim)
        pair = 2.0**-0.5
        for index in range(output_dim):
            base[index, 2 * index : 2 * index + 2] = pair
        self.register_buffer("base", base, persistent=False)

    def rotation(self) -> torch.Tensor:
        raw = self.raw.float()
        skew = raw.new_zeros(self.num_heads, self.input_dim, self.input_dim)
        skew[:, self.row_indices, self.col_indices] = raw
        skew[:, self.col_indices, self.row_indices] = -raw
        identity = torch.eye(
            self.input_dim,
            device=raw.device,
            dtype=raw.dtype,
        ).expand(self.num_heads, -1, -1)
        return torch.linalg.solve(identity - 0.5 * skew, identity + 0.5 * skew)

    def matrix(self) -> torch.Tensor:
        return torch.einsum("ck,hkj->hcj", self.base.float(), self.rotation())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...hk,hck->...hc",
            tensor,
            self.matrix().to(dtype=tensor.dtype),
        )

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        matrix = self.matrix()
        identity = torch.eye(
            self.output_dim,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        gram = matrix @ matrix.transpose(-1, -2)
        return {
            "basis_delta_fro_mean": float(
                (matrix - self.base.float().unsqueeze(0))
                .square()
                .sum(dim=(-1, -2))
                .sqrt()
                .mean()
                .item()
            ),
            "row_orthogonality_max_error": float((gram - identity).abs().max().item()),
        }


class CommittedDeltaGatedDeltaNet2(nn.Module):
    """Parent GDN2 plus a bounded correction state written with exact v_new."""

    def __init__(self, base: nn.Module, *, learned_subspace: bool) -> None:
        super().__init__()
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-025 requires exactly four matched heads")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-025 requires K32/V32")
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-025 requires D128")
        self.base = base
        self.correction_k_dim = 16
        self.learned_subspace = bool(learned_subspace)
        self.correction_read_logit = nn.Parameter(torch.zeros(1, 1, 4, 1))
        self.correction_subspace = (
            LearnedCorrectionSubspace(4, 32, 16)
            if self.learned_subspace
            else None
        )

        fixed = torch.zeros(16, 32)
        pair = 2.0**-0.5
        for index in range(16):
            fixed[index, 2 * index : 2 * index + 2] = pair
        self.register_buffer("fixed_subspace", fixed, persistent=False)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def layer_idx(self) -> int:
        return int(self.base.layer_idx)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    def _project_address(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.correction_subspace is not None:
            return self.correction_subspace(tensor)
        return torch.einsum(
            "...hk,ck->...hc",
            tensor,
            self.fixed_subspace.to(dtype=tensor.dtype),
        )

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
            raise ValueError("P-GDN3-025 uses fixed unpadded MQAR sequences")
        if self.base.mode != "chunk":
            raise RuntimeError("P-GDN3-025 requires the official chunk kernel")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-025 does not use packed sequences")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        v, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        b = layer.b_proj(hidden_states).sigmoid()
        w = layer.w_proj(hidden_states).sigmoid()
        q, k, g = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=layer.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=layer.head_v_dim)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        with capture_committed_edit() as captured:
            main_output, recurrent_state = chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
            )
        if len(captured) != 1:
            raise RuntimeError(f"Expected one committed edit tensor, got {len(captured)}")
        committed_edit = captured[0].detach()

        correction_q = self._project_address(q)
        correction_k = self._project_address(k)
        correction_g = g.float().mean(dim=-1, keepdim=True).expand(
            *g.shape[:-1], self.correction_k_dim
        )
        correction_output, correction_state = chunk_gdn2(
            q=correction_q,
            k=correction_k,
            v=committed_edit,
            g=correction_g,
            b=torch.ones_like(correction_k),
            w=torch.ones_like(committed_edit),
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        read_gate = torch.tanh(self.correction_read_logit).to(
            dtype=main_output.dtype
        )
        output = main_output + read_gate * correction_output
        with torch.no_grad():
            main_rms = main_output.float().square().mean().sqrt().clamp_min(1e-8)
            committed_board = committed_edit.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            correction_board = correction_output.float().square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            state_rms = correction_state.float().square().mean().sqrt()
            main_state_rms = (
                recurrent_state.float().square().mean().sqrt().clamp_min(1e-8)
                if recurrent_state is not None
                else state_rms.new_tensor(1.0)
            )
            self.last_diagnostics = {
                "committed_edit_rms": committed_edit.float().square().mean().sqrt(),
                "committed_edit_board_std": committed_board.std(unbiased=False),
                "correction_output_relative_rms": (
                    correction_output.float().square().mean().sqrt() / main_rms
                ),
                "correction_output_board_std": correction_board.std(unbiased=False),
                "correction_state_rms": state_rms,
                "correction_state_to_main_rms": state_rms / main_state_rms,
                "correction_read_abs_gate": read_gate.float().abs().mean(),
                "correction_q_token_std": correction_q.float().mean(dim=-1).std(
                    unbiased=False
                ),
                "correction_k_token_std": correction_k.float().mean(dim=-1).std(
                    unbiased=False
                ),
            }

        output = layer.o_norm(
            output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=layer.head_v_dim,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologySharedCommittedDeltaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """FutureSeed GDN2 with a fixed-basis committed-delta correction state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = CommittedDeltaGatedDeltaNet2(
            self.layer,
            learned_subspace=False,
        )


class ZoologyClusteredCommittedDeltaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """FutureSeed GDN2 with a learned clustered committed-delta state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = CommittedDeltaGatedDeltaNet2(
            self.layer,
            learned_subspace=True,
        )


def committed_delta_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            (
                ZoologySharedCommittedDeltaFutureSeedMixer,
                ZoologyClusteredCommittedDeltaFutureSeedMixer,
            ),
        )
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two committed-delta mixers, got {len(mixers)}")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Committed-delta diagnostics were not populated")
        row: dict[str, Any] = {
            "layer_idx": mixer.layer_idx,
            **{name: float(value.item()) for name, value in diagnostics.items()},
        }
        if mixer.layer.correction_subspace is not None:
            row.update(mixer.layer.correction_subspace.diagnostics())
        else:
            row.update(
                {
                    "basis_delta_fro_mean": 0.0,
                    "row_orthogonality_max_error": 0.0,
                }
            )
        rows.append(row)
    correction_parameters = sum(
        parameter.numel()
        for mixer in mixers
        for name, parameter in mixer.layer.named_parameters()
        if name.startswith("correction_")
    )
    return {
        "active_layers": len(rows),
        "learned_subspace": all(
            mixer.layer.learned_subspace for mixer in mixers
        ),
        "main_state_values_per_layer": 4 * 32 * 32,
        "correction_state_values_per_layer": 4 * 16 * 32,
        "total_state_values_per_layer": 4 * 32 * 32 + 4 * 16 * 32,
        "official_scans_per_layer": 2,
        "correction_parameters": correction_parameters,
        "committed_edit_detached_from_main_backward": True,
        "per_layer": rows,
    }


def parent_parameter_hash(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    rows = []
    for name, parameter in model.named_parameters():
        if ".layer.correction_" in name:
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, parameter in sorted(rows):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
