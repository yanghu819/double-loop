from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


BLOCK_SIZE = 4
EXPECTED_HEAD_DIM = 32
EXPECTED_GROUPS = EXPECTED_HEAD_DIM // BLOCK_SIZE
EXPECTED_PARAMETER_DELTA = 8
TRANSIENT_GEOMETRY_VALUES_PER_LAYER = 4 * EXPECTED_GROUPS * BLOCK_SIZE**2


@triton.jit
def _block_rls_precision_kernel(
    key,
    log_decay,
    transported,
    terminal,
    minimum_denominator,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid % GROUPS
    bh = pid // GROUPS
    head = bh % H
    batch = bh // H

    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)
    diagonal = row[:, None] == col[None, :]
    precision = tl.where(diagonal, 1.0, 0.0).to(tl.float32)
    min_denom = 1.0e30

    for token in tl.range(0, T):
        vector_offset = ((batch * T + token) * H + head) * K + group * BLOCK
        key_t = tl.load(key + vector_offset + col).to(tl.float32)
        decay_t = tl.exp(
            tl.load(log_decay + vector_offset + col).to(tl.float32)
        )
        transported_t = (
            precision * decay_t[:, None] * decay_t[None, :]
            + tl.where(diagonal, 1.0 - decay_t[:, None] * decay_t[:, None], 0.0)
        )
        output_offset = (
            ((((batch * T + token) * H + head) * GROUPS + group) * BLOCK)
            * BLOCK
        )
        tl.store(
            transported + output_offset + row[:, None] * BLOCK + col[None, :],
            transported_t,
        )

        preconditioned = tl.sum(transported_t * key_t[None, :], axis=1)
        denom = 1.0 + tl.sum(key_t * preconditioned, axis=0)
        min_denom = tl.minimum(min_denom, denom)
        precision = (
            transported_t
            - preconditioned[:, None] * preconditioned[None, :] / denom
        )

    terminal_offset = (
        (((batch * H + head) * GROUPS + group) * BLOCK) * BLOCK
    )
    tl.store(
        terminal + terminal_offset + row[:, None] * BLOCK + col[None, :],
        precision,
    )
    tl.store(
        minimum_denominator + (batch * H + head) * GROUPS + group,
        min_denom,
    )


def block_rls_precision_triton(
    key: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pre-update block precision, terminal precision, and min denominator."""
    if not key.is_cuda or not log_decay.is_cuda:
        raise RuntimeError("P-GDN3-054 block RLS is CUDA-only")
    if key.shape != log_decay.shape or key.ndim != 4:
        raise ValueError("key and log_decay must both have shape [B,T,H,K]")
    batch, length, heads, key_dim = key.shape
    if key_dim != EXPECTED_HEAD_DIM:
        raise ValueError(f"P-GDN3-054 requires K{EXPECTED_HEAD_DIM}")
    if key_dim % BLOCK_SIZE:
        raise ValueError("Key width must be divisible by the fixed block size")
    groups = key_dim // BLOCK_SIZE
    key = key.contiguous()
    log_decay = log_decay.contiguous()
    transported = torch.empty(
        batch,
        length,
        heads,
        groups,
        BLOCK_SIZE,
        BLOCK_SIZE,
        device=key.device,
        dtype=torch.float32,
    )
    terminal = torch.empty(
        batch,
        heads,
        groups,
        BLOCK_SIZE,
        BLOCK_SIZE,
        device=key.device,
        dtype=torch.float32,
    )
    minimum_denominator = torch.empty(
        batch,
        heads,
        groups,
        device=key.device,
        dtype=torch.float32,
    )
    _block_rls_precision_kernel[(batch * heads * groups,)](
        key,
        log_decay,
        transported,
        terminal,
        minimum_denominator,
        T=length,
        H=heads,
        K=key_dim,
        GROUPS=groups,
        BLOCK=BLOCK_SIZE,
        num_warps=1,
        num_stages=2,
    )
    return transported, terminal, minimum_denominator


def block_rls_precision_torch(
    key: torch.Tensor,
    log_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small-shape reference used only by the strict CUDA contract."""
    batch, length, heads, key_dim = key.shape
    if key.shape != log_decay.shape or key_dim % BLOCK_SIZE:
        raise ValueError("Invalid block-RLS reference shapes")
    groups = key_dim // BLOCK_SIZE
    key_blocks = key.float().reshape(
        batch, length, heads, groups, BLOCK_SIZE
    )
    decay_blocks = log_decay.float().exp().reshape_as(key_blocks)
    precision = torch.eye(
        BLOCK_SIZE,
        device=key.device,
        dtype=torch.float32,
    ).expand(batch, heads, groups, -1, -1).clone()
    transported_rows = []
    minimum_denominator = torch.full(
        (batch, heads, groups),
        float("inf"),
        device=key.device,
        dtype=torch.float32,
    )
    identity = torch.eye(BLOCK_SIZE, device=key.device, dtype=torch.float32)
    for token in range(length):
        decay_t = decay_blocks[:, token]
        transported_t = (
            precision * decay_t.unsqueeze(-1) * decay_t.unsqueeze(-2)
            + identity * (1.0 - decay_t.square()).unsqueeze(-1)
        )
        transported_rows.append(transported_t)
        key_t = key_blocks[:, token]
        preconditioned = torch.einsum(
            "bhgij,bhgj->bhgi", transported_t, key_t
        )
        denominator = 1.0 + (key_t * preconditioned).sum(dim=-1)
        minimum_denominator = torch.minimum(minimum_denominator, denominator)
        precision = transported_t - (
            preconditioned.unsqueeze(-1) * preconditioned.unsqueeze(-2)
            / denominator[..., None, None]
        )
    return (
        torch.stack(transported_rows, dim=1),
        precision,
        minimum_denominator,
    )


def constrained_block_rls_address(
    key: torch.Tensor,
    erase_gate: torch.Tensor,
    transported: torch.Tensor,
    mix_logit: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch, length, heads, key_dim = key.shape
    groups = key_dim // BLOCK_SIZE
    key_float = key.float()
    key_blocks = key_float.reshape(batch, length, heads, groups, BLOCK_SIZE)
    preconditioned = torch.einsum(
        "bthgij,bthgj->bthgi", transported, key_blocks
    ).reshape_as(key_float)
    erase_address = erase_gate.float() * key_float
    erase_energy = erase_address.square().sum(dim=-1, keepdim=True)
    correction = (
        (erase_address * key_float).sum(dim=-1, keepdim=True)
        - (erase_address * preconditioned).sum(dim=-1, keepdim=True)
    ) / erase_energy.clamp_min(1e-8)
    constrained = preconditioned + erase_address * correction
    constrained = torch.where(
        erase_energy > 1e-8,
        constrained,
        preconditioned,
    )
    mix = torch.tanh(mix_logit.float()).view(1, 1, heads, 1)
    address = key_float + mix * (constrained - key_float)
    diagnostics = {
        "mix": mix,
        "preconditioned": preconditioned,
        "erase_address": erase_address,
        "constrained": constrained,
        "address": address,
        "erase_energy": erase_energy,
    }
    return address, diagnostics


class BlockRLSGatedDeltaNet2(nn.Module):
    """One official DPLR scan with a coherent block-RLS committed address."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.num_heads != base.num_v_heads:
            raise ValueError("P-GDN3-054 requires matched QK/V heads")
        if base.head_k_dim != EXPECTED_HEAD_DIM or base.head_v_dim != 32:
            raise ValueError("P-GDN3-054 fixes K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-054 requires chunk GDN2 and ShortConv")
        if base.allow_neg_eigval:
            raise ValueError("P-GDN3-054 requires bounded positive erase gates")
        self.base = base
        self.block_rls_mix_logit = nn.Parameter(torch.zeros(base.num_heads))
        self._capture = False
        self._captured: dict[str, torch.Tensor] = {}

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

    @property
    def layer_idx(self) -> int:
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-054 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

    @staticmethod
    def _unit(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=1e-6).to(tensor.dtype)

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
            raise ValueError("P-GDN3-054 uses fixed unpadded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-054 does not use packed sequences")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        q_state, k_state, v_state = None, None, None
        if last_state is not None:
            q_state, k_state, v_state = last_state["conv_state"]
        q, q_state = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=q_state,
            output_final_state=use_cache,
        )
        key, k_state = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=k_state,
            output_final_state=use_cache,
        )
        value, v_state = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=v_state,
            output_final_state=use_cache,
        )

        log_decay = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, key, log_decay, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, key, log_decay, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        log_decay = -layer.A_log.float().exp().unsqueeze(-1) * log_decay
        q = self._unit(q)
        key = self._unit(key)

        transported, terminal_precision, minimum_denominator = (
            block_rls_precision_triton(key.detach(), log_decay.detach())
        )
        address_float, address_diagnostics = constrained_block_rls_address(
            key,
            erase_gate,
            transported,
            self.block_rls_mix_logit,
        )
        address = address_float.to(key.dtype)
        erase_address = address_diagnostics["erase_address"]
        alpha = (log_decay.float().exp() * erase_address).to(key.dtype)
        beta = -address
        payload = write_gate * value
        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_dplr_delta_rule(
            q=q,
            k=address,
            v=payload,
            a=alpha,
            b=beta,
            gk=log_decay,
            scale=layer.head_k_dim**-0.5,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            safe_gate=False,
            chunk_size=16,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(q_state, k_state, v_state),
            offset=q_len,
        )

        if self._capture:
            with torch.no_grad():
                key_float = key.float()
                delta = address_float - key_float
                key_scale = key_float.square().mean().sqrt().clamp_min(1e-8)
                delta_board = delta.square().mean(dim=(1, 2, 3)).sqrt()
                terminal_symmetric = 0.5 * (
                    terminal_precision + terminal_precision.transpose(-1, -2)
                )
                eigenvalues = torch.linalg.eigvalsh(terminal_symmetric)
                diagonal = torch.diagonal(
                    terminal_symmetric,
                    dim1=-2,
                    dim2=-1,
                )
                off_diagonal = terminal_symmetric - torch.diag_embed(diagonal)
                precision_scale = terminal_symmetric.square().mean().sqrt().clamp_min(
                    1e-8
                )
                z = erase_address
                constraint = (
                    (z * address_float).sum(dim=-1)
                    - (z * key_float).sum(dim=-1)
                ).abs()
                self._captured = {
                    "mix": address_diagnostics["mix"].detach(),
                    "address_relative_rms": (delta.square().mean().sqrt() / key_scale),
                    "address_board_std": delta_board.std(unbiased=False),
                    "constraint_error_max": constraint.max(),
                    "minimum_denominator": minimum_denominator.min(),
                    "terminal_eigen_min": eigenvalues.min(),
                    "terminal_eigen_max": eigenvalues.max(),
                    "terminal_off_diagonal_ratio": (
                        off_diagonal.square().mean().sqrt() / precision_scale
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


class ZoologyBlockRLSGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = BlockRLSGatedDeltaNet2(self.layer)


def _is_extra_parameter(name: str) -> bool:
    return name.endswith(".sequence_mixer.layer.block_rls_mix_logit")


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extras = 0
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = torch.zeros_like(tensor)
            extras += tensor.numel()
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    if extras != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_DELTA} new parameters, found {extras}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    tensors = []
    for name, parameter in model.named_parameters():
        if _is_extra_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def block_rls_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyBlockRLSGDN2FutureSeedMixer)
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use block-RLS GDN2")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    rows = []
    active_paths = 0
    for mixer in mixers:
        if not mixer.layer._captured:
            raise RuntimeError("Block-RLS capture was not produced")
        captured = mixer.layer._captured
        mix_values = captured["mix"].float().flatten()
        active_paths += int((mix_values.abs() >= 1e-3).sum().item())
        row = {
            "layer_idx": mixer.layer_idx,
            "mix_abs_mean": float(mix_values.abs().mean().item()),
            "mix_abs_min": float(mix_values.abs().min().item()),
            "mix_abs_max": float(mix_values.abs().max().item()),
            "mix_values": [float(value) for value in mix_values.tolist()],
        }
        row.update(
            {
                name: float(value.float().mean().item())
                for name, value in captured.items()
                if name != "mix"
            }
        )
        rows.append(row)
    return {
        "active_layers": len(rows),
        "active_paths": active_paths,
        "total_paths": len(rows) * mixers[0].layer.num_heads,
        "block_size": BLOCK_SIZE,
        "groups_per_head": EXPECTED_GROUPS,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "transient_geometry_values_per_layer": TRANSIENT_GEOMETRY_VALUES_PER_LAYER,
        "official_dplr_scans_per_layer": 1,
        "history_gradient": "stopped",
        "current_key_local_jacobian": True,
        "per_layer": rows,
    }
