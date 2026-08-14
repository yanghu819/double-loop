from __future__ import annotations

import hashlib
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


LINEAR_DIM = 32
FACTOR_DIM = 8
PRODUCT_DIM = FACTOR_DIM * FACTOR_DIM
TOTAL_KEY_DIM = LINEAR_DIM + PRODUCT_DIM
STATE_VALUES_PER_LAYER = 4 * TOTAL_KEY_DIM * 32


class LinearProductStateGatedDeltaNet2(nn.Module):
    """One coherent GDN2 state with linear and exact-product address blocks."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-040 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-040 requires H4")
        if base.head_k_dim != LINEAR_DIM or base.head_v_dim != 32:
            raise ValueError("P-GDN3-040 requires native K32/V32")
        if base.mode != "chunk":
            raise ValueError("P-GDN3-040 requires the official chunk path")
        self.base = base
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return TOTAL_KEY_DIM

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    @property
    def layer_idx(self) -> int:
        return int(self.base.layer_idx)

    @staticmethod
    def address_features(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return native and exact-product unit blocks in one direct sum."""

        native = F.normalize(tensor.float(), dim=-1, eps=1e-6)
        axes = tensor.float()[..., : 2 * FACTOR_DIM].reshape(
            *tensor.shape[:-1], 2, FACTOR_DIM
        )
        axes = F.normalize(axes, dim=-1, eps=1e-6)
        first, second = axes.unbind(dim=-2)
        product = torch.einsum("...i,...j->...ij", first, second).flatten(-2)
        combined = torch.cat((native, product), dim=-1)
        return combined.to(tensor.dtype), native, product, axes

    @staticmethod
    def transition_features(
        decay: torch.Tensor,
        erase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift stable one-axis decay/erase gates to the exact product block."""

        decay_axes = decay.float()[..., : 2 * FACTOR_DIM].reshape(
            *decay.shape[:-1], 2, FACTOR_DIM
        )
        decay_first, decay_second = decay_axes.unbind(dim=-2)
        product_decay = 0.5 * (
            decay_first.unsqueeze(-1) + decay_second.unsqueeze(-2)
        ).flatten(-2)

        erase_axes = erase.float()[..., : 2 * FACTOR_DIM].reshape(
            *erase.shape[:-1], 2, FACTOR_DIM
        )
        erase_first, erase_second = erase_axes.unbind(dim=-2)
        product_erase = (
            erase_first.unsqueeze(-1) * erase_second.unsqueeze(-2)
        ).clamp_min(1e-8).sqrt().flatten(-2)
        return (
            torch.cat((decay.float(), product_decay), dim=-1).to(decay.dtype),
            torch.cat((erase.float(), product_erase), dim=-1).to(erase.dtype),
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
            raise ValueError("P-GDN3-040 uses fixed unpadded MQAR sequences")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-040 does not use packed sequences")

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
            rearrange(tensor, "... (h d) -> ... h d", d=LINEAR_DIM)
            for tensor in (q, k, g)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=layer.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=LINEAR_DIM)
        w = rearrange(w, "... (h d) -> ... h d", d=layer.head_v_dim)
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            raise RuntimeError("P-GDN3-040 requires bounded positive erase gates")

        q, q_native, q_product, q_axes = self.address_features(q)
        k, k_native, k_product, k_axes = self.address_features(k)
        g, b = self.transition_features(g, b)

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if recurrent_state is not None and recurrent_state.shape[1:] != (
            self.num_heads,
            TOTAL_KEY_DIM,
            self.head_v_dim,
        ):
            raise RuntimeError(
                f"Unexpected linear-product state shape: {tuple(recurrent_state.shape)}"
            )
        output, recurrent_state = chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            scale=LINEAR_DIM**-0.5,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=False,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        if recurrent_state is not None:
            with torch.no_grad():
                state = recurrent_state.float()
                native_state = state[..., :LINEAR_DIM, :]
                product_state = state[..., LINEAR_DIM:, :]
                native_read = torch.einsum(
                    "bthk,bhkv->bthv", q.float()[..., :LINEAR_DIM], native_state
                )
                product_read = torch.einsum(
                    "bthk,bhkv->bthv", q.float()[..., LINEAR_DIM:], product_state
                )
                product_state_board_rms = product_state.square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                native_state_rms = native_state.square().mean().sqrt().clamp_min(1e-8)
                product_state_rms = product_state.square().mean().sqrt()
                q_similarity = (q_native * k_native).sum(dim=-1)
                product_similarity = (q_product * k_product).sum(dim=-1)
                direct_sum_similarity = (q.float() * k.float()).sum(dim=-1)
                expected_similarity = q_similarity + product_similarity
                self.last_diagnostics = {
                    "q_direct_sum_norm_max_error": (
                        q.float().norm(dim=-1) - 2**0.5
                    ).abs().max(),
                    "k_direct_sum_norm_max_error": (
                        k.float().norm(dim=-1) - 2**0.5
                    ).abs().max(),
                    "similarity_identity_max_error": (
                        direct_sum_similarity - expected_similarity
                    ).abs().max(),
                    "q_product_rms": q_product.square().mean().sqrt(),
                    "k_product_rms": k_product.square().mean().sqrt(),
                    "q_product_token_std": q_product.std(dim=1, unbiased=False).mean(),
                    "k_product_token_std": k_product.std(dim=1, unbiased=False).mean(),
                    "q_product_board_std": q_product.mean(dim=1).square().mean(
                        dim=(1, 2)
                    ).sqrt().std(unbiased=False),
                    "k_product_board_std": k_product.mean(dim=1).square().mean(
                        dim=(1, 2)
                    ).sqrt().std(unbiased=False),
                    "q_factor_abs_cosine": (q_axes[..., 0, :] * q_axes[..., 1, :]).sum(
                        dim=-1
                    ).abs().mean(),
                    "k_factor_abs_cosine": (k_axes[..., 0, :] * k_axes[..., 1, :]).sum(
                        dim=-1
                    ).abs().mean(),
                    "native_state_rms": native_state_rms,
                    "product_state_rms": product_state_rms,
                    "product_native_state_rms_ratio": product_state_rms
                    / native_state_rms,
                    "product_state_board_std": product_state_board_rms.std(
                        unbiased=False
                    ),
                    "product_native_read_rms_ratio": product_read.square().mean().sqrt()
                    / native_read.square().mean().sqrt().clamp_min(1e-8),
                    "terminal_state_rms": state.square().mean().sqrt(),
                    "decay_max": g.float().max(),
                    "erase_min": b.float().min(),
                    "erase_max": b.float().max(),
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


class ZoologyLinearProductStateFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over one linear-plus-product coherent state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = LinearProductStateGatedDeltaNet2(self.layer)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
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
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def linear_product_state_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyLinearProductStateFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two linear-product state mixers")
    rows = []
    for mixer in mixers:
        diagnostics = mixer.layer.last_diagnostics
        if not diagnostics:
            raise RuntimeError("Linear-product diagnostics were not populated")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                **{name: float(value.item()) for name, value in diagnostics.items()},
            }
        )
    return {
        "active_layers": len(rows),
        "linear_dim": LINEAR_DIM,
        "factor_dim": FACTOR_DIM,
        "product_dim": PRODUCT_DIM,
        "total_key_dim": TOTAL_KEY_DIM,
        "state_values_per_layer": STATE_VALUES_PER_LAYER,
        "new_parameters": 0,
        "official_scans_per_layer": 1,
        "per_layer": rows,
    }
