from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.ops.gdn2 import chunk_gdn2
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_NEW_PARAMETERS = 16_384
EXPECTED_STATE_VALUES = 4_096


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class StableTokenAddressGDN2(nn.Module):
    """Official GDN2 with one external stable residual shared by Q and K."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-046 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-046 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-046 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-046 requires chunk GDN2 with short convolution")
        self.base = base
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
            raise RuntimeError("P-GDN3-046 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
        if enabled:
            self._captured.clear()

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
            raise ValueError("P-GDN3-046 uses fixed unpadded MQAR batches")
        address_residual = kwargs.pop("address_residual", None)
        if address_residual is None:
            raise ValueError("Stable token address residual is required")
        if address_residual.shape != hidden_states.shape:
            raise ValueError(
                "Address residual shape mismatch: "
                f"{tuple(address_residual.shape)} != {tuple(hidden_states.shape)}"
            )

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        cu_seqlens = kwargs.get("cu_seqlens")
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
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        g = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, k, g, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=layer.head_k_dim)
            for tensor in (q, k, g, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=layer.head_v_dim)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=layer.head_v_dim,
        )
        address = rearrange(
            address_residual,
            "b t (h d) -> b t h d",
            h=layer.num_heads,
            d=layer.head_k_dim,
        ).to(dtype=q.dtype)
        transformed_q = q + address
        transformed_k = k + address
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        output, recurrent_state = chunk_gdn2(
            q=transformed_q,
            k=transformed_k,
            v=value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-046 capture requires terminal state")
            self._captured = {
                "q": q.detach(),
                "k": k.detach(),
                "transformed_q": transformed_q.detach(),
                "transformed_k": transformed_k.detach(),
                "address_residual": address_residual.detach(),
                "terminal_state": recurrent_state.detach(),
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


class ZoologyStableTokenAddressFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over stable-token-address official GDN2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = StableTokenAddressGDN2(self.layer)

    def forward_with_address_state(
        self,
        hidden_states: torch.Tensor,
        *,
        address_residual: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                address_residual=address_residual,
                past_key_values=cache,
                use_cache=True,
            )
        layer_cache = cache[self.layer_idx]
        terminal_state = layer_cache["recurrent_state"]
        if terminal_state is None:
            raise RuntimeError("Official GDN2 did not return a terminal state")
        return output.to(hidden_states.dtype), terminal_state


class StableTokenAddressBackbone(FutureSeedLMBackbone):
    """One token-only address namespace shared by every GDN2 layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.shared_address_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        nn.init.zeros_(self.shared_address_proj.weight)
        self.last_address_input: Optional[torch.Tensor] = None
        self.last_address_residual: Optional[torch.Tensor] = None

    @staticmethod
    def normalize_anchor(anchor: torch.Tensor) -> torch.Tensor:
        rms = anchor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return anchor / rms.to(dtype=anchor.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_anchor = self.embeddings.word_embeddings(input_ids)
        if self.embeddings.project_in is not None:
            token_anchor = self.embeddings.project_in(token_anchor)
        address_input = self.normalize_anchor(token_anchor)
        address_residual = self.shared_address_proj(address_input)
        self.last_address_input = address_input.detach()
        self.last_address_residual = address_residual.detach()

        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        terminal_state = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyStableTokenAddressFutureSeedMixer):
                raise TypeError("Stable-address backbone requires strict mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            hidden_states, terminal_state = mixer.forward_with_address_state(
                hidden_states,
                address_residual=address_residual,
                initial_state=initial_state,
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class StableTokenAddressLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = StableTokenAddressBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        nn.init.zeros_(self.backbone.shared_address_proj.weight)
        if config.learnable_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state=None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        del state
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        if return_embeddings:
            return hidden_states
        return self.lm_head(hidden_states)

    def state_size(self, sequence_length: int) -> int:
        return _compute_state_size(self.backbone.layers, sequence_length)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name == "backbone.shared_address_proj.weight":
            loaded[name] = torch.zeros_like(tensor)
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
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if name == "backbone.shared_address_proj.weight":
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _repeated_token_max_error(
    input_ids: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[float, int]:
    max_error = 0.0
    groups = 0
    ids = input_ids.detach().cpu()
    values = residual.detach().float().cpu()
    for board_ids, board_values in zip(ids, values):
        for token in board_ids.unique():
            positions = torch.nonzero(board_ids == token, as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            groups += 1
            reference = board_values[positions[0]]
            error = (board_values[positions] - reference).abs().max().item()
            max_error = max(max_error, float(error))
    return max_error, groups


@torch.no_grad()
def stable_token_address_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, StableTokenAddressBackbone):
        raise TypeError("Expected StableTokenAddressBackbone")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyStableTokenAddressFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two stable-token-address GDN2 mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    residual = model.backbone.last_address_residual
    address_input = model.backbone.last_address_input
    if residual is None or address_input is None:
        raise RuntimeError("Stable address capture is missing")
    repeat_error, repeat_groups = _repeated_token_max_error(inputs, residual)
    rows = []
    address_ptrs = set()
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "q",
            "k",
            "transformed_q",
            "transformed_k",
            "address_residual",
            "terminal_state",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete stable-address capture: {set(captured)}")
        q = captured["q"].float()
        k = captured["k"].float()
        tq = captured["transformed_q"].float()
        tk = captured["transformed_k"].float()
        state = captured["terminal_state"].float()
        state_board = state.square().mean(dim=(-1, -2, -3)).sqrt()
        address_ptrs.add(captured["address_residual"].data_ptr())
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "q_change_relative_rms": float(
                    (_rms(tq - q) / _rms(q).clamp_min(1e-8)).item()
                ),
                "k_change_relative_rms": float(
                    (_rms(tk - k) / _rms(k).clamp_min(1e-8)).item()
                ),
                "terminal_state_rms": float(_rms(state).item()),
                "terminal_state_board_std": float(
                    state_board.std(unbiased=False).item()
                ),
            }
        )
    residual_board = residual.float().square().mean(dim=(-1, -2)).sqrt()
    return {
        "active_layers": len(rows),
        "shared_projection_count": 1,
        "shared_residual_data_ptr_count": len(address_ptrs),
        "new_parameters": model.backbone.shared_address_proj.weight.numel(),
        "new_persistent_state_values": 0,
        "logical_scans_per_layer": 1,
        "token_only_anchor": True,
        "uses_position_anchor": False,
        "projection_weight_rms": float(
            _rms(model.backbone.shared_address_proj.weight).item()
        ),
        "address_input_rms": float(_rms(address_input).item()),
        "address_residual_rms": float(_rms(residual).item()),
        "address_residual_token_std": float(
            residual.float().std(dim=1, unbiased=False).mean().item()
        ),
        "address_residual_board_std": float(
            residual_board.std(unbiased=False).item()
        ),
        "repeated_token_groups": repeat_groups,
        "repeated_token_max_error": repeat_error,
        "per_layer": rows,
    }
