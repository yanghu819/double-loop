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


CANONICAL_K_DIM = 16
EXPECTED_NEW_PARAMETERS = 520
EXPECTED_MAIN_STATE_VALUES = 4_096
EXPECTED_COMPANION_STATE_VALUES = 2_048


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _normalize_last(tensor: torch.Tensor) -> torch.Tensor:
    rms = tensor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / rms.to(dtype=tensor.dtype)


class CanonicalAddressCompanionGDN2(nn.Module):
    """Native GDN2 plus a smaller ownership-preserving companion state."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != 128:
            raise ValueError("P-GDN3-049 requires D128")
        if base.num_heads != 4 or base.num_v_heads != 4:
            raise ValueError("P-GDN3-049 requires H4")
        if base.head_k_dim != 32 or base.head_v_dim != 32:
            raise ValueError("P-GDN3-049 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-049 requires chunk GDN2 with short convolution")
        self.base = base
        self.companion_read_logit = nn.Parameter(torch.zeros(1, 1, 4, 1))
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
            raise RuntimeError("P-GDN3-049 requires an explicit layer index")
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
            raise ValueError("P-GDN3-049 uses fixed unpadded MQAR batches")
        address_projector = kwargs.pop("address_projector", None)
        if not isinstance(address_projector, nn.Linear):
            raise TypeError("Canonical-address projector is required")
        if (
            address_projector.in_features != self.head_k_dim
            or address_projector.out_features != CANONICAL_K_DIM
        ):
            raise ValueError("Canonical-address projector shape drifted")

        layer = self.base
        q_len = hidden_states.shape[1]
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-049 does not use packed sequences")
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
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
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
        g = -layer.A_log.float().exp().unsqueeze(-1) * g
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        main_output, recurrent_state = chunk_gdn2(
            q=q,
            k=k,
            v=value,
            g=g,
            b=erase_gate,
            w=write_gate,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )

        # A single learned map shared across layers and heads defines a compact
        # address namespace. The companion keeps the full native value payload.
        address_source = _normalize_last(q + k)
        canonical_address = _normalize_last(address_projector(address_source))
        companion_g = g.mean(dim=-1, keepdim=True).expand(
            *g.shape[:-1], CANONICAL_K_DIM
        )
        companion_b = erase_gate.mean(dim=-1, keepdim=True).expand(
            *erase_gate.shape[:-1], CANONICAL_K_DIM
        )
        companion_output, companion_state = chunk_gdn2(
            q=canonical_address,
            k=canonical_address,
            v=value,
            g=companion_g,
            b=companion_b,
            w=write_gate,
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

        gate = torch.tanh(self.companion_read_logit).to(dtype=main_output.dtype)
        output = main_output + gate * _normalize_last(companion_output)
        if self._capture:
            if recurrent_state is None:
                raise RuntimeError("P-GDN3-049 capture requires main terminal state")
            self._captured = {
                "main_output": main_output.detach(),
                "main_state": recurrent_state.detach(),
                "companion_output": companion_output.detach(),
                "companion_state": companion_state.detach(),
                "companion_gate": gate.detach(),
                "address_source": address_source.detach(),
                "canonical_address": canonical_address.detach(),
                "companion_g": companion_g.detach(),
                "companion_b": companion_b.detach(),
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


class ZoologyCanonicalAddressCompanionFutureSeedMixer(
    ZoologyGDN2FutureSeedMixer
):
    """Native FutureSeed over main state; companion state stays layer-local."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = CanonicalAddressCompanionGDN2(self.layer)

    def forward_with_companion_state(
        self,
        hidden_states: torch.Tensor,
        *,
        address_projector: nn.Linear,
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
                address_projector=address_projector,
                past_key_values=cache,
                use_cache=True,
            )
        terminal_state = cache[self.layer_idx]["recurrent_state"]
        if terminal_state is None:
            raise RuntimeError("Official GDN2 did not return a terminal state")
        return output.to(hidden_states.dtype), terminal_state


class CanonicalAddressCompanionBackbone(FutureSeedLMBackbone):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.shared_address_proj = nn.Linear(
            32,
            CANONICAL_K_DIM,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        terminal_state = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(
                mixer, ZoologyCanonicalAddressCompanionFutureSeedMixer
            ):
                raise TypeError("Canonical-address backbone requires strict mixers")

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
            hidden_states, terminal_state = mixer.forward_with_companion_state(
                hidden_states,
                address_projector=self.shared_address_proj,
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


class CanonicalAddressCompanionLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = CanonicalAddressCompanionBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        for block in self.backbone.layers:
            nn.init.zeros_(block.sequence_mixer.layer.companion_read_logit)
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


def _is_extra_parameter(name: str) -> bool:
    return name == "backbone.shared_address_proj.weight" or name.endswith(
        ".companion_read_logit"
    )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = tensor
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
        if _is_extra_parameter(name):
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


@torch.no_grad()
def canonical_address_companion_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, CanonicalAddressCompanionBackbone):
        raise TypeError("Expected CanonicalAddressCompanionBackbone")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyCanonicalAddressCompanionFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two canonical-address companion mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    projection = model.backbone.shared_address_proj.weight.detach().float()
    rows = []
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "main_output",
            "main_state",
            "companion_output",
            "companion_state",
            "companion_gate",
            "address_source",
            "canonical_address",
            "companion_g",
            "companion_b",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete companion capture: {set(captured)}")
        main_output = captured["main_output"].float()
        companion_output = captured["companion_output"].float()
        companion_state = captured["companion_state"].float()
        main_state = captured["main_state"].float()
        address = captured["canonical_address"].float()
        output_board = companion_output.square().mean(dim=(1, 2, 3)).sqrt()
        state_board = companion_state.square().mean(dim=(1, 2, 3)).sqrt()
        address_board = address.square().mean(dim=(1, 2, 3)).sqrt()
        output_cosine = F.cosine_similarity(
            main_output.flatten(1),
            companion_output.flatten(1),
            dim=-1,
            eps=1e-8,
        )
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "companion_gate_abs_mean": float(
                    captured["companion_gate"].float().abs().mean().item()
                ),
                "active_gate_heads": int(
                    (
                        captured["companion_gate"].float().abs().flatten()
                        >= 1e-4
                    ).sum().item()
                ),
                "canonical_address_rms": float(_rms(address).item()),
                "canonical_address_token_std": float(
                    address.mean(dim=-1).std(dim=1, unbiased=False).mean().item()
                ),
                "canonical_address_board_std": float(
                    address_board.std(unbiased=False).item()
                ),
                "companion_output_rms": float(_rms(companion_output).item()),
                "companion_output_token_std": float(
                    companion_output.mean(dim=-1)
                    .std(dim=1, unbiased=False)
                    .mean()
                    .item()
                ),
                "companion_output_board_std": float(
                    output_board.std(unbiased=False).item()
                ),
                "companion_state_rms": float(_rms(companion_state).item()),
                "companion_state_board_std": float(
                    state_board.std(unbiased=False).item()
                ),
                "companion_to_main_state_rms": float(
                    (_rms(companion_state) / _rms(main_state).clamp_min(1e-8)).item()
                ),
                "main_companion_output_cosine_mean": float(
                    output_cosine.mean().item()
                ),
                "companion_decay_rms": float(
                    _rms(captured["companion_g"]).item()
                ),
                "companion_erase_rms": float(
                    _rms(captured["companion_b"]).item()
                ),
            }
        )
    return {
        "active_layers": len(rows),
        "active_gate_heads": sum(row["active_gate_heads"] for row in rows),
        "new_parameters": EXPECTED_NEW_PARAMETERS,
        "shared_projection_parameters": projection.numel(),
        "shared_projection_rms": float(_rms(projection).item()),
        "shared_projection_rank": int(torch.linalg.matrix_rank(projection).item()),
        "main_persistent_state_values_per_layer": EXPECTED_MAIN_STATE_VALUES,
        "companion_transient_state_values_per_layer": (
            EXPECTED_COMPANION_STATE_VALUES
        ),
        "official_scans_per_layer": 2,
        "companion_futureseed_routes": 0,
        "shared_across_layers_and_heads": True,
        "per_layer": rows,
    }
