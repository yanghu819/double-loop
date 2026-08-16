from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
from torch import nn

from fla.models.utils import Cache
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import FutureSeedLMBackbone
from experiments.zoology_mqar.momentum_futureseed import (
    MODEL_HEADS,
    HEAD_DIM,
    ZoologyMomentumDeltaFutureSeedMixer,
    momentum_futureseed_diagnostics,
)


PREFILL_TOKENS = 4
PREFILL_STREAMS = 3
PREFILL_PARAMETERS = PREFILL_STREAMS * MODEL_HEADS


class ZoologyMomentumConvPrefillFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Native Momentum FutureSeed plus receiver-native short-conv prefill."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.layer_idx > 0:
            self.conv_prefill_raw = nn.Parameter(
                torch.zeros(PREFILL_STREAMS, MODEL_HEADS)
            )
        else:
            self.register_parameter("conv_prefill_raw", None)
        self.prefill_calls = 0
        self.last_prefill_evidence_rms: Optional[torch.Tensor] = None
        self.last_prefill_evidence_board_std: Optional[torch.Tensor] = None
        self.last_prefill_raw_cache_rms: Optional[torch.Tensor] = None
        self.last_prefill_cache_rms: Optional[torch.Tensor] = None
        self.last_prefill_cache_board_std: Optional[torch.Tensor] = None
        self.last_prefill_mix_abs: Optional[torch.Tensor] = None

    def make_conv_prefill(
        self,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.conv_prefill_raw is None:
            raise RuntimeError("Only receiving Momentum layers own conv prefill")
        if evidence.ndim != 3 or evidence.shape[1:] != (PREFILL_TOKENS, 128):
            raise ValueError(
                "P-FS2-015 requires four D128 producer terminal tokens, got "
                f"{tuple(evidence.shape)}"
            )
        with torch.autocast(
            device_type=evidence.device.type,
            dtype=torch.bfloat16,
            enabled=evidence.is_cuda,
        ):
            projected = (
                self.layer.q_proj(evidence),
                self.layer.k_proj(evidence),
                self.layer.v_proj(evidence),
            )
            convolutions = (
                self.layer.q_conv1d,
                self.layer.k_conv1d,
                self.layer.v_conv1d,
            )
            raw_caches = tuple(
                convolution(x=value, output_final_state=True)[1]
                for convolution, value in zip(convolutions, projected)
            )
        if any(cache is None for cache in raw_caches):
            raise RuntimeError("Triton short convolution did not return final state")
        if any(cache.shape != (evidence.shape[0], 128, PREFILL_TOKENS) for cache in raw_caches):
            raise RuntimeError(
                f"Unexpected receiver conv-cache geometry: {[tuple(x.shape) for x in raw_caches]}"
            )

        mix = torch.tanh(self.conv_prefill_raw)
        mixed_caches = []
        for stream, cache in enumerate(raw_caches):
            gate = mix[stream].view(1, MODEL_HEADS, 1, 1)
            cache_heads = cache.view(
                evidence.shape[0], MODEL_HEADS, HEAD_DIM, PREFILL_TOKENS
            )
            mixed_caches.append(
                (cache_heads * gate.to(cache_heads.dtype)).reshape_as(cache)
            )

        evidence_rms = evidence.float().square().mean(dim=(-1, -2)).sqrt()
        raw_stack = torch.stack([cache.float() for cache in raw_caches])
        mixed_stack = torch.stack([cache.float() for cache in mixed_caches])
        mixed_board_rms = mixed_stack.square().mean(dim=(0, 2, 3)).sqrt()
        self.prefill_calls += 1
        self.last_prefill_evidence_rms = evidence_rms.mean().detach()
        self.last_prefill_evidence_board_std = evidence_rms.std(
            unbiased=False
        ).detach()
        self.last_prefill_raw_cache_rms = raw_stack.square().mean().sqrt().detach()
        self.last_prefill_cache_rms = mixed_stack.square().mean().sqrt().detach()
        self.last_prefill_cache_board_std = mixed_board_rms.std(
            unbiased=False
        ).detach()
        self.last_prefill_mix_abs = mix.abs().mean().detach()
        return tuple(mixed_caches)

    def forward_with_prefill(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        initial_conv_state: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = Cache()
        for index in range(self.layer_idx):
            cache.update(layer_idx=index, offset=0)
        if initial_state is not None or initial_conv_state is not None:
            cache.update(
                recurrent_state=initial_state,
                conv_state=(None, None, None)
                if initial_conv_state is None
                else initial_conv_state,
                layer_idx=self.layer_idx,
                offset=0,
            )
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                past_key_values=cache,
                use_cache=True,
            )
        terminal_state = cache[self.layer_idx]["recurrent_state"]
        if terminal_state is None:
            raise RuntimeError("Momentum layer did not return terminal [S,M]")
        state_rms = terminal_state[0].float().square().mean().sqrt()
        momentum_rms = terminal_state[1].float().square().mean().sqrt()
        self.last_state_rms = state_rms.detach()
        self.last_momentum_rms = momentum_rms.detach()
        self.last_momentum_to_state_rms = (
            momentum_rms / state_rms.clamp_min(1e-8)
        ).detach()
        return output.to(hidden_states.dtype), terminal_state


class MomentumConvPrefillBackbone(FutureSeedLMBackbone):
    """Pass terminal Momentum state and a receiver-native conv boundary."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        terminal_state = None
        prefill_evidence = None
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyMomentumConvPrefillFutureSeedMixer:
                raise TypeError("Conv-prefill backbone requires exact Momentum mixers")
            if mixer.future_seed_scale != 1.0:
                raise RuntimeError("Conv-prefill FutureSeed scale must remain one")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            mixer_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            initial_conv_state = (
                None
                if prefill_evidence is None
                else mixer.make_conv_prefill(prefill_evidence)
            )
            hidden_states, terminal_state = mixer.forward_with_prefill(
                mixer_input,
                initial_state=initial_state,
                initial_conv_state=initial_conv_state,
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)
            if layer_index + 1 < len(self.layers):
                prefill_evidence = hidden_states[:, -PREFILL_TOKENS:, :]

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class MomentumConvPrefillLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1 or config.n_layers != 2 or config.d_model != 128:
            raise ValueError("P-FS2-015 fixes D128/L2 and multiplier one")
        self.backbone = MomentumConvPrefillBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
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


def momentum_conv_prefill_diagnostics(model: nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    receiving = [mixer for mixer in mixers if mixer.conv_prefill_raw is not None]
    rows = []
    for mixer in receiving:
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "calls": mixer.prefill_calls,
                "mix_abs": None
                if mixer.last_prefill_mix_abs is None
                else float(mixer.last_prefill_mix_abs),
                "evidence_rms": None
                if mixer.last_prefill_evidence_rms is None
                else float(mixer.last_prefill_evidence_rms),
                "evidence_board_std": None
                if mixer.last_prefill_evidence_board_std is None
                else float(mixer.last_prefill_evidence_board_std),
                "raw_cache_rms": None
                if mixer.last_prefill_raw_cache_rms is None
                else float(mixer.last_prefill_raw_cache_rms),
                "mixed_cache_rms": None
                if mixer.last_prefill_cache_rms is None
                else float(mixer.last_prefill_cache_rms),
                "mixed_cache_board_std": None
                if mixer.last_prefill_cache_board_std is None
                else float(mixer.last_prefill_cache_board_std),
            }
        )
    return {
        **base,
        "mechanism": "receiver_native_terminal_hidden_conv_prefill",
        "prefill_tokens": PREFILL_TOKENS,
        "prefill_streams": PREFILL_STREAMS,
        "prefill_parameters": PREFILL_PARAMETERS,
        "persistent_state_delta": 0,
        "scan_delta": 0,
        "receiving_routes": len(rows),
        "active_receiving_routes": sum(row["calls"] > 0 for row in rows),
        "per_route": rows,
    }
