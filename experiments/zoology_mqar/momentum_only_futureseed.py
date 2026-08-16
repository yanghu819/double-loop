from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
from torch import nn

from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import FutureSeedLMBackbone
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_STATE_VALUES_PER_LAYER,
    MODEL_HEADS,
    HEAD_DIM,
    ZoologyMomentumDeltaFutureSeedMixer,
    momentum_futureseed_diagnostics,
)


MOMENTUM_PAYLOAD_VALUES_PER_ROUTE = MODEL_HEADS * HEAD_DIM * HEAD_DIM
FULL_STATE_VALUES_PER_ROUTE = EXPECTED_STATE_VALUES_PER_LAYER


class ZoologyMomentumOnlyFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum DeltaNet whose cross-layer FutureSeed carries only M."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pack_calls = 0
        self.last_payload_rms: Optional[torch.Tensor] = None
        self.last_seed_state_rms: Optional[torch.Tensor] = None
        self.last_seed_momentum_rms: Optional[torch.Tensor] = None

    def pack_future_seed(self, terminal_state: torch.Tensor) -> torch.Tensor:
        if terminal_state.ndim != 5 or terminal_state.shape[0] != 2:
            raise ValueError(
                "Momentum-only FutureSeed expects terminal [S,M], got "
                f"{tuple(terminal_state.shape)}"
            )
        payload = terminal_state[1]
        if payload.shape[1:] != (MODEL_HEADS, HEAD_DIM, HEAD_DIM):
            raise ValueError(f"Unexpected Momentum payload: {tuple(payload.shape)}")
        self.pack_calls += 1
        self.last_payload_rms = payload.float().square().mean().sqrt().detach()
        return payload

    def make_initial_state(self, payload: torch.Tensor) -> torch.Tensor:
        if payload.ndim != 4 or payload.shape[1:] != (
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
        ):
            raise ValueError(
                "Momentum-only FutureSeed expects [B,H,K,V], got "
                f"{tuple(payload.shape)}"
            )
        momentum_seed = super().make_initial_state(payload)
        initial_state = torch.stack(
            (torch.zeros_like(momentum_seed), momentum_seed),
            dim=0,
        )
        self.last_seed_state_rms = (
            initial_state[0].float().square().mean().sqrt().detach()
        )
        self.last_seed_momentum_rms = (
            initial_state[1].float().square().mean().sqrt().detach()
        )
        return initial_state


class MomentumOnlyFutureSeedBackbone(FutureSeedLMBackbone):
    """Carry the producer M plane and reconstruct [0,M] in the receiver."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        payload = None
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyMomentumOnlyFutureSeedMixer:
                raise TypeError("Momentum-only backbone requires exact mixers")
            if mixer.future_seed_scale != 1.0:
                raise RuntimeError("Momentum-only FutureSeed scale must remain one")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            mixer_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None if payload is None else mixer.make_initial_state(payload)
            )
            hidden_states, terminal_state = mixer.forward_with_state(
                mixer_input,
                initial_state=initial_state,
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)
            if layer_index + 1 < len(self.layers):
                payload = mixer.pack_future_seed(terminal_state)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class MomentumOnlyFutureSeedLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1 or config.n_layers != 2 or config.d_model != 128:
            raise ValueError("P-FS2-016 fixes D128/L2 and multiplier one")
        self.backbone = MomentumOnlyFutureSeedBackbone(config=config)
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


def momentum_only_futureseed_diagnostics(model: nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    rows = []
    for mixer in mixers:
        if type(mixer) is not ZoologyMomentumOnlyFutureSeedMixer:
            raise TypeError("Diagnostics require exact Momentum-only mixers")
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "pack_calls": mixer.pack_calls,
                "payload_rms": (
                    None
                    if mixer.last_payload_rms is None
                    else float(mixer.last_payload_rms)
                ),
                "seed_state_rms": (
                    None
                    if mixer.last_seed_state_rms is None
                    else float(mixer.last_seed_state_rms)
                ),
                "seed_momentum_rms": (
                    None
                    if mixer.last_seed_momentum_rms is None
                    else float(mixer.last_seed_momentum_rms)
                ),
            }
        )
    return {
        **base,
        "mechanism": "momentum_only_cross_layer_futureseed",
        "full_state_values_per_route": FULL_STATE_VALUES_PER_ROUTE,
        "transported_values_per_route": MOMENTUM_PAYLOAD_VALUES_PER_ROUTE,
        "transport_ratio": (
            MOMENTUM_PAYLOAD_VALUES_PER_ROUTE / FULL_STATE_VALUES_PER_ROUTE
        ),
        "parameter_delta_vs_p059": 0,
        "persistent_state_delta_vs_p059": 0,
        "scan_delta_vs_p059": 0,
        "per_route": rows,
    }
