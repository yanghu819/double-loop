from __future__ import annotations

import math
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


AUXILIARY_WEIGHT = 0.25
EXPECTED_PARAMETER_DELTA = 0
EXPECTED_STATE_DELTA = 0
EXPECTED_SCAN_DELTA = 0


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _unit_rms(tensor: torch.Tensor) -> torch.Tensor:
    denominator = tensor.float().square().mean(dim=-1, keepdim=True).sqrt()
    return tensor.float() / denominator.clamp_min(1e-6)


class ReceiverReadCreditBackbone(FutureSeedLMBackbone):
    """Native FutureSeed with a producer-only receiver-read training loss."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config=config)
        self._auxiliary_loss: Optional[torch.Tensor] = None
        self._last_credit_diagnostics: Optional[dict[str, Any]] = None

    @staticmethod
    def _receiver_query(
        mixer: ZoologyGDN2FutureSeedMixer,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer = mixer.layer
        with torch.no_grad(), torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            if layer.use_short_conv:
                query, _conv_state = layer.q_conv1d(
                    x=layer.q_proj(hidden_states),
                    cache=None,
                    output_final_state=False,
                )
            else:
                query = F.silu(layer.q_proj(hidden_states))
        query = rearrange(
            query,
            "b t (h d) -> b t h d",
            d=layer.head_k_dim,
        )
        if layer.num_v_heads > layer.num_heads:
            query = repeat(
                query,
                "b t h d -> b t (h g) d",
                g=layer.num_v_heads // layer.num_heads,
            )
        return F.normalize(query.float(), dim=-1) / math.sqrt(layer.head_k_dim)

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self._auxiliary_loss = None
            return super().layers_forward(hidden_states)

        scales = {
            float(layer.sequence_mixer.future_seed_scale)
            for layer in self.layers
            if isinstance(layer.sequence_mixer, ZoologyGDN2FutureSeedMixer)
        }
        if scales != {1.0}:
            raise RuntimeError("Receiver read credit requires native FutureSeed scale 1")

        residual = None
        terminal_state = None
        route_losses: list[torch.Tensor] = []
        route_rows: list[dict[str, float | int]] = []
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyGDN2FutureSeedMixer:
                raise TypeError("Receiver read credit requires the native GDN2 mixer")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            receiver_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            hidden_states, terminal_state = mixer.forward_with_state(
                receiver_input,
                initial_state=initial_state,
            )

            if initial_state is not None:
                query = self._receiver_query(mixer, receiver_input)
                inherited_read = torch.einsum(
                    "bthk,bhkv->bthv",
                    query,
                    initial_state.float(),
                )
                live_terminal_read = torch.einsum(
                    "bthk,bhkv->bthv",
                    query,
                    terminal_state.detach().float(),
                )
                normalized_inherited = _unit_rms(inherited_read)
                normalized_live = _unit_rms(live_terminal_read)
                route_loss = (
                    normalized_inherited - normalized_live
                ).square().mean()
                route_losses.append(route_loss)
                route_rows.append(
                    {
                        "receiving_layer": layer_index,
                        "unweighted_loss": float(route_loss.detach().item()),
                        "inherited_read_rms": float(
                            _rms(inherited_read.detach()).item()
                        ),
                        "live_terminal_read_rms": float(
                            _rms(live_terminal_read.detach()).item()
                        ),
                        "read_cosine": float(
                            F.cosine_similarity(
                                inherited_read.detach().float(),
                                live_terminal_read.detach().float(),
                                dim=-1,
                            ).mean().item()
                        ),
                        "query_board_std": float(
                            query.detach().float().square().mean(
                                dim=(-1, -2, -3)
                            ).sqrt().std(unbiased=False).item()
                        ),
                    }
                )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        if not route_losses:
            raise RuntimeError("Receiver read credit found no FutureSeed route")
        unweighted_loss = torch.stack(route_losses).mean()
        self._auxiliary_loss = AUXILIARY_WEIGHT * unweighted_loss
        self._last_credit_diagnostics = {
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "active_routes": len(route_rows),
            "unweighted_loss": float(unweighted_loss.detach().item()),
            "weighted_loss": float(self._auxiliary_loss.detach().item()),
            "routes": route_rows,
            "teacher_detached": True,
            "receiver_query_detached": True,
            "uses_targets": False,
            "inference_parameter_delta": EXPECTED_PARAMETER_DELTA,
            "inference_state_delta": EXPECTED_STATE_DELTA,
            "inference_scan_delta": EXPECTED_SCAN_DELTA,
        }

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._auxiliary_loss is None:
            raise RuntimeError("Receiver read credit was requested without a train forward")
        return self._auxiliary_loss

    def credit_diagnostics(self) -> dict[str, Any]:
        if self._last_credit_diagnostics is None:
            raise RuntimeError("Receiver read credit has not executed a train forward")
        return self._last_credit_diagnostics


class ReceiverReadCreditLanguageModel(nn.Module):
    """Parameter-identical FutureSeed LM with training-only read credit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = ReceiverReadCreditBackbone(config=config)
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


def receiver_read_credit_diagnostics(
    model: ReceiverReadCreditLanguageModel,
) -> dict[str, Any]:
    return model.backbone.credit_diagnostics()
