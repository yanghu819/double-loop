from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import FutureSeedLMBackbone
from experiments.zoology_mqar.gdn2_log_spd import (
    BoundedLogSPDAddressMetric,
    ZoologyLogSPDGDN2FutureSeedMixer,
)


def metric_read_pullback(
    terminal_state: torch.Tensor,
    *,
    producer_metric: BoundedLogSPDAddressMetric,
    receiver_metric: BoundedLogSPDAddressMetric,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pull producer state into the receiver metric's read coordinates."""
    if terminal_state.ndim != 4:
        raise ValueError(f"Expected [B,H,K,V] terminal state, got {terminal_state.shape}")
    with torch.autocast(device_type=terminal_state.device.type, enabled=False):
        producer = producer_metric.applied_matrix(torch.bfloat16)
        receiver = receiver_metric.applied_matrix(torch.bfloat16)
        pullback = torch.linalg.solve(receiver, producer)
        identity = torch.eye(
            pullback.shape[-1],
            device=pullback.device,
            dtype=pullback.dtype,
        )
        delta = pullback - identity
        state_fp32 = terminal_state.float()
        residual = torch.einsum("hij,bhjv->bhiv", delta, state_fp32)
        transported_fp32 = state_fp32 + residual
    transported = terminal_state + residual.to(dtype=terminal_state.dtype)

    producer_rms = state_fp32.square().mean(dim=(-1, -2)).sqrt()
    transported_rms = transported_fp32.square().mean(dim=(-1, -2)).sqrt()
    residual_rms = residual.square().mean(dim=(-1, -2)).sqrt()
    relative = residual_rms / producer_rms.clamp_min(1e-8)
    singular_values = torch.linalg.svdvals(pullback)
    condition = singular_values[..., 0] / singular_values[..., -1].clamp_min(1e-8)
    diagnostics = {
        "pullback": pullback.detach(),
        "delta": delta.detach(),
        "condition": condition.detach(),
        "producer_rms": producer_rms.detach(),
        "transported_rms": transported_rms.detach(),
        "residual_relative_rms": relative.detach(),
    }
    return transported, diagnostics


class MetricPullbackFutureSeedBackbone(FutureSeedLMBackbone):
    """Native FutureSeed with private-metric receiver-read pullback."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scales = {
            float(layer.sequence_mixer.future_seed_scale)
            for layer in self.layers
            if isinstance(
                layer.sequence_mixer,
                ZoologyLogSPDGDN2FutureSeedMixer,
            )
        }
        if scales != {1.0}:
            raise RuntimeError("Metric pullback requires FutureSeed scale exactly one")

        residual = None
        terminal_state: Optional[torch.Tensor] = None
        producer_mixer: Optional[ZoologyLogSPDGDN2FutureSeedMixer] = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyLogSPDGDN2FutureSeedMixer):
                raise TypeError("Metric pullback requires private Log-SPD GDN2 mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )

            if terminal_state is None:
                initial_state = None
                mixer.last_metric_pullback = None
            else:
                if producer_mixer is None:
                    raise RuntimeError("Missing producer metric for FutureSeed edge")
                transported, diagnostics = metric_read_pullback(
                    terminal_state,
                    producer_metric=producer_mixer.layer.address_metric,
                    receiver_metric=mixer.layer.address_metric,
                )
                mixer.last_metric_pullback = diagnostics
                initial_state = mixer.make_initial_state(transported)

            hidden_states, terminal_state = mixer.forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
            producer_mixer = mixer

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class MetricPullbackFutureSeedLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Metric pullback requires multiplier=1")

        self.backbone = MetricPullbackFutureSeedBackbone(config=config)
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
        state: Any = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        del state
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        if return_embeddings:
            return hidden_states
        return self.lm_head(hidden_states)

    def state_size(self, sequence_length: int) -> int:
        return _compute_state_size(self.backbone.layers, sequence_length)


def metric_pullback_diagnostics(model: nn.Module) -> dict[str, Any]:
    rows = []
    for layer_index, layer in enumerate(model.backbone.layers):
        mixer = layer.sequence_mixer
        diagnostics = getattr(mixer, "last_metric_pullback", None)
        if diagnostics is None:
            continue
        relative = diagnostics["residual_relative_rms"].float()
        producer_rms = diagnostics["producer_rms"].float()
        transported_rms = diagnostics["transported_rms"].float()
        rows.append(
            {
                "receiver_layer": layer_index,
                "pullback_delta_fro_mean": float(
                    diagnostics["delta"].float().square().sum(dim=(-1, -2)).sqrt().mean().item()
                ),
                "pullback_condition_max": float(diagnostics["condition"].float().amax().item()),
                "residual_relative_rms_mean": float(relative.mean().item()),
                "residual_relative_rms_head_std": float(relative.mean(dim=0).std().item()),
                "residual_relative_rms_board_std": float(relative.mean(dim=1).std().item()),
                "transported_to_producer_rms_max": float(
                    (transported_rms / producer_rms.clamp_min(1e-8)).amax().item()
                ),
                "finite": bool(
                    all(torch.isfinite(value).all() for value in diagnostics.values())
                ),
            }
        )
    return {
        "active_routes": len(rows),
        "new_parameters": 0,
        "new_recurrent_state": 0,
        "new_scans": 0,
        "per_receiver": rows,
    }
