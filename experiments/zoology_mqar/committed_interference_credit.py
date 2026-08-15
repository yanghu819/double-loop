from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


AUXILIARY_WEIGHT = 0.03
CAUSAL_WINDOW = 128
EXPECTED_PARAMETER_DELTA = 0
EXPECTED_STATE_DELTA = 0
EXPECTED_INFERENCE_SCAN_DELTA = 0
EXPECTED_TRAINING_CONV_DELTA = 2


def _training_key(
    mixer: ZoologyGDN2FutureSeedMixer,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Recompute the native post-convolution K with gradients only into K."""

    layer = mixer.layer
    with torch.autocast(
        device_type=hidden_states.device.type,
        dtype=torch.bfloat16,
        enabled=hidden_states.is_cuda,
    ):
        if layer.use_short_conv:
            key, _conv_state = layer.k_conv1d(
                x=layer.k_proj(hidden_states.detach()),
                cache=None,
                output_final_state=False,
            )
        else:
            key = F.silu(layer.k_proj(hidden_states.detach()))
    if layer.num_v_heads > layer.num_heads:
        key = rearrange(key, "b t (h k) -> b t h k", k=layer.head_k_dim)
        key = repeat(
            key,
            "b t h k -> b t (h g) k",
            g=layer.num_v_heads // layer.num_heads,
        )
        key = rearrange(key, "b t h k -> b t (h k)")
    return key


def committed_interference_terms(
    key: torch.Tensor,
    committed_edit: torch.Tensor,
    *,
    head_dim: int,
    causal_window: int = CAUSAL_WINDOW,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize high-surprise writes that collide with recent write addresses.

    The exact official committed edit supplies a detached importance weight. The
    only trainable path is through normalized native K. A finite causal window
    avoids turning the objective into the global whitening loss closed by P045.
    """

    if key.ndim != 3 or committed_edit.ndim != 4:
        raise ValueError("Expected key [B,T,H*K] and committed edit [B,T,H,V]")
    if key.shape[:2] != committed_edit.shape[:2]:
        raise ValueError("Key and committed-edit token geometry differs")
    if causal_window <= 0 or causal_window >= key.shape[1]:
        raise ValueError("Causal window must be in [1, sequence_length)")

    key = rearrange(key, "b t (h k) -> b t h k", k=head_dim)
    if key.shape[2] != committed_edit.shape[2]:
        raise ValueError("Key and committed-edit head geometry differs")
    key = F.normalize(key.float(), dim=-1, eps=1e-6)

    surprise = committed_edit.detach().float().square().mean(dim=-1).sqrt()
    prefix_count = torch.arange(
        1,
        surprise.shape[1] + 1,
        device=surprise.device,
        dtype=surprise.dtype,
    ).view(1, -1, 1)
    causal_surprise_mean = (surprise.cumsum(dim=1) / prefix_count).clamp_min(1e-6)
    weight = surprise / causal_surprise_mean

    outer = key.unsqueeze(-1) * key.unsqueeze(-2)
    weighted_outer = outer * weight.unsqueeze(-1).unsqueeze(-1)
    cumulative_gram = weighted_outer.cumsum(dim=1)
    zero_gram = cumulative_gram.new_zeros(
        cumulative_gram.shape[0],
        1,
        cumulative_gram.shape[2],
        cumulative_gram.shape[3],
        cumulative_gram.shape[4],
    )
    before = torch.cat((zero_gram, cumulative_gram[:, :-1]), dim=1)
    older = torch.zeros_like(before)
    older[:, causal_window + 1 :] = cumulative_gram[:, : -(causal_window + 1)]
    local_gram = before - older

    cumulative_mass = weight.cumsum(dim=1)
    zero_mass = cumulative_mass.new_zeros(
        cumulative_mass.shape[0], 1, cumulative_mass.shape[2]
    )
    mass_before = torch.cat((zero_mass, cumulative_mass[:, :-1]), dim=1)
    mass_older = torch.zeros_like(mass_before)
    mass_older[:, causal_window + 1 :] = cumulative_mass[
        :, : -(causal_window + 1)
    ]
    local_mass = mass_before - mass_older

    overlap = torch.einsum(
        "bthk,bthkl,bthl->bth",
        key,
        local_gram,
        key,
    )
    valid = local_mass > 1e-6
    normalized_overlap = torch.where(
        valid,
        overlap / local_mass.clamp_min(1e-6),
        torch.zeros_like(overlap),
    )
    relative_interference = head_dim * normalized_overlap
    excess = F.relu(relative_interference - 1.0)
    weighted_excess = weight * excess
    loss = weighted_excess[valid].mean()

    with torch.no_grad():
        valid_interference = relative_interference[valid]
        valid_weight = weight[valid]
        centered_weight = valid_weight - valid_weight.mean()
        centered_interference = valid_interference - valid_interference.mean()
        correlation = (
            (centered_weight * centered_interference).mean()
            / (
                centered_weight.square().mean().sqrt()
                * centered_interference.square().mean().sqrt()
            ).clamp_min(1e-8)
        )
        board_surprise = surprise.mean(dim=(1, 2))
        diagnostics = {
            "loss": float(loss.detach().item()),
            "committed_edit_rms": float(
                committed_edit.detach().float().square().mean().sqrt().item()
            ),
            "surprise_mean": float(surprise.mean().item()),
            "surprise_cv": float(
                (surprise.std(unbiased=False) / surprise.mean().clamp_min(1e-8)).item()
            ),
            "surprise_board_std": float(board_surprise.std(unbiased=False).item()),
            "relative_interference_mean": float(valid_interference.mean().item()),
            "above_random_fraction": float((valid_interference > 1.0).float().mean().item()),
            "weighted_excess_mean": float(weighted_excess[valid].mean().item()),
            "surprise_interference_correlation": float(correlation.item()),
            "local_mass_mean": float(local_mass[valid].mean().item()),
            "valid_fraction": float(valid.float().mean().item()),
        }
    return loss, diagnostics


class CommittedInterferenceBackbone(FutureSeedLMBackbone):
    """Native FutureSeed with training-only committed-edit interference credit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config=config)
        self._auxiliary_loss: Optional[torch.Tensor] = None
        self._last_diagnostics: Optional[dict[str, Any]] = None

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self._auxiliary_loss = None
            return super().layers_forward(hidden_states)

        residual = None
        terminal_state = None
        losses: list[torch.Tensor] = []
        rows: list[dict[str, float | int]] = []
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyGDN2FutureSeedMixer:
                raise TypeError("Committed-interference credit requires native GDN2")
            if float(mixer.future_seed_scale) != 1.0:
                raise RuntimeError("Committed-interference credit requires native FutureSeed")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            receiver_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            key = _training_key(mixer, receiver_input)

            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            with capture_committed_edit() as captured:
                hidden_states, terminal_state = mixer.forward_with_state(
                    receiver_input,
                    initial_state=initial_state,
                )
            if len(captured) != 1:
                raise RuntimeError(
                    f"Expected one official committed edit, got {len(captured)}"
                )
            committed_edit = captured[0].detach()
            layer_loss, row = committed_interference_terms(
                key,
                committed_edit,
                head_dim=mixer.layer.head_k_dim,
            )
            losses.append(layer_loss)
            rows.append({"layer": layer_index, **row})

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        unweighted = torch.stack(losses).mean()
        self._auxiliary_loss = AUXILIARY_WEIGHT * unweighted
        self._last_diagnostics = {
            "auxiliary_weight": AUXILIARY_WEIGHT,
            "causal_window": CAUSAL_WINDOW,
            "active_layers": len(rows),
            "unweighted_loss": float(unweighted.detach().item()),
            "weighted_loss": float(self._auxiliary_loss.detach().item()),
            "layers": rows,
            "input_detached": True,
            "committed_edit_detached": True,
            "uses_targets": False,
            "random_interference_floor": 1.0,
            "inference_parameter_delta": EXPECTED_PARAMETER_DELTA,
            "inference_state_delta": EXPECTED_STATE_DELTA,
            "inference_scan_delta": EXPECTED_INFERENCE_SCAN_DELTA,
            "training_short_conv_delta": EXPECTED_TRAINING_CONV_DELTA,
        }

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._auxiliary_loss is None:
            raise RuntimeError("Committed-interference credit requires a training forward")
        return self._auxiliary_loss

    def interference_diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics is None:
            raise RuntimeError("Committed-interference credit has not run")
        return self._last_diagnostics


class CommittedInterferenceLanguageModel(nn.Module):
    """Parameter-identical native FutureSeed LM with causal interference credit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = CommittedInterferenceBackbone(config=config)
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


def committed_interference_diagnostics(
    model: CommittedInterferenceLanguageModel,
) -> dict[str, Any]:
    return model.backbone.interference_diagnostics()
