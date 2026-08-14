from __future__ import annotations

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


AUXILIARY_WEIGHT = 0.10
EXPECTED_PARAMETER_DELTA = 0
EXPECTED_STATE_DELTA = 0
EXPECTED_INFERENCE_SCAN_DELTA = 0
EXPECTED_TRAINING_CONV_DELTA = 2


def _spectrum_terms(
    key: torch.Tensor,
    *,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    key = rearrange(key, "b t (h k) -> b t h k", k=head_dim)
    key = F.normalize(key.float(), dim=-1, eps=1e-6)
    covariance = torch.einsum("bthk,bthl->bhkl", key, key)
    covariance = covariance * (head_dim / key.shape[1])
    identity = torch.eye(
        head_dim,
        device=covariance.device,
        dtype=covariance.dtype,
    )
    error = covariance - identity
    loss = error.square().mean()

    detached = covariance.detach()
    eigenvalues = torch.linalg.eigvalsh(detached).clamp_min(0)
    probability = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    )
    diagonal = torch.diagonal(detached, dim1=-2, dim2=-1)
    off_diagonal = detached - torch.diag_embed(diagonal)
    return loss, {
        "loss": float(loss.detach().item()),
        "effective_rank_mean": float(effective_rank.mean().item()),
        "effective_rank_min": float(effective_rank.min().item()),
        "anisotropy_mean": float(
            (eigenvalues[..., -1] / eigenvalues.mean(dim=-1).clamp_min(1e-12))
            .mean()
            .item()
        ),
        "minimum_eigenvalue_mean": float(eigenvalues[..., 0].mean().item()),
        "off_diagonal_rms": float(off_diagonal.square().mean().sqrt().item()),
        "board_loss_std": float(
            error.square().mean(dim=(-1, -2)).std(unbiased=False).item()
        ),
    }


def _training_key(
    mixer: ZoologyGDN2FutureSeedMixer,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
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


class CoherentKeySpectrumBackbone(FutureSeedLMBackbone):
    """Native FutureSeed with training-only coherent-key spectrum credit."""

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
                raise TypeError("Key-spectrum credit requires native GDN2 mixers")
            if float(mixer.future_seed_scale) != 1.0:
                raise RuntimeError("Key-spectrum credit requires native FutureSeed")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            receiver_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )

            key = _training_key(mixer, receiver_input)
            layer_loss, row = _spectrum_terms(
                key,
                head_dim=mixer.layer.head_k_dim,
            )
            losses.append(layer_loss)
            rows.append({"layer": layer_index, **row})

            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            hidden_states, terminal_state = mixer.forward_with_state(
                receiver_input,
                initial_state=initial_state,
            )

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
            "active_layers": len(rows),
            "unweighted_loss": float(unweighted.detach().item()),
            "weighted_loss": float(self._auxiliary_loss.detach().item()),
            "layers": rows,
            "input_detached": True,
            "uses_targets": False,
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
            raise RuntimeError("Key-spectrum credit requires a training forward")
        return self._auxiliary_loss

    def spectrum_diagnostics(self) -> dict[str, Any]:
        if self._last_diagnostics is None:
            raise RuntimeError("Key-spectrum credit has not run")
        return self._last_diagnostics


class CoherentKeySpectrumLanguageModel(nn.Module):
    """Parameter-identical native FutureSeed LM with training-only key credit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = CoherentKeySpectrumBackbone(config=config)
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


def coherent_key_spectrum_diagnostics(
    model: CoherentKeySpectrumLanguageModel,
) -> dict[str, Any]:
    return model.backbone.spectrum_diagnostics()
