from __future__ import annotations

from functools import partial
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_committed_delta import capture_committed_edit
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


RIDGE_SCALE = 2.0**-8


class ZoologySurpriseRegressionGDN2FutureSeedMixer(
    ZoologyGDN2FutureSeedMixer
):
    """Official GDN2 with a receiver-native all-token regression seed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_regression_input: Optional[torch.Tensor] = None
        self.last_surprise_scores: Optional[torch.Tensor] = None
        self.last_regression_system: Optional[torch.Tensor] = None
        self.last_regression_keys: Optional[torch.Tensor] = None
        self.last_regression_payload: Optional[torch.Tensor] = None
        self.last_regression_weights: Optional[torch.Tensor] = None
        self.last_native_seed: Optional[torch.Tensor] = None
        self.last_bounded_delta: Optional[torch.Tensor] = None
        self.last_regression_diagnostics: dict[str, torch.Tensor] = {}
        self.collect_regression_diagnostics = False

    def forward_with_committed_edit(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with capture_committed_edit() as captured:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        if len(captured) != 1:
            raise RuntimeError(
                f"Expected one official committed-edit tensor, got {len(captured)}"
            )
        committed_edit = captured[0].detach()
        if committed_edit.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                "Official committed-edit shape does not match the producer tokens: "
                f"{committed_edit.shape} versus {hidden_states.shape}"
            )
        return output, terminal_state, committed_edit

    def regression_seed(
        self,
        receiver_input: torch.Tensor,
        committed_edit: torch.Tensor,
        native_terminal_state: torch.Tensor,
    ) -> torch.Tensor:
        if receiver_input.ndim != 3 or committed_edit.ndim != 4:
            raise ValueError("Expected evidence [B,L,D] and committed edit [B,L,H,V]")
        if receiver_input.shape[:2] != committed_edit.shape[:2]:
            raise ValueError("Evidence and committed edit must cover the same tokens")
        if receiver_input.shape[-1] != self.layer.hidden_size:
            raise ValueError("Canonical evidence width does not match the receiver")
        if committed_edit.shape[2] != self.layer.num_v_heads:
            raise ValueError("Committed-edit value-head count does not match the receiver")
        if committed_edit.shape[3] != self.layer.head_v_dim:
            raise ValueError("Committed-edit value width does not match the receiver")
        if native_terminal_state.shape != (
            receiver_input.shape[0],
            self.layer.num_v_heads,
            self.layer.head_k_dim,
            self.layer.head_v_dim,
        ):
            raise ValueError(
                "Native terminal state does not match the receiver state layout: "
                f"{native_terminal_state.shape}"
            )

        batch, tokens, _width = receiver_input.shape
        if tokens == 0:
            raise ValueError("Surprise regression requires at least one token")

        surprise = committed_edit.float().square().sum(
            dim=(-1, -2)
        ).sqrt().detach()
        if not torch.isfinite(surprise).all():
            raise RuntimeError("Committed-edit surprise contains non-finite values")
        surprise_sum = surprise.sum(dim=1, keepdim=True)
        if not torch.isfinite(surprise_sum).all() or (surprise_sum <= 0).any():
            raise RuntimeError("Committed-edit surprise must have positive finite mass")
        weights = (float(tokens) * surprise / surprise_sum).detach()

        with torch.autocast(
            device_type=receiver_input.device.type,
            dtype=torch.bfloat16,
            enabled=receiver_input.is_cuda,
        ):
            receiver_k, _ = self.layer.k_conv1d(
                x=self.layer.k_proj(receiver_input),
                cache=None,
                output_final_state=False,
            )
            receiver_v, _ = self.layer.v_conv1d(
                x=self.layer.v_proj(receiver_input),
                cache=None,
                output_final_state=False,
            )
            receiver_w = torch.sigmoid(self.layer.w_proj(receiver_input))

        if self.collect_regression_diagnostics:
            self.last_regression_input = receiver_input
            if receiver_input.requires_grad:
                receiver_input.retain_grad()

        receiver_k = receiver_k.unflatten(
            -1,
            (self.layer.num_heads, self.layer.head_k_dim),
        )
        receiver_v = receiver_v.unflatten(
            -1,
            (self.layer.num_v_heads, self.layer.head_v_dim),
        )
        receiver_w = receiver_w.unflatten(
            -1,
            (self.layer.num_v_heads, self.layer.head_v_dim),
        )
        if self.layer.num_v_heads > self.layer.num_heads:
            receiver_k = receiver_k.repeat_interleave(
                self.layer.num_v_heads // self.layer.num_heads,
                dim=2,
            )
        if receiver_k.shape[2] != receiver_v.shape[2]:
            raise RuntimeError("Receiver key and value head layouts are incompatible")

        with torch.autocast(
            device_type=receiver_input.device.type,
            enabled=False,
        ):
            keys = F.normalize(receiver_k.float(), p=2.0, dim=-1, eps=1e-6)
            payload = receiver_w.float() * receiver_v.float()
            gram = torch.einsum(
                "bl,blhk,blhj->bhkj",
                weights.float(),
                keys,
                keys,
            )
            gram = 0.5 * (gram + gram.transpose(-1, -2))
            native_seed = self.make_initial_state(native_terminal_state)
            native_prediction = torch.einsum(
                "blhk,bhkv->blhv", keys, native_seed.float()
            )
            residual_target = payload - native_prediction
            cross_covariance = torch.einsum(
                "bl,blhk,blhv->bhkv",
                weights.float(),
                keys,
                residual_target,
            )
            trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            ridge = (
                RIDGE_SCALE
                * (trace / float(self.layer.head_k_dim) + 1e-6)
            ).detach()
            identity = torch.eye(
                self.layer.head_k_dim,
                device=gram.device,
                dtype=gram.dtype,
            )
            system = gram + ridge.unsqueeze(-1).unsqueeze(-1) * identity
            factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
            if not torch.equal(info, torch.zeros_like(info)):
                failures = int((info != 0).sum().item())
                raise RuntimeError(
                    f"Surprise-regression Cholesky failed for {failures} board-heads"
                )
            regression_delta = torch.cholesky_solve(cross_covariance, factor)

        if not torch.isfinite(regression_delta).all():
            raise RuntimeError("Surprise-regression delta contains non-finite values")
        native_head_rms = native_seed.float().square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        delta_head_rms = regression_delta.square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        residual_scale = (native_head_rms / delta_head_rms).clamp(max=1.0)
        bounded_delta = regression_delta * residual_scale
        seed = native_seed.float() + bounded_delta
        seed = seed.to(dtype=native_seed.dtype)

        if self.collect_regression_diagnostics:
            with torch.no_grad():
                raw_prediction = torch.einsum(
                    "blhk,bhkv->blhv", keys, regression_delta
                )
                raw_weighted_write_fit_error = (
                    weights.float().unsqueeze(-1).unsqueeze(-1)
                    * (raw_prediction - residual_target).square()
                ).sum() / (
                    weights.float().sum()
                    * float(payload.shape[2] * payload.shape[3])
                ).clamp_min(1e-12)
                candidate_prediction = torch.einsum(
                    "blhk,bhkv->blhv", keys, seed.float()
                )
                weighted_denominator = (
                    weights.float().sum()
                    * float(payload.shape[2] * payload.shape[3])
                ).clamp_min(1e-12)
                candidate_seed_write_fit_mse = (
                    weights.float().unsqueeze(-1).unsqueeze(-1)
                    * (candidate_prediction - payload).square()
                ).sum() / weighted_denominator
                native_seed_write_fit_mse = (
                    weights.float().unsqueeze(-1).unsqueeze(-1)
                    * (native_prediction - payload).square()
                ).sum() / weighted_denominator
                solve_residual = (
                    torch.matmul(system, regression_delta) - cross_covariance
                ).norm(dim=(-2, -1)) / cross_covariance.norm(
                    dim=(-2, -1)
                ).clamp_min(1e-12)
                normalized_weights = weights / float(tokens)
                weight_entropy = -(
                    normalized_weights
                    * normalized_weights.clamp_min(1e-12).log()
                ).sum(dim=1)
                effective_tokens = weight_entropy.exp()
                raw_state_board_rms = regression_delta.square().mean(
                    dim=(-1, -2, -3)
                ).sqrt()
                seed_board_rms = seed.float().square().mean(
                    dim=(-1, -2, -3)
                ).sqrt()
                native_seed_board_rms = native_seed.float().square().mean(
                    dim=(-1, -2, -3)
                ).sqrt()
                self.last_surprise_scores = surprise
                self.last_regression_system = system.detach()
                self.last_regression_keys = keys.detach()
                self.last_regression_payload = payload.detach()
                self.last_regression_weights = weights.detach()
                self.last_native_seed = native_seed.detach()
                self.last_bounded_delta = bounded_delta.detach()
                self.last_regression_diagnostics = {
                    "committed_edit_rms": committed_edit.float().square().mean().sqrt(),
                    "surprise_mean": surprise.mean(),
                    "surprise_token_std": surprise.std(unbiased=False),
                    "surprise_board_std": surprise.mean(dim=1).std(unbiased=False),
                    "weight_max": weights.max(),
                    "weight_token_std": weights.std(unbiased=False),
                    "effective_tokens_mean": effective_tokens.mean(),
                    "ridge_mean": ridge.mean(),
                    "ridge_min": ridge.min(),
                    "ridge_max": ridge.max(),
                    "receiver_key_unit_error": (keys.norm(dim=-1) - 1.0).abs().max(),
                    "receiver_payload_rms": payload.square().mean().sqrt(),
                    "raw_weighted_receiver_write_fit_mse": (
                        raw_weighted_write_fit_error
                    ),
                    "weighted_candidate_seed_write_fit_mse": (
                        candidate_seed_write_fit_mse
                    ),
                    "weighted_native_seed_write_fit_mse": (
                        native_seed_write_fit_mse
                    ),
                    "weighted_seed_write_fit_mse_ratio": (
                        candidate_seed_write_fit_mse
                        / native_seed_write_fit_mse.clamp_min(1e-12)
                    ),
                    "solve_relative_residual": solve_residual.max(),
                    "raw_regression_state_rms": raw_state_board_rms.mean(),
                    "raw_regression_state_board_std": raw_state_board_rms.std(
                        unbiased=False
                    ),
                    "seed_rms": seed_board_rms.mean(),
                    "seed_board_std": seed_board_rms.std(unbiased=False),
                    "seed_abs_max": seed.float().abs().max(),
                    "native_seed_rms": native_seed_board_rms.mean(),
                    "bounded_residual_rms": bounded_delta.square().mean().sqrt(),
                    "bounded_residual_to_native_head_rms_max": (
                        bounded_delta.square().mean(dim=(-1, -2)).sqrt()
                        / native_seed.float().square().mean(dim=(-1, -2)).sqrt()
                        .clamp_min(1e-6)
                    ).max(),
                    "residual_scale_mean": residual_scale.mean(),
                    "residual_scale_min": residual_scale.min(),
                    "candidate_to_native_seed_rms_ratio": (
                        seed_board_rms.mean()
                        / native_seed_board_rms.mean().clamp_min(1e-12)
                    ),
                    "candidate_to_native_head_rms_ratio_max": (
                        seed.float().square().mean(dim=(-1, -2)).sqrt()
                        / native_seed.float().square().mean(dim=(-1, -2)).sqrt()
                        .clamp_min(1e-6)
                    ).max(),
                }
        return seed


class SurpriseRegressionFutureSeedLMBackbone(FutureSeedLMBackbone):
    """Pass all producer evidence through a receiver-native regression seed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.force_native_future_seed = False

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        committed_edit = None
        native_terminal_state = None
        for layer_index, layer in enumerate(self.layers):
            mixer = layer.sequence_mixer
            if not isinstance(
                mixer,
                ZoologySurpriseRegressionGDN2FutureSeedMixer,
            ):
                raise TypeError(
                    "Surprise-regression backbone requires strict regression mixers"
                )
            if mixer.future_seed_scale != 1.0:
                raise RuntimeError(
                    "Surprise-regression FutureSeed scale must remain exactly one"
                )

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            mixer_input = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if committed_edit is None or native_terminal_state is None
                else (
                    mixer.make_initial_state(native_terminal_state)
                    if self.force_native_future_seed
                    else mixer.regression_seed(
                        mixer_input,
                        committed_edit,
                        native_terminal_state,
                    )
                )
            )
            if layer_index + 1 < len(self.layers):
                hidden_states, native_terminal_state, committed_edit = (
                    mixer.forward_with_committed_edit(
                        mixer_input,
                        initial_state=initial_state,
                    )
                )
            else:
                hidden_states, _terminal_state = mixer.forward_with_state(
                    mixer_input,
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


class SurpriseRegressionFutureSeedLanguageModel(nn.Module):
    """Zoology LM using receiver-native surprise-regression FutureSeed."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Surprise-regression model requires multiplier=1")

        self.backbone = SurpriseRegressionFutureSeedLMBackbone(config=config)
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


def surprise_regression_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologySurpriseRegressionGDN2FutureSeedMixer,
        )
    ]
    if len(mixers) < 2:
        raise RuntimeError(
            f"Expected at least two surprise-regression mixers, got {len(mixers)}"
        )
    receivers = [mixer for mixer in mixers if mixer.last_regression_diagnostics]
    if len(receivers) != len(mixers) - 1:
        raise RuntimeError("Surprise-regression diagnostics were not populated")

    per_receiver = []
    for mixer in receivers:
        if mixer.last_regression_system is None:
            raise RuntimeError("Regression system was not retained for diagnostics")
        singular_values = torch.linalg.svdvals(mixer.last_regression_system.float())
        condition = singular_values[..., 0] / singular_values[..., -1].clamp_min(
            1e-12
        )
        row = {
            "layer_idx": mixer.layer_idx,
            "condition_mean": float(condition.mean().item()),
            "condition_max": float(condition.max().item()),
            **{
                name: float(value.item())
                for name, value in mixer.last_regression_diagnostics.items()
            },
        }
        per_receiver.append(row)

    return {
        "mechanism": "receiver_native_surprise_regression",
        "ridge_scale": RIDGE_SCALE,
        "new_parameters": 0,
        "persistent_state_delta": 0,
        "main_official_scans": len(mixers),
        "additional_official_scans": 0,
        "receiver_routes": len(receivers),
        "admission": "all_tokens_weighted_by_stop_gradient_committed_edit_norm",
        "receiver_projection": "existing_causal_conv_k_v_and_pointwise_w",
        "seed_update": "native_seed_plus_per_board_head_rms_bounded_regression_residual",
        "linear_system": "fp32_cholesky_no_fallback",
        "per_receiver": per_receiver,
    }
