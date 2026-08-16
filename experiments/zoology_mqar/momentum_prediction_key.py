from __future__ import annotations

import importlib
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_STATE_VALUES_PER_LAYER,
    ZoologyMomentumDeltaFutureSeedMixer,
    load_external_momentum_layer,
    load_matched_parent_state as load_momentum_parent_state,
    momentum_futureseed_diagnostics,
)


MODEL_WIDTH = 128
MODEL_HEADS = 4
HEAD_DIM = 32
CONV_SIZE = 4
EXPECTED_PARAMETER_DELTA_VS_MOMENTUM = 2 * (
    MODEL_WIDTH * MODEL_WIDTH + MODEL_WIDTH * CONV_SIZE
)


class PredictionKeyOperation:
    """Supply one receiver-local prediction key to the native Momentum scan."""

    def __init__(
        self,
        operation,
        prediction_key: torch.Tensor,
        *,
        collect_diagnostics: bool,
    ) -> None:
        self.operation = operation
        self.prediction_key = prediction_key
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_stats: Optional[dict[str, Any]] = None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_alpha: torch.Tensor,
        log_mu: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = True,
        use_p_times_alpha: bool = True,
    ):
        if p is not None:
            raise RuntimeError("Prediction-key scope received an unexpected parent p")
        if cu_seqlens is not None:
            raise RuntimeError("P-GDN3-063 fixes equal-length inputs")
        if self.prediction_key.shape != k.shape:
            raise RuntimeError(
                f"Prediction-key geometry drifted: {self.prediction_key.shape} != {k.shape}"
            )
        if self.prediction_key.device != k.device or self.prediction_key.dtype != k.dtype:
            raise RuntimeError("Prediction-key device or dtype drifted")

        if self.collect_diagnostics:
            prediction = self.prediction_key.detach().float()
            owner = k.detach().float()
            delta = prediction - owner
            prediction_unit = F.normalize(prediction, dim=-1)
            owner_unit = F.normalize(owner, dim=-1)
            cosine = (prediction_unit * owner_unit).sum(dim=-1)
            board_rms = delta.square().mean(dim=(1, 2, 3)).sqrt()
            token_rms = delta.square().mean(dim=(0, 2, 3)).sqrt()
            head_rms = delta.square().mean(dim=(0, 1, 3)).sqrt()
            self.last_stats = {
                "relative_rms": float(
                    delta.square().mean().sqrt()
                    / owner.square().mean().sqrt().clamp_min(1e-8)
                ),
                "cosine_mean": float(cosine.mean()),
                "cosine_min": float(cosine.min()),
                "cosine_max": float(cosine.max()),
                "board_rms_std": float(board_rms.std(unbiased=False)),
                "token_rms_std": float(token_rms.std(unbiased=False)),
                "head_rms_std": float(head_rms.std(unbiased=False)),
                "prediction_rms": float(prediction.square().mean().sqrt()),
                "owner_rms": float(owner.square().mean().sqrt()),
            }

        return self.operation(
            q=q,
            k=k,
            v=v,
            p=self.prediction_key,
            log_alpha=log_alpha,
            log_mu=log_mu,
            beta=beta,
            eta=eta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_p_times_alpha=use_p_times_alpha,
        )


@contextmanager
def scoped_prediction_key(
    prediction_key: torch.Tensor,
    *,
    collect_diagnostics: bool,
) -> Iterator[PredictionKeyOperation]:
    layer_class = load_external_momentum_layer()
    layer_module = importlib.import_module(layer_class.__module__)
    original = layer_module.chunk_mode_rule
    if isinstance(original, PredictionKeyOperation):
        raise RuntimeError("Nested prediction-key scope is unsupported")
    wrapped = PredictionKeyOperation(
        original,
        prediction_key,
        collect_diagnostics=collect_diagnostics,
    )
    layer_module.chunk_mode_rule = wrapped
    try:
        yield wrapped
    finally:
        if layer_module.chunk_mode_rule is not wrapped:
            raise RuntimeError("Momentum operation changed inside prediction-key scope")
        layer_module.chunk_mode_rule = original


class ZoologyMomentumPredictionKeyFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum Delta with a distinct residual-prediction key and native owner K."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.layer.hidden_size != MODEL_WIDTH
            or self.layer.num_heads != MODEL_HEADS
            or self.layer.head_k_dim != HEAD_DIM
            or self.layer.conv_size != CONV_SIZE
        ):
            raise ValueError("P-GDN3-063 fixes D128/H4/K32/Conv4")
        self.prediction_key_proj = deepcopy(self.layer.k_proj)
        self.prediction_key_conv = deepcopy(self.layer.k_conv1d)
        self.last_prediction_stats: Optional[dict[str, Any]] = None

    def _prediction_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            prediction, _conv_state = self.prediction_key_conv(
                x=self.prediction_key_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=None,
            )
        return rearrange(
            prediction,
            "... (h d) -> ... h d",
            h=self.layer.num_heads,
            d=self.layer.head_k_dim,
        )

    def _record_prediction(self, operation: PredictionKeyOperation) -> None:
        self.last_prediction_stats = (
            None if operation.last_stats is None else dict(operation.last_stats)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        prediction_key = self._prediction_key(hidden_states)
        with scoped_prediction_key(
            prediction_key,
            collect_diagnostics=not self.training,
        ) as operation:
            output = super().forward(hidden_states)
        self._record_prediction(operation)
        return output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_key = self._prediction_key(hidden_states)
        with scoped_prediction_key(
            prediction_key,
            collect_diagnostics=not self.training,
        ) as operation:
            output, terminal_state = super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        self._record_prediction(operation)
        return output, terminal_state

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_STATE_VALUES_PER_LAYER


def load_matched_parent_state(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> None:
    load_momentum_parent_state(model, parent_state)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyMomentumPredictionKeyFutureSeedMixer
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two prediction-key mixers, got {len(mixers)}")
    tied_names = []
    for mixer in mixers:
        mixer.prediction_key_proj.load_state_dict(mixer.layer.k_proj.state_dict())
        mixer.prediction_key_conv.load_state_dict(mixer.layer.k_conv1d.state_dict())
        if any(
            not torch.equal(left, right)
            for left, right in zip(
                mixer.prediction_key_proj.state_dict().values(),
                mixer.layer.k_proj.state_dict().values(),
                strict=True,
            )
        ):
            raise RuntimeError("Prediction projection did not tie exactly to K")
        if any(
            not torch.equal(left, right)
            for left, right in zip(
                mixer.prediction_key_conv.state_dict().values(),
                mixer.layer.k_conv1d.state_dict().values(),
                strict=True,
            )
        ):
            raise RuntimeError("Prediction convolution did not tie exactly to K")
        tied_names.extend(
            [
                f"layer{mixer.layer_idx}.prediction_key_proj",
                f"layer{mixer.layer_idx}.prediction_key_conv",
            ]
        )
    model._prediction_key_parent_tie = {
        "exact": True,
        "paths": tied_names,
        "parameter_delta_vs_momentum": EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
    }


def momentum_prediction_key_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    momentum = momentum_futureseed_diagnostics(model)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyMomentumPredictionKeyFutureSeedMixer
    ]
    rows = [
        {
            "layer_idx": mixer.layer_idx,
            "prediction": mixer.last_prediction_stats,
        }
        for mixer in mixers
    ]
    return {
        "momentum": momentum,
        "prediction_layers": len(rows),
        "active_prediction_layers": sum(row["prediction"] is not None for row in rows),
        "parameter_delta_vs_momentum": EXPECTED_PARAMETER_DELTA_VS_MOMENTUM,
        "persistent_state_delta_vs_momentum": 0,
        "parent_tie": getattr(model, "_prediction_key_parent_tie", None),
        "per_layer_prediction": rows,
    }
