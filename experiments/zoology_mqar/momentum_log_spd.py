from __future__ import annotations

from typing import Any, Optional

import torch
from einops import rearrange

from experiments.zoology_mqar.gdn2_log_spd import BoundedLogSPDAddressMetric
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_STATE_VALUES_PER_LAYER,
    ZoologyMomentumDeltaFutureSeedMixer,
    momentum_futureseed_diagnostics,
)


MODEL_HEADS = 4
HEAD_DIM = 32
EXPECTED_METRIC_PARAMETERS_PER_LAYER = MODEL_HEADS * (
    HEAD_DIM * (HEAD_DIM + 1) // 2 - 1
)
EXPECTED_PARAMETER_DELTA_VS_MOMENTUM = 2 * EXPECTED_METRIC_PARAMETERS_PER_LAYER


class ZoologyMomentumLogSPDFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum DeltaNet with one coherent bounded Q/K metric per layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.layer.num_heads != MODEL_HEADS or self.layer.head_k_dim != HEAD_DIM:
            raise ValueError("P-GDN3-060 fixes H4/K32")
        self.address_metric = BoundedLogSPDAddressMetric(
            num_heads=self.layer.num_heads,
            head_dim=self.layer.head_k_dim,
        )
        self._address_delta: Optional[torch.Tensor] = None
        self._q_hook_handle = self.layer.q_conv1d.register_forward_hook(
            self._transform_short_conv_output
        )
        self._k_hook_handle = self.layer.k_conv1d.register_forward_hook(
            self._transform_short_conv_output
        )

    def _transform_short_conv_output(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._address_delta is None:
            raise RuntimeError("Address metric hook escaped the recurrent forward")
        tensor, conv_state = output
        shaped = rearrange(
            tensor,
            "... (h d) -> ... h d",
            h=self.layer.num_heads,
            d=self.layer.head_k_dim,
        )
        transformed = self.address_metric.transform_with_delta(
            shaped,
            self._address_delta,
        )
        return rearrange(transformed, "... h d -> ... (h d)"), conv_state

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._address_delta is not None:
            raise RuntimeError("Nested Momentum Log-SPD forward is unsupported")
        identity = torch.eye(
            self.layer.head_k_dim,
            device=self.address_metric.raw.device,
            dtype=torch.float32,
        )
        self._address_delta = self.address_metric.matrix() - identity
        try:
            return super().forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
        finally:
            self._address_delta = None

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_STATE_VALUES_PER_LAYER


def momentum_log_spd_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer) is ZoologyMomentumLogSPDFutureSeedMixer
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two Momentum Log-SPD mixers, got {len(mixers)}")
    return {
        "momentum": momentum_futureseed_diagnostics(model),
        "active_metric_layers": len(mixers),
        "parameter_delta_vs_momentum": sum(
            mixer.address_metric.raw.numel() for mixer in mixers
        ),
        "per_layer_metric": [
            mixer.address_metric.diagnostics() for mixer in mixers
        ],
    }
