from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from experiments.zoology_mqar.momentum_futureseed import (
    ZoologyMomentumDeltaFutureSeedMixer,
    momentum_futureseed_diagnostics,
)


REMOVED_QK_CONV_PARAMETERS_PER_LAYER = 1_024
REMOVED_QK_CONV_PARAMETERS_TOTAL = 2_048


class PointwiseSiluAddress(nn.Module):
    """Keep the parent's address nonlinearity without temporal convolution."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.calls = 0
        self.last_input_rms: Optional[torch.Tensor] = None
        self.last_output_rms: Optional[torch.Tensor] = None
        self.last_token_std: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None or mask is not None or cache is not None:
            raise RuntimeError("P-GDN3-069 fixes a stateless pointwise address path")
        if cu_seqlens is not None:
            raise RuntimeError("P-GDN3-069 fixes equal-length MQAR inputs")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Address width drifted: {x.shape[-1]}")
        del output_final_state
        output = F.silu(x)
        self.calls += 1
        self.last_input_rms = x.detach().float().square().mean().sqrt()
        self.last_output_rms = output.detach().float().square().mean().sqrt()
        token_rms = output.detach().float().square().mean(dim=-1).sqrt()
        self.last_token_std = token_rms.std(unbiased=False)
        return output, None


class ZoologyMomentumAddressDeblurFutureSeedMixer(
    ZoologyMomentumDeltaFutureSeedMixer
):
    """Momentum DeltaNet with pointwise Q/K addresses and convolved payload."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        q_width = int(self.layer.q_proj.out_features)
        k_width = int(self.layer.k_proj.out_features)
        if q_width != k_width:
            raise ValueError("P-GDN3-069 requires matched Q/K address widths")
        q_parameters = sum(p.numel() for p in self.layer.q_conv1d.parameters())
        k_parameters = sum(p.numel() for p in self.layer.k_conv1d.parameters())
        if q_parameters + k_parameters != REMOVED_QK_CONV_PARAMETERS_PER_LAYER:
            raise RuntimeError(
                "Unexpected parent Q/K convolution parameter count: "
                f"{q_parameters + k_parameters}"
            )
        self.layer.q_conv1d = PointwiseSiluAddress(q_width)
        self.layer.k_conv1d = PointwiseSiluAddress(k_width)


def momentum_address_deblur_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    base = momentum_futureseed_diagnostics(model)
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if type(block.sequence_mixer)
        is ZoologyMomentumAddressDeblurFutureSeedMixer
    ]
    rows = []
    for mixer in mixers:
        q_path = mixer.layer.q_conv1d
        k_path = mixer.layer.k_conv1d
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "q_calls": q_path.calls,
                "k_calls": k_path.calls,
                "q_input_rms": None
                if q_path.last_input_rms is None
                else float(q_path.last_input_rms),
                "q_output_rms": None
                if q_path.last_output_rms is None
                else float(q_path.last_output_rms),
                "q_token_std": None
                if q_path.last_token_std is None
                else float(q_path.last_token_std),
                "k_input_rms": None
                if k_path.last_input_rms is None
                else float(k_path.last_input_rms),
                "k_output_rms": None
                if k_path.last_output_rms is None
                else float(k_path.last_output_rms),
                "k_token_std": None
                if k_path.last_token_std is None
                else float(k_path.last_token_std),
                "value_conv_backend": mixer.layer.v_conv1d.backend,
                "value_conv_kernel_size": int(mixer.layer.v_conv1d.kernel_size[0]),
            }
        )
    return {
        **base,
        "address_deblur_layers": len(rows),
        "active_address_deblur_layers": sum(
            row["q_calls"] > 0 and row["k_calls"] > 0 for row in rows
        ),
        "address_deblur_per_layer": rows,
        "parameter_delta_vs_momentum": -REMOVED_QK_CONV_PARAMETERS_TOTAL,
        "persistent_state_delta_vs_momentum": 0,
        "scan_delta_vs_momentum": 0,
    }
