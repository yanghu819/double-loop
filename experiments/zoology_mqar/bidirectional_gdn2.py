from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2


PINNED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"


class ExplicitBidirectionalGDN2Mixer(nn.Module):
    """Independent forward and reverse official-FLA GDN2 streams."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
        head_dim: int = 32,
        expand_v: float = 1.0,
        conv_size: int = 4,
    ) -> None:
        super().__init__()
        if d_model != num_heads * head_dim:
            raise ValueError("d_model must equal num_heads * head_dim")
        if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
            raise RuntimeError("Pinned official FLA source SHA was not asserted")

        module_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
        expected_root = Path("/huyang2/double-loop/.cache/fla-versions")
        if expected_root not in module_path.parents:
            raise RuntimeError(f"Unexpected FLA GDN2 source: {module_path}")

        kwargs = {
            "hidden_size": d_model,
            "expand_v": expand_v,
            "head_dim": head_dim,
            "num_heads": num_heads,
            "mode": "chunk",
            "use_short_conv": True,
            "conv_size": conv_size,
            "layer_idx": layer_idx,
        }
        self.forward_layer = GatedDeltaNet2(**kwargs)
        self.backward_layer = GatedDeltaNet2(**kwargs)
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)
        self.source_path = str(module_path)
        self.last_forward_rms: Optional[torch.Tensor] = None
        self.last_backward_rms: Optional[torch.Tensor] = None

    @staticmethod
    def _output(layer: GatedDeltaNet2, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _attentions, _cache = layer(hidden_states)
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            forward = self._output(self.forward_layer, hidden_states)
            backward = self._output(
                self.backward_layer,
                hidden_states.flip(1).contiguous(),
            ).flip(1)
            merged = self.merge(torch.cat((forward, backward), dim=-1))
        self.last_forward_rms = forward.detach().float().square().mean().sqrt()
        self.last_backward_rms = backward.detach().float().square().mean().sqrt()
        return merged.to(hidden_states.dtype)

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        one_direction = (
            self.forward_layer.num_v_heads
            * self.forward_layer.head_k_dim
            * self.forward_layer.head_v_dim
        )
        return 2 * one_direction


def bidirectional_diagnostics(model: nn.Module) -> dict[str, object]:
    mixers = [
        module
        for module in model.modules()
        if isinstance(module, ExplicitBidirectionalGDN2Mixer)
    ]
    return {
        "mixer_count": len(mixers),
        "official_gdn2_streams": 2 * len(mixers),
        "scan_directions": ["forward", "reverse"],
        "fusion": "concatenate_then_learned_linear_projection",
        "forward_rms": [
            None if mixer.last_forward_rms is None else float(mixer.last_forward_rms)
            for mixer in mixers
        ],
        "backward_rms": [
            None
            if mixer.last_backward_rms is None
            else float(mixer.last_backward_rms)
            for mixer in mixers
        ],
    }
