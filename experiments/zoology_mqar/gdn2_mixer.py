from __future__ import annotations

import inspect
import os
from pathlib import Path

import torch
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2


class ZoologyGDN2Mixer(nn.Module):
    """Thin shape/dtype adapter from Zoology blocks to official FLA GDN2."""

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
        expected_sha = os.environ.get("FLA_EXPECTED_SOURCE_SHA")
        if expected_sha != "9c8e42e762fce087c27b673af4922795d9edb85e":
            raise RuntimeError("Pinned official FLA source SHA was not asserted")

        module_path = Path(inspect.getfile(GatedDeltaNet2)).resolve()
        expected_root = Path("/huyang2/double-loop/.cache/fla-versions")
        if expected_root not in module_path.parents:
            raise RuntimeError(f"Unexpected FLA GDN2 source: {module_path}")

        self.layer = GatedDeltaNet2(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=conv_size,
            layer_idx=layer_idx,
        )
        self.source_path = str(module_path)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, _cache = self.layer(hidden_states)
        return output.to(hidden_states.dtype)

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return (
            self.layer.num_v_heads
            * self.layer.head_k_dim
            * self.layer.head_v_dim
        )
