from __future__ import annotations

import inspect
import os
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2
from fla.models.utils import Cache
from zoology.config import ModelConfig
from zoology.model import LMBackbone, _compute_state_size, _init_weights


PINNED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"


class ZoologyGDN2FutureSeedMixer(nn.Module):
    """Official FLA GDN2 plus native cross-layer terminal-state seeding."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
        head_dim: int = 32,
        expand_v: float = 1.0,
        conv_size: int = 4,
        future_seed_scale: float = 0.0,
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

        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
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
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.layer.num_v_heads, 1, 1)
        )
        self.source_path = str(module_path)
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, _cache = self.layer(hidden_states)
        return output.to(hidden_states.dtype)

    def make_initial_state(self, terminal_state: torch.Tensor) -> torch.Tensor:
        rms = terminal_state.float().square().mean(
            dim=(-1, -2),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.future_seed_logit).to(
            device=terminal_state.device,
            dtype=terminal_state.dtype,
        )
        normalized = terminal_state / rms.to(dtype=terminal_state.dtype)
        seed = normalized * gate * self.future_seed_scale
        self.last_seed_rms = rms.detach()
        self.last_seed_norm = seed.detach().float().norm(dim=(-1, -2)).mean()
        self.last_seed_gate = gate.detach().float().mean()
        return seed

    def _new_cache(self, initial_state: Optional[torch.Tensor]) -> Cache:
        cache = Cache()
        for index in range(self.layer_idx):
            cache.update(layer_idx=index, offset=0)
        if initial_state is not None:
            cache.update(
                recurrent_state=initial_state,
                conv_state=(None, None, None),
                layer_idx=self.layer_idx,
                offset=0,
            )
        return cache

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                past_key_values=cache,
                use_cache=True,
            )
        layer_cache = cache[self.layer_idx]
        terminal_state = layer_cache["recurrent_state"]
        if terminal_state is None:
            raise RuntimeError("Official GDN2 did not return a terminal recurrent state")
        return output.to(hidden_states.dtype), terminal_state

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return (
            self.layer.num_v_heads
            * self.layer.head_k_dim
            * self.layer.head_v_dim
        )


class FutureSeedLMBackbone(LMBackbone):
    """Upstream Zoology backbone with one generic state-passing hook."""

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scales = {
            float(layer.sequence_mixer.future_seed_scale)
            for layer in self.layers
            if isinstance(layer.sequence_mixer, ZoologyGDN2FutureSeedMixer)
        }
        if len(scales) != 1:
            raise RuntimeError("All GDN2 layers must use the same FutureSeed scale")
        future_seed_scale = scales.pop()
        if future_seed_scale == 0.0:
            return super().layers_forward(hidden_states)

        residual = None
        terminal_state = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyGDN2FutureSeedMixer):
                raise TypeError("FutureSeed backbone requires the strict GDN2 mixer")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = (
                None
                if terminal_state is None
                else mixer.make_initial_state(terminal_state)
            )
            hidden_states, terminal_state = mixer.forward_with_state(
                hidden_states,
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


class FutureSeedLanguageModel(nn.Module):
    """Upstream Zoology language model with FutureSeedLMBackbone only."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = FutureSeedLMBackbone(config=config)
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


def futureseed_diagnostics(
    model: FutureSeedLanguageModel,
) -> dict[str, Any]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyGDN2FutureSeedMixer)
    ]
    target = mixers[-1]
    per_layer = [
        {
            "layer_idx": mixer.layer_idx,
            "seed_applied": mixer.last_seed_gate is not None,
            "future_seed_gate": (
                None
                if mixer.last_seed_gate is None
                else float(mixer.last_seed_gate.item())
            ),
            "future_seed_raw_rms": (
                None
                if mixer.last_seed_rms is None
                else float(mixer.last_seed_rms.mean().item())
            ),
            "future_seed_norm": (
                None
                if mixer.last_seed_norm is None
                else float(mixer.last_seed_norm.item())
            ),
        }
        for mixer in mixers
    ]
    return {
        "future_seed_scale": target.future_seed_scale,
        "future_seed_gate": (
            None
            if target.last_seed_gate is None
            else float(target.last_seed_gate.item())
        ),
        "future_seed_raw_rms": (
            None
            if target.last_seed_rms is None
            else float(target.last_seed_rms.mean().item())
        ),
        "future_seed_norm": (
            None
            if target.last_seed_norm is None
            else float(target.last_seed_norm.item())
        ),
        "active_seed_routes": sum(row["seed_applied"] for row in per_layer),
        "per_layer": per_layer,
    }
