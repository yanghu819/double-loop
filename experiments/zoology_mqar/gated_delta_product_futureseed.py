from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers.gated_deltaproduct import GatedDeltaProduct

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


class ZoologyGatedDeltaProductFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Pinned official GatedDeltaProduct with native cross-layer FutureSeed."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = 4,
        head_dim: int = 32,
        expand_v: float = 1.0,
        conv_size: int = 4,
        future_seed_scale: float = 1.0,
        num_householder: int = 2,
        use_forget_gate: bool = True,
        allow_neg_eigval: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        if d_model != num_heads * head_dim:
            raise ValueError("d_model must equal num_heads * head_dim")
        if num_householder != 2:
            raise ValueError("P-GDN3-029 fixes num_householder=2")
        if not use_forget_gate:
            raise ValueError("P-GDN3-029 requires the native forget gate")
        if allow_neg_eigval:
            raise ValueError("P-GDN3-029 isolates update rank from signed eigenvalues")
        if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
            raise RuntimeError("Pinned official FLA source SHA was not asserted")

        module_path = Path(inspect.getfile(GatedDeltaProduct)).resolve()
        expected_root = Path("/huyang2/double-loop/.cache/fla-versions")
        if expected_root not in module_path.parents:
            raise RuntimeError(f"Unexpected FLA GatedDeltaProduct source: {module_path}")

        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = GatedDeltaProduct(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=conv_size,
            use_forget_gate=use_forget_gate,
            allow_neg_eigval=allow_neg_eigval,
            num_householder=num_householder,
            layer_idx=layer_idx,
        )
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.layer.num_v_heads, 1, 1)
        )
        self.source_path = str(module_path)
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None
        self._capture_product = False
        self._captured_product: dict[str, torch.Tensor] = {}
        self.layer.k_conv1d.register_forward_hook(self._capture_hook("key"))
        self.layer.v_conv1d.register_forward_hook(self._capture_hook("value"))
        self.layer.b_proj.register_forward_hook(self._capture_hook("beta_logit"))

    def _capture_hook(self, name: str):
        def hook(_module, _inputs, output) -> None:
            if not self._capture_product:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            self._captured_product[name] = tensor.detach()

        return hook

    def set_product_capture(self, enabled: bool) -> None:
        self._capture_product = bool(enabled)
        if enabled:
            self._captured_product.clear()

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, terminal_state = super().forward_with_state(
            hidden_states,
            initial_state=initial_state,
        )
        if self._capture_product:
            self._captured_product["terminal_state"] = terminal_state.detach()
        return output, terminal_state


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _pair_diagnostics(mixer: ZoologyGatedDeltaProductFutureSeedMixer) -> dict[str, Any]:
    captures = mixer._captured_product
    required = {"key", "value", "beta_logit", "terminal_state"}
    if set(captures) != required:
        raise RuntimeError(f"Incomplete GatedDeltaProduct diagnostics: {set(captures)}")

    layer = mixer.layer
    batch, length = captures["key"].shape[:2]
    key = captures["key"].reshape(
        batch,
        length,
        layer.num_householder,
        layer.num_heads,
        layer.head_k_dim,
    ).float()
    value = captures["value"].reshape(
        batch,
        length,
        layer.num_householder,
        layer.num_v_heads,
        layer.head_v_dim,
    ).float()
    beta = captures["beta_logit"].reshape(
        batch,
        length,
        layer.num_householder,
        layer.num_v_heads,
    ).float().sigmoid()
    first_key, second_key = key.unbind(dim=2)
    first_value, second_value = value.unbind(dim=2)
    first_beta, second_beta = beta.unbind(dim=2)
    key_cosine = F.cosine_similarity(first_key, second_key, dim=-1)
    value_cosine = F.cosine_similarity(first_value, second_value, dim=-1)
    terminal = captures["terminal_state"].float()
    terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()

    return {
        "layer_idx": mixer.layer_idx,
        "num_householder": layer.num_householder,
        "key_pair_cosine_mean": float(key_cosine.mean().item()),
        "key_pair_cosine_abs_mean": float(key_cosine.abs().mean().item()),
        "key_pair_relative_rms": float(
            (_rms(first_key - second_key) / _rms(key).clamp_min(1e-8)).item()
        ),
        "value_pair_cosine_mean": float(value_cosine.mean().item()),
        "value_pair_cosine_abs_mean": float(value_cosine.abs().mean().item()),
        "value_pair_relative_rms": float(
            (_rms(first_value - second_value) / _rms(value).clamp_min(1e-8)).item()
        ),
        "beta_mean": float(beta.mean().item()),
        "beta_std": float(beta.std(unbiased=False).item()),
        "beta_pair_abs_difference": float(
            (first_beta - second_beta).abs().mean().item()
        ),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }


@torch.no_grad()
def gated_delta_product_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(
            layer.sequence_mixer,
            ZoologyGatedDeltaProductFutureSeedMixer,
        )
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use GatedDeltaProduct")
    for mixer in mixers:
        mixer.set_product_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.set_product_capture(False)

    rows = [_pair_diagnostics(mixer) for mixer in mixers]
    return {
        "active_layers": len(rows),
        "num_householder": 2,
        "new_persistent_state_values": 0,
        "per_layer": rows,
    }
