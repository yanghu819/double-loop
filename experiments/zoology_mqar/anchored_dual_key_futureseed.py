from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F

from experiments.zoology_mqar.decoupled_key_futureseed import (
    NATIVE_MIXER_NAME,
    DirectDecoupledKeyGDN2,
    _module_delta_rms,
    _relative_rms,
    _rms,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


RESIDUAL_CAP = 0.5
COSINE_FLOOR = 1.0 / math.sqrt(1.0 + RESIDUAL_CAP**2)


def _rms_ratio(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((_rms(value) / _rms(reference).clamp_min(1e-8)).item())


class AnchoredDualKeyGDN2(DirectDecoupledKeyGDN2):
    """GDN2 erase key with a bounded tangent correction to the write key."""

    residual_cap = RESIDUAL_CAP

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__(base)
        self._anchored_capture: dict[str, torch.Tensor] = {}

    @staticmethod
    def anchored_key(
        k_write: torch.Tensor,
        k_erase_raw: torch.Tensor,
        *,
        residual_cap: float = RESIDUAL_CAP,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        write = k_write.float()
        raw = k_erase_raw.float()
        write_sq = write.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        tangent = raw - (
            (raw * write).sum(dim=-1, keepdim=True) / write_sq
        ) * write
        correction = float(residual_cap) * tangent
        anchored = (write + correction) / (
            1.0 + correction.square().sum(dim=-1, keepdim=True)
        ).sqrt()
        return (
            anchored.to(k_write.dtype),
            tangent.to(k_write.dtype),
            correction.to(k_write.dtype),
        )

    def prepare_erase_key(
        self,
        k_write: torch.Tensor,
        k_erase_raw: torch.Tensor,
    ) -> torch.Tensor:
        anchored, tangent, correction = self.anchored_key(
            k_write,
            k_erase_raw,
            residual_cap=self.residual_cap,
        )
        if self._capture:
            self._anchored_capture = {
                "k_erase_raw": k_erase_raw.detach(),
                "erase_tangent": tangent.detach(),
                "erase_correction": correction.detach(),
            }
        return anchored

    def extra_capture(self) -> dict[str, torch.Tensor]:
        if not self._capture:
            return {}
        return dict(self._anchored_capture)


class ZoologyAnchoredDualKeyGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed over an anchored dual-key GDN2 transition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = AnchoredDualKeyGDN2(self.layer)


def make_tied_anchored_dual_key_model(
    model_config: Any,
) -> FutureSeedLanguageModel:
    """Build an anchored candidate with the exact native parent parameters."""
    parent_config = copy.deepcopy(model_config)
    parent_config.sequence_mixer.name = NATIVE_MIXER_NAME
    parent = FutureSeedLanguageModel(parent_config)
    candidate = FutureSeedLanguageModel(copy.deepcopy(model_config))
    candidate_state = candidate.state_dict()
    with torch.no_grad():
        for name, tensor in parent.state_dict().items():
            target = name.replace(
                ".sequence_mixer.layer.",
                ".sequence_mixer.layer.base.",
            )
            if target not in candidate_state:
                target = name
            if target not in candidate_state or candidate_state[target].shape != tensor.shape:
                raise RuntimeError(f"Cannot map native parent tensor {name} -> {target}")
            candidate_state[target].copy_(tensor)
    candidate.load_state_dict(candidate_state)
    for block in candidate.backbone.layers:
        mixer = block.sequence_mixer
        if not isinstance(mixer, ZoologyAnchoredDualKeyGDN2FutureSeedMixer):
            raise RuntimeError(f"Unexpected candidate mixer: {type(mixer)}")
        mixer.layer.reset_erase_from_write()
    return candidate


def _layer_diagnostics(
    mixer: ZoologyAnchoredDualKeyGDN2FutureSeedMixer,
    *,
    max_transition_samples: int = 256,
) -> dict[str, float | int]:
    layer = mixer.layer
    captured = layer._captured
    required = {
        "q",
        "k_write",
        "k_erase",
        "k_erase_raw",
        "erase_tangent",
        "erase_correction",
        "alpha",
        "beta",
        "g",
        "erase_gate",
        "write",
        "terminal_state",
    }
    if set(captured) != required:
        raise RuntimeError(f"Incomplete anchored-key capture: {set(captured)}")

    k_write = captured["k_write"].float()
    k_erase = captured["k_erase"].float()
    correction = captured["erase_correction"].float()
    tangent = captured["erase_tangent"].float()
    alpha = captured["alpha"].float().flatten(0, -2)
    beta = captured["beta"].float().flatten(0, -2)
    decay = captured["g"].float().exp().flatten(0, -2)
    sample_count = min(max_transition_samples, alpha.shape[0])
    indices = torch.linspace(
        0,
        alpha.shape[0] - 1,
        steps=sample_count,
        device=alpha.device,
    ).round().long()
    transition = (
        torch.diag_embed(decay.index_select(0, indices))
        + beta.index_select(0, indices).unsqueeze(-1)
        * alpha.index_select(0, indices).unsqueeze(-2)
    )
    singular_values = torch.linalg.svdvals(transition)
    terminal = captured["terminal_state"].float()
    terminal_board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
    cosine = F.cosine_similarity(k_write, k_erase, dim=-1)
    correction_board_rms = correction.square().mean(dim=(1, 2, 3)).sqrt()
    correction_token_rms = correction.square().mean(dim=(0, 2, 3)).sqrt()
    tangent_dot = (tangent * k_write).sum(dim=-1).abs()
    return {
        "layer_idx": mixer.layer_idx,
        "initial_tie_max_error": layer.initial_tie_max_error,
        "residual_cap": layer.residual_cap,
        "registered_cosine_floor": COSINE_FLOOR,
        "key_cosine_mean": float(cosine.mean().item()),
        "key_cosine_min": float(cosine.min().item()),
        "key_relative_rms": _relative_rms(k_erase, k_write),
        "raw_tangent_relative_rms": _rms_ratio(tangent, k_write),
        "correction_relative_rms": _rms_ratio(correction, k_write),
        "correction_board_std": float(
            correction_board_rms.std(unbiased=False).item()
        ),
        "correction_token_std": float(
            correction_token_rms.std(unbiased=False).item()
        ),
        "tangent_orthogonality_max_error": float(tangent_dot.max().item()),
        "projection_delta_rms": _module_delta_rms(
            layer.k_erase_proj,
            layer.base.k_proj,
        ),
        "convolution_delta_rms": _module_delta_rms(
            layer.k_erase_conv1d,
            layer.base.k_conv1d,
        ),
        "erase_gate_mean": float(captured["erase_gate"].float().mean().item()),
        "erase_gate_std": float(
            captured["erase_gate"].float().std(unbiased=False).item()
        ),
        "write_rms": float(_rms(captured["write"]).item()),
        "transition_samples": int(sample_count),
        "transition_spectral_norm_max": float(singular_values[:, 0].max().item()),
        "terminal_state_rms": float(_rms(terminal).item()),
        "terminal_state_board_std": float(
            terminal_board_rms.std(unbiased=False).item()
        ),
    }


@torch.no_grad()
def anchored_dual_key_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyAnchoredDualKeyGDN2FutureSeedMixer,
        )
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use anchored dual-key GDN2")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    rows = [_layer_diagnostics(mixer) for mixer in mixers]
    return {
        "active_layers": len(rows),
        "residual_cap": RESIDUAL_CAP,
        "registered_cosine_floor": COSINE_FLOOR,
        "new_persistent_state_values": 0,
        "logical_scans_per_layer": 1,
        "per_layer": rows,
    }
