from __future__ import annotations

import hashlib
from typing import Any

import torch.nn as nn

from experiments.zoology_mqar.gdn2_futureseed import FutureSeedLanguageModel
from experiments.zoology_mqar.gdn2_log_spd import (
    ZoologyLogSPDGDN2FutureSeedMixer,
)


def shared_log_spd_mixers(model: nn.Module) -> list[ZoologyLogSPDGDN2FutureSeedMixer]:
    mixers = [
        layer.sequence_mixer
        for layer in model.backbone.layers
        if isinstance(layer.sequence_mixer, ZoologyLogSPDGDN2FutureSeedMixer)
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use the Log-SPD FutureSeed mixer")
    return mixers


class SharedLogSPDFutureSeedLanguageModel(FutureSeedLanguageModel):
    """Native FutureSeed whose GDN2 layers use one common bounded Q/K metric."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        mixers = shared_log_spd_mixers(self)
        if len(mixers) < 2:
            raise ValueError("Cross-layer sharing requires at least two GDN2 layers")
        shared_metric = mixers[0].layer.address_metric
        for mixer in mixers[1:]:
            mixer.layer.address_metric = shared_metric


def shared_log_spd_diagnostics(model: nn.Module) -> dict[str, Any]:
    mixers = shared_log_spd_mixers(model)
    metrics = [mixer.layer.address_metric for mixer in mixers]
    raw_pointers = [metric.raw.data_ptr() for metric in metrics]
    unique_parameters = {id(metric.raw): metric.raw for metric in metrics}
    return {
        "active_layers": len(mixers),
        "shared_object": len({id(metric) for metric in metrics}) == 1,
        "shared_parameter": len(unique_parameters) == 1,
        "raw_data_ptrs": raw_pointers,
        "parameter_delta": sum(parameter.numel() for parameter in unique_parameters.values()),
        "per_layer": [metric.diagnostics() for metric in metrics],
    }


def parent_parameter_hash(model: nn.Module) -> str:
    """Hash candidate parent parameters under the original native GDN2 names."""
    tensors = []
    for name, parameter in model.named_parameters():
        if name.endswith("layer.address_metric.raw"):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        tensors.append((parent_name, parameter))
    digest = hashlib.sha256()
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
