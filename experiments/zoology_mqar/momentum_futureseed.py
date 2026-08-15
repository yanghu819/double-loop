from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_MDN_SHA = "c6e77fa261fb0c002fae1a14b6209a5b28d2edc9"
EXPECTED_LAYER_SHA256 = (
    "32b1d6d89b7484b4eab09efd96512c5ec35baadcefb2f23deb7850087154e1a3"
)
EXPECTED_CHUNK_SHA256 = (
    "ef9c5340ae94ea1bd870f57760ab17dc6c707f003a30422b5e4d2d8db969f56b"
)
EXPECTED_RECURRENT_SHA256 = (
    "19f58968d5c0c967f75ceca56cbed5e164109e410a254ad000170ec9d1c059d4"
)
MODEL_HEADS = 4
HEAD_DIM = 32
STATE_COMPONENTS = 2
EXPECTED_STATE_VALUES_PER_LAYER = (
    STATE_COMPONENTS * MODEL_HEADS * HEAD_DIM * HEAD_DIM
)
EXPECTED_PARAMETER_DELTA_VS_GDN2 = -61_912


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _append_package_path(package_name: str, path: Path) -> None:
    package = importlib.import_module(package_name)
    package_path = str(path.resolve())
    if package_path not in package.__path__:
        package.__path__.append(package_path)


def load_external_momentum_layer():
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if os.environ.get("MDN_EXPECTED_SHA") != EXPECTED_MDN_SHA:
        raise RuntimeError("Momentum DeltaNet source SHA was not asserted")
    if _git_head(repo_root) != EXPECTED_MDN_SHA:
        raise RuntimeError(f"Unexpected Momentum DeltaNet checkout: {repo_root}")
    if fla_root != repo_root / "flash-linear-attention":
        raise RuntimeError(f"Unexpected Momentum DeltaNet FLA root: {fla_root}")

    layer_path = fla_root / "fla" / "layers" / "momentum_deltanet.py"
    chunk_path = fla_root / "fla" / "ops" / "momentum_delta_rule" / "chunk.py"
    recurrent_path = (
        fla_root / "fla" / "ops" / "momentum_delta_rule" / "fused_recurrent.py"
    )
    expected_hashes = {
        layer_path: EXPECTED_LAYER_SHA256,
        chunk_path: EXPECTED_CHUNK_SHA256,
        recurrent_path: EXPECTED_RECURRENT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Momentum DeltaNet source drifted: {path}")

    _append_package_path("fla.layers", fla_root / "fla" / "layers")
    _append_package_path("fla.ops", fla_root / "fla" / "ops")
    importlib.invalidate_caches()
    module = importlib.import_module("fla.layers.momentum_deltanet")
    layer_class = module.MomentumDeltaNet
    resolved = Path(inspect.getfile(layer_class)).resolve()
    if resolved != layer_path:
        raise RuntimeError(f"Unexpected Momentum DeltaNet module: {resolved}")
    return layer_class


def _tensor_hash(rows: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(rows):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ZoologyMomentumDeltaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Second-order Momentum DeltaNet with native state-and-momentum FutureSeed."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = MODEL_HEADS,
        head_dim: int = HEAD_DIM,
        expand_v: float = 1.0,
        conv_size: int = 4,
        future_seed_scale: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if num_heads != MODEL_HEADS or head_dim != HEAD_DIM or expand_v != 1.0:
            raise ValueError("P-GDN3-059 fixes H4/K32/V32")
        layer_class = load_external_momentum_layer()
        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = layer_class(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=conv_size,
            layer_idx=layer_idx,
            use_output_correction=True,
        )
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.layer.num_v_heads, 1, 1)
        )
        self.source_path = str(Path(inspect.getfile(layer_class)).resolve())
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None
        self.last_state_rms: Optional[torch.Tensor] = None
        self.last_momentum_rms: Optional[torch.Tensor] = None
        self.last_momentum_to_state_rms: Optional[torch.Tensor] = None

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
        if terminal_state.ndim != 5 or terminal_state.shape[0] != STATE_COMPONENTS:
            raise RuntimeError(
                f"Momentum state must be [2,B,H,K,V], got {terminal_state.shape}"
            )
        state_rms = terminal_state[0].float().square().mean().sqrt()
        momentum_rms = terminal_state[1].float().square().mean().sqrt()
        self.last_state_rms = state_rms.detach()
        self.last_momentum_rms = momentum_rms.detach()
        self.last_momentum_to_state_rms = (
            momentum_rms / state_rms.clamp_min(1e-8)
        ).detach()
        return output, terminal_state

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_STATE_VALUES_PER_LAYER


def _shared_parent_rows(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    rows = []
    for name, tensor in model.named_parameters():
        parent = parent_state.get(name)
        if parent is None or parent.shape != tensor.shape:
            continue
        rows.append((name, parent))
    return rows


def load_matched_parent_state(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> None:
    target = model.state_dict()
    rows = _shared_parent_rows(model, parent_state)
    if not rows:
        raise RuntimeError("No shared parent tensors were found")
    for name, parent in rows:
        target[name] = parent.detach().clone()
    model.load_state_dict(target, strict=True)
    loaded_rows = [(name, model.state_dict()[name]) for name, _ in rows]
    source_hash = _tensor_hash(rows)
    loaded_hash = _tensor_hash(loaded_rows)
    if source_hash != loaded_hash:
        raise RuntimeError("Shared parent tensor mapping is not exact")
    model._momentum_parent_metadata = {
        "tensor_count": len(rows),
        "numel": sum(tensor.numel() for _, tensor in rows),
        "source_hash": source_hash,
        "loaded_hash": loaded_hash,
        "names": [name for name, _ in rows],
    }


def parent_parameter_hash(model: torch.nn.Module) -> str:
    metadata = getattr(model, "_momentum_parent_metadata", None)
    if metadata is None:
        raise RuntimeError("Matched parent metadata is missing")
    names = set(metadata["names"])
    rows = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in names
    ]
    return _tensor_hash(rows)


def momentum_futureseed_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyMomentumDeltaFutureSeedMixer,
        )
    ]
    rows = []
    for mixer in mixers:
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "state_rms": None if mixer.last_state_rms is None else float(mixer.last_state_rms),
                "momentum_rms": None if mixer.last_momentum_rms is None else float(mixer.last_momentum_rms),
                "momentum_to_state_rms": (
                    None
                    if mixer.last_momentum_to_state_rms is None
                    else float(mixer.last_momentum_to_state_rms)
                ),
                "seed_applied": mixer.last_seed_gate is not None,
                "seed_gate": None if mixer.last_seed_gate is None else float(mixer.last_seed_gate),
            }
        )
    return {
        "external_sha": EXPECTED_MDN_SHA,
        "external_layer_sha256": EXPECTED_LAYER_SHA256,
        "state_components": STATE_COMPONENTS,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "active_layers": sum(row["momentum_rms"] is not None for row in rows),
        "active_futureseed_routes": sum(row["seed_applied"] for row in rows),
        "per_layer": rows,
        "matched_parent": getattr(model, "_momentum_parent_metadata", None),
    }
