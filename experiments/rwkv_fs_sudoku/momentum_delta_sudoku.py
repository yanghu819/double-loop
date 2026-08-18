from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


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
EXPECTED_HOST_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"
MOMENTUM_STATE_COMPONENTS = 2
_FLA_COMPATIBILITY: Optional[dict[str, Any]] = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _append_package_path(package_name: str, path: Path) -> None:
    package = importlib.import_module(package_name)
    resolved = str(path.resolve())
    if resolved not in package.__path__:
        package.__path__.append(resolved)


def _install_fla_compatibility() -> dict[str, Any]:
    global _FLA_COMPATIBILITY
    if _FLA_COMPATIBILITY is not None:
        return dict(_FLA_COMPATIBILITY)

    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != EXPECTED_HOST_FLA_SHA:
        raise RuntimeError("Pinned host FLA SHA was not asserted")
    if os.environ.get("FLA_USE_CUDA_GRAPH", "0") != "0":
        raise RuntimeError("Momentum DeltaNet requires FLA_USE_CUDA_GRAPH=0")

    host_utils = importlib.import_module("fla.utils")
    host_path = Path(inspect.getfile(host_utils)).resolve()
    host_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if not host_path.is_relative_to(host_root):
        raise RuntimeError(f"Unexpected host FLA utils module: {host_path}")

    required = {
        "IS_NVIDIA_HOPPER",
        "autotune_cache_kwargs",
        "check_shared_mem",
        "is_tf32_supported",
        "input_guard",
        "autocast_custom_bwd",
        "autocast_custom_fwd",
    }
    missing = sorted(name for name in required if not hasattr(host_utils, name))
    if missing:
        raise RuntimeError(f"Pinned host FLA lacks required utilities: {missing}")

    injected = not hasattr(host_utils, "USE_CUDA_GRAPH")
    if injected:
        host_utils.USE_CUDA_GRAPH = False
    if host_utils.USE_CUDA_GRAPH is not False:
        raise RuntimeError("Momentum DeltaNet requires disabled CUDA graph scheduling")

    _FLA_COMPATIBILITY = {
        "host_utils_path": str(host_path),
        "host_fla_sha": EXPECTED_HOST_FLA_SHA,
        "missing_required_symbols": missing,
        "injected_use_cuda_graph": injected,
        "use_cuda_graph": bool(host_utils.USE_CUDA_GRAPH),
    }
    return dict(_FLA_COMPATIBILITY)


def load_external_momentum_layer() -> type:
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

    _install_fla_compatibility()
    _append_package_path("fla.layers", fla_root / "fla" / "layers")
    _append_package_path("fla.ops", fla_root / "fla" / "ops")
    importlib.invalidate_caches()
    module = importlib.import_module("fla.layers.momentum_deltanet")
    layer_class = module.MomentumDeltaNet
    if Path(inspect.getfile(layer_class)).resolve() != layer_path:
        raise RuntimeError("Momentum DeltaNet resolved outside the pinned checkout")
    return layer_class


def momentum_source_summary() -> dict[str, Any]:
    layer_class = load_external_momentum_layer()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    return {
        "external_sha": EXPECTED_MDN_SHA,
        "layer_path": str(Path(inspect.getfile(layer_class)).resolve()),
        "layer_sha256": EXPECTED_LAYER_SHA256,
        "chunk_sha256": EXPECTED_CHUNK_SHA256,
        "recurrent_sha256": EXPECTED_RECURRENT_SHA256,
        "chunk_path": str(
            fla_root / "fla" / "ops" / "momentum_delta_rule" / "chunk.py"
        ),
        "compatibility": dict(_FLA_COMPATIBILITY or {}),
    }


class MomentumDeltaTimeMix(nn.Module):
    """Pinned Momentum DeltaNet with explicit two-plane recurrent state I/O."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        layer_idx: int,
        expand_v: float,
        use_short_conv: bool,
        conv_size: int,
    ) -> None:
        super().__init__()
        if d_model != heads * head_dim:
            raise ValueError("Momentum Sudoku keeps d_model == heads * head_dim")
        if not use_short_conv:
            raise ValueError("The registered Momentum transfer keeps short convolution")
        head_v_dim = int(head_dim * float(expand_v))
        if not math.isclose(
            float(head_v_dim),
            head_dim * float(expand_v),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Momentum expand_v must produce an integer head value width")

        layer_class = load_external_momentum_layer()
        self.layer_idx = int(layer_idx)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.head_v_dim = int(head_v_dim)
        self.core = layer_class(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=heads,
            mode="chunk",
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            layer_idx=layer_idx,
            use_output_correction=True,
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self.last_terminal_state_shape: Optional[tuple[int, ...]] = None

    def _new_cache(self, initial_state: Optional[torch.Tensor]):
        from fla.models.utils import Cache

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            raise RuntimeError("Momentum DeltaNet Sudoku is CUDA-only")
        batch_size = x.shape[0]
        expected = (
            MOMENTUM_STATE_COMPONENTS,
            batch_size,
            self.heads,
            self.head_dim,
            self.head_v_dim,
        )
        if initial_state is not None and tuple(initial_state.shape) != expected:
            raise ValueError(
                f"Momentum initial state {tuple(initial_state.shape)} does not match {expected}"
            )

        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            output, _attentions, returned_cache = self.core(
                x,
                past_key_values=cache,
                use_cache=True,
            )
        if returned_cache is None or len(returned_cache) <= self.layer_idx:
            raise RuntimeError("Momentum DeltaNet did not return its recurrent cache")
        terminal_state = returned_cache[self.layer_idx]["recurrent_state"]
        if terminal_state is None or tuple(terminal_state.shape) != expected:
            raise RuntimeError(
                "Momentum terminal state must be [2,B,H,K,V], got "
                f"{None if terminal_state is None else tuple(terminal_state.shape)}"
            )
        if not torch.isfinite(terminal_state).all():
            raise RuntimeError("Momentum terminal state is nonfinite")
        self.last_terminal_state_shape = tuple(terminal_state.shape)

        state = terminal_state[0].float()
        velocity = terminal_state[1].float()
        state_rms = state.square().mean().sqrt()
        velocity_rms = velocity.square().mean().sqrt()
        state_board = state.square().mean(dim=(-1, -2, -3)).sqrt()
        velocity_board = velocity.square().mean(dim=(-1, -2, -3)).sqrt()
        self.last_diagnostics = {
            "momentum_state_rms": state_rms.detach(),
            "momentum_velocity_rms": velocity_rms.detach(),
            "momentum_velocity_to_state_rms": (
                velocity_rms / state_rms.clamp_min(1e-8)
            ).detach(),
            "momentum_state_board_std": state_board.std(unbiased=False).detach(),
            "momentum_velocity_board_std": velocity_board.std(unbiased=False).detach(),
            "momentum_active_layers_sum": x.new_ones(()),
        }
        return output.to(dtype=x.dtype), terminal_state

    def state_elements_per_head(self) -> int:
        return MOMENTUM_STATE_COMPONENTS * self.head_dim * self.head_v_dim
