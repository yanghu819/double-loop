from __future__ import annotations

import importlib
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.momentum_futureseed import (
    EXPECTED_MDN_SHA,
    _git_head,
    _install_fla_compatibility,
    _sha256,
    _shared_parent_rows,
    _tensor_hash,
)


EXPECTED_LAYER_SHA256 = (
    "d35bc7d3cb768b68a8af107ecb4acd470cb56faf1f1e06da4e5ccfd1acd62600"
)
EXPECTED_CHUNK_SHA256 = (
    "bb19c6ae51ac2f3d1a95d133ad3f56baf71facfe315decd3106eabf736d5c406"
)
EXPECTED_RECURRENT_SHA256 = (
    "aa73a9abe5820d8104b5f6399a016c6fd4e1922798ca78759e89766a5ef5bba5"
)
EXPECTED_WY_SHA256 = (
    "6e4edc5886fda8eca4be2d67066521db16c91e49a73f34e0dfc2b874adf69959"
)
EXPECTED_COMPAT_WY_SHA256 = (
    "2de7bebba43ccef9fb1aef8598da0c0f075664cd3769789bdaa85806f9d6a109"
)
_WY_DTYPE_PATCHES = (
    (
        "tl.dot(b_dw, tl.trans(b_p_beta_g0))",
        "tl.dot(b_dw, tl.trans(b_p_beta_g0.to(b_dw.dtype)))",
    ),
    (
        "tl.dot(b_A, b_kb)",
        "tl.dot(b_A, b_kb.to(b_A.dtype))",
    ),
)
MODEL_HEADS = 4
HEAD_DIM = 32
EXPECTED_STATE_VALUES_PER_LAYER = MODEL_HEADS * HEAD_DIM * HEAD_DIM
EXPECTED_PARAMETER_DELTA_VS_GDN2 = -63_976
EXPECTED_PARAMETER_DELTA_VS_MOMENTUM = -2_064
_COMBA_LAYER_CLASS = None
_COMBA_COMPAT_METADATA: Optional[dict[str, Any]] = None


def _prepend_package_path(package_name: str, path: Path) -> None:
    package = importlib.import_module(package_name)
    package_path = str(path.resolve())
    while package_path in package.__path__:
        package.__path__.remove(package_path)
    package.__path__.insert(0, package_path)


def _build_comba_compat_overlay(fla_root: Path) -> dict[str, Any]:
    run_root = Path(os.environ["COMBA_RUN_ROOT"]).resolve()
    compat_root = Path(os.environ["COMBA_COMPAT_ROOT"]).resolve()
    if compat_root != run_root / "comba-compat":
        raise RuntimeError(f"Unexpected Comba compatibility root: {compat_root}")

    source_package = fla_root / "fla" / "ops" / "comba"
    source_wy = source_package / "wy_fast.py"
    if _sha256(source_wy) != EXPECTED_WY_SHA256:
        raise RuntimeError(f"Comba WY source drifted: {source_wy}")

    if compat_root.exists():
        shutil.rmtree(compat_root)
    effective_package = compat_root / "fla" / "ops" / "comba"
    shutil.copytree(
        source_package,
        effective_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    effective_wy = effective_package / "wy_fast.py"
    text = effective_wy.read_text()
    for old, new in _WY_DTYPE_PATCHES:
        if text.count(old) != 1:
            raise RuntimeError(f"Comba WY compatibility site drifted: {old}")
        text = text.replace(old, new)
    effective_wy.write_text(text)
    if _sha256(effective_wy) != EXPECTED_COMPAT_WY_SHA256:
        raise RuntimeError("Comba WY compatibility output drifted")
    return {
        "root": str(compat_root),
        "ops_root": str(compat_root / "fla" / "ops"),
        "source_wy_path": str(source_wy),
        "source_wy_sha256": EXPECTED_WY_SHA256,
        "effective_wy_path": str(effective_wy),
        "effective_wy_sha256": EXPECTED_COMPAT_WY_SHA256,
        "patch_count": len(_WY_DTYPE_PATCHES),
    }


def comba_compatibility_metadata() -> dict[str, Any]:
    if _COMBA_COMPAT_METADATA is None:
        raise RuntimeError("Comba compatibility overlay has not been loaded")
    return dict(_COMBA_COMPAT_METADATA)


def load_external_comba_layer():
    global _COMBA_COMPAT_METADATA, _COMBA_LAYER_CLASS
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if os.environ.get("MDN_EXPECTED_SHA") != EXPECTED_MDN_SHA:
        raise RuntimeError("Comba source SHA was not asserted")
    if _git_head(repo_root) != EXPECTED_MDN_SHA:
        raise RuntimeError(f"Unexpected Comba checkout: {repo_root}")
    if fla_root != repo_root / "flash-linear-attention":
        raise RuntimeError(f"Unexpected Comba FLA root: {fla_root}")

    layer_path = fla_root / "fla" / "layers" / "comba.py"
    chunk_path = fla_root / "fla" / "ops" / "comba" / "chunk.py"
    recurrent_path = fla_root / "fla" / "ops" / "comba" / "fused_recurrent.py"
    wy_path = fla_root / "fla" / "ops" / "comba" / "wy_fast.py"
    expected_hashes = {
        layer_path: EXPECTED_LAYER_SHA256,
        chunk_path: EXPECTED_CHUNK_SHA256,
        recurrent_path: EXPECTED_RECURRENT_SHA256,
        wy_path: EXPECTED_WY_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Comba source drifted: {path}")

    _install_fla_compatibility()
    _prepend_package_path("fla.layers", fla_root / "fla" / "layers")
    _prepend_package_path("fla.ops", fla_root / "fla" / "ops")
    if _COMBA_LAYER_CLASS is not None:
        resolved = Path(inspect.getfile(_COMBA_LAYER_CLASS)).resolve()
        if resolved != layer_path:
            raise RuntimeError(f"Cached Comba module drifted: {resolved}")
        metadata = comba_compatibility_metadata()
        if (
            Path(metadata["root"]).resolve()
            != Path(os.environ["COMBA_COMPAT_ROOT"]).resolve()
            or _sha256(Path(metadata["effective_wy_path"]))
            != EXPECTED_COMPAT_WY_SHA256
        ):
            raise RuntimeError("Cached Comba compatibility overlay drifted")
        return _COMBA_LAYER_CLASS

    _COMBA_COMPAT_METADATA = _build_comba_compat_overlay(fla_root)
    _prepend_package_path("fla.ops", Path(_COMBA_COMPAT_METADATA["ops_root"]))
    for name in list(sys.modules):
        if name == "fla.layers.comba" or name.startswith("fla.ops.comba"):
            del sys.modules[name]
    importlib.invalidate_caches()
    module = importlib.import_module("fla.layers.comba")
    resolved = Path(inspect.getfile(module.Comba)).resolve()
    if resolved != layer_path:
        raise RuntimeError(f"Unexpected Comba module: {resolved}")
    effective_chunk = Path(
        inspect.getfile(importlib.import_module("fla.ops.comba.chunk"))
    ).resolve()
    effective_wy = Path(
        inspect.getfile(importlib.import_module("fla.ops.comba.wy_fast"))
    ).resolve()
    if (
        _sha256(effective_chunk) != EXPECTED_CHUNK_SHA256
        or effective_wy != Path(_COMBA_COMPAT_METADATA["effective_wy_path"])
        or _sha256(effective_wy) != EXPECTED_COMPAT_WY_SHA256
    ):
        raise RuntimeError("Comba compatibility modules escaped the pinned overlay")
    _COMBA_COMPAT_METADATA.update(
        {
            "effective_chunk_path": str(effective_chunk),
            "effective_chunk_sha256": _sha256(effective_chunk),
        }
    )
    _COMBA_LAYER_CLASS = module.Comba
    return _COMBA_LAYER_CLASS


class ZoologyCombaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Closed-loop Comba recurrence with native terminal-state FutureSeed."""

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
            raise ValueError("P-GDN3-065 fixes D128/H4/K32/V32")
        layer_class = load_external_comba_layer()
        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = layer_class(
            hidden_size=d_model,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=True,
            use_output_gate=True,
            use_output_correction=True,
            use_inner_decay=True,
            correction_factor=1.0,
            conv_size=conv_size,
            layer_idx=layer_idx,
        )
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.layer.num_v_heads, 1, 1)
        )
        self.source_path = str(Path(inspect.getfile(layer_class)).resolve())
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None
        self.last_state_rms: Optional[torch.Tensor] = None
        self.last_state_board_std: Optional[torch.Tensor] = None

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
        if terminal_state.ndim != 4:
            raise RuntimeError(
                f"Comba state must be [B,H,K,V], got {terminal_state.shape}"
            )
        board_rms = terminal_state.float().square().mean(dim=(-1, -2, -3)).sqrt()
        self.last_state_rms = board_rms.mean().detach()
        self.last_state_board_std = board_rms.std(unbiased=False).detach()
        return output, terminal_state

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_STATE_VALUES_PER_LAYER


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
    model._comba_parent_metadata = {
        "tensor_count": len(rows),
        "numel": sum(tensor.numel() for _, tensor in rows),
        "source_hash": source_hash,
        "loaded_hash": loaded_hash,
        "names": [name for name, _ in rows],
    }


def parent_parameter_hash(model: torch.nn.Module) -> str:
    metadata = getattr(model, "_comba_parent_metadata", None)
    if metadata is None:
        raise RuntimeError("Matched parent metadata is missing")
    names = set(metadata["names"])
    return _tensor_hash(
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name in names
        ]
    )


def _variation(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    board_rms = tensor.square().flatten(1).mean(dim=1).sqrt()
    token_rms = tensor.square().flatten(2).mean(dim=2).sqrt()
    return {
        "rms": float(tensor.square().mean().sqrt()),
        "board_std": float(board_rms.std(unbiased=False)),
        "token_std": float(token_rms.std(unbiased=False)),
    }


def comba_diagnostics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyCombaFutureSeedMixer)
    ]
    module = importlib.import_module("fla.layers.comba")
    original = module.chunk_comba
    calls: list[dict[str, Any]] = []

    def capture(*args, **kwargs):
        names = ("q", "k", "v", "p", "g", "beta")
        values = dict(zip(names, args))
        values.update(kwargs)
        q, k, p = values["q"], values["k"], values["p"]
        g, beta = values["g"], values["beta"]
        cosine = torch.nn.functional.cosine_similarity(
            k.detach().float(), p.detach().float(), dim=-1
        )
        norm_ratio = p.detach().float().norm(dim=-1) / k.detach().float().norm(
            dim=-1
        ).clamp_min(1e-8)
        calls.append(
            {
                "q": _variation(q),
                "k": _variation(k),
                "p": _variation(p),
                "p_k_cosine_mean": float(cosine.mean()),
                "p_k_cosine_min": float(cosine.min()),
                "p_k_norm_ratio_mean": float(norm_ratio.mean()),
                "p_k_norm_ratio_min": float(norm_ratio.min()),
                "p_k_norm_ratio_max": float(norm_ratio.max()),
                "g_min": float(g.detach().float().min()),
                "g_max": float(g.detach().float().max()),
                "beta_min": float(beta.detach().float().min()),
                "beta_max": float(beta.detach().float().max()),
            }
        )
        return original(*args, **kwargs)

    module.chunk_comba = capture
    try:
        with torch.no_grad():
            model.eval()(inputs)
    finally:
        module.chunk_comba = original

    rows = []
    for mixer, call in zip(mixers, calls, strict=True):
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "state_rms": None if mixer.last_state_rms is None else float(mixer.last_state_rms),
                "state_board_std": (
                    None
                    if mixer.last_state_board_std is None
                    else float(mixer.last_state_board_std)
                ),
                "seed_applied": mixer.last_seed_gate is not None,
                "seed_gate": None if mixer.last_seed_gate is None else float(mixer.last_seed_gate),
                "decay_sigmoid_mean": float(mixer.layer.decay.detach().float().sigmoid().mean()),
                "correction_abs_mean": float(mixer.layer.D.detach().float().abs().mean()),
                **call,
            }
        )
    return {
        "external_sha": EXPECTED_MDN_SHA,
        "external_layer_sha256": EXPECTED_LAYER_SHA256,
        "external_chunk_sha256": EXPECTED_CHUNK_SHA256,
        "external_recurrent_sha256": EXPECTED_RECURRENT_SHA256,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "active_layers": sum(row["state_rms"] is not None for row in rows),
        "active_futureseed_routes": sum(row["seed_applied"] for row in rows),
        "per_layer": rows,
        "matched_parent": getattr(model, "_comba_parent_metadata", None),
    }
