from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)


EXPECTED_SDM_SHA = "183e7df809131b80ad4393741029d0f20fc3640b"
EXPECTED_SDM_TREE_SHA256 = (
    "ed55e7196c8d59e8f416d938f82fa280dff4a3672d73b64f5741b0fc4ff11d29"
)
EXPECTED_SDM_SOURCE_SHA256 = {
    "layer.py": "0b14fbc760423e3c39e810a8e583681f1e819d33f683c220433563337cb55da6",
    "memory_ops.py": "b4885bdb9d0c8fd72c730c8ffbfa1c37a25c3717c226eb3900787302f00f4468",
    "cache.py": "3c529ba708280946cd63d6e95499090f33c04758d9ebfabf40b2ade570f48926",
}
MODEL_WIDTH = 128
HOST_HEADS = 4
HOST_HEAD_DIM = 32
SDM_HEADS = 1
SDM_SLOTS = 1024
SDM_READS = 8
SDM_WRITES = 8
SDM_BLOCK_SIZE = 64
EXPECTED_STATE_VALUES_PER_LAYER = SDM_HEADS * SDM_SLOTS * MODEL_WIDTH


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_tree_hash(root: Path, relative: str) -> str:
    files = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", relative], text=True
    ).splitlines()
    if not files:
        raise RuntimeError(f"No tracked SDM files under {relative}")
    digest = hashlib.sha256()
    for name in sorted(files):
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Tracked SDM source is missing: {path}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def external_sdm_provenance() -> dict[str, Any]:
    root_text = os.environ.get("SDM_REPO_ROOT")
    expected = os.environ.get("SDM_EXPECTED_SHA")
    if not root_text or expected != EXPECTED_SDM_SHA:
        raise RuntimeError("Pinned official SDM source SHA was not asserted")
    root = Path(root_text).resolve()
    if _git_head(root) != EXPECTED_SDM_SHA:
        raise RuntimeError(f"Official SDM checkout drifted: {root}")
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(f"Official SDM checkout is dirty: {root}")
    package = root / "lingua" / "sparse_delta_memory"
    hashes = {name: _sha256(package / name) for name in EXPECTED_SDM_SOURCE_SHA256}
    tree_hash = _tracked_tree_hash(root, "lingua/sparse_delta_memory")
    if hashes != EXPECTED_SDM_SOURCE_SHA256 or tree_hash != EXPECTED_SDM_TREE_SHA256:
        raise RuntimeError(
            f"Official SDM source hashes drifted: tree={tree_hash}, files={hashes}"
        )
    return {
        "root": str(root),
        "sha": EXPECTED_SDM_SHA,
        "tree_sha256": tree_hash,
        "source_sha256": hashes,
        "license": "CC-BY-NC-4.0",
        "redistributed_source": False,
    }


def load_external_sdm():
    provenance = external_sdm_provenance()
    root = Path(provenance["root"])
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("lingua.sparse_delta_memory")
    cache_module = importlib.import_module("lingua.sparse_delta_memory.cache")
    layer_class = module.SparseDeltaMemory
    args_class = module.SparseDeltaMemoryArgs
    layer_path = Path(inspect.getfile(layer_class)).resolve()
    if layer_path != root / "lingua" / "sparse_delta_memory" / "layer.py":
        raise RuntimeError(f"Unexpected official SDM module: {layer_path}")
    return layer_class, args_class, cache_module.SDMLayerState, provenance


def _tensor_hash(rows: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(rows):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ZoologyOfficialSDMFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Pinned official sparse delta memory with full-bank native FutureSeed."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = HOST_HEADS,
        head_dim: int = HOST_HEAD_DIM,
        expand_v: float = 1.0,
        conv_size: int = 4,
        future_seed_scale: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if (
            d_model != MODEL_WIDTH
            or num_heads != HOST_HEADS
            or head_dim != HOST_HEAD_DIM
            or expand_v != 1.0
            or conv_size != 4
        ):
            raise ValueError("P-GDN3-064 fixes D128 host H4/K32/V32")
        layer_class, args_class, cache_class, provenance = load_external_sdm()
        args = args_class(
            dim=d_model,
            num_writes=SDM_WRITES,
            num_reads=SDM_READS,
            slots_per_head=SDM_SLOTS,
            memory_block_size=SDM_BLOCK_SIZE,
            normalize_readings=True,
            backprop_on_memory=False,
            output_gate=True,
            log_memory_access_stats=True,
            log_memory_norms=False,
            num_heads=SDM_HEADS,
        )
        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = layer_class(args, layer_id=self.layer_idx)
        self.layer.args.log_memory_access_stats = False
        self.cache_class = cache_class
        self.future_seed_logit = nn.Parameter(torch.zeros(1, SDM_HEADS, 1, 1))
        self.source_path = str(Path(inspect.getfile(layer_class)).resolve())
        self.external_provenance = provenance
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None
        self.last_state_rms: Optional[torch.Tensor] = None
        self.last_state_board_std: Optional[torch.Tensor] = None
        self.last_active_slots: Optional[int] = None
        self.last_terminal_requires_grad: Optional[bool] = None
        self.last_terminal_grad_fn: Optional[str] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _terminal = self.forward_with_state(
            hidden_states,
            initial_state=None,
        )
        return output

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = hidden_states.shape[0]
        cache = None
        if initial_state is not None:
            expected = (batch, SDM_HEADS, SDM_SLOTS, MODEL_WIDTH)
            if tuple(initial_state.shape) != expected:
                raise RuntimeError(
                    f"Official SDM FutureSeed must be {expected}, got {tuple(initial_state.shape)}"
                )
            cache = self.cache_class(
                initial_state.reshape(batch * SDM_HEADS * SDM_SLOTS, MODEL_WIDTH),
                seq_len=0,
            )
        source_dtype = hidden_states.dtype
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, new_cache = self.layer(
                hidden_states.to(torch.bfloat16) if hidden_states.is_cuda else hidden_states,
                cache=cache,
            )
        if new_cache is None:
            raise RuntimeError("Official SDM did not return terminal memory")
        terminal = new_cache.memory.reshape(
            batch, SDM_HEADS, SDM_SLOTS, MODEL_WIDTH
        )
        per_board = terminal.float().square().mean(dim=(-1, -2)).sqrt()
        active = terminal.float().norm(dim=-1) > 1e-6
        self.last_state_rms = per_board.mean().detach()
        self.last_state_board_std = per_board.std(unbiased=False).detach()
        self.last_active_slots = int(active.any(dim=(0, 1)).sum().item())
        self.last_terminal_requires_grad = bool(terminal.requires_grad)
        self.last_terminal_grad_fn = (
            None if terminal.grad_fn is None else type(terminal.grad_fn).__name__
        )
        return output.to(source_dtype), terminal

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_STATE_VALUES_PER_LAYER


def _shared_parent_rows(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, parent)
        for name, tensor in model.named_parameters()
        if (parent := parent_state.get(name)) is not None and parent.shape == tensor.shape
    ]


def load_matched_parent_state(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> None:
    target = model.state_dict()
    rows = _shared_parent_rows(model, parent_state)
    if len(rows) < 20:
        raise RuntimeError(f"Too few shared parent tensors: {len(rows)}")
    for name, parent in rows:
        target[name] = parent.detach().clone()
    model.load_state_dict(target, strict=True)
    loaded = [(name, model.state_dict()[name]) for name, _ in rows]
    source_hash = _tensor_hash(rows)
    loaded_hash = _tensor_hash(loaded)
    if source_hash != loaded_hash:
        raise RuntimeError("Shared parent tensor mapping is not exact")
    model._official_sdm_parent_metadata = {
        "tensor_count": len(rows),
        "numel": sum(tensor.numel() for _, tensor in rows),
        "source_hash": source_hash,
        "loaded_hash": loaded_hash,
        "names": [name for name, _ in rows],
    }


def parent_parameter_hash(model: torch.nn.Module) -> str:
    metadata = getattr(model, "_official_sdm_parent_metadata", None)
    if metadata is None:
        raise RuntimeError("Official SDM matched-parent metadata is missing")
    names = set(metadata["names"])
    return _tensor_hash(
        [(name, parameter) for name, parameter in model.named_parameters() if name in names]
    )


@torch.no_grad()
def official_sdm_diagnostics(
    model: torch.nn.Module,
    diagnostic_inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyOfficialSDMFutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two official SDM layers, got {len(mixers)}")
    for mixer in mixers:
        mixer.layer.args.log_memory_access_stats = True
        mixer.layer.get_memory_access_stats(reset=True)
    was_training = model.training
    model.eval()(diagnostic_inputs)
    rows = []
    try:
        for mixer in mixers:
            access = mixer.layer.get_memory_access_stats(reset=True)
            rows.append(
                {
                    "layer_idx": mixer.layer_idx,
                    "state_rms": None if mixer.last_state_rms is None else float(mixer.last_state_rms),
                    "state_board_std": (
                        None
                        if mixer.last_state_board_std is None
                        else float(mixer.last_state_board_std)
                    ),
                    "active_slots": mixer.last_active_slots,
                    "terminal_requires_grad": mixer.last_terminal_requires_grad,
                    "terminal_grad_fn": mixer.last_terminal_grad_fn,
                    "seed_applied": mixer.last_seed_gate is not None,
                    "seed_gate": None if mixer.last_seed_gate is None else float(mixer.last_seed_gate),
                    "access": access,
                }
            )
    finally:
        for mixer in mixers:
            mixer.layer.args.log_memory_access_stats = False
        model.train(was_training)
    return {
        "external": mixers[0].external_provenance,
        "heads": SDM_HEADS,
        "slots_per_head": SDM_SLOTS,
        "reads": SDM_READS,
        "writes": SDM_WRITES,
        "block_size": SDM_BLOCK_SIZE,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "active_layers": sum(row["active_slots"] is not None for row in rows),
        "active_futureseed_routes": sum(row["seed_applied"] for row in rows),
        "producer_terminal_gradient_connected": bool(rows[0]["terminal_requires_grad"]),
        "per_layer": rows,
        "matched_parent": getattr(model, "_official_sdm_parent_metadata", None),
    }
