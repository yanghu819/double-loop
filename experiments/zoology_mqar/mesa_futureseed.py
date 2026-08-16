from __future__ import annotations

import hashlib
import inspect
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.mesa_net import MesaNet
from fla.ops.mesa_net import chunk_mesa_net
from fla.ops.mesa_net.chunk import ChunkMesaNetFunction

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    ZoologyGDN2FutureSeedMixer,
)


MODEL_WIDTH = 128
NUM_HEADS = 4
HEAD_DIM = 32
MAX_CG_ITERATIONS = 30
LAMBDA_LOWER_BOUND = 0.25
EXPECTED_STATE_VALUES_PER_LAYER = 2 * NUM_HEADS * HEAD_DIM * HEAD_DIM
EXPECTED_MIXER_PARAMETERS = 84_140
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_MESA_SOURCE_SHA256 = {
    "fla/layers/mesa_net.py": (
        "19db3cd2c15e25a3350d312b298877ccc91a3c45fe8f79ba44cf5e8b5dace3db"
    ),
    "fla/ops/mesa_net/chunk.py": (
        "27b5ee4058ac8a12bc1c819fb6ffc87c09fc763df3a0b6050e7c84318aadfcb3"
    ),
    "fla/ops/mesa_net/chunk_h_fwd.py": (
        "8edaa0b49f04a43ab4b5cad83daadfa1ef0b12b6e3f564422c0dcb23c00c0094"
    ),
    "fla/ops/mesa_net/naive.py": (
        "cefde34feef551e80d6f427220a9cfbfc39d772cc7bb8c1e1044edb749c9bf30"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def official_mesa_provenance() -> dict[str, Any]:
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned official FLA source SHA was not asserted")
    root_text = os.environ.get("FLA_SOURCE_ROOT")
    if root_text is None or Path(root_text).resolve() != EXPECTED_FLA_ROOT:
        raise RuntimeError(f"Unexpected pinned FLA root: {root_text}")

    resolved = {
        relative: _sha256(EXPECTED_FLA_ROOT / relative)
        for relative in EXPECTED_MESA_SOURCE_SHA256
    }
    if resolved != EXPECTED_MESA_SOURCE_SHA256:
        raise RuntimeError(f"Pinned official Mesa source drifted: {resolved}")

    expected_paths = {
        Path(inspect.getfile(MesaNet)).resolve(): EXPECTED_FLA_ROOT
        / "fla/layers/mesa_net.py",
        Path(inspect.getfile(chunk_mesa_net)).resolve(): EXPECTED_FLA_ROOT
        / "fla/ops/mesa_net/chunk.py",
        Path(inspect.getfile(ChunkMesaNetFunction)).resolve(): EXPECTED_FLA_ROOT
        / "fla/ops/mesa_net/chunk.py",
    }
    if any(actual != expected for actual, expected in expected_paths.items()):
        raise RuntimeError(f"Official Mesa import escaped pinned FLA: {expected_paths}")
    return {
        "root": str(EXPECTED_FLA_ROOT),
        "source_sha": PINNED_FLA_SHA,
        "source_sha256": resolved,
        "layer_source": str(Path(inspect.getfile(MesaNet)).resolve()),
        "operator_source": str(Path(inspect.getfile(chunk_mesa_net)).resolve()),
        "license": "MIT",
        "redistributed_source": False,
    }


def _tensor_hash(rows: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(rows):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ZoologyMesaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Official Mesa least-squares memory with native two-statistic FutureSeed."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        expand_v: float = 1.0,
        conv_size: int = 4,
        future_seed_scale: float = 1.0,
    ) -> None:
        nn.Module.__init__(self)
        if (
            d_model != MODEL_WIDTH
            or num_heads != NUM_HEADS
            or head_dim != HEAD_DIM
            or expand_v != 1.0
            or conv_size != 4
        ):
            raise ValueError("P-GDN3-066 fixes D128/H4/K32/V32 and conv4")
        self.external_provenance = official_mesa_provenance()
        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = MesaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            mode="chunk",
            use_output_gate=True,
            use_short_conv=True,
            conv_size=conv_size,
            layer_idx=layer_idx,
            lambda_lower_bound=LAMBDA_LOWER_BOUND,
            max_cg_step_training=MAX_CG_ITERATIONS,
            max_cg_step_decoding=MAX_CG_ITERATIONS,
        )
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, self.layer.num_heads, 1, 1)
        )
        self.source_path = self.external_provenance["layer_source"]
        self.collect_diagnostics = False
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None
        self.last_terminal_requires_grad: Optional[bool] = None
        self.last_terminal_grad_fn: Optional[str] = None
        self.last_terminal_state: Optional[torch.Tensor] = None
        self.last_cg_output_relative_rms: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _terminal = self.forward_with_state(
            hidden_states,
            initial_state=None,
        )
        return output

    def make_initial_state(self, terminal_state: torch.Tensor) -> torch.Tensor:
        if terminal_state.ndim != 5 or terminal_state.shape[0] != 2:
            raise RuntimeError(
                "Mesa FutureSeed must stack [Hkk,Hkv] as [2,B,H,K,K]"
            )
        # One positive scale preserves the Hkk/Hkv sufficient-statistic ratio.
        rms = terminal_state.float().square().mean(
            dim=(0, 3, 4),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.future_seed_logit).float().unsqueeze(0)
        seed = terminal_state.float() / rms * gate * self.future_seed_scale
        self.last_seed_rms = rms.detach()
        self.last_seed_norm = seed.detach().norm(dim=(-1, -2)).mean()
        self.last_seed_gate = gate.detach().mean()
        return seed

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = hidden_states.shape[0]
        h_kk_init = None
        h_kv_init = None
        if initial_state is not None:
            expected = (2, batch, NUM_HEADS, HEAD_DIM, HEAD_DIM)
            if tuple(initial_state.shape) != expected:
                raise RuntimeError(
                    f"Mesa FutureSeed must be {expected}, got {tuple(initial_state.shape)}"
                )
            h_kk_init = initial_state[0].float().contiguous()
            h_kv_init = initial_state[1].float().contiguous()

        source_dtype = hidden_states.dtype
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            q, _ = self.layer.q_conv1d(
                x=self.layer.q_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=None,
            )
            k, _ = self.layer.k_conv1d(
                x=self.layer.k_proj(hidden_states),
                cache=None,
                output_final_state=False,
                cu_seqlens=None,
            )
            v = self.layer.v_proj(hidden_states)
            q, k = map(
                lambda tensor: rearrange(
                    tensor,
                    "b t (h d) -> b t h d",
                    d=HEAD_DIM,
                ),
                (q, k),
            )
            v = rearrange(v, "b t (h d) -> b t h d", d=HEAD_DIM)
            beta = self.layer.b_proj(hidden_states).float().sigmoid()
            g = F.logsigmoid(self.layer.a_proj(hidden_states).float())
            lamb = F.softplus(self.layer.lambda_params.float())
            lamb = lamb.reshape(NUM_HEADS, HEAD_DIM) + LAMBDA_LOWER_BOUND
            raw_output, h_kk, h_kv = chunk_mesa_net(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                lamb=lamb,
                h_kk_init=h_kk_init,
                h_kv_init=h_kv_init,
                output_final_state=True,
                max_CG_iteration=MAX_CG_ITERATIONS,
                use_qk_l2norm_in_kernel=True,
            )
            output_gate = rearrange(
                self.layer.g_proj(hidden_states),
                "b t (h d) -> b t h d",
                d=HEAD_DIM,
            )
            output = self.layer.o_norm(raw_output, output_gate)
            output = rearrange(output, "b t h d -> b t (h d)")
            output = self.layer.o_proj(output)

        if h_kk is None or h_kv is None:
            raise RuntimeError("Official Mesa operator did not return terminal states")
        terminal = torch.stack((h_kk.float(), h_kv.float()), dim=0)
        self.last_terminal_requires_grad = bool(terminal.requires_grad)
        self.last_terminal_grad_fn = (
            None if terminal.grad_fn is None else type(terminal.grad_fn).__name__
        )
        if self.collect_diagnostics:
            self.last_terminal_state = terminal.detach()
            with torch.no_grad():
                q_final = F.normalize(q[:, -1].float(), dim=-1)
                system = h_kk.float() + torch.diag_embed(lamb)[None]
                exact_query = torch.linalg.solve(system, q_final.unsqueeze(-1)).squeeze(-1)
                exact_output = torch.einsum(
                    "bhk,bhkv->bhv",
                    exact_query,
                    h_kv.float(),
                )
                numerator = (
                    raw_output[:, -1].float() - exact_output
                ).square().mean().sqrt()
                denominator = exact_output.square().mean().sqrt().clamp_min(1e-8)
                self.last_cg_output_relative_rms = (numerator / denominator).detach()
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
    model._mesa_parent_metadata = {
        "tensor_count": len(rows),
        "numel": sum(tensor.numel() for _, tensor in rows),
        "source_hash": source_hash,
        "loaded_hash": loaded_hash,
        "names": [name for name, _ in rows],
    }


def parent_parameter_hash(model: torch.nn.Module) -> str:
    metadata = getattr(model, "_mesa_parent_metadata", None)
    if metadata is None:
        raise RuntimeError("Mesa matched-parent metadata is missing")
    names = set(metadata["names"])
    return _tensor_hash(
        [(name, parameter) for name, parameter in model.named_parameters() if name in names]
    )


@torch.no_grad()
def mesa_futureseed_diagnostics(
    model: torch.nn.Module,
    diagnostic_inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyMesaFutureSeedMixer)
    ]
    if len(mixers) != 2:
        raise RuntimeError(f"Expected two official Mesa layers, got {len(mixers)}")
    was_training = model.training
    for mixer in mixers:
        mixer.collect_diagnostics = True
    try:
        model.eval()(diagnostic_inputs)
    finally:
        for mixer in mixers:
            mixer.collect_diagnostics = False
        model.train(was_training)

    rows = []
    for mixer in mixers:
        if mixer.last_terminal_state is None:
            raise RuntimeError("Mesa terminal diagnostics were not collected")
        h_kk, h_kv = mixer.last_terminal_state.float()
        symmetric = 0.5 * (h_kk + h_kk.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        per_board_rms = mixer.last_terminal_state.float().square().mean(
            dim=(0, 3, 4)
        ).sqrt().mean(dim=-1)
        trace = eigenvalues.clamp_min(0).sum(dim=-1)
        effective_rank = trace.square() / eigenvalues.clamp_min(0).square().sum(
            dim=-1
        ).clamp_min(1e-8)
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "state_rms": float(mixer.last_terminal_state.float().square().mean().sqrt()),
                "state_board_std": float(per_board_rms.std(unbiased=False)),
                "hkk_rms": float(h_kk.square().mean().sqrt()),
                "hkv_rms": float(h_kv.square().mean().sqrt()),
                "hkk_symmetry_max_abs": float(
                    (h_kk - h_kk.transpose(-1, -2)).abs().max()
                ),
                "hkk_min_eigenvalue": float(eigenvalues.min()),
                "hkk_effective_rank": float(effective_rank.mean()),
                "cg_output_relative_rms": (
                    None
                    if mixer.last_cg_output_relative_rms is None
                    else float(mixer.last_cg_output_relative_rms)
                ),
                "terminal_requires_grad": mixer.last_terminal_requires_grad,
                "terminal_grad_fn": mixer.last_terminal_grad_fn,
                "seed_applied": mixer.last_seed_gate is not None,
                "seed_gate": (
                    None
                    if mixer.last_seed_gate is None
                    else float(mixer.last_seed_gate)
                ),
                "seed_joint_rms": (
                    None
                    if mixer.last_seed_rms is None
                    else float(mixer.last_seed_rms.mean())
                ),
            }
        )
    return {
        "external": mixers[0].external_provenance,
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "max_cg_iterations": MAX_CG_ITERATIONS,
        "lambda_lower_bound": LAMBDA_LOWER_BOUND,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "active_layers": len(rows),
        "active_futureseed_routes": sum(row["seed_applied"] for row in rows),
        "producer_terminal_gradient_connected": bool(rows[0]["terminal_requires_grad"]),
        "per_layer": rows,
        "matched_parent": getattr(model, "_mesa_parent_metadata", None),
    }
