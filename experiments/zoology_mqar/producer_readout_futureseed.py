from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


READOUT_RANK = 32
RESIDUAL_CAP = 0.5
EXPECTED_PARAMETER_DELTA = 12_288


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


class ProducerNativeReadoutFusion(nn.Module):
    """Decode a transferred state in producer coordinates and fuse it boundedly."""

    def __init__(self, d_model: int, *, rank: int = READOUT_RANK) -> None:
        super().__init__()
        if d_model != 128 or rank != READOUT_RANK:
            raise ValueError("P-FS2-010 is fixed at D128 and rank32")
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.hidden_proj = nn.Linear(d_model, rank, bias=False)
        self.read_proj = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.enabled = True
        self.capture = False
        self.last_capture: dict[str, torch.Tensor] = {}

    def reset_identity(self) -> None:
        nn.init.zeros_(self.out_proj.weight)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_capture(self, enabled: bool) -> None:
        self.capture = bool(enabled)
        if enabled:
            self.last_capture.clear()

    @staticmethod
    def decode(
        producer: ZoologyGDN2FutureSeedMixer,
        hidden_states: torch.Tensor,
        transferred_state: torch.Tensor,
    ) -> torch.Tensor:
        layer = producer.layer
        if type(layer) is not GatedDeltaNet2:
            raise TypeError("Producer-native readout requires exact official GDN2")
        batch_size, sequence_length, _channels = hidden_states.shape
        expected_state = (
            batch_size,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
        )
        if tuple(transferred_state.shape) != expected_state:
            raise ValueError(
                "Transferred state shape mismatch: "
                f"{tuple(transferred_state.shape)} != {expected_state}"
            )

        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            query, _conv_state = layer.q_conv1d(
                x=layer.q_proj(hidden_states),
                cache=None,
                output_final_state=False,
            )
            query = F.normalize(
                query.float().view(
                    batch_size,
                    sequence_length,
                    layer.num_heads,
                    layer.head_k_dim,
                ),
                dim=-1,
                p=2.0,
            )
            readout = torch.einsum(
                "bthk,bhkv->bthv",
                query,
                transferred_state.float(),
            )
            gate = layer.g_proj(hidden_states).view(
                batch_size,
                sequence_length,
                layer.num_v_heads,
                layer.head_v_dim,
            )
            readout = layer.o_norm(readout.to(hidden_states.dtype), gate)
            decoded = layer.o_proj(
                readout.reshape(batch_size, sequence_length, -1)
            )
        return decoded.to(hidden_states.dtype)

    def fuse(
        self,
        hidden_states: torch.Tensor,
        decoded_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = F.silu(self.hidden_proj(hidden_states)) * self.read_proj(
            decoded_state
        )
        raw = self.out_proj(feature)
        hidden_rms = hidden_states.float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        residual = RESIDUAL_CAP * hidden_rms * torch.tanh(
            raw.float() / hidden_rms
        )
        residual = residual.to(hidden_states.dtype)
        return hidden_states + residual, feature, residual

    def forward(
        self,
        producer: ZoologyGDN2FutureSeedMixer,
        hidden_states: torch.Tensor,
        transferred_state: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return hidden_states
        decoded = self.decode(producer, hidden_states, transferred_state)
        output, feature, residual = self.fuse(hidden_states, decoded)
        if self.capture:
            self.last_capture = {
                "hidden": hidden_states.detach(),
                "state": transferred_state.detach(),
                "decoded": decoded.detach(),
                "feature": feature.detach(),
                "residual": residual.detach(),
            }
        return output


class ProducerReadoutFutureSeedBackbone(FutureSeedLMBackbone):
    """Native FutureSeed plus one producer-native readout on each adjacent edge."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config=config)
        if config.d_model != 128 or config.n_layers != 2:
            raise ValueError("P-FS2-010 is fixed at D128/L2")
        self.readout_fusions = nn.ModuleList(
            ProducerNativeReadoutFusion(config.d_model)
            for _ in range(config.n_layers - 1)
        )

    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scales = {
            float(layer.sequence_mixer.future_seed_scale)
            for layer in self.layers
            if isinstance(layer.sequence_mixer, ZoologyGDN2FutureSeedMixer)
        }
        if scales != {1.0}:
            raise RuntimeError("P-FS2-010 requires native FutureSeed scale 1")

        residual = None
        terminal_state = None
        producer = None
        edge_index = 0
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if type(mixer) is not ZoologyGDN2FutureSeedMixer:
                raise TypeError("P-FS2-010 requires exact native GDN2 mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            initial_state = None
            if terminal_state is not None:
                if producer is None:
                    raise RuntimeError("Missing producer for transferred state")
                initial_state = mixer.make_initial_state(terminal_state)
                hidden_states = self.readout_fusions[edge_index](
                    producer,
                    hidden_states,
                    initial_state,
                )
                edge_index += 1
            hidden_states, terminal_state = mixer.forward_with_state(
                hidden_states,
                initial_state=initial_state,
            )
            producer = mixer

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        if edge_index != len(self.readout_fusions):
            raise RuntimeError("Not all producer-native readout edges executed")
        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class ProducerReadoutFutureSeedLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("P-FS2-010 requires multiplier=1")

        self.backbone = ProducerReadoutFutureSeedBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        for fusion in self.backbone.readout_fusions:
            fusion.reset_identity()
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

    def set_producer_readout_enabled(self, enabled: bool) -> None:
        for fusion in self.backbone.readout_fusions:
            fusion.set_enabled(enabled)


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("backbone.readout_fusions."):
            loaded[name] = tensor
            continue
        if name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {name}")
        parent = control_state[name]
        if parent.shape != tensor.shape:
            raise RuntimeError(
                f"Matched tensor shape mismatch for {name}: "
                f"{tuple(parent.shape)} != {tuple(tensor.shape)}"
            )
        loaded[name] = parent
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.readout_fusions.")
    ]
    for name, parameter in sorted(rows):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def producer_readout_diagnostics(
    model: ProducerReadoutFutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    fusions = list(model.backbone.readout_fusions)
    for fusion in fusions:
        fusion.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for fusion in fusions:
            fusion.set_capture(False)

    rows = []
    for edge_index, fusion in enumerate(fusions):
        captured = fusion.last_capture
        required = {"hidden", "state", "decoded", "feature", "residual"}
        if set(captured) != required:
            raise RuntimeError(f"Incomplete P-FS2-010 capture: {set(captured)}")
        hidden = captured["hidden"].float()
        state = captured["state"].float()
        decoded = captured["decoded"].float()
        feature = captured["feature"].float()
        residual = captured["residual"].float()
        read_board = decoded.square().mean(dim=(1, 2)).sqrt()
        read_token = decoded.square().mean(dim=(0, 2)).sqrt()
        state_board = state.square().mean(dim=(1, 2, 3)).sqrt()
        out_rms = _rms(fusion.out_proj.weight.detach())
        rows.append(
            {
                "edge": edge_index,
                "read_rms": float(_rms(decoded).item()),
                "read_board_std": float(read_board.std(unbiased=False).item()),
                "read_token_std": float(read_token.std(unbiased=False).item()),
                "feature_rms": float(_rms(feature).item()),
                "residual_rms": float(_rms(residual).item()),
                "residual_relative_rms": float(
                    (_rms(residual) / _rms(hidden).clamp_min(1e-8)).item()
                ),
                "residual_abs_max_over_hidden_rms": float(
                    (
                        residual.abs().amax(dim=-1)
                        / hidden.square().mean(dim=-1).sqrt().clamp_min(1e-8)
                    ).max().item()
                ),
                "state_rms": float(_rms(state).item()),
                "state_board_std": float(state_board.std(unbiased=False).item()),
                "out_projection_rms": float(out_rms.item()),
            }
        )
    return {
        "active_edges": len(rows),
        "rank": READOUT_RANK,
        "residual_cap": RESIDUAL_CAP,
        "new_parameters": sum(
            parameter.numel()
            for fusion in fusions
            for parameter in fusion.parameters()
        ),
        "new_persistent_state_values": 0,
        "new_official_scans_per_layer": 0,
        "per_edge": rows,
    }
