from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


BLOCK_SIZE = 128
MODEL_WIDTH = 128
MODEL_HEADS = 4
LOCAL_HEAD_DIM = MODEL_WIDTH // MODEL_HEADS
EXPECTED_PARAMETER_DELTA_PER_LAYER = 4 * MODEL_WIDTH * MODEL_WIDTH + MODEL_HEADS
EXPECTED_PARAMETER_DELTA = 2 * EXPECTED_PARAMETER_DELTA_PER_LAYER


class ZoologyLocalBindingGDN2FutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native GDN2 plus a block-local causal binding path."""

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        *,
        num_heads: int = MODEL_HEADS,
        head_dim: int = LOCAL_HEAD_DIM,
        **kwargs: Any,
    ) -> None:
        if d_model != MODEL_WIDTH or num_heads != MODEL_HEADS:
            raise ValueError("P-GDN3-055 is fixed to D128/H4")
        if d_model != num_heads * head_dim:
            raise ValueError("Local attention requires d_model == heads * head_dim")
        super().__init__(
            d_model=d_model,
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            **kwargs,
        )
        self.local_num_heads = int(num_heads)
        self.local_head_dim = int(head_dim)
        self.local_q = nn.Linear(d_model, d_model, bias=False)
        self.local_k = nn.Linear(d_model, d_model, bias=False)
        self.local_v = nn.Linear(d_model, d_model, bias=False)
        self.local_o = nn.Linear(d_model, d_model, bias=False)
        self.local_gate = nn.Parameter(torch.zeros(num_heads))
        self.capture_local_binding = False
        self._local_binding_capture: dict[str, torch.Tensor] = {}

    def set_local_binding_capture(self, enabled: bool) -> None:
        self.capture_local_binding = bool(enabled)
        if enabled:
            self._local_binding_capture.clear()

    def _shape_local(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _width = tensor.shape
        blocks = length // BLOCK_SIZE
        return (
            tensor.reshape(
                batch,
                blocks,
                BLOCK_SIZE,
                self.local_num_heads,
                self.local_head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(
                batch * blocks,
                self.local_num_heads,
                BLOCK_SIZE,
                self.local_head_dim,
            )
        )

    def _local_binding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden_states.shape
        if length % BLOCK_SIZE:
            raise ValueError(
                f"P-GDN3-055 requires sequence length divisible by {BLOCK_SIZE}"
            )
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            query = self._shape_local(self.local_q(hidden_states))
            key = self._shape_local(self.local_k(hidden_states))
            value = self._shape_local(self.local_v(hidden_states))
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=True,
                    scale=self.local_head_dim**-0.5,
                )
            blocks = length // BLOCK_SIZE
            attended = (
                attended.reshape(
                    batch,
                    blocks,
                    self.local_num_heads,
                    BLOCK_SIZE,
                    self.local_head_dim,
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(batch, length, width)
            )
            gate = torch.tanh(self.local_gate).to(attended.dtype)
            gated = (
                attended.reshape(
                    batch,
                    length,
                    self.local_num_heads,
                    self.local_head_dim,
                )
                * gate.view(1, 1, self.local_num_heads, 1)
            ).reshape(batch, length, width)
            local_output = self.local_o(gated)

        if self.capture_local_binding:
            with torch.no_grad():
                local_float = local_output.float()
                board_rms = local_float.square().mean(dim=(1, 2)).sqrt()
                token_rms = local_float.square().mean(dim=-1).sqrt()
                self._local_binding_capture = {
                    "gate": gate.detach().float(),
                    "attended_rms": attended.detach().float().square().mean().sqrt(),
                    "local_output_rms": local_float.square().mean().sqrt(),
                    "local_output_board_std": board_rms.std(unbiased=False),
                    "local_output_token_std": token_rms.std(unbiased=False),
                }
        return local_output.to(hidden_states.dtype)

    def _merge(
        self,
        hidden_states: torch.Tensor,
        main_output: torch.Tensor,
    ) -> torch.Tensor:
        local_output = self._local_binding(hidden_states)
        if self.capture_local_binding:
            with torch.no_grad():
                main_rms = main_output.float().square().mean().sqrt().clamp_min(1e-8)
                self._local_binding_capture["local_relative_rms"] = (
                    local_output.float().square().mean().sqrt() / main_rms
                )
        return main_output + local_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        main_output = super().forward(hidden_states)
        return self._merge(hidden_states, main_output)

    def forward_with_state(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_output, terminal_state = super().forward_with_state(
            hidden_states,
            initial_state=initial_state,
        )
        return self._merge(hidden_states, main_output), terminal_state


def _is_extra_parameter(name: str) -> bool:
    return (
        ".sequence_mixer.local_" in name
        or name.endswith(".sequence_mixer.local_gate")
    )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extras = 0
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = (
                torch.zeros_like(tensor)
                if name.endswith(".sequence_mixer.local_gate")
                else tensor
            )
            extras += tensor.numel()
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
    if extras != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_DELTA} new parameters, found {extras}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not _is_extra_parameter(name)
    ]
    for name, parameter in sorted(tensors):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def local_binding_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(
            block.sequence_mixer,
            ZoologyLocalBindingGDN2FutureSeedMixer,
        )
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use the local-binding mixer")
    for mixer in mixers:
        mixer.set_local_binding_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.set_local_binding_capture(False)

    rows = []
    active_paths = 0
    for mixer in mixers:
        captured = mixer._local_binding_capture
        if not captured:
            raise RuntimeError("Local-binding capture was not produced")
        gates = captured["gate"].flatten()
        active_paths += int((gates.abs() >= 1e-3).sum().item())
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "gate_abs_mean": float(gates.abs().mean().item()),
                "gate_abs_min": float(gates.abs().min().item()),
                "gate_abs_max": float(gates.abs().max().item()),
                "gate_values": [float(value) for value in gates.tolist()],
                **{
                    name: float(value.float().mean().item())
                    for name, value in captured.items()
                    if name != "gate"
                },
            }
        )
    return {
        "active_layers": len(rows),
        "active_paths": active_paths,
        "total_paths": len(rows) * MODEL_HEADS,
        "block_size": BLOCK_SIZE,
        "local_heads": MODEL_HEADS,
        "local_head_dim": LOCAL_HEAD_DIM,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "persistent_state_delta": 0,
        "official_gdn2_scans_per_layer": 1,
        "flash_sdpa_calls_per_layer": 1,
        "layers": rows,
    }
