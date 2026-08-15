from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2
from zoology.config import ModelConfig
from zoology.model import _compute_state_size, _init_weights

from experiments.zoology_mqar.gdn2_futureseed import (
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
)


MODEL_WIDTH = 128
MODEL_HEADS = 4
HEAD_DIM = 32
TRACE_WIDTH = 4
MAIN_STATE_VALUES = MODEL_HEADS * HEAD_DIM * HEAD_DIM
COMPANION_STATE_VALUES = MAIN_STATE_VALUES
SHARED_PROJECTION_PARAMETERS = MODEL_WIDTH * MODEL_WIDTH
SHARED_TRACE_PARAMETERS = MODEL_WIDTH * TRACE_WIDTH
READ_GATE_PARAMETERS = 2 * MODEL_HEADS
SEED_GATE_PARAMETERS = 2 * MODEL_HEADS
EXPECTED_PARAMETER_DELTA = (
    SHARED_PROJECTION_PARAMETERS
    + SHARED_TRACE_PARAMETERS
    + READ_GATE_PARAMETERS
    + SEED_GATE_PARAMETERS
)


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _normalize_last(tensor: torch.Tensor) -> torch.Tensor:
    rms = tensor.float().square().mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / rms.to(dtype=tensor.dtype)


class SharedEligibilityCompanionGDN2(nn.Module):
    """Native GDN2 plus a role-decoupled companion event memory."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != MODEL_WIDTH:
            raise ValueError("P-GDN3-056 requires D128")
        if base.num_heads != MODEL_HEADS or base.num_v_heads != MODEL_HEADS:
            raise ValueError("P-GDN3-056 requires H4")
        if base.head_k_dim != HEAD_DIM or base.head_v_dim != HEAD_DIM:
            raise ValueError("P-GDN3-056 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-056 requires chunk GDN2 with short convolution")
        if base.allow_neg_eigval:
            raise ValueError("P-GDN3-056 requires bounded positive erase gates")
        self.base = base
        self.companion_read_logit = nn.Parameter(torch.zeros(MODEL_HEADS))
        self.companion_seed_logit = nn.Parameter(torch.zeros(MODEL_HEADS))
        self.last_companion_state: Optional[torch.Tensor] = None
        self.capture_eligibility = False
        self._captured: dict[str, torch.Tensor] = {}

    @property
    def num_heads(self) -> int:
        return int(self.base.num_heads)

    @property
    def num_v_heads(self) -> int:
        return int(self.base.num_v_heads)

    @property
    def head_k_dim(self) -> int:
        return int(self.base.head_k_dim)

    @property
    def head_v_dim(self) -> int:
        return int(self.base.head_v_dim)

    @property
    def layer_idx(self) -> int:
        if self.base.layer_idx is None:
            raise RuntimeError("P-GDN3-056 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self.capture_eligibility = bool(enabled)
        if enabled:
            self._captured.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        del output_attentions
        if attention_mask is not None:
            raise ValueError("P-GDN3-056 uses fixed unpadded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-056 does not use packed sequences")
        read_address = kwargs.pop("read_address", None)
        write_address = kwargs.pop("write_address", None)
        companion_initial_state = kwargs.pop("companion_initial_state", None)
        if not isinstance(read_address, torch.Tensor) or not isinstance(
            write_address, torch.Tensor
        ):
            raise TypeError("Shared read and eligibility-write addresses are required")
        if read_address.shape != hidden_states.shape:
            raise ValueError("Read-address shape drifted")
        if write_address.shape != hidden_states.shape:
            raise ValueError("Write-address shape drifted")
        if companion_initial_state is not None and companion_initial_state.shape[1:] != (
            MODEL_HEADS,
            HEAD_DIM,
            HEAD_DIM,
        ):
            raise ValueError("Companion initial-state shape drifted")

        layer = self.base
        q_len = hidden_states.shape[1]
        last_state = get_layer_cache(layer, past_key_values)
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = layer.q_conv1d(
            x=layer.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
        )
        k, conv_state_k = layer.k_conv1d(
            x=layer.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
        )
        value, conv_state_v = layer.v_conv1d(
            x=layer.v_proj(hidden_states),
            cache=conv_state_v,
            output_final_state=use_cache,
        )
        decay = F.softplus(layer.f_proj(hidden_states).float() + layer.dt_bias)
        erase = layer.b_proj(hidden_states).sigmoid()
        write = layer.w_proj(hidden_states).sigmoid()
        q, k, decay, erase = (
            rearrange(tensor, "... (h d) -> ... h d", d=HEAD_DIM)
            for tensor in (q, k, decay, erase)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=HEAD_DIM)
        write = rearrange(write, "... (h d) -> ... h d", d=HEAD_DIM)
        decay = -layer.A_log.float().exp().unsqueeze(-1) * decay

        main_initial_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        main_output, main_state = chunk_gdn2(
            q=q,
            k=k,
            v=value,
            g=decay,
            b=erase,
            w=write,
            initial_state=main_initial_state,
            output_final_state=use_cache,
            use_qk_l2norm_in_kernel=True,
        )

        companion_q = rearrange(
            read_address,
            "b t (h d) -> b t h d",
            h=MODEL_HEADS,
            d=HEAD_DIM,
        ).to(dtype=q.dtype)
        companion_k = rearrange(
            write_address,
            "b t (h d) -> b t h d",
            h=MODEL_HEADS,
            d=HEAD_DIM,
        ).to(dtype=k.dtype)
        companion_output, companion_state = chunk_gdn2(
            q=companion_q,
            k=companion_k,
            v=value,
            g=decay,
            b=erase,
            w=write,
            initial_state=companion_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        if companion_state is None:
            raise RuntimeError("Companion official scan returned no terminal state")
        self.last_companion_state = companion_state

        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=main_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )

        read_gate = torch.tanh(self.companion_read_logit).to(main_output.dtype)
        companion_unit = _normalize_last(companion_output)
        output = main_output + (
            companion_unit
            * read_gate.view(1, 1, MODEL_HEADS, 1)
        )
        if self.capture_eligibility:
            if main_state is None:
                raise RuntimeError("Eligibility capture requires main terminal state")
            with torch.no_grad():
                companion_board = companion_state.float().square().mean(
                    dim=(-1, -2, -3)
                ).sqrt()
                output_board = companion_output.float().square().mean(
                    dim=(1, 2, 3)
                ).sqrt()
                output_token = companion_output.float().square().mean(
                    dim=-1
                ).sqrt()
                q_unit = F.normalize(companion_q.float(), dim=-1, eps=1e-6)
                k_unit = F.normalize(companion_k.float(), dim=-1, eps=1e-6)
                self._captured = {
                    "read_gate": read_gate.detach().float(),
                    "read_address": companion_q.detach(),
                    "write_address": companion_k.detach(),
                    "address_cosine": (q_unit * k_unit).sum(dim=-1).mean(),
                    "address_change_relative_rms": _rms(companion_k - companion_q)
                    / _rms(companion_q).clamp_min(1e-8),
                    "companion_output_rms": _rms(companion_output),
                    "companion_output_relative_rms": _rms(companion_output)
                    / _rms(main_output).clamp_min(1e-8),
                    "companion_output_board_std": output_board.std(unbiased=False),
                    "companion_output_token_std": output_token.std(unbiased=False),
                    "companion_state_rms": _rms(companion_state),
                    "companion_state_board_std": companion_board.std(unbiased=False),
                    "main_state_rms": _rms(main_state),
                }

        output = layer.o_norm(
            output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=HEAD_DIM,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologySharedEligibilityFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Native FutureSeed plus a shared-basis eligibility companion state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = SharedEligibilityCompanionGDN2(self.layer)
        self.last_companion_seed_gate: Optional[torch.Tensor] = None
        self.last_companion_seed_rms: Optional[torch.Tensor] = None

    def make_companion_initial_state(
        self,
        terminal_state: torch.Tensor,
    ) -> torch.Tensor:
        rms = terminal_state.float().square().mean(
            dim=(-1, -2),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.layer.companion_seed_logit).to(
            device=terminal_state.device,
            dtype=terminal_state.dtype,
        ).view(1, MODEL_HEADS, 1, 1)
        self.last_companion_seed_gate = gate.detach().float().mean()
        self.last_companion_seed_rms = rms.detach().float().mean()
        return terminal_state / rms.to(terminal_state.dtype) * gate

    def forward_with_dual_state(
        self,
        hidden_states: torch.Tensor,
        *,
        read_address: torch.Tensor,
        write_address: torch.Tensor,
        main_initial_state: Optional[torch.Tensor],
        companion_initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._new_cache(main_initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                read_address=read_address,
                write_address=write_address,
                companion_initial_state=companion_initial_state,
                past_key_values=cache,
                use_cache=True,
            )
        main_state = cache[self.layer_idx]["recurrent_state"]
        companion_state = self.layer.last_companion_state
        if main_state is None or companion_state is None:
            raise RuntimeError("Dual-state forward did not return both states")
        return output.to(hidden_states.dtype), main_state, companion_state


class SharedEligibilityBackbone(FutureSeedLMBackbone):
    """Build one shared event namespace and transport both memory planes."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.d_model != MODEL_WIDTH:
            raise ValueError("P-GDN3-056 requires D128")
        self.shared_event_projection = nn.Linear(
            MODEL_WIDTH,
            MODEL_WIDTH,
            bias=False,
        )
        self.shared_eligibility_conv = ShortConvolution(
            hidden_size=MODEL_WIDTH,
            kernel_size=TRACE_WIDTH,
            bias=False,
            activation=None,
        )
        self.last_event_read: Optional[torch.Tensor] = None
        self.last_event_write: Optional[torch.Tensor] = None

    def reset_event_path(self) -> None:
        with torch.no_grad():
            self.shared_event_projection.weight.copy_(
                torch.eye(
                    MODEL_WIDTH,
                    dtype=self.shared_event_projection.weight.dtype,
                    device=self.shared_event_projection.weight.device,
                )
            )
            self.shared_eligibility_conv.weight.zero_()
            self.shared_eligibility_conv.weight[:, 0, -1] = 1.0

    @staticmethod
    def normalize_anchor(anchor: torch.Tensor) -> torch.Tensor:
        return _normalize_last(anchor)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_anchor = self.embeddings.word_embeddings(input_ids)
        if self.embeddings.project_in is not None:
            token_anchor = self.embeddings.project_in(token_anchor)
        token_anchor = self.normalize_anchor(token_anchor)
        event_read = self.shared_event_projection(token_anchor)
        event_write, _ = self.shared_eligibility_conv(
            x=event_read,
            cache=None,
            output_final_state=False,
        )
        self.last_event_read = event_read.detach()
        self.last_event_write = event_write.detach()

        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        main_terminal = None
        companion_terminal = None
        for block in self.layers:
            mixer = block.sequence_mixer
            if not isinstance(mixer, ZoologySharedEligibilityFutureSeedMixer):
                raise TypeError("Shared-eligibility backbone requires strict mixers")

            dropped = block.drop_path1(block.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = block.norm1(
                residual.to(dtype=block.norm1.weight.dtype)
            )
            main_initial = (
                None
                if main_terminal is None
                else mixer.make_initial_state(main_terminal)
            )
            companion_initial = (
                None
                if companion_terminal is None
                else mixer.make_companion_initial_state(companion_terminal)
            )
            hidden_states, main_terminal, companion_terminal = (
                mixer.forward_with_dual_state(
                    hidden_states,
                    read_address=event_read,
                    write_address=event_write,
                    main_initial_state=main_initial,
                    companion_initial_state=companion_initial,
                )
            )

            dropped = block.drop_path2(block.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = block.norm2(
                residual.to(dtype=block.norm2.weight.dtype)
            )
            hidden_states = block.state_mixer(hidden_states)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class SharedEligibilityLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = SharedEligibilityBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        self.backbone.reset_event_path()
        for block in self.backbone.layers:
            mixer = block.sequence_mixer
            nn.init.zeros_(mixer.layer.companion_read_logit)
            nn.init.zeros_(mixer.layer.companion_seed_logit)
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


def _is_extra_parameter(name: str) -> bool:
    return (
        name == "backbone.shared_event_projection.weight"
        or name == "backbone.shared_eligibility_conv.weight"
        or name.endswith(".sequence_mixer.layer.companion_read_logit")
        or name.endswith(".sequence_mixer.layer.companion_seed_logit")
    )


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extras = 0
    for name, tensor in model.state_dict().items():
        if _is_extra_parameter(name):
            loaded[name] = tensor
            extras += tensor.numel()
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        if parent_name not in control_state:
            raise RuntimeError(f"Missing matched parent tensor: {parent_name}")
        parent = control_state[parent_name]
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
    rows = []
    for name, parameter in model.named_parameters():
        if _is_extra_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, parameter in sorted(rows):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def shared_eligibility_diagnostics(
    model: SharedEligibilityLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, SharedEligibilityBackbone):
        raise TypeError("Expected SharedEligibilityBackbone")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologySharedEligibilityFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two shared-eligibility mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    read = model.backbone.last_event_read
    write = model.backbone.last_event_write
    if read is None or write is None:
        raise RuntimeError("Shared event addresses were not captured")
    trace = model.backbone.shared_eligibility_conv.weight.detach().float()[:, 0]
    history_abs = trace[:, :-1].abs().mean()
    current_abs = trace[:, -1].abs().mean()
    total_abs = trace.abs().sum().clamp_min(1e-8)
    lagged_cosine = F.cosine_similarity(
        write[:, 1:].float(),
        read[:, :-1].float(),
        dim=-1,
    ).mean()
    current_cosine = F.cosine_similarity(
        write.float(),
        read.float(),
        dim=-1,
    ).mean()
    read_board = read.float().square().mean(dim=(1, 2)).sqrt()
    write_board = write.float().square().mean(dim=(1, 2)).sqrt()
    rows = []
    active_paths = 0
    for mixer in mixers:
        captured = mixer.layer._captured
        if not captured:
            raise RuntimeError("Shared-eligibility capture was not produced")
        gates = captured["read_gate"].flatten()
        active_paths += int((gates.abs() >= 1e-3).sum().item())
        row = {
            "layer_idx": mixer.layer_idx,
            "read_gate_abs_mean": float(gates.abs().mean().item()),
            "read_gate_abs_min": float(gates.abs().min().item()),
            "read_gate_abs_max": float(gates.abs().max().item()),
            "read_gate_values": [float(value) for value in gates.tolist()],
            "companion_seed_applied": mixer.last_companion_seed_gate is not None,
            "companion_seed_gate": (
                None
                if mixer.last_companion_seed_gate is None
                else float(mixer.last_companion_seed_gate.item())
            ),
            "companion_seed_raw_rms": (
                None
                if mixer.last_companion_seed_rms is None
                else float(mixer.last_companion_seed_rms.item())
            ),
        }
        row.update(
            {
                name: float(value.float().mean().item())
                for name, value in captured.items()
                if name not in {"read_gate", "read_address", "write_address"}
            }
        )
        rows.append(row)
    return {
        "active_layers": len(rows),
        "active_read_paths": active_paths,
        "total_read_paths": 2 * MODEL_HEADS,
        "companion_seed_routes": sum(
            row["companion_seed_applied"] for row in rows
        ),
        "shared_projection_count": 1,
        "shared_trace_count": 1,
        "trace_width": TRACE_WIDTH,
        "parameter_delta": EXPECTED_PARAMETER_DELTA,
        "main_state_values_per_layer": MAIN_STATE_VALUES,
        "companion_state_values_per_layer": COMPANION_STATE_VALUES,
        "official_scans_per_layer": 2,
        "event_projection_rms": float(_rms(model.backbone.shared_event_projection.weight).item()),
        "trace_history_abs_mean": float(history_abs.item()),
        "trace_current_abs_mean": float(current_abs.item()),
        "trace_history_mass_fraction": float((trace[:, :-1].abs().sum() / total_abs).item()),
        "event_read_rms": float(_rms(read).item()),
        "event_write_rms": float(_rms(write).item()),
        "event_write_read_relative_rms": float(
            (_rms(write - read) / _rms(read).clamp_min(1e-8)).item()
        ),
        "event_current_cosine": float(current_cosine.item()),
        "event_lagged_cosine": float(lagged_cosine.item()),
        "event_read_board_std": float(read_board.std(unbiased=False).item()),
        "event_write_board_std": float(write_board.std(unbiased=False).item()),
        "layers": rows,
    }
