from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.utils import get_layer_cache, update_layer_cache
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
EXPECTED_NEW_PARAMETERS = 2 * MODEL_HEADS
EXPECTED_MAIN_STATE_VALUES = MODEL_HEADS * HEAD_DIM * HEAD_DIM
EXPECTED_REVERSE_STATE_VALUES = EXPECTED_MAIN_STATE_VALUES


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    scale = tensor.float().square().sum(dim=-1, keepdim=True).add(1e-6).sqrt()
    return tensor / scale.to(dtype=tensor.dtype)


class CycleConsistencyGDN2(nn.Module):
    """Native key->value memory plus a value->key consistency recurrence."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        if base.hidden_size != MODEL_WIDTH:
            raise ValueError("P-GDN3-058 requires D128")
        if base.num_heads != MODEL_HEADS or base.num_v_heads != MODEL_HEADS:
            raise ValueError("P-GDN3-058 requires H4")
        if base.head_k_dim != HEAD_DIM or base.head_v_dim != HEAD_DIM:
            raise ValueError("P-GDN3-058 requires K32/V32")
        if base.mode != "chunk" or not base.use_short_conv:
            raise ValueError("P-GDN3-058 requires chunk GDN2 with short convolution")
        self.base = base
        self.cycle_gate = nn.Parameter(torch.zeros(1, 1, MODEL_HEADS, 1))
        self.last_reverse_state: Optional[torch.Tensor] = None
        self._capture = False
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
            raise RuntimeError("P-GDN3-058 requires an explicit layer index")
        return int(self.base.layer_idx)

    def set_capture(self, enabled: bool) -> None:
        self._capture = bool(enabled)
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
            raise ValueError("P-GDN3-058 uses fixed unpadded MQAR batches")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("P-GDN3-058 does not use packed sequences")
        reverse_initial_state = kwargs.pop("reverse_initial_state", None)

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
        key, conv_state_k = layer.k_conv1d(
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
        erase_gate = layer.b_proj(hidden_states).sigmoid()
        write_gate = layer.w_proj(hidden_states).sigmoid()
        q, key, decay, erase_gate = (
            rearrange(tensor, "... (h d) -> ... h d", d=HEAD_DIM)
            for tensor in (q, key, decay, erase_gate)
        )
        value = rearrange(value, "... (h d) -> ... h d", d=HEAD_DIM)
        write_gate = rearrange(
            write_gate,
            "... (h d) -> ... h d",
            d=HEAD_DIM,
        )
        decay = -layer.A_log.float().exp().unsqueeze(-1) * decay
        if layer.allow_neg_eigval:
            erase_gate = erase_gate * 2.0

        main_initial_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        main_output, main_state = chunk_gdn2(
            q=q,
            k=key,
            v=value,
            g=decay,
            b=erase_gate,
            w=write_gate,
            initial_state=main_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        # The dual memory stores key evidence under the value coordinate. Gates
        # are swapped coherently because K32 and V32 have equal dimensions.
        reverse_decay = decay.mean(dim=-1, keepdim=True).expand_as(decay)
        reverse_output, reverse_state = chunk_gdn2(
            q=main_output,
            k=value,
            v=_l2_normalize(key),
            g=reverse_decay,
            b=write_gate,
            w=erase_gate,
            initial_state=reverse_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        gate = torch.tanh(self.cycle_gate).to(dtype=q.dtype)
        owner_residual = _l2_normalize(q) - _l2_normalize(reverse_output)
        refined_q = q + gate * owner_residual.to(dtype=q.dtype)
        refined_output, _ = chunk_gdn2(
            q=refined_q,
            k=key,
            v=value,
            g=decay,
            b=erase_gate,
            w=write_gate,
            initial_state=main_initial_state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        update_layer_cache(
            layer,
            past_key_values,
            recurrent_state=main_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len,
        )
        self.last_reverse_state = reverse_state

        if self._capture:
            if main_state is None or reverse_state is None:
                raise RuntimeError("P-GDN3-058 capture requires terminal states")
            self._captured = {
                "q": q.detach(),
                "key": key.detach(),
                "value": value.detach(),
                "main_output": main_output.detach(),
                "refined_output": refined_output.detach(),
                "main_state": main_state.detach(),
                "reverse_output": reverse_output.detach(),
                "reverse_state": reverse_state.detach(),
                "owner_residual": owner_residual.detach(),
                "refined_q": refined_q.detach(),
                "cycle_gate": gate.detach(),
            }

        output = layer.o_norm(
            refined_output,
            rearrange(
                layer.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=HEAD_DIM,
            ),
        )
        output = rearrange(output, "b t h d -> b t (h d)")
        return layer.o_proj(output), None, past_key_values


class ZoologyCycleMemoryFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Cycle-consistent GDN2 with native FutureSeed on both dual states."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layer = CycleConsistencyGDN2(self.layer)
        self.last_reverse_seed_rms: Optional[torch.Tensor] = None
        self.last_reverse_seed_norm: Optional[torch.Tensor] = None

    def make_initial_states(
        self,
        main_terminal_state: torch.Tensor,
        reverse_terminal_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_seed = super().make_initial_state(main_terminal_state)
        rms = reverse_terminal_state.float().square().mean(
            dim=(-1, -2), keepdim=True
        ).sqrt().clamp_min(1e-6)
        gate = torch.sigmoid(self.future_seed_logit).to(
            device=reverse_terminal_state.device,
            dtype=reverse_terminal_state.dtype,
        )
        reverse_seed = (
            reverse_terminal_state
            / rms.to(dtype=reverse_terminal_state.dtype)
            * gate
            * self.future_seed_scale
        )
        self.last_reverse_seed_rms = rms.detach()
        self.last_reverse_seed_norm = (
            reverse_seed.detach().float().norm(dim=(-1, -2)).mean()
        )
        return main_seed, reverse_seed

    def forward_with_cycle_states(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
        reverse_initial_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._new_cache(initial_state)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=torch.bfloat16,
            enabled=hidden_states.is_cuda,
        ):
            output, _attentions, cache = self.layer(
                hidden_states,
                reverse_initial_state=reverse_initial_state,
                past_key_values=cache,
                use_cache=True,
            )
        layer_cache = cache[self.layer_idx]
        main_terminal_state = layer_cache["recurrent_state"]
        reverse_terminal_state = self.layer.last_reverse_state
        if main_terminal_state is None or reverse_terminal_state is None:
            raise RuntimeError("P-GDN3-058 did not produce both terminal states")
        return (
            output.to(hidden_states.dtype),
            main_terminal_state,
            reverse_terminal_state,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, _main, _reverse = self.forward_with_cycle_states(
            hidden_states,
            initial_state=None,
            reverse_initial_state=None,
        )
        return output

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return EXPECTED_MAIN_STATE_VALUES + EXPECTED_REVERSE_STATE_VALUES


class CycleMemoryBackbone(FutureSeedLMBackbone):
    def layers_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        main_terminal_state = None
        reverse_terminal_state = None
        for layer in self.layers:
            mixer = layer.sequence_mixer
            if not isinstance(mixer, ZoologyCycleMemoryFutureSeedMixer):
                raise TypeError("Cycle-memory backbone requires strict mixers")

            dropped = layer.drop_path1(layer.dropout1(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm1(
                residual.to(dtype=layer.norm1.weight.dtype)
            )
            if main_terminal_state is None:
                main_initial_state = None
                reverse_initial_state = None
            else:
                if reverse_terminal_state is None:
                    raise RuntimeError("Reverse terminal state is missing")
                main_initial_state, reverse_initial_state = mixer.make_initial_states(
                    main_terminal_state,
                    reverse_terminal_state,
                )
            (
                hidden_states,
                main_terminal_state,
                reverse_terminal_state,
            ) = mixer.forward_with_cycle_states(
                hidden_states,
                initial_state=main_initial_state,
                reverse_initial_state=reverse_initial_state,
            )

            dropped = layer.drop_path2(layer.dropout2(hidden_states))
            residual = dropped + residual if residual is not None else dropped
            hidden_states = layer.norm2(
                residual.to(dtype=layer.norm2.weight.dtype)
            )
            hidden_states = layer.state_mixer(hidden_states)

        dropped = self.drop_f(hidden_states)
        residual = dropped + residual if residual is not None else dropped
        return self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))


class CycleMemoryLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.multiplier != 1:
            raise ValueError("Discrete FutureSeed model requires multiplier=1")

        self.backbone = CycleMemoryBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(
            partial(
                _init_weights,
                n_layers=config.n_layers,
                block_type=config.block_type,
            )
        )
        for block in self.backbone.layers:
            nn.init.zeros_(block.sequence_mixer.layer.cycle_gate)
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


def _is_cycle_parameter(name: str) -> bool:
    return name.endswith(".cycle_gate")


def load_matched_parent_state(
    model: nn.Module,
    control_state: dict[str, torch.Tensor],
) -> None:
    loaded: dict[str, torch.Tensor] = {}
    extras = 0
    for name, tensor in model.state_dict().items():
        if _is_cycle_parameter(name):
            loaded[name] = torch.zeros_like(tensor)
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
    if extras != EXPECTED_NEW_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_NEW_PARAMETERS} new parameters, found {extras}"
        )
    model.load_state_dict(loaded, strict=True)


def parent_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, parameter in model.named_parameters():
        if _is_cycle_parameter(name):
            continue
        parent_name = name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        )
        rows.append((parent_name, parameter))
    for name, value in sorted(rows):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cycle_memory_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    if not isinstance(model.backbone, CycleMemoryBackbone):
        raise TypeError("Expected CycleMemoryBackbone")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyCycleMemoryFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected two cycle-memory mixers")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)

    rows = []
    for mixer in mixers:
        captured = mixer.layer._captured
        required = {
            "q",
            "key",
            "value",
            "main_output",
            "refined_output",
            "main_state",
            "reverse_output",
            "reverse_state",
            "owner_residual",
            "refined_q",
            "cycle_gate",
        }
        if set(captured) != required:
            raise RuntimeError(f"Incomplete cycle-memory capture: {set(captured)}")
        q = captured["q"].float()
        main_output = captured["main_output"].float()
        refined_output = captured["refined_output"].float()
        reverse_output = captured["reverse_output"].float()
        owner_residual = captured["owner_residual"].float()
        refined_q = captured["refined_q"].float()
        main_state = captured["main_state"].float()
        reverse_state = captured["reverse_state"].float()
        reverse_board = reverse_state.square().mean(dim=(1, 2, 3)).sqrt()
        reverse_token = reverse_output.square().mean(dim=-1).sqrt()
        cycle_delta = refined_q - q
        gate = captured["cycle_gate"].float().flatten()
        rows.append(
            {
                "layer_idx": mixer.layer_idx,
                "cycle_gate_abs_mean": float(gate.abs().mean().item()),
                "cycle_gate_abs_min": float(gate.abs().min().item()),
                "cycle_gate_abs_max": float(gate.abs().max().item()),
                "active_gate_heads": int((gate.abs() >= 1e-3).sum().item()),
                "reverse_output_rms": float(_rms(reverse_output).item()),
                "reverse_output_board_std": float(
                    reverse_output.square().mean(dim=(1, 2, 3)).sqrt().std(
                        unbiased=False
                    ).item()
                ),
                "reverse_output_token_std": float(
                    reverse_token.std(dim=1, unbiased=False).mean().item()
                ),
                "reverse_state_rms": float(_rms(reverse_state).item()),
                "reverse_state_board_std": float(
                    reverse_board.std(unbiased=False).item()
                ),
                "reverse_to_main_state_rms": float(
                    (_rms(reverse_state) / _rms(main_state).clamp_min(1e-8)).item()
                ),
                "owner_residual_rms": float(_rms(owner_residual).item()),
                "cycle_q_relative_rms": float(
                    (_rms(cycle_delta) / _rms(q).clamp_min(1e-8)).item()
                ),
                "cycle_q_board_std": float(
                    cycle_delta.square().mean(dim=(1, 2, 3)).sqrt().std(
                        unbiased=False
                    ).item()
                ),
                "cycle_q_token_std": float(
                    cycle_delta.square().mean(dim=-1).sqrt().std(
                        dim=1, unbiased=False
                    ).mean().item()
                ),
                "refined_output_relative_change": float(
                    (_rms(refined_output - main_output)
                     / _rms(main_output).clamp_min(1e-8)).item()
                ),
                "query_owner_cosine": float(
                    F.cosine_similarity(
                        q.flatten(0, 2),
                        reverse_output.flatten(0, 2),
                        dim=-1,
                        eps=1e-8,
                    ).mean().item()
                ),
                "reverse_futureseed_rms": (
                    None
                    if mixer.last_reverse_seed_rms is None
                    else float(mixer.last_reverse_seed_rms.mean().item())
                ),
                "reverse_futureseed_norm": (
                    None
                    if mixer.last_reverse_seed_norm is None
                    else float(mixer.last_reverse_seed_norm.item())
                ),
            }
        )
    return {
        "active_layers": len(rows),
        "active_gate_heads": sum(row["active_gate_heads"] for row in rows),
        "new_parameters": sum(
            mixer.layer.cycle_gate.numel() for mixer in mixers
        ),
        "main_state_values_per_layer": EXPECTED_MAIN_STATE_VALUES,
        "reverse_state_values_per_layer": EXPECTED_REVERSE_STATE_VALUES,
        "official_scans_per_layer": 3,
        "reverse_futureseed_routes": sum(
            row["reverse_futureseed_rms"] is not None for row in rows
        ),
        "per_layer": rows,
    }
