from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_product import chunk_gated_delta_product

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLanguageModel,
    ZoologyGDN2FutureSeedMixer,
)


NUM_HEADS = 4
HEAD_DIM = 32
NUM_MICROSTEPS = 2
STATE_VALUES_PER_LAYER = NUM_HEADS * HEAD_DIM * HEAD_DIM


class EraseThenDelta(nn.Module):
    """Structured erase-then-correct recurrence on the official product op."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        conv_size: int,
        layer_idx: int,
    ) -> None:
        super().__init__()
        if hidden_size != 128 or num_heads != NUM_HEADS or head_dim != HEAD_DIM:
            raise ValueError("P-GDN3-067 fixes D128/H4/K32/V32")
        if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
            raise RuntimeError("Pinned official FLA source SHA was not asserted")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_heads
        self.num_householder = NUM_MICROSTEPS
        self.layer_idx = int(layer_idx)
        self.mode = "chunk"
        self.use_forget_gate = True
        self.allow_neg_eigval = False

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(
            hidden_size,
            NUM_MICROSTEPS * hidden_size,
            bias=False,
        )
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(
            hidden_size,
            NUM_MICROSTEPS * num_heads,
            bias=False,
        )
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)

        a = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.q_conv1d = ShortConvolution(
            hidden_size=hidden_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=NUM_MICROSTEPS * hidden_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=hidden_size,
            kernel_size=conv_size,
            bias=False,
            activation="silu",
        )
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.capture_enabled = False
        self.captures: dict[str, torch.Tensor] = {}

    def set_capture(self, enabled: bool) -> None:
        self.capture_enabled = bool(enabled)
        if enabled:
            self.captures.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        del output_attentions, kwargs
        if attention_mask is not None:
            raise ValueError("P-GDN3-067 fixes unpadded MQAR batches")
        last_state = get_layer_cache(self, past_key_values)
        recurrent_state = (
            None if last_state is None else last_state["recurrent_state"]
        )
        conv_q = conv_k = conv_v = None
        if last_state is not None:
            conv_q, conv_k, conv_v = last_state["conv_state"]

        q, conv_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_q,
            output_final_state=use_cache,
        )
        pair_key, conv_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_k,
            output_final_state=use_cache,
        )
        value, conv_v = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=conv_v,
            output_final_state=use_cache,
        )

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)
        pair_key = rearrange(
            pair_key,
            "b t (n h d) -> b t n h d",
            n=NUM_MICROSTEPS,
            h=self.num_heads,
        )
        value = rearrange(
            value,
            "b t (h d) -> b t h d",
            h=self.num_v_heads,
        )
        pair_value = torch.stack((torch.zeros_like(value), value), dim=2)
        pair_beta = rearrange(
            self.b_proj(hidden_states).sigmoid(),
            "b t (n h) -> b t n h",
            n=NUM_MICROSTEPS,
            h=self.num_v_heads,
        )
        decay = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )

        if self.capture_enabled:
            self.captures = {
                "initial_state": (
                    torch.zeros(
                        hidden_states.shape[0],
                        self.num_heads,
                        self.head_k_dim,
                        self.head_v_dim,
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
                    if recurrent_state is None
                    else recurrent_state.detach()
                ),
                "pair_key": pair_key.detach(),
                "pair_value": pair_value.detach(),
                "pair_beta": pair_beta.detach(),
                "decay": decay.detach(),
            }

        output, recurrent_state = chunk_gated_delta_product(
            q=q,
            k=rearrange(pair_key, "b t n h d -> b (t n) h d"),
            v=rearrange(pair_value, "b t n h d -> b (t n) h d"),
            g=decay,
            beta=rearrange(pair_beta, "b t n h -> b (t n) h"),
            initial_state=recurrent_state,
            output_final_state=use_cache,
            num_householder=NUM_MICROSTEPS,
            use_qk_l2norm_in_kernel=True,
        )
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_q, conv_k, conv_v),
            offset=hidden_states.shape[1],
        )
        if self.capture_enabled:
            if recurrent_state is None:
                raise RuntimeError("EDA capture requires a terminal state")
            self.captures["terminal_state"] = recurrent_state.detach()

        output_gate = rearrange(
            self.g_proj(hidden_states),
            "b t (h d) -> b t h d",
            h=self.num_v_heads,
        )
        output = self.o_norm(output, output_gate)
        output = self.o_proj(rearrange(output, "b t h d -> b t (h d)"))
        return output, None, past_key_values


class ZoologyEraseThenDeltaFutureSeedMixer(ZoologyGDN2FutureSeedMixer):
    """Erase-then-Delta recurrence with native full-state FutureSeed."""

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
        if expand_v != 1.0:
            raise ValueError("P-GDN3-067 fixes V32")
        self.layer_idx = int(layer_idx)
        self.future_seed_scale = float(future_seed_scale)
        self.layer = EraseThenDelta(
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            conv_size=conv_size,
            layer_idx=layer_idx,
        )
        self.future_seed_logit = nn.Parameter(
            torch.zeros(1, num_heads, 1, 1)
        )
        chunk_module = importlib.import_module("fla.ops.gated_delta_product.chunk")
        self.source_path = str(Path(chunk_module.__file__).resolve())
        self.last_seed_rms: Optional[torch.Tensor] = None
        self.last_seed_norm: Optional[torch.Tensor] = None
        self.last_seed_gate: Optional[torch.Tensor] = None

    def state_size(self, sequence_length: int = 2048) -> int:
        del sequence_length
        return STATE_VALUES_PER_LAYER


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def _explicit_replay(captures: dict[str, torch.Tensor]) -> dict[str, float]:
    pair_key = F.normalize(captures["pair_key"].float(), dim=-1)
    pair_value = captures["pair_value"].float()
    pair_beta = captures["pair_beta"].float()
    decay = captures["decay"].float().exp()
    state = captures["initial_state"].float().clone()
    erase_square = torch.zeros((), device=state.device)
    decay_square = torch.zeros((), device=state.device)
    steps = pair_key.shape[1]
    for token in range(steps):
        state = state * decay[:, token, :, None, None]
        erased = torch.einsum(
            "bhk,bhkv->bhv",
            pair_key[:, token, 0],
            state,
        )
        erase_delta = (
            pair_beta[:, token, 0, :, None, None]
            * pair_key[:, token, 0, :, :, None]
            * erased[:, :, None, :]
        )
        erase_square += erase_delta.square().mean()
        decay_square += state.square().mean()
        state = state - erase_delta
        predicted = torch.einsum(
            "bhk,bhkv->bhv",
            pair_key[:, token, 1],
            state,
        )
        correction = pair_value[:, token, 1] - predicted
        state = state + (
            pair_beta[:, token, 1, :, None, None]
            * pair_key[:, token, 1, :, :, None]
            * correction[:, :, None, :]
        )
    terminal = captures["terminal_state"].float()
    return {
        "erase_relative_rms": float(
            (erase_square / steps).sqrt()
            .div((decay_square / steps).sqrt().clamp_min(1e-8))
            .item()
        ),
        "explicit_terminal_relative_rms": float(
            (_rms(state - terminal) / _rms(terminal).clamp_min(1e-8)).item()
        ),
    }


@torch.no_grad()
def erase_then_delta_diagnostics(
    model: FutureSeedLanguageModel,
    inputs: torch.Tensor,
) -> dict[str, Any]:
    mixers = [
        block.sequence_mixer
        for block in model.backbone.layers
        if isinstance(block.sequence_mixer, ZoologyEraseThenDeltaFutureSeedMixer)
    ]
    if len(mixers) != len(model.backbone.layers):
        raise RuntimeError("Every layer must use Erase-then-Delta")
    for mixer in mixers:
        mixer.layer.set_capture(True)
    try:
        model.eval()(inputs)
        rows = []
        for mixer in mixers:
            captures = mixer.layer.captures
            key = F.normalize(captures["pair_key"].float(), dim=-1)
            erase_key, write_key = key.unbind(dim=2)
            gate = captures["pair_beta"].float()
            gamma, beta = gate.unbind(dim=2)
            terminal = captures["terminal_state"].float()
            board_rms = terminal.square().mean(dim=(-1, -2, -3)).sqrt()
            replay = _explicit_replay(captures)
            rows.append(
                {
                    "layer_idx": mixer.layer_idx,
                    "erase_write_key_relative_rms": float(
                        (_rms(erase_key - write_key) / _rms(key).clamp_min(1e-8)).item()
                    ),
                    "erase_write_key_abs_cosine": float(
                        F.cosine_similarity(erase_key, write_key, dim=-1)
                        .abs()
                        .mean()
                        .item()
                    ),
                    "gamma_mean": float(gamma.mean().item()),
                    "gamma_std": float(gamma.std(unbiased=False).item()),
                    "beta_mean": float(beta.mean().item()),
                    "beta_std": float(beta.std(unbiased=False).item()),
                    "first_payload_max_abs": float(
                        captures["pair_value"][:, :, 0].abs().max().item()
                    ),
                    "terminal_state_rms": float(_rms(terminal).item()),
                    "terminal_state_board_std": float(
                        board_rms.std(unbiased=False).item()
                    ),
                    **replay,
                }
            )
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    return {
        "active_layers": len(rows),
        "microsteps_per_token": NUM_MICROSTEPS,
        "state_values_per_layer": STATE_VALUES_PER_LAYER,
        "first_microstep": "independent_erase_zero_payload",
        "second_microstep": "standard_delta_correction",
        "per_layer": rows,
    }
