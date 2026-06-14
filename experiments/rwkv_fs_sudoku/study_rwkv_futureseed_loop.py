#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    from rwkv7_cuda import StatePassingRWKV7, WindRWKV7, statepassing_available, wind_available
except Exception:  # pragma: no cover - CUDA extension is optional for CPU smoke.
    StatePassingRWKV7 = None
    WindRWKV7 = None

    def statepassing_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"

    def wind_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"


N = 9
BOX_ROWS = 3
BOX_COLS = 3
CELLS = 81
BLANK = 9
VOCAB = 10
ROWS: List[List[int]] = []
COLS: List[List[int]] = []
BOXES: List[List[int]] = []
UNITS: List[List[int]] = []


def configure_sudoku(size: int, box_rows: int, box_cols: int) -> None:
    global N, BOX_ROWS, BOX_COLS, CELLS, BLANK, VOCAB, ROWS, COLS, BOXES, UNITS
    if box_rows <= 0 or box_cols <= 0:
        inferred = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4), 25: (5, 5)}
        if size not in inferred:
            raise ValueError("Use a supported Sudoku size, or pass --box_rows and --box_cols.")
        box_rows, box_cols = inferred[size]
    if box_rows <= 1 or box_cols <= 1 or box_rows * box_cols != size:
        raise ValueError("Sudoku needs size == box_rows * box_cols with both box dimensions > 1.")

    N = int(size)
    BOX_ROWS = int(box_rows)
    BOX_COLS = int(box_cols)
    CELLS = N * N
    BLANK = N
    VOCAB = N + 1
    ROWS = [[r * N + c for c in range(N)] for r in range(N)]
    COLS = [[r * N + c for r in range(N)] for c in range(N)]
    BOXES = [
        [(br + dr) * N + (bc + dc) for dr in range(BOX_ROWS) for dc in range(BOX_COLS)]
        for br in range(0, N, BOX_ROWS)
        for bc in range(0, N, BOX_COLS)
    ]
    UNITS = ROWS + COLS + BOXES


configure_sudoku(N, BOX_ROWS, BOX_COLS)


@dataclass
class FullMetrics:
    label_exact: float
    valid_sudoku: float
    solved_valid_clue: float
    clue_ok: float
    blank_acc: float
    avg_filled: float


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def forward_autocast(forward_dtype: str, device: torch.device):
    if forward_dtype == "float32":
        return nullcontext()
    if forward_dtype != "bfloat16":
        raise ValueError("--forward_dtype must be 'float32' or 'bfloat16'")
    if device.type != "cuda":
        raise ValueError("--forward_dtype=bfloat16 requires a CUDA device")
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def pattern(row: int, col: int) -> int:
    return (BOX_COLS * (row % BOX_ROWS) + row // BOX_ROWS + col) % N


def shuffled_solution(rng: random.Random) -> List[int]:
    rows = [
        g * BOX_ROWS + r
        for g in rng.sample(range(BOX_COLS), BOX_COLS)
        for r in rng.sample(range(BOX_ROWS), BOX_ROWS)
    ]
    cols = [
        g * BOX_COLS + c
        for g in rng.sample(range(BOX_ROWS), BOX_ROWS)
        for c in rng.sample(range(BOX_COLS), BOX_COLS)
    ]
    digits = rng.sample(range(N), N)
    return [digits[pattern(r, c)] for r in rows for c in cols]


def choose_blank_cells(holes: int, rng: random.Random, hole_pattern: str) -> List[int]:
    if hole_pattern == "random":
        return rng.sample(range(CELLS), holes)
    raise ValueError("--hole_pattern now supports only 'random'; structured hole patterns were Sudoku-specific probes.")


def make_batch(
    batch_size: int,
    holes_min: int,
    holes_max: int,
    hole_pattern: str,
    rng: random.Random,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.full((batch_size, CELLS), BLANK, dtype=torch.long)
    labels = torch.empty((batch_size, CELLS), dtype=torch.long)
    clue_mask = torch.zeros((batch_size, CELLS), dtype=torch.bool)

    for b in range(batch_size):
        solution = shuffled_solution(rng)
        holes = rng.randint(holes_min, holes_max)
        blank_cells = set(choose_blank_cells(holes, rng, hole_pattern))
        puzzle = [BLANK if i in blank_cells else solution[i] for i in range(CELLS)]
        inputs[b] = torch.tensor(puzzle, dtype=torch.long)
        labels[b] = torch.tensor(solution, dtype=torch.long)
        clue_mask[b] = torch.tensor([i not in blank_cells for i in range(CELLS)], dtype=torch.bool)

    return inputs.to(device), labels.to(device), clue_mask.to(device)


class RWKVTimeMix(nn.Module):
    """RWKV-style WKV block with CUDA RWKV7 paths and a torch fallback."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        if d_model != heads * head_dim:
            raise ValueError("d_model must equal heads * head_dim")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = head_dim
        self.rwkv_kernel = rwkv_kernel
        self._fallback_warned = False

        ratio_0_to_1 = layer_id / max(layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / max(layers, 1))
        ddd = torch.arange(d_model, dtype=torch.float32).view(1, 1, d_model) / max(d_model - 1, 1)
        self.mix_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.mix_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.mix_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.mix_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.mix_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.mix_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        base_decay = torch.zeros(d_model)
        zigzag = torch.zeros(d_model)
        linear = torch.zeros(d_model)
        for n in range(d_model):
            frac = n / max(d_model - 1, 1)
            linear[n] = frac - 0.5
            base_decay[n] = -6.0 + 6.0 * frac ** (1.0 + ratio_0_to_1**0.3)
            local = ((n % head_dim) - ((head_dim - 1) / 2.0)) / max((head_dim - 1) / 2.0, 1.0)
            zigzag[n] = local * abs(local)
        self.time_decay = nn.Parameter((base_decay + 0.5 + zigzag * 2.5).view(1, 1, d_model))
        self.decay_delta = nn.Linear(d_model, d_model, bias=False)
        self.state_lr = nn.Linear(d_model, d_model, bias=True)
        self.key_scale = nn.Parameter((0.71 - linear * 0.1).view(1, 1, d_model))
        self.key_lr_mix = nn.Parameter(torch.full((1, 1, d_model), 1.02))

        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(heads, d_model, eps=64e-5)

        scale = d_model**0.5
        self.receptance.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        self.key.weight.data.uniform_(-0.05 / scale, 0.05 / scale)
        self.value.weight.data.uniform_(-0.5 / scale, 0.5 / scale)
        nn.init.zeros_(self.decay_delta.weight)
        nn.init.zeros_(self.state_lr.weight)
        self.state_lr.bias.data.copy_(-0.19 + zigzag * 0.3 + linear * 0.4)
        nn.init.zeros_(self.out.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.shape
        heads, head_dim = self.heads, self.head_dim

        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        delta = shifted - x

        xr = x + delta * self.mix_r
        xw = x + delta * self.mix_w
        xk = x + delta * self.mix_k
        xv = x + delta * self.mix_v
        xa = x + delta * self.mix_a
        xg = x + delta * self.mix_g

        r = torch.sigmoid(self.receptance(xr)).view(batch_size, seq_len, heads, head_dim)
        w_raw = self.time_decay + torch.tanh(self.decay_delta(xw))
        w_log = -F.softplus(-w_raw) - 0.5
        state_lr = torch.sigmoid(self.state_lr(xa)).view(batch_size, seq_len, heads, head_dim)
        k_flat = self.key(xk)
        kk = F.normalize((k_flat * self.key_scale).view(batch_size, seq_len, heads, head_dim), dim=-1, p=2.0)
        k = (k_flat * (1.0 + (state_lr.reshape(batch_size, seq_len, channels) - 1.0) * self.key_lr_mix)).view(
            batch_size, seq_len, heads, head_dim
        )
        v = self.value(xv).view(batch_size, seq_len, heads, head_dim)
        g = torch.sigmoid(self.gate(xg))

        if self.rwkv_kernel in {"auto", "cuda", "statepassing"}:
            if self._can_use_statepassing(x, initial_state):
                y, terminal_state = self._forward_statepassing(
                    r=r,
                    w_raw=w_raw.view(batch_size, seq_len, heads, head_dim),
                    k=k,
                    v=v,
                    kk=kk,
                    state_lr=state_lr,
                    gate=g,
                    initial_state=initial_state,
                    original_dtype=x.dtype,
                )
                return y, terminal_state
            if self.rwkv_kernel in {"cuda", "statepassing"}:
                ok, reason = statepassing_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV_KERNEL={self.rwkv_kernel} requested but unavailable: {reason}")
                raise RuntimeError(f"RWKV_KERNEL={self.rwkv_kernel} requested but inputs are not CUDA tensors")

        if self.rwkv_kernel in {"auto", "wind"}:
            if self._can_use_wind(x, initial_state):
                y, terminal_state = self._forward_wind(
                    r=r,
                    w_log=w_log.view(batch_size, seq_len, heads, head_dim),
                    k=k,
                    v=v,
                    kk=kk,
                    state_lr=state_lr,
                    gate=g,
                    initial_state=initial_state,
                    original_dtype=x.dtype,
                )
                return y, terminal_state
            if self.rwkv_kernel == "wind":
                ok, reason = wind_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV_KERNEL=wind requested but unavailable: {reason}")
                raise RuntimeError("RWKV_KERNEL=wind requested but inputs are not CUDA tensors")
            if not self._fallback_warned:
                sp_ok, sp_reason = statepassing_available(self.head_dim)
                ok, reason = wind_available(self.head_dim)
                print(
                    f"rwkv_kernel=auto using torch fallback: statepassing={sp_ok}:{sp_reason}; wind={ok}:{reason}",
                    flush=True,
                )
                self._fallback_warned = True

        return self._forward_torch(
            r=r,
            w_log=w_log,
            k=k,
            v=v,
            kk=kk,
            state_lr=state_lr,
            gate=g,
            initial_state=initial_state,
        )

    def _can_use_statepassing(self, x: torch.Tensor, initial_state: Optional[torch.Tensor]) -> bool:
        if StatePassingRWKV7 is None or not x.is_cuda:
            return False
        ok, _reason = statepassing_available(self.head_dim)
        if not ok:
            return False
        if initial_state is not None:
            expected = (x.shape[0], self.heads, self.head_dim, self.head_dim)
            if tuple(initial_state.shape) != expected:
                return False
        return True

    def _can_use_wind(self, x: torch.Tensor, initial_state: Optional[torch.Tensor]) -> bool:
        if WindRWKV7 is None or not x.is_cuda:
            return False
        ok, _reason = wind_available(self.head_dim)
        if not ok:
            return False
        if initial_state is not None:
            expected = (x.shape[0], self.heads, self.head_dim, self.head_dim)
            if tuple(initial_state.shape) != expected:
                return False
        return True

    @staticmethod
    def _pad_time(t: torch.Tensor, pad_len: int, value: float = 0.0) -> torch.Tensor:
        if pad_len <= 0:
            return t
        pad_shape = (t.shape[0], pad_len, *t.shape[2:])
        pad = t.new_full(pad_shape, value)
        return torch.cat([t, pad], dim=1)

    def _forward_wind(
        self,
        *,
        r: torch.Tensor,
        w_log: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        original_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, heads, head_dim = r.shape
        pad_len = (-seq_len) % 16
        q_bf16 = self._pad_time(r, pad_len).to(torch.bfloat16)
        w_bf16 = self._pad_time(w_log, pad_len, value=-60.0).to(torch.bfloat16)
        k_bf16 = self._pad_time(k, pad_len).to(torch.bfloat16)
        v_bf16 = self._pad_time(v, pad_len).to(torch.bfloat16)
        z_bf16 = self._pad_time(-kk, pad_len).to(torch.bfloat16)
        a_bf16 = self._pad_time(kk * state_lr, pad_len).to(torch.bfloat16)

        if initial_state is None:
            s0 = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.bfloat16)
        else:
            s0 = initial_state.to(device=r.device, dtype=torch.bfloat16)

        y_bf16, terminal_state_bf16 = WindRWKV7.apply(w_bf16, q_bf16, k_bf16, v_bf16, z_bf16, a_bf16, s0)
        y = y_bf16[:, :seq_len].reshape(batch_size, seq_len, heads * head_dim).to(original_dtype)
        y = self.group_norm(y.reshape(batch_size * seq_len, heads * head_dim)).reshape(batch_size, seq_len, heads * head_dim)
        return self.out(y * gate), terminal_state_bf16.to(original_dtype)

    def _forward_statepassing(
        self,
        *,
        r: torch.Tensor,
        w_raw: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        original_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, heads, head_dim = r.shape
        pad_len = (-seq_len) % 16
        r_bf16 = self._pad_time(r, pad_len).to(torch.bfloat16)
        w_bf16 = self._pad_time(w_raw, pad_len, value=-60.0).to(torch.bfloat16)
        k_bf16 = self._pad_time(k, pad_len).to(torch.bfloat16)
        v_bf16 = self._pad_time(v, pad_len).to(torch.bfloat16)
        a_bf16 = self._pad_time(-kk, pad_len).to(torch.bfloat16)
        b_bf16 = self._pad_time(kk * state_lr, pad_len).to(torch.bfloat16)

        if initial_state is None:
            s0 = torch.zeros(batch_size, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
        else:
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            s0 = initial_state.to(device=r.device, dtype=torch.float32)

        y_bf16, terminal_state = StatePassingRWKV7.apply(s0, r_bf16, w_bf16, k_bf16, v_bf16, a_bf16, b_bf16)
        y = y_bf16[:, :seq_len].reshape(batch_size, seq_len, heads * head_dim).to(original_dtype)
        y = self.group_norm(y.reshape(batch_size * seq_len, heads * head_dim)).reshape(batch_size, seq_len, heads * head_dim)
        return self.out(y * gate), terminal_state.to(original_dtype)

    def _forward_torch(
        self,
        *,
        r: torch.Tensor,
        w_log: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kk: torch.Tensor,
        state_lr: torch.Tensor,
        gate: torch.Tensor,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = gate.shape
        heads, head_dim = self.heads, self.head_dim
        w = torch.exp(-torch.exp(w_log.float())).to(gate.dtype).view(batch_size, seq_len, heads, head_dim)

        if initial_state is None:
            memory = torch.zeros(batch_size, heads, head_dim, head_dim, device=gate.device, dtype=gate.dtype)
        else:
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            memory = initial_state.to(device=gate.device, dtype=gate.dtype)

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            erase = torch.einsum("bhij,bhj->bhi", memory, -kk[:, t])
            write_back = (kk[:, t] * state_lr[:, t]).unsqueeze(-2)
            memory = (
                memory * w[:, t].unsqueeze(-2)
                + erase.unsqueeze(-1) * write_back
                + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
            )
            y_t = torch.einsum("bhij,bhj->bhi", memory, r[:, t]).reshape(batch_size, channels)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = self.group_norm(y.reshape(batch_size * seq_len, channels)).reshape(batch_size, seq_len, channels)
        return self.out(y * gate), memory


class ChannelMix(nn.Module):
    def __init__(self, d_model: int, mult: int) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, mult * d_model, bias=False)
        self.value = nn.Linear(mult * d_model, d_model, bias=False)
        self.key.weight.data.uniform_(-0.5 / (d_model**0.5), 0.5 / (d_model**0.5))
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(F.relu(self.key(x)).square())


class RWKVBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        layer_id: int,
        layers: int,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        self.ln_time = nn.LayerNorm(d_model)
        self.ln_channel = nn.LayerNorm(d_model)
        self.time_mix = RWKVTimeMix(d_model, heads, head_dim, layer_id=layer_id, layers=layers, rwkv_kernel=rwkv_kernel)
        self.channel_mix = ChannelMix(d_model, channel_mult)
        self.future_seed_logit = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_out, terminal_state = self.time_mix(self.ln_time(x), initial_state=initial_state)
        x = x + time_out
        x = x + self.channel_mix(self.ln_channel(x))
        return x, terminal_state


class FutureSeedRWKV(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        *,
        future_seed_scale: float = 1.0,
        future_seed_decay: float = 0.0,
        future_seed_update: str = "fixed",
        activation_checkpoint: bool = False,
        rwkv_kernel: str = "auto",
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("FutureSeed needs at least two layers.")
        if future_seed_update not in {"fixed", "learned", "loop_residual"}:
            raise ValueError("future_seed_update must be one of: fixed, learned, loop_residual.")
        self.future_seed_scale = float(future_seed_scale)
        self.future_seed_decay = float(future_seed_decay)
        self.future_seed_update = future_seed_update
        self.activation_checkpoint = bool(activation_checkpoint)
        if future_seed_update in {"learned", "loop_residual"}:
            update_init = min(max(1.0 - self.future_seed_decay, 1e-4), 1.0 - 1e-4)
            update_logit = math.log(update_init / (1.0 - update_init))
            self.future_seed_update_logit = nn.Parameter(
                torch.full((layers - 1, 1, heads, 1, 1), float(update_logit))
            )
        else:
            self.register_parameter("future_seed_update_logit", None)
        self.blocks = nn.ModuleList(
            [
                RWKVBlock(
                    d_model,
                    heads,
                    head_dim,
                    channel_mult,
                    layer_id=layer_id,
                    layers=layers,
                    rwkv_kernel=rwkv_kernel,
                )
                for layer_id in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        seed_memory: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        if seed_memory is not None and len(seed_memory) != len(self.blocks) - 1:
            raise ValueError(f"seed_memory has {len(seed_memory)} entries, expected {len(self.blocks) - 1}")
        previous_state: Optional[torch.Tensor] = None
        seed_state: Optional[torch.Tensor] = None
        next_seed_memory: Optional[List[torch.Tensor]] = [] if self.future_seed_update == "loop_residual" else None
        gates = []
        update_gates = []
        state_norms = []
        memory_norms = []
        memory_delta_norms = []
        for layer_idx, block in enumerate(self.blocks):
            initial_state = None
            if layer_idx > 0:
                assert previous_state is not None
                gate = torch.sigmoid(block.future_seed_logit) * self.future_seed_scale
                gates.append(gate.mean())
                if self.future_seed_scale > 0:
                    if self.future_seed_update == "loop_residual":
                        assert self.future_seed_update_logit is not None
                        update_gate = torch.sigmoid(self.future_seed_update_logit[layer_idx - 1]).to(
                            device=previous_state.device,
                            dtype=previous_state.dtype,
                        )
                        prior_state = None if seed_memory is None else seed_memory[layer_idx - 1]
                        if prior_state is None:
                            seed_state = previous_state
                            memory_delta_norms.append(x.new_zeros(()))
                        else:
                            prior_state = prior_state.to(device=previous_state.device, dtype=previous_state.dtype)
                            delta = previous_state - prior_state
                            seed_state = prior_state + update_gate * delta
                            memory_delta_norms.append(delta.norm(dim=(-1, -2)).mean())
                        update_gates.append(update_gate.mean())
                        memory_norms.append(seed_state.norm(dim=(-1, -2)).mean())
                        assert next_seed_memory is not None
                        next_seed_memory.append(seed_state)
                    elif seed_state is None:
                        seed_state = previous_state
                        update_gates.append(x.new_ones(()))
                    elif self.future_seed_update == "learned":
                        assert self.future_seed_update_logit is not None
                        update_gate = torch.sigmoid(self.future_seed_update_logit[layer_idx - 1]).to(
                            device=previous_state.device,
                            dtype=previous_state.dtype,
                        )
                        seed_state = seed_state + update_gate * (previous_state - seed_state)
                        update_gates.append(update_gate.mean())
                    elif self.future_seed_decay <= 0:
                        seed_state = previous_state
                        update_gates.append(x.new_ones(()))
                    else:
                        keep = self.future_seed_decay
                        seed_state = keep * seed_state + (1.0 - keep) * previous_state
                        update_gates.append(x.new_tensor(1.0 - keep))
                    denom = seed_state.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                    normalized_state = seed_state / denom
                    initial_state = normalized_state * gate
                    state_norms.append(initial_state.norm(dim=(-1, -2)).mean())
                else:
                    update_gates.append(x.new_zeros(()))
                    state_norms.append(x.new_zeros(()))
                    if next_seed_memory is not None:
                        next_seed_memory.append(previous_state)
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                if initial_state is None:
                    x, previous_state = torch_checkpoint(
                        lambda block_input: block(block_input, initial_state=None),
                        x,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    x, previous_state = torch_checkpoint(
                        lambda block_input, block_state: block(block_input, initial_state=block_state),
                        x,
                        initial_state,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                x, previous_state = block(x, initial_state=initial_state)

        if gates:
            out = {
                "fs_gate_mean": torch.stack(gates).mean(),
                "fs_update_mean": torch.stack(update_gates).mean(),
                "fs_state_norm": torch.stack(state_norms).mean(),
                "fs_decay": x.new_tensor(self.future_seed_decay),
            }
            if memory_norms:
                out["fs_memory_norm"] = torch.stack(memory_norms).mean()
            if memory_delta_norms:
                out["fs_memory_delta_norm"] = torch.stack(memory_delta_norms).mean()
            return x, out, next_seed_memory
        zero = x.new_zeros(())
        return (
            x,
            {"fs_gate_mean": zero, "fs_update_mean": zero, "fs_state_norm": zero, "fs_decay": zero},
            next_seed_memory,
        )


class FeatureNoiseBuffer:
    def __init__(self, *, capacity: int, feature_dim: int, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.dtype = dtype
        self.data = torch.empty(self.capacity, self.feature_dim, device=device, dtype=dtype)
        self.count = 0
        self.write_pos = 0

    def add(self, features: torch.Tensor, *, max_items: int) -> None:
        if self.capacity <= 0 or max_items <= 0:
            return
        flat = features.detach().reshape(-1, self.feature_dim).to(device=self.device, dtype=self.dtype)
        if flat.shape[0] > max_items:
            idx = torch.randperm(flat.shape[0], device=flat.device)[:max_items]
            flat = flat.index_select(0, idx)
        n_items = min(flat.shape[0], self.capacity)
        if n_items <= 0:
            return
        flat = flat[:n_items]
        first = min(n_items, self.capacity - self.write_pos)
        self.data[self.write_pos : self.write_pos + first].copy_(flat[:first])
        remaining = n_items - first
        if remaining > 0:
            self.data[:remaining].copy_(flat[first:])
        self.write_pos = (self.write_pos + n_items) % self.capacity
        self.count = min(self.capacity, self.count + n_items)

    def sample_diff_like(self, reference: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(reference)
        flat_ref = reference.detach().reshape(-1, self.feature_dim)
        n_items = flat_ref.shape[0]
        idx_a = torch.randint(0, self.count, (n_items,), device=self.device)
        idx_b = torch.randint(0, self.count, (n_items,), device=self.device)
        diff = self.data.index_select(0, idx_a) - self.data.index_select(0, idx_b)
        diff_rms = diff.square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        ref_rms = flat_ref.to(self.dtype).square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        perturb = diff / diff_rms * ref_rms
        return perturb.to(dtype=reference.dtype).reshape_as(reference)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "feature_dim": self.feature_dim,
            "count": self.count,
            "write_pos": self.write_pos,
            "data": self.data.detach().cpu(),
            "dtype": str(self.dtype),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError("FeatureNoiseBuffer capacity mismatch in training checkpoint")
        if int(state.get("feature_dim", self.feature_dim)) != self.feature_dim:
            raise ValueError("FeatureNoiseBuffer feature_dim mismatch in training checkpoint")
        data = state.get("data")
        if data is not None:
            self.data.copy_(data.to(device=self.device, dtype=self.dtype))
        self.count = int(state.get("count", 0))
        self.write_pos = int(state.get("write_pos", 0))


def train_checkpoint_dir(args: argparse.Namespace) -> Path:
    if str(args.train_checkpoint_dir).strip():
        return Path(args.train_checkpoint_dir)
    return Path(args.out_dir).parent / "checkpoints"


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def save_training_checkpoint(
    *,
    args: argparse.Namespace,
    model: "FutureSeedLoopSudoku",
    opt: torch.optim.Optimizer,
    feature_buffer: FeatureNoiseBuffer,
    rng: random.Random,
    device: torch.device,
    global_step: int,
    stage_idx: int,
    holes_min: int,
    holes_max: int,
    t0: float,
    checkpoint_evals: Dict[str, Any],
    last_metrics: Dict[str, Any],
    reason: str,
) -> Path:
    out_dir = train_checkpoint_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "saved_at_step": int(global_step),
        "stage": int(stage_idx),
        "holes_min": int(holes_min),
        "holes_max": int(holes_max),
        "elapsed_sec": time.time() - t0,
        "reason": reason,
        "args": vars(args),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "feature_buffer": feature_buffer.state_dict(),
        "rng_python": rng.getstate(),
        "rng_torch": capture_rng_state(device),
        "checkpoint_evals": checkpoint_evals,
        "last_metrics": last_metrics,
    }
    step_path = out_dir / f"train_state_step{global_step:06d}.pt"
    tmp_path = out_dir / f".train_state_step{global_step:06d}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(step_path)
    latest_tmp = out_dir / ".latest.pt.tmp"
    latest_path = out_dir / "latest.pt"
    torch.save(payload, latest_tmp)
    latest_tmp.replace(latest_path)
    return step_path


def load_training_checkpoint(
    path: str,
    *,
    model: "FutureSeedLoopSudoku",
    opt: torch.optim.Optimizer,
    feature_buffer: FeatureNoiseBuffer,
    rng: random.Random,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    feature_buffer.load_state_dict(checkpoint.get("feature_buffer", {}))
    if "rng_python" in checkpoint:
        rng.setstate(checkpoint["rng_python"])
    restore_rng_state(checkpoint.get("rng_torch", {}), device)
    return checkpoint


class FutureSeedLoopSudoku(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        layers: int,
        heads: int,
        head_dim: int,
        channel_mult: int,
        l_cycles: int,
        lambda_: float,
        future_seed_scale: float,
        future_seed_decay: float,
        future_seed_update: str,
        loop_feedback_scale: float,
        loop_time_scale: float,
        scratch_mode: str,
        scratch_scale: float,
        scratch_noise_scale: float,
        scratch_gauss_projections: int,
        scratch_gate_bias: float,
        scratch_decay_bias: float,
        activation_checkpoint: bool,
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        self.l_cycles = int(l_cycles)
        self.lambda_ = float(lambda_)
        self.loop_feedback_scale = float(loop_feedback_scale)
        self.loop_time_scale = float(loop_time_scale)
        if scratch_mode not in {"none", "gated"}:
            raise ValueError("scratch_mode must be 'none' or 'gated'.")
        self.scratch_mode = scratch_mode
        self.scratch_scale = float(scratch_scale)
        self.scratch_noise_scale = float(scratch_noise_scale)
        self.embed = nn.Embedding(VOCAB, d_model)
        self.position = nn.Embedding(CELLS, d_model)
        self.reasoner = FutureSeedRWKV(
            d_model,
            layers,
            heads,
            head_dim,
            channel_mult,
            future_seed_scale=future_seed_scale,
            future_seed_decay=future_seed_decay,
            future_seed_update=future_seed_update,
            activation_checkpoint=activation_checkpoint,
            rwkv_kernel=rwkv_kernel,
        )
        self.h_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.l_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N, bias=False)
        if self.loop_time_scale > 0:
            self.loop_time = nn.Linear(2, d_model, bias=False)
            nn.init.normal_(self.loop_time.weight, mean=0.0, std=0.02)
        else:
            self.loop_time = None
        if self.loop_feedback_scale > 0:
            self.loop_feedback = nn.Linear(N, d_model, bias=False)
            nn.init.zeros_(self.loop_feedback.weight)
        else:
            self.loop_feedback = None
        if self.scratch_mode == "gated":
            self.scratch_init = nn.Parameter(torch.zeros(1, 1, d_model))
            self.scratch_norm = nn.LayerNorm(d_model)
            self.scratch_update_norm = nn.LayerNorm(d_model)
            self.scratch_residual = nn.Linear(d_model, d_model, bias=False)
            self.scratch_gate = nn.Linear(d_model, d_model)
            self.scratch_decay = nn.Linear(d_model, d_model)
            nn.init.normal_(self.scratch_residual.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.scratch_gate.weight)
            nn.init.zeros_(self.scratch_decay.weight)
            nn.init.constant_(self.scratch_gate.bias, float(scratch_gate_bias))
            nn.init.constant_(self.scratch_decay.bias, float(scratch_decay_bias))
            if scratch_gauss_projections > 0:
                projection = torch.randn(int(scratch_gauss_projections), d_model)
                projection = projection / projection.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                projection = torch.empty(0, d_model)
            self.register_buffer("scratch_projection", projection)
        else:
            self.register_parameter("scratch_init", None)
            self.scratch_norm = None
            self.scratch_update_norm = None
            self.scratch_residual = None
            self.scratch_gate = None
            self.scratch_decay = None
            self.register_buffer("scratch_projection", torch.empty(0, d_model))

    def input_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(CELLS, dtype=torch.long, device=inputs.device)
        return self.embed(inputs) + self.position(positions).unsqueeze(0)

    def scratch_gaussian_loss(self, residual: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.scratch_projection.numel() == 0:
            zero = residual.new_zeros(())
            return zero, {
                "scratch_proj_mean_abs": zero,
                "scratch_proj_var_mean": zero,
                "scratch_proj_var_rank": zero,
            }
        z = F.layer_norm(residual.float(), (residual.shape[-1],))
        projection = self.scratch_projection.to(device=z.device, dtype=z.dtype)
        projected = z.reshape(-1, z.shape[-1]) @ projection.t()
        mean = projected.mean(dim=0)
        var = projected.var(dim=0, unbiased=False)
        loss = mean.square().mean() + (var - 1.0).square().mean()
        var_detached = var.detach().clamp(min=1e-8)
        var_rank = var_detached.sum().square() / var_detached.square().sum().clamp(min=1e-8)
        return loss.to(dtype=residual.dtype), {
            "scratch_proj_mean_abs": mean.detach().abs().mean().to(dtype=residual.dtype),
            "scratch_proj_var_mean": var.detach().mean().to(dtype=residual.dtype),
            "scratch_proj_var_rank": var_rank.to(dtype=residual.dtype),
        }

    def scratch_input(
        self,
        scratch: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if scratch is None or self.scratch_mode == "none":
            zero = self.h_init.new_zeros(())
            return None, zero
        if self.training and self.scratch_noise_scale > 0:
            noise = torch.randn_like(scratch) * self.scratch_noise_scale
            return scratch + noise, noise.norm(dim=-1).mean()
        return scratch, scratch.new_zeros(())

    def update_scratch(
        self,
        scratch: Optional[torch.Tensor],
        hidden: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.scratch_mode == "none":
            zero = hidden.new_zeros(())
            return None, {
                "scratch_gate_mean": zero,
                "scratch_decay_mean": zero,
                "scratch_residual_norm": zero,
                "scratch_delta_norm": zero,
                "scratch_gauss_loss": zero,
                "scratch_proj_mean_abs": zero,
                "scratch_proj_var_mean": zero,
                "scratch_proj_var_rank": zero,
            }
        assert scratch is not None
        assert self.scratch_norm is not None
        assert self.scratch_update_norm is not None
        assert self.scratch_residual is not None
        assert self.scratch_gate is not None
        assert self.scratch_decay is not None
        h = self.scratch_norm(hidden)
        residual = torch.tanh(self.scratch_residual(h))
        gate = torch.sigmoid(self.scratch_gate(h))
        decay = torch.sigmoid(self.scratch_decay(h))
        updated = self.scratch_update_norm(decay * scratch + gate * residual)
        delta = updated - scratch
        gauss_loss, gauss_diag = self.scratch_gaussian_loss(residual)
        diag = {
            "scratch_gate_mean": gate.mean(),
            "scratch_decay_mean": decay.mean(),
            "scratch_residual_norm": residual.norm(dim=-1).mean(),
            "scratch_delta_norm": delta.norm(dim=-1).mean(),
            "scratch_gauss_loss": gauss_loss,
            **gauss_diag,
        }
        return updated, diag

    def depth_update(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        noise_scale: float,
        *,
        feature_buffer: Optional[FeatureNoiseBuffer],
        update_feature_buffer: bool,
        feature_buffer_add: int,
        allow_noise: bool,
        seed_memory: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        updated, diag, next_seed_memory = self.reasoner(hidden + injection, seed_memory=seed_memory)
        out = (1.0 - self.lambda_) * hidden + self.lambda_ * updated
        if allow_noise and noise_scale > 0:
            if feature_buffer is None:
                raise ValueError("feature-difference noise requires a FeatureNoiseBuffer")
            out = out + feature_buffer.sample_diff_like(out) * float(noise_scale)
        if allow_noise and update_feature_buffer and feature_buffer is not None:
            feature_buffer.add(out, max_items=feature_buffer_add)
        return out, diag, next_seed_memory

    def forward_trace(
        self,
        inputs: torch.Tensor,
        *,
        loops: int,
        noise_scale: float,
        feature_buffer: Optional[FeatureNoiseBuffer] = None,
        update_feature_buffer: bool = False,
        feature_buffer_add: int = 2048,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        x = self.input_sequence(inputs)
        batch_size, seq_len, _channels = x.shape
        z_h = self.h_init.expand(batch_size, seq_len, -1)
        z_l = self.l_init.expand(batch_size, seq_len, -1)
        feedback: Optional[torch.Tensor] = None
        scratch: Optional[torch.Tensor] = None
        if self.scratch_mode == "gated":
            assert self.scratch_init is not None
            scratch = self.scratch_init.expand(batch_size, seq_len, -1)
        h_seed_memory: Optional[List[torch.Tensor]] = None
        l_seed_memory: Optional[List[torch.Tensor]] = None
        zero = x.new_zeros(())

        loop_logits: List[torch.Tensor] = []
        fs_trace: List[Dict[str, torch.Tensor]] = []
        loop_count = int(loops)
        for loop_idx in range(loop_count):
            feedback_in_norm = zero if feedback is None else feedback.norm(dim=-1).mean()
            loop_context = x if feedback is None else x + feedback
            scratch_context, scratch_noise_norm = self.scratch_input(scratch)
            if scratch_context is not None and self.scratch_scale > 0:
                loop_context = loop_context + scratch_context * self.scratch_scale
            loop_time_norm = zero
            if self.loop_time is not None and self.loop_time_scale > 0:
                phase = float(loop_idx + 1) / float(max(loop_count, 1))
                features = x.new_tensor([[math.sin(math.pi * phase), math.cos(math.pi * phase)]])
                loop_time = self.loop_time(features).to(dtype=x.dtype).view(1, 1, -1) * self.loop_time_scale
                loop_context = loop_context + loop_time
                loop_time_norm = loop_time.norm(dim=-1).mean()
            for _ in range(self.l_cycles):
                z_l, _diag, l_seed_memory = self.depth_update(
                    z_l,
                    z_h + loop_context,
                    noise_scale,
                    feature_buffer=feature_buffer,
                    update_feature_buffer=update_feature_buffer,
                    feature_buffer_add=feature_buffer_add,
                    allow_noise=True,
                    seed_memory=l_seed_memory,
                )
            z_h, fs_diag, h_seed_memory = self.depth_update(
                z_h,
                z_l,
                noise_scale,
                feature_buffer=feature_buffer,
                update_feature_buffer=update_feature_buffer,
                feature_buffer_add=feature_buffer_add,
                allow_noise=True,
                seed_memory=h_seed_memory,
            )
            board_h = self.out_norm(z_h[:, :CELLS])
            logits = self.head(board_h)
            loop_logits.append(logits)
            fs_diag = dict(fs_diag)
            fs_diag["loop_feedback_in_norm"] = feedback_in_norm
            fs_diag["loop_time_norm"] = loop_time_norm
            fs_diag["scratch_noise_norm"] = scratch_noise_norm
            scratch, scratch_diag = self.update_scratch(scratch, z_h)
            fs_diag.update(scratch_diag)
            if self.loop_feedback is not None and self.loop_feedback_scale > 0:
                feedback = self.loop_feedback(logits.softmax(dim=-1)) * self.loop_feedback_scale
                fs_diag["loop_feedback_next_norm"] = feedback.norm(dim=-1).mean()
            else:
                feedback = None
                fs_diag["loop_feedback_next_norm"] = zero
            fs_trace.append(fs_diag)
        return loop_logits, fs_trace


def valid_board(row: torch.Tensor) -> bool:
    board = [int(x) for x in row.detach().cpu().tolist()]
    target = set(range(N))
    if any(x < 0 or x >= N for x in board):
        return False
    return all(set(board[i] for i in unit) == target for unit in UNITS)


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, clue_mask: torch.Tensor) -> Tuple[FullMetrics, torch.Tensor]:
    pred = logits.argmax(dim=-1)
    return metrics_from_predictions(pred, labels, clue_mask), pred


@torch.no_grad()
def metrics_from_predictions(pred: torch.Tensor, labels: torch.Tensor, clue_mask: torch.Tensor) -> FullMetrics:
    exact = (pred == labels).all(dim=1)
    clue_ok = ((pred == labels) | ~clue_mask).all(dim=1)
    valid = torch.tensor([valid_board(pred[i]) for i in range(pred.shape[0])], device=pred.device)
    blanks = ~clue_mask
    blank_acc = ((pred == labels) & blanks).sum().float() / blanks.sum().clamp_min(1)
    return FullMetrics(
        label_exact=exact.float().mean().item(),
        valid_sudoku=valid.float().mean().item(),
        solved_valid_clue=(valid & clue_ok).float().mean().item(),
        clue_ok=clue_ok.float().mean().item(),
        blank_acc=blank_acc.item(),
        avg_filled=pred.ne(BLANK).float().mean().item(),
    )


def loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    *,
    blank_weight: float,
) -> torch.Tensor:
    loss = F.cross_entropy(logits.float().reshape(-1, N), labels.reshape(-1), reduction="none").view_as(labels)
    if blank_weight == 1.0:
        return loss.mean()
    weights = torch.where(clue_mask, torch.ones_like(loss), torch.full_like(loss, float(blank_weight)))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def loop_weight_tensor(
    num_loops: int,
    *,
    mode: str,
    start: int,
    power: float,
    min_weight: float,
    device: torch.device,
) -> torch.Tensor:
    if num_loops <= 0:
        raise ValueError("num_loops must be positive")
    weights = torch.zeros(num_loops, dtype=torch.float32, device=device)
    if mode == "final":
        weights[-1] = 1.0
        return weights
    if mode == "all":
        weights.fill_(1.0 / num_loops)
        return weights

    first = min(max(int(start), 1), num_loops) - 1
    active = num_loops - first
    ramp = torch.linspace(1.0 / active, 1.0, active, dtype=torch.float32, device=device).pow(float(power))
    if mode == "shaped":
        early = torch.full((first,), float(min_weight), dtype=torch.float32, device=device)
        weights = torch.cat((early, torch.clamp(ramp, min=float(min_weight))))
    elif mode == "delayed":
        weights[first:] = ramp
    else:
        raise ValueError(f"Unknown loop loss mode {mode!r}")
    return weights / weights.sum().clamp_min(1e-8)


def weighted_loop_loss(losses: List[torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = loop_weight_tensor(
        len(losses),
        mode=args.loop_loss,
        start=args.loop_loss_start,
        power=args.loop_loss_power,
        min_weight=args.loop_loss_min_weight,
        device=losses[-1].device,
    ).to(dtype=losses[-1].dtype)
    stacked = torch.stack(losses)
    return (stacked * weights).sum(), weights


@torch.no_grad()
def fs_metrics_from_trace(trace: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in trace.items()}


def metric_line(m: Dict[str, float]) -> str:
    return (
        f"exact={m['label_exact']:.4f}, valid={m['valid_sudoku']:.4f}, "
        f"solved={m['solved_valid_clue']:.4f}, blank_acc={m['blank_acc']:.4f}"
    )


def coupling_line(m: Dict[str, float], holes: int) -> str:
    independent = float(m["blank_acc"]) ** int(holes)
    ratio = float(m["label_exact"]) / independent if independent > 0 else float("nan")
    return f"indep={independent:.4f}, exact/indep={ratio:.2f}"


def fs_line(m: Dict[str, float]) -> str:
    parts = [
        f"fs_gate={m['fs_gate_mean']:.3f}",
        f"fs_update={m.get('fs_update_mean', 0.0):.3f}",
        f"fs_state_norm={m['fs_state_norm']:.3f}",
    ]
    if "fs_decay" in m:
        parts.append(f"fs_decay={m['fs_decay']:.2f}")
    if "fs_memory_norm" in m:
        parts.append(f"fs_mem={m['fs_memory_norm']:.3f}")
    if "fs_memory_delta_norm" in m:
        parts.append(f"fs_mem_delta={m['fs_memory_delta_norm']:.3f}")
    if "loop_feedback_in_norm" in m:
        parts.append(f"fb_in={m['loop_feedback_in_norm']:.3f}")
    if "loop_feedback_next_norm" in m:
        parts.append(f"fb_next={m['loop_feedback_next_norm']:.3f}")
    if "loop_time_norm" in m:
        parts.append(f"loop_time={m['loop_time_norm']:.3f}")
    if "scratch_gate_mean" in m:
        parts.append(f"scratch_gate={m['scratch_gate_mean']:.3f}")
        parts.append(f"scratch_decay={m.get('scratch_decay_mean', 0.0):.3f}")
        parts.append(f"scratch_delta={m.get('scratch_delta_norm', 0.0):.3f}")
        parts.append(f"scratch_resid={m.get('scratch_residual_norm', 0.0):.3f}")
    if m.get("scratch_noise_norm", 0.0) > 0:
        parts.append(f"scratch_noise={m['scratch_noise_norm']:.3f}")
    if m.get("scratch_proj_var_rank", 0.0) > 0:
        parts.append(f"scratch_var={m.get('scratch_proj_var_mean', 0.0):.3f}")
        parts.append(f"scratch_rank={m['scratch_proj_var_rank']:.1f}")
    return ", ".join(parts)


def train_model(args: argparse.Namespace, *, device: torch.device) -> Tuple[FutureSeedLoopSudoku, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rng = random.Random(args.seed + 1000)
    model = FutureSeedLoopSudoku(
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        channel_mult=args.channel_mult,
        l_cycles=args.l_cycles,
        lambda_=args.lambda_,
        future_seed_scale=args.future_seed_scale,
        future_seed_decay=args.future_seed_decay,
        future_seed_update=args.future_seed_update,
        loop_feedback_scale=args.loop_feedback_scale,
        loop_time_scale=args.loop_time_scale,
        scratch_mode=args.scratch_mode,
        scratch_scale=args.scratch_scale,
        scratch_noise_scale=args.scratch_noise_scale,
        scratch_gauss_projections=args.scratch_gauss_projections,
        scratch_gate_bias=args.scratch_gate_bias,
        scratch_decay_bias=args.scratch_decay_bias,
        activation_checkpoint=args.activation_checkpoint,
        rwkv_kernel=args.rwkv_kernel,
    ).to(device)
    feature_buffer = FeatureNoiseBuffer(
        capacity=args.feature_buffer_size,
        feature_dim=args.d_model,
        device=device,
        dtype=torch.float32,
    )
    model.feature_noise_buffer = feature_buffer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.time()
    resume_info: Dict[str, Any] = {}
    saved_train_checkpoints: List[str] = []
    last_ce_loss = 0.0
    last_total_loss = 0.0
    last_loop1_loss = 0.0
    last_loop_last_loss = 0.0
    last_loop_weights: List[float] = []
    last_scratch_gauss_loss = 0.0
    last_scratch_gate = 0.0
    last_scratch_decay = 0.0
    last_scratch_delta = 0.0
    last_scratch_residual = 0.0
    last_scratch_proj_var = 0.0
    last_scratch_proj_rank = 0.0
    stages = parse_hole_stages(args)
    checkpoint_steps = parse_eval_checkpoint_steps(args, stages)
    checkpoint_step_set = set(checkpoint_steps)
    checkpoint_evals: Dict[str, Any] = {}
    checkpoint_batches = {}
    if checkpoint_steps:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_holes = parse_eval_checkpoint_holes(args)
        checkpoint_batches = {
            holes: make_batch(
                args.eval_n,
                holes,
                holes,
                args.hole_pattern,
                random.Random(args.seed + 13000 + holes * 17),
                device=device,
            )
            for holes in checkpoint_holes
        }
        print(
            "checkpoint_eval "
            f"steps={','.join(str(step) for step in checkpoint_steps)} "
            f"holes={','.join(str(holes) for holes in checkpoint_holes)}",
            flush=True,
        )

    global_step = 0
    if str(args.resume_train_checkpoint).strip():
        checkpoint = load_training_checkpoint(
            args.resume_train_checkpoint,
            model=model,
            opt=opt,
            feature_buffer=feature_buffer,
            rng=rng,
            device=device,
        )
        global_step = int(checkpoint.get("saved_at_step", 0))
        checkpoint_evals = dict(checkpoint.get("checkpoint_evals", {}))
        last_metrics = dict(checkpoint.get("last_metrics", {}))
        last_ce_loss = float(last_metrics.get("ce_loss", 0.0))
        last_total_loss = float(last_metrics.get("total_loss", 0.0))
        last_loop1_loss = float(last_metrics.get("loop1_loss", 0.0))
        last_loop_last_loss = float(last_metrics.get("loop_last_loss", 0.0))
        last_loop_weights = [float(x) for x in last_metrics.get("loop_weights", [])]
        last_scratch_gauss_loss = float(last_metrics.get("scratch_gauss_loss", 0.0))
        last_scratch_gate = float(last_metrics.get("scratch_gate", 0.0))
        last_scratch_decay = float(last_metrics.get("scratch_decay", 0.0))
        last_scratch_delta = float(last_metrics.get("scratch_delta", 0.0))
        last_scratch_residual = float(last_metrics.get("scratch_residual", 0.0))
        last_scratch_proj_var = float(last_metrics.get("scratch_proj_var", 0.0))
        last_scratch_proj_rank = float(last_metrics.get("scratch_proj_rank", 0.0))
        resume_info = {
            "path": str(args.resume_train_checkpoint),
            "saved_at_step": global_step,
            "checkpoint_version": checkpoint.get("version"),
            "checkpoint_reason": checkpoint.get("reason"),
            "checkpoint_elapsed_sec": checkpoint.get("elapsed_sec"),
        }
        print(
            f"resumed_train_checkpoint path={args.resume_train_checkpoint} "
            f"step={global_step} reason={checkpoint.get('reason', '')}",
            flush=True,
        )

    total_steps = sum(stage_steps for _lo, _hi, stage_steps in stages)
    if global_step > total_steps:
        raise ValueError(f"resume step {global_step} exceeds configured total steps {total_steps}")
    last_saved_step = -1
    completed_before_stage = 0
    for stage_idx, (holes_min, holes_max, stage_steps) in enumerate(stages, start=1):
        stage_end = completed_before_stage + stage_steps
        if global_step >= stage_end:
            completed_before_stage = stage_end
            continue
        for _ in range(global_step - completed_before_stage, stage_steps):
            global_step += 1
            model.train()
            inputs, labels, clue_mask = make_batch(
                args.batch,
                holes_min,
                holes_max,
                args.hole_pattern,
                rng,
                device=device,
            )
            with forward_autocast(args.forward_dtype, device):
                loop_logits, _fs_trace = model.forward_trace(
                    inputs,
                    loops=args.max_loops,
                    noise_scale=args.noise_scale,
                    feature_buffer=feature_buffer,
                    update_feature_buffer=model.training,
                    feature_buffer_add=args.feature_buffer_add,
                )
            loop_losses = [
                loss_from_logits(logits, labels, clue_mask, blank_weight=args.blank_loss_weight)
                for logits in loop_logits
            ]
            ce_loss = loop_losses[-1]
            supervised_loss, loop_weights = weighted_loop_loss(loop_losses, args)
            scratch_gauss_terms = [
                trace["scratch_gauss_loss"].to(dtype=supervised_loss.dtype)
                for trace in _fs_trace
                if "scratch_gauss_loss" in trace
            ]
            scratch_gauss_loss = (
                torch.stack(scratch_gauss_terms).mean()
                if scratch_gauss_terms
                else supervised_loss.new_zeros(())
            )
            loss = supervised_loss + float(args.scratch_gauss_weight) * scratch_gauss_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_ce_loss = float(ce_loss.detach().cpu())
            last_total_loss = float(loss.detach().cpu())
            last_loop1_loss = float(loop_losses[0].detach().cpu())
            last_loop_last_loss = float(loop_losses[-1].detach().cpu())
            last_loop_weights = [float(x) for x in loop_weights.detach().cpu().tolist()]
            last_scratch_gauss_loss = float(scratch_gauss_loss.detach().cpu())
            if _fs_trace:
                trace_last = _fs_trace[-1]
                last_scratch_gate = float(trace_last.get("scratch_gate_mean", ce_loss.new_zeros(())).detach().cpu())
                last_scratch_decay = float(trace_last.get("scratch_decay_mean", ce_loss.new_zeros(())).detach().cpu())
                last_scratch_delta = float(trace_last.get("scratch_delta_norm", ce_loss.new_zeros(())).detach().cpu())
                last_scratch_residual = float(trace_last.get("scratch_residual_norm", ce_loss.new_zeros(())).detach().cpu())
                last_scratch_proj_var = float(trace_last.get("scratch_proj_var_mean", ce_loss.new_zeros(())).detach().cpu())
                last_scratch_proj_rank = float(trace_last.get("scratch_proj_var_rank", ce_loss.new_zeros(())).detach().cpu())
            if args.log_every and global_step % args.log_every == 0:
                print(
                    f"[future_seed_loop stage={stage_idx}:{holes_min}-{holes_max}] "
                    f"step={global_step:04d} ce={last_ce_loss:.4f} total={last_total_loss:.4f} "
                    f"loop1={last_loop1_loss:.4f} loop_last={last_loop_last_loss:.4f} "
                    f"scratch_delta={last_scratch_delta:.3f} scratch_gauss={last_scratch_gauss_loss:.4f}",
                    flush=True,
                )
            if global_step in checkpoint_step_set:
                model.eval()
                checkpoint = {
                    "step": global_step,
                    "stage": stage_idx,
                    "holes_min": holes_min,
                    "holes_max": holes_max,
                    "elapsed_sec": time.time() - t0,
                    "train": {
                        "ce_loss": last_ce_loss,
                        "total_loss": last_total_loss,
                        "loop1_loss": last_loop1_loss,
                        "loop_last_loss": last_loop_last_loss,
                        "scratch_gauss_loss": last_scratch_gauss_loss,
                        "scratch_gate": last_scratch_gate,
                        "scratch_decay": last_scratch_decay,
                        "scratch_delta": last_scratch_delta,
                        "scratch_residual": last_scratch_residual,
                        "scratch_proj_var": last_scratch_proj_var,
                        "scratch_proj_rank": last_scratch_proj_rank,
                    },
                    "eval_by_holes": {},
                }
                for holes, eval_batch in checkpoint_batches.items():
                    clean, _preds = evaluate_model(
                        model,
                        eval_batch,
                        max_loops=args.max_loops,
                        noise_scale=0.0,
                        seed=args.seed + 14000 + global_step + holes,
                        forward_dtype=args.forward_dtype,
                    )
                    checkpoint["eval_by_holes"][f"holes{holes}"] = {"eval_clean": clean}
                checkpoint_key = f"step{global_step}"
                checkpoint_evals[checkpoint_key] = checkpoint
                checkpoint_path = Path(args.out_dir) / f"checkpoint_eval_step{global_step:06d}.json"
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                target_key = f"holes{args.eval_holes}"
                if target_key in checkpoint["eval_by_holes"]:
                    last_key = f"loop{args.max_loops}"
                    target_metrics = checkpoint["eval_by_holes"][target_key]["eval_clean"][last_key]
                    print(
                        f"[checkpoint_eval step={global_step:04d} {target_key}] "
                        f"loop{args.max_loops} exact={target_metrics['label_exact']:.4f} "
                        f"blank={target_metrics['blank_acc']:.4f}",
                        flush=True,
                    )
                model.train()
                if args.save_train_checkpoint_every >= 0:
                    path = save_training_checkpoint(
                        args=args,
                        model=model,
                        opt=opt,
                        feature_buffer=feature_buffer,
                        rng=rng,
                        device=device,
                        global_step=global_step,
                        stage_idx=stage_idx,
                        holes_min=holes_min,
                        holes_max=holes_max,
                        t0=t0,
                        checkpoint_evals=checkpoint_evals,
                        last_metrics={
                            "ce_loss": last_ce_loss,
                            "total_loss": last_total_loss,
                            "loop1_loss": last_loop1_loss,
                            "loop_last_loss": last_loop_last_loss,
                            "loop_weights": last_loop_weights,
                            "scratch_gauss_loss": last_scratch_gauss_loss,
                            "scratch_gate": last_scratch_gate,
                            "scratch_decay": last_scratch_decay,
                            "scratch_delta": last_scratch_delta,
                            "scratch_residual": last_scratch_residual,
                            "scratch_proj_var": last_scratch_proj_var,
                            "scratch_proj_rank": last_scratch_proj_rank,
                        },
                        reason="eval_checkpoint",
                    )
                    saved_train_checkpoints.append(str(path))
                    last_saved_step = global_step
                    print(f"[train_checkpoint step={global_step:04d}] path={path}", flush=True)

            if (
                args.save_train_checkpoint_every > 0
                and global_step % args.save_train_checkpoint_every == 0
                and global_step != last_saved_step
            ):
                path = save_training_checkpoint(
                    args=args,
                    model=model,
                    opt=opt,
                    feature_buffer=feature_buffer,
                    rng=rng,
                    device=device,
                    global_step=global_step,
                    stage_idx=stage_idx,
                    holes_min=holes_min,
                    holes_max=holes_max,
                    t0=t0,
                    checkpoint_evals=checkpoint_evals,
                    last_metrics={
                        "ce_loss": last_ce_loss,
                        "total_loss": last_total_loss,
                        "loop1_loss": last_loop1_loss,
                        "loop_last_loss": last_loop_last_loss,
                        "loop_weights": last_loop_weights,
                        "scratch_gauss_loss": last_scratch_gauss_loss,
                        "scratch_gate": last_scratch_gate,
                        "scratch_decay": last_scratch_decay,
                        "scratch_delta": last_scratch_delta,
                        "scratch_residual": last_scratch_residual,
                        "scratch_proj_var": last_scratch_proj_var,
                        "scratch_proj_rank": last_scratch_proj_rank,
                    },
                    reason="periodic",
                )
                saved_train_checkpoints.append(str(path))
                last_saved_step = global_step
                print(f"[train_checkpoint step={global_step:04d}] path={path}", flush=True)
        completed_before_stage = stage_end

    train_stats = {
        "train_ce_loss": last_ce_loss,
        "train_total_loss": last_total_loss,
        "train_loop1_loss": last_loop1_loss,
        "train_loop_last_loss": last_loop_last_loss,
        "loop_loss_mode": args.loop_loss,
        "loop_loss_start": args.loop_loss_start,
        "loop_loss_power": args.loop_loss_power,
        "loop_loss_min_weight": args.loop_loss_min_weight,
        "loop_loss_weights": last_loop_weights,
        "noise_mode": "feature_diff",
        "scratch_mode": args.scratch_mode,
        "scratch_scale": args.scratch_scale,
        "scratch_noise_scale": args.scratch_noise_scale,
        "scratch_gauss_weight": args.scratch_gauss_weight,
        "scratch_gauss_projections": args.scratch_gauss_projections,
        "scratch_gate_bias": args.scratch_gate_bias,
        "scratch_decay_bias": args.scratch_decay_bias,
        "scratch_gauss_loss": last_scratch_gauss_loss,
        "scratch_gate": last_scratch_gate,
        "scratch_decay": last_scratch_decay,
        "scratch_delta": last_scratch_delta,
        "scratch_residual": last_scratch_residual,
        "scratch_proj_var": last_scratch_proj_var,
        "scratch_proj_rank": last_scratch_proj_rank,
        "feature_buffer_count": feature_buffer.count,
        "train_sec": time.time() - t0,
        "rwkv_kernel": args.rwkv_kernel,
        "forward_dtype": args.forward_dtype,
        "activation_checkpoint": bool(args.activation_checkpoint),
        "resume_train_checkpoint": resume_info,
        "save_train_checkpoint_every": args.save_train_checkpoint_every,
        "saved_train_checkpoints": saved_train_checkpoints,
        "cuda_max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "cuda_max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else 0.0
        ),
        "future_seed_update": args.future_seed_update,
        "loop_feedback_scale": args.loop_feedback_scale,
        "loop_time_scale": args.loop_time_scale,
        "eval_checkpoint_steps": checkpoint_steps,
        "eval_checkpoint_holes": sorted(checkpoint_batches),
        "checkpoint_evals": checkpoint_evals,
    }
    return model.eval(), train_stats


@torch.no_grad()
def evaluate_model(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    max_loops: int,
    noise_scale: float,
    seed: int,
    forward_dtype: str = "float32",
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    if noise_scale > 0:
        torch.manual_seed(seed)
    inputs, labels, clue_mask = batch
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    with forward_autocast(forward_dtype, inputs.device):
        loop_logits, fs_trace = model.forward_trace(
            inputs,
            loops=max_loops,
            noise_scale=noise_scale,
            feature_buffer=feature_buffer,
            update_feature_buffer=False,
        )
    out: Dict[str, Any] = {}
    loop_preds: List[torch.Tensor] = []
    for loop_idx, logits in enumerate(loop_logits, start=1):
        metrics, pred = metrics_from_logits(logits, labels, clue_mask)
        out[f"loop{loop_idx}"] = asdict(metrics)
        out[f"loop{loop_idx}/future_seed"] = fs_metrics_from_trace(fs_trace[loop_idx - 1])
        loop_preds.append(pred)
    return out, loop_preds


def parse_rollout_ks(args: argparse.Namespace) -> List[int]:
    raw = str(args.rollout_ks).strip()
    if not raw:
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("--rollout_ks values must be positive")
        if value not in values:
            values.append(value)
    return sorted(values)


def parse_rollout_loop_values(args: argparse.Namespace) -> List[int]:
    raw = str(args.rollout_loop_values).strip()
    if not raw:
        return []
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > args.max_loops:
            raise ValueError(f"--rollout_loop_values entries must be in [1, {args.max_loops}]")
        if value not in values:
            values.append(value)
    return sorted(values)


@torch.no_grad()
def selected_metrics(
    logits: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
) -> Dict[str, float]:
    batch_size = labels.shape[0]
    selected = scores.argmax(dim=0)
    batch_idx = torch.arange(batch_size, device=labels.device)
    selected_logits = logits[selected, batch_idx]
    metrics, _pred = metrics_from_logits(selected_logits, labels, clue_mask)
    return asdict(metrics)


@torch.no_grad()
def evaluate_rollouts(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    max_loops: int,
    noise_scale: float,
    ks: List[int],
    seed: int,
    forward_dtype: str = "float32",
) -> Dict[str, Any]:
    if not ks:
        return {}
    inputs, labels, clue_mask = batch
    max_k = max(ks)
    last_logits_list = []
    prev_logits_list = []
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    for rollout_idx in range(max_k):
        if noise_scale > 0:
            torch.manual_seed(seed + rollout_idx)
        with forward_autocast(forward_dtype, inputs.device):
            loop_logits, _fs_trace = model.forward_trace(
                inputs,
                loops=max_loops,
                noise_scale=noise_scale,
                feature_buffer=feature_buffer,
                update_feature_buffer=False,
            )
        last_logits_list.append(loop_logits[-1])
        prev_logits_list.append(loop_logits[-2] if max_loops > 1 else loop_logits[-1])

    last_logits = torch.stack(last_logits_list, dim=0)
    prev_logits = torch.stack(prev_logits_list, dim=0)
    preds = last_logits.argmax(dim=-1)
    labels_k = labels.unsqueeze(0)
    clue_k = clue_mask.unsqueeze(0)
    blank_k = ~clue_k
    exact = (preds == labels_k).all(dim=-1)
    clue_ok = ((preds == labels_k) | ~clue_k).all(dim=-1)
    valid_rows = []
    for rollout_idx in range(max_k):
        valid_rows.append(torch.tensor([valid_board(preds[rollout_idx, i]) for i in range(labels.shape[0])], device=labels.device))
    valid = torch.stack(valid_rows, dim=0)
    solved = valid & clue_ok

    probs = last_logits.float().softmax(dim=-1)
    max_prob = probs.max(dim=-1).values
    confidence_scores = (max_prob * blank_k).sum(dim=-1) / blank_k.sum(dim=-1).clamp_min(1)
    residual = (last_logits.float().softmax(dim=-1) - prev_logits.float().softmax(dim=-1)).square().mean(dim=-1)
    residual_scores = -((residual * blank_k).sum(dim=-1) / blank_k.sum(dim=-1).clamp_min(1))

    out: Dict[str, Any] = {}
    for k in ks:
        subset_logits = last_logits[:k]
        subset_preds = preds[:k]
        subset_exact = exact[:k]
        subset_solved = solved[:k]
        mode_pred = F.one_hot(subset_preds, num_classes=N).sum(dim=0).argmax(dim=-1)
        consistency_scores = ((subset_preds == mode_pred.unsqueeze(0)) & blank_k).float().sum(dim=-1)
        consistency_scores = consistency_scores / blank_k.sum(dim=-1).clamp_min(1)
        mode_metrics = asdict(metrics_from_predictions(mode_pred, labels, clue_mask))
        token_disagreement = 1.0 - float(consistency_scores.mean().detach().cpu())

        confidence_metrics = selected_metrics(subset_logits, confidence_scores[:k], labels, clue_mask)
        consistency_metrics = selected_metrics(subset_logits, consistency_scores, labels, clue_mask)
        residual_metrics = selected_metrics(subset_logits, residual_scores[:k], labels, clue_mask)

        oracle_exact = float(subset_exact.any(dim=0).float().mean().detach().cpu())
        oracle_solved = float(subset_solved.any(dim=0).float().mean().detach().cpu())
        out[f"K{k}"] = {
            "oracle_label_exact": oracle_exact,
            "oracle_solved_valid_clue": oracle_solved,
            "trajectory_token_disagreement": token_disagreement,
            "selector_confidence": confidence_metrics,
            "selector_consistency": consistency_metrics,
            "selector_residual": residual_metrics,
            "selector_majority": mode_metrics,
            "selector_gap_confidence": oracle_exact - float(confidence_metrics["label_exact"]),
            "selector_gap_consistency": oracle_exact - float(consistency_metrics["label_exact"]),
            "selector_gap_residual": oracle_exact - float(residual_metrics["label_exact"]),
            "selector_gap_majority": oracle_exact - float(mode_metrics["label_exact"]),
        }
    return out


@torch.no_grad()
def evaluate_rollouts_by_loop(
    model: FutureSeedLoopSudoku,
    batch,
    *,
    loop_values: List[int],
    noise_scale: float,
    ks: List[int],
    seed: int,
    forward_dtype: str = "float32",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for loops in loop_values:
        out[f"loop{loops}"] = evaluate_rollouts(
            model,
            batch,
            max_loops=loops,
            noise_scale=noise_scale,
            ks=ks,
            seed=seed + loops * 1009,
            forward_dtype=forward_dtype,
        )
    return out


def board_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().view(-1).tolist()]


def digit(v: int) -> str:
    return "." if int(v) == BLANK else str(int(v) + 1)


def coord(idx: int) -> str:
    row, col = divmod(int(idx), N)
    return f"R{row + 1}C{col + 1}"


def grid_html(board: List[int], solution: List[int], clue: List[bool]) -> str:
    cells = []
    for i, value in enumerate(board):
        row, col = divmod(i, N)
        cls = ["cell"]
        if clue[i]:
            cls.append("clue")
        elif value == solution[i]:
            cls.append("correct")
        else:
            cls.append("wrong")
        style = []
        if col == N - 1:
            style.append("border-right: 0")
        elif (col + 1) % BOX_COLS == 0:
            style.append("border-right: 2px solid #0f172a")
        if row == N - 1:
            style.append("border-bottom: 0")
        elif (row + 1) % BOX_ROWS == 0:
            style.append("border-bottom: 2px solid #0f172a")
        style_attr = f' style="{"; ".join(style)}"' if style else ""
        cells.append(f'<div class="{" ".join(cls)}"{style_attr}>{html.escape(digit(value))}</div>')
    cols = " ".join(["24px"] * N)
    rows = " ".join(["24px"] * N)
    grid_style = f"grid-template-columns: {cols}; grid-template-rows: {rows}"
    return f'<div class="grid" style="{grid_style}">' + "".join(cells) + "</div>"


def write_case_html(
    path: Path,
    *,
    title: str,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    clue_mask: torch.Tensor,
    loop_preds: List[torch.Tensor],
) -> None:
    clue = [bool(x) for x in clue_mask.detach().cpu().tolist()]
    solution = board_list(labels)
    puzzle = board_list(inputs)
    sections = []
    for loop_idx, pred in enumerate(loop_preds, start=1):
        sections.append(
            f'<div class="panel"><h3>loop {loop_idx}</h3>{grid_html(board_list(pred), solution, clue)}</div>'
        )

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 22px; background: #f8fafc; color: #0f172a; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
h3 {{ margin: 0 0 8px; font-size: 13px; }}
.meta {{ color: #475569; margin: 0 0 16px; }}
.row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
.panel {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; }}
.grid {{ display: grid; border: 2px solid #0f172a; width: max-content; }}
.cell {{ width: 24px; height: 24px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-right: 1px solid #94a3b8; border-bottom: 1px solid #94a3b8; font-size: 13px; font-weight: 750; }}
.clue {{ background: #e2e8f0; color: #0f172a; }}
.correct {{ background: #bbf7d0; color: #14532d; }}
.wrong {{ background: #fecaca; color: #7f1d1d; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">FutureSeed RWKV with depth-loop refinement. Green matches sampled solution, red differs, gray is clue.</p>
<div class="row">
  <div class="panel"><h3>puzzle</h3>{grid_html(puzzle, solution, clue)}</div>
  <div class="panel"><h3>solution</h3>{grid_html(solution, solution, [False] * CELLS)}</div>
</div>
<div class="row" style="margin-top: 12px;">{"".join(sections)}</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def parse_int_csv(raw: str, *, name: str, low: int, high: int) -> List[int]:
    values: List[int] = []
    if not str(raw).strip():
        return values
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < low or value > high:
            raise ValueError(f"{name} entries must be in [{low}, {high}]")
        if value not in values:
            values.append(value)
    return values


def conflict_info(board: List[int]) -> Tuple[set[int], List[str]]:
    conflict_cells: set[int] = set()
    conflict_units: List[str] = []
    unit_names = [f"row {i + 1}" for i in range(N)] + [f"col {i + 1}" for i in range(N)] + [
        f"box {r + 1},{c + 1}" for r in range(N // BOX_ROWS) for c in range(N // BOX_COLS)
    ]
    for name, unit in zip(unit_names, UNITS):
        by_digit: Dict[int, List[int]] = {}
        for idx in unit:
            value = int(board[idx])
            if 0 <= value < N:
                by_digit.setdefault(value, []).append(idx)
        repeated = {digit_value: locs for digit_value, locs in by_digit.items() if len(locs) > 1}
        if not repeated:
            continue
        bits = []
        for digit_value, locs in sorted(repeated.items()):
            conflict_cells.update(locs)
            bits.append(f"{digit(digit_value)}@" + ",".join(coord(idx) for idx in locs))
        conflict_units.append(f"{name}: " + "; ".join(bits))
    return conflict_cells, conflict_units


def case_grid_html(
    board: List[int],
    solution: List[int],
    clue: List[bool],
    *,
    conflict_cells: set[int] | None = None,
    previous: List[int] | None = None,
    mode: str = "prediction",
) -> str:
    conflict_cells = conflict_cells or set()
    cell_px = 34 if N <= 12 else 30 if N <= 16 else 26
    font_px = 13 if N <= 16 else 11
    cells = []
    for i, value in enumerate(board):
        row, col = divmod(i, N)
        cls = ["cell"]
        title = coord(i)
        if mode == "puzzle":
            if clue[i]:
                cls.append("clue")
                title += " clue"
            else:
                cls.append("hole")
                title += " hidden"
        elif mode == "solution":
            cls.append("solution")
            if clue[i]:
                cls.append("given-solution")
        elif clue[i]:
            cls.append("clue")
        elif value == solution[i]:
            cls.append("correct")
        else:
            cls.append("wrong")
            title += f" truth={digit(solution[i])}"
        if i in conflict_cells:
            cls.append("conflict")
            title += " duplicate-conflict"
        if previous is not None and not clue[i] and int(previous[i]) != int(value):
            cls.append("changed")
            title += f" changed-from={digit(previous[i])}"
        style = [f"width:{cell_px}px", f"height:{cell_px}px", f"font-size:{font_px}px"]
        if col == N - 1:
            style.append("border-right:0")
        elif (col + 1) % BOX_COLS == 0:
            style.append("border-right:2px solid #0f172a")
        if row == N - 1:
            style.append("border-bottom:0")
        elif (row + 1) % BOX_ROWS == 0:
            style.append("border-bottom:2px solid #0f172a")
        cells.append(
            f'<div class="{" ".join(cls)}" style="{";".join(style)}" title="{html.escape(title)}">'
            f"{html.escape(digit(value))}</div>"
        )
    grid_style = f"grid-template-columns: repeat({N}, {cell_px}px); grid-template-rows: repeat({N}, {cell_px}px)"
    return f'<div class="grid" style="{grid_style}">' + "".join(cells) + "</div>"


def case_panel_html(
    title: str,
    subtitle: str,
    board: List[int],
    solution: List[int],
    clue: List[bool],
    *,
    conflict_cells: set[int] | None = None,
    previous: List[int] | None = None,
    mode: str = "prediction",
) -> str:
    return (
        '<section class="case-panel">'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p>"
        f"{case_grid_html(board, solution, clue, conflict_cells=conflict_cells, previous=previous, mode=mode)}"
        "</section>"
    )


def write_case_bank_case_html(path: Path, case: Dict[str, Any], loop_values: List[int]) -> None:
    puzzle = case["_puzzle"]
    solution = case["_solution"]
    clue = case["_clue"]
    panels = [
        case_panel_html("Puzzle", f"{case['holes']} hidden cells", puzzle, solution, clue, mode="puzzle"),
        case_panel_html("Solution", "sampled target board", solution, solution, clue, mode="solution"),
    ]
    previous: Optional[List[int]] = None
    for loop in loop_values:
        loop_key = f"loop{loop}"
        row = case["loops"][loop_key]
        panels.append(
            case_panel_html(
                f"Loop {loop}",
                (
                    f"wrong={row['wrong_count']}/{case['holes']}; "
                    f"changed={row['changed_from_previous']}; "
                    f"entropy={row['hidden_entropy_mean']:.3f}; "
                    f"conflicts={row['conflict_unit_count']}"
                ),
                row["_board"],
                solution,
                clue,
                conflict_cells=set(row["_conflict_cells"]),
                previous=previous,
            )
        )
        previous = row["_board"]
    wrong_lines = []
    for loop in loop_values:
        loop_key = f"loop{loop}"
        wrong = case["loops"][loop_key]["wrong_blanks"]
        wrong_lines.append(
            f"<li><code>{loop_key}</code>: "
            + (", ".join(f"<code>{html.escape(item['coord'])}</code>" for item in wrong[:32]) if wrong else "none")
            + (" ..." if len(wrong) > 32 else "")
            + "</li>"
        )

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(case['title'])}</title>
<style>
body {{ margin: 22px; background: #f6f8fa; color: #24292f; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
h2 {{ margin: 18px 0 8px; font-size: 17px; letter-spacing: 0; }}
h3 {{ margin: 0 0 5px; font-size: 14px; letter-spacing: 0; }}
p {{ margin: 0; color: #57606a; font-size: 13px; line-height: 1.35; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }}
.case-panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 10px; }}
.case-panel p {{ min-height: 35px; max-width: 420px; }}
.grid {{ display: grid; border: 2px solid #0f172a; width: max-content; margin-top: 9px; }}
.cell {{ box-sizing: border-box; display: flex; align-items: center; justify-content: center; border-right: 1px solid #94a3b8; border-bottom: 1px solid #94a3b8; font-weight: 760; position: relative; }}
.clue {{ background: #e2e8f0; color: #0f172a; }}
.hole {{ background: #fff7ed; color: #9a3412; }}
.solution {{ background: #e0f2fe; color: #0c4a6e; }}
.given-solution {{ box-shadow: inset 0 0 0 2px rgba(15,23,42,0.13); }}
.correct {{ background: #bbf7d0; color: #14532d; }}
.wrong {{ background: #fecaca; color: #7f1d1d; }}
.conflict::after {{ content: ""; position: absolute; inset: 3px; border: 3px solid #f59e0b; border-radius: 5px; pointer-events: none; }}
.changed::before {{ content: ""; position: absolute; left: 7px; right: 7px; bottom: 4px; height: 3px; background: #1f6feb; border-radius: 4px; }}
.notes {{ max-width: 1200px; margin-top: 14px; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 14px; }}
.notes ul {{ margin: 8px 0 0 18px; padding: 0; }}
.notes li {{ margin: 4px 0; font-size: 13px; color: #57606a; }}
</style>
</head>
<body>
<h1>{html.escape(case['title'])}</h1>
<p>Kind: <code>{html.escape(case['kind'])}</code>. Batch index {case['batch_index']}. Blue underline marks hidden cells changed since the previous shown loop; orange outline marks duplicate conflicts.</p>
<div class="row" style="margin-top: 14px;">{''.join(panels)}</div>
<div class="notes">
  <h2>Wrong hidden cells</h2>
  <ul>{''.join(wrong_lines)}</ul>
</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def json_ready_case(case: Dict[str, Any]) -> Dict[str, Any]:
    out = {key: value for key, value in case.items() if not key.startswith("_")}
    loops = {}
    for loop_key, row in case["loops"].items():
        loops[loop_key] = {key: value for key, value in row.items() if not key.startswith("_")}
    out["loops"] = loops
    return out


def case_cell_diagnostic(
    *,
    cell: int,
    loop: int,
    board: List[int],
    solution: List[int],
    preds_cpu: List[torch.Tensor],
    margins: List[torch.Tensor],
    entropies: List[torch.Tensor],
    idx: int,
    loop_values: List[int],
) -> Dict[str, Any]:
    pred_value = board[cell]
    previous_pred: Optional[int] = None
    previous_loop: Optional[int] = None
    for candidate_loop in reversed([value for value in loop_values if value < loop]):
        previous_pred = int(preds_cpu[candidate_loop - 1][idx].view(-1)[cell].item())
        previous_loop = candidate_loop
        break

    stable_since = loop
    for candidate_loop in reversed([value for value in loop_values if value < loop]):
        candidate_pred = int(preds_cpu[candidate_loop - 1][idx].view(-1)[cell].item())
        if candidate_pred != pred_value:
            break
        stable_since = candidate_loop

    row = {
        "coord": coord(cell),
        "pred": digit(pred_value),
        "truth": digit(solution[cell]),
        "margin": float(margins[loop - 1][idx].view(-1)[cell].item()),
        "entropy": float(entropies[loop - 1][idx].view(-1)[cell].item()),
        "stable_since_loop": stable_since,
        "changed_from_previous_loop": previous_pred is not None and previous_pred != pred_value,
    }
    if previous_pred is not None and previous_loop is not None:
        row["previous_loop"] = previous_loop
        row["previous_pred"] = digit(previous_pred)
    return row


def write_case_bank_index(path: Path, *, holes: int, selected: Dict[str, List[Dict[str, Any]]], summary: Dict[str, Any]) -> None:
    cards = []
    for kind, cases in selected.items():
        rows = []
        for case in cases:
            final_loop = f"loop{summary['final_loop']}"
            row = case["loops"][final_loop]
            rows.append(
                "<tr>"
                f"<td><a href=\"{html.escape(Path(case['html_path']).name)}\">{html.escape(case['stem'])}</a></td>"
                f"<td>{case['batch_index']}</td>"
                f"<td>{row['wrong_count']}</td>"
                f"<td>{row['blank_acc']:.3f}</td>"
                f"<td>{row['hidden_entropy_mean']:.3f}</td>"
                f"<td>{row['conflict_unit_count']}</td>"
                "</tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="6">no cases found in this eval sample</td></tr>')
        cards.append(
            f"""
<section class="panel">
  <h2>{html.escape(kind.replace("_", " "))}</h2>
  <table><thead><tr><th>case</th><th>batch</th><th>final wrong</th><th>blank acc</th><th>entropy</th><th>conflict units</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</section>
"""
        )

    body = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{N}x{N} h{holes} case bank</title>
<style>
body {{ margin: 24px; background: #f6f8fa; color: #24292f; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }}
h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
p {{ margin: 0 0 14px; color: #57606a; line-height: 1.45; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
.panel {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #d0d7de; padding: 8px 7px; text-align: left; }}
th {{ color: #57606a; background: #f0f3f6; }}
tr:last-child td {{ border-bottom: 0; }}
a {{ color: #1f6feb; }}
</style>
</head>
<body>
<h1>{N}x{N} h{holes} FutureSeed loop case bank</h1>
<p>Eval sample: <code>{summary['eval_n']}</code>. Final loop exact: <code>{summary['final_exact']:.4f}</code>. Final loop blank accuracy: <code>{summary['final_blank_acc']:.4f}</code>. This artifact is diagnostic only; it does not change training.</p>
<div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


@torch.no_grad()
def export_case_bank(
    model: "FutureSeedLoopSudoku",
    out_dir: Path,
    *,
    holes_values: List[int],
    eval_n: int,
    cases_per_kind: int,
    loop_values: List[int],
    seed: int,
    forward_dtype: str,
) -> Dict[str, Any]:
    if cases_per_kind <= 0 or not holes_values:
        return {}
    bank_root = out_dir / "case_bank"
    bank_root.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    feature_buffer = getattr(model, "feature_noise_buffer", None)
    artifacts: Dict[str, Any] = {"case_bank_root": str(bank_root.resolve()), "holes": {}}
    for holes in holes_values:
        batch = make_batch(eval_n, holes, holes, "random", random.Random(seed + 91000 + holes * 97), device=device)
        inputs, labels, clue_mask = batch
        max_loop = max(loop_values)
        with forward_autocast(forward_dtype, device):
            loop_logits, _fs_trace = model.forward_trace(
                inputs,
                loops=max_loop,
                noise_scale=0.0,
                feature_buffer=feature_buffer,
                update_feature_buffer=False,
            )
        loop_preds = [logits.argmax(dim=-1) for logits in loop_logits]
        entropies = []
        margins = []
        for logits in loop_logits:
            probs = logits.float().softmax(dim=-1)
            entropies.append((-(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).detach().cpu())
            top2 = probs.topk(k=2, dim=-1).values
            margins.append((top2[..., 0] - top2[..., 1]).detach().cpu())

        holes_dir = bank_root / f"h{holes}"
        holes_dir.mkdir(parents=True, exist_ok=True)
        clue_cpu = clue_mask.detach().cpu()
        labels_cpu = labels.detach().cpu()
        inputs_cpu = inputs.detach().cpu()
        preds_cpu = [pred.detach().cpu() for pred in loop_preds]
        selected: Dict[str, List[Dict[str, Any]]] = {"solved_by_loop": [], "almost_solved": [], "hard_failure": []}
        candidates: Dict[str, List[Tuple[Tuple[float, ...], Dict[str, Any]]]] = {key: [] for key in selected}

        for idx in range(eval_n):
            clue = [bool(x) for x in clue_cpu[idx].tolist()]
            blank_indices = [cell for cell, is_clue in enumerate(clue) if not is_clue]
            solution = [int(x) for x in labels_cpu[idx].view(-1).tolist()]
            puzzle = [int(x) for x in inputs_cpu[idx].view(-1).tolist()]
            loops: Dict[str, Any] = {}
            previous: Optional[List[int]] = None
            for loop in loop_values:
                board = [int(x) for x in preds_cpu[loop - 1][idx].view(-1).tolist()]
                wrong = [cell for cell in blank_indices if board[cell] != solution[cell]]
                conflict_cells, conflict_units = conflict_info(board)
                hidden_entropy = float(entropies[loop - 1][idx][blank_indices].mean().item()) if blank_indices else 0.0
                changed = (
                    sum(1 for cell in blank_indices if previous is not None and previous[cell] != board[cell])
                    if previous is not None
                    else 0
                )
                loops[f"loop{loop}"] = {
                    "wrong_count": len(wrong),
                    "blank_acc": 1.0 - (len(wrong) / max(len(blank_indices), 1)),
                    "exact": len(wrong) == 0 and all(board[cell] == solution[cell] for cell in range(CELLS)),
                    "valid_board": valid_board(torch.tensor(board)),
                    "changed_from_previous": changed,
                    "hidden_entropy_mean": hidden_entropy,
                    "conflict_unit_count": len(conflict_units),
                    "conflict_units": conflict_units[:32],
                    "wrong_blanks": [
                        case_cell_diagnostic(
                            cell=cell,
                            loop=loop,
                            board=board,
                            solution=solution,
                            preds_cpu=preds_cpu,
                            margins=margins,
                            entropies=entropies,
                            idx=idx,
                            loop_values=loop_values,
                        )
                        for cell in wrong[:64]
                    ],
                    "_board": board,
                    "_conflict_cells": sorted(conflict_cells),
                }
                previous = board
            first_key = f"loop{loop_values[0]}"
            final_key = f"loop{loop_values[-1]}"
            first_wrong = loops[first_key]["wrong_count"]
            final_wrong = loops[final_key]["wrong_count"]
            final_blank_acc = loops[final_key]["blank_acc"]
            final_conflicts = loops[final_key]["conflict_unit_count"]
            case = {
                "holes": holes,
                "batch_index": idx,
                "loops": loops,
                "_puzzle": puzzle,
                "_solution": solution,
                "_clue": clue,
            }
            if first_wrong > 0 and final_wrong == 0:
                kind = "solved_by_loop"
                score = (-float(first_wrong), float(loops[final_key]["hidden_entropy_mean"]), float(idx))
            elif 1 <= final_wrong <= 4:
                kind = "almost_solved"
                score = (float(final_wrong), float(final_conflicts), -float(final_blank_acc), float(idx))
            elif final_wrong > 4:
                kind = "hard_failure"
                score = (-float(final_blank_acc), float(final_wrong), float(final_conflicts), float(idx))
            else:
                continue
            case["kind"] = kind
            candidates[kind].append((score, case))

        used: set[int] = set()
        for kind, rows in candidates.items():
            rows.sort(key=lambda item: item[0])
            for _score, case in rows:
                if case["batch_index"] in used:
                    continue
                used.add(case["batch_index"])
                stem = f"h{holes}_{kind}_{len(selected[kind]) + 1:02d}_b{case['batch_index']:04d}"
                case["stem"] = stem
                case["title"] = f"{N}x{N} h{holes} {kind} #{len(selected[kind]) + 1}"
                html_path = holes_dir / f"{stem}.html"
                write_case_bank_case_html(html_path, case, loop_values)
                case["html_path"] = str(html_path.resolve())
                selected[kind].append(case)
                if len(selected[kind]) >= cases_per_kind:
                    break

        final_logits = loop_logits[loop_values[-1] - 1]
        final_metrics, _final_preds = metrics_from_logits(final_logits, labels, clue_mask)
        holes_summary = {
            "holes": holes,
            "eval_n": eval_n,
            "loop_values": loop_values,
            "final_loop": loop_values[-1],
            "final_exact": final_metrics.label_exact,
            "final_blank_acc": final_metrics.blank_acc,
            "selected_counts": {kind: len(cases) for kind, cases in selected.items()},
        }
        json_path = holes_dir / "cases.json"
        index_path = holes_dir / "index.html"
        json_path.write_text(
            json.dumps(
                {
                    "summary": holes_summary,
                    "selected": {kind: [json_ready_case(case) for case in cases] for kind, cases in selected.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_case_bank_index(index_path, holes=holes, selected=selected, summary=holes_summary)
        artifacts["holes"][f"h{holes}"] = {
            "index_html": str(index_path.resolve()),
            "cases_json": str(json_path.resolve()),
            "summary": holes_summary,
        }
    return artifacts


def parse_eval_holes(args: argparse.Namespace) -> List[int]:
    values = [int(args.eval_holes)]
    if str(args.eval_holes_list).strip():
        for raw in str(args.eval_holes_list).split(","):
            raw = raw.strip()
            if raw:
                values.append(int(raw))
    deduped = []
    for value in values:
        if value < 1 or value > CELLS:
            raise ValueError(f"--eval_holes values must be in [1, {CELLS}]")
        if value not in deduped:
            deduped.append(value)
    return deduped


def parse_eval_checkpoint_holes(args: argparse.Namespace) -> List[int]:
    raw = str(args.eval_checkpoint_holes_list).strip()
    if not raw:
        return parse_eval_holes(args)
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1 or value > CELLS:
            raise ValueError(f"--eval_checkpoint_holes_list values must be in [1, {CELLS}]")
        if value not in values:
            values.append(value)
    return values


def parse_hole_stages(args: argparse.Namespace) -> List[Tuple[int, int, int]]:
    raw = str(args.hole_stages).strip()
    if not raw:
        return [(int(args.holes_min), int(args.holes_max), int(args.steps))]
    stages = []
    for item in raw.split(","):
        span, steps = item.split(":", 1)
        lo, hi = span.split(":", 1) if ":" in span else span.split("-", 1)
        stages.append((int(lo), int(hi), int(steps)))
    for lo, hi, steps in stages:
        if lo < 1 or hi < lo or steps < 1:
            raise ValueError("--hole_stages entries must look like 2-4:100 with 1 <= lo <= hi and steps > 0")
    return stages


def parse_eval_checkpoint_steps(args: argparse.Namespace, stages: List[Tuple[int, int, int]]) -> List[int]:
    total_steps = sum(stage_steps for _lo, _hi, stage_steps in stages)
    steps: List[int] = []
    for value in parse_int_csv(args.eval_checkpoint_steps, name="--eval_checkpoint_steps", low=1, high=total_steps):
        if value not in steps:
            steps.append(value)
    final_stage_start = total_steps - stages[-1][2]
    for offset in parse_int_csv(
        args.eval_checkpoint_stage_offsets,
        name="--eval_checkpoint_stage_offsets",
        low=1,
        high=stages[-1][2],
    ):
        step = final_stage_start + offset
        if step not in steps:
            steps.append(step)
    return sorted(steps)


def write_report(path: Path, metrics: Dict[str, Any], artifacts: Dict[str, str]) -> None:
    task = metrics.get("task", {})
    lines = [
        f"# {N}x{N} RWKV FutureSeed Loop Study",
        "",
        "Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.",
        "",
        f"Board: {N}x{N}, box: {BOX_ROWS}x{BOX_COLS}, hole pattern: `{task.get('hole_pattern', 'random')}`.",
        "",
        "## Loop Metrics",
        "",
    ]
    train = metrics.get("train", {})
    if train:
        lines.append(
            "- train: "
            f"ce={train.get('train_ce_loss', 0.0):.4f}, "
            f"total={train.get('train_total_loss', 0.0):.4f}, "
            f"loop_loss={train.get('loop_loss_mode', 'final')}, "
            f"noise={train.get('noise_mode', 'feature_diff')}, "
            f"buffer={train.get('feature_buffer_count', 0)}, "
            f"dtype={train.get('forward_dtype', 'float32')}, "
            f"fs_update={train.get('future_seed_update', 'fixed')}, "
            f"loop_fb={train.get('loop_feedback_scale', 0.0):.2f}, "
            f"loop_time={train.get('loop_time_scale', 0.0):.2f}, "
            f"scratch={train.get('scratch_mode', 'none')}, "
            f"scratch_delta={train.get('scratch_delta', 0.0):.3f}, "
            f"scratch_gauss={train.get('scratch_gauss_loss', 0.0):.4f}, "
            f"loop1_loss={train.get('train_loop1_loss', 0.0):.4f}, "
            f"loop_last_loss={train.get('train_loop_last_loss', 0.0):.4f}, "
            f"sec={train.get('train_sec', 0.0):.1f}"
        )
    for loop in range(1, int(task.get("max_loops", 3)) + 1):
        loop_key = f"loop{loop}"
        if loop_key not in metrics["eval_clean"]:
            continue
        lines.append(f"- loop {loop}: {metric_line(metrics['eval_clean'][loop_key])}")
        fs = metrics["eval_clean"].get(f"{loop_key}/future_seed")
        if fs:
            lines.append(f"  future_seed: {fs_line(fs)}")
    if "eval_noisy" in metrics:
        lines.extend(["", "## Eval Noise", ""])
        max_loop = task.get("max_loops", 3)
        last_loop_key = f"loop{max_loop}"
        lines.append(f"- clean eval loop{max_loop}: {metric_line(metrics['eval_clean'][last_loop_key])}")
        lines.append(f"- noisy eval loop{max_loop}: {metric_line(metrics['eval_noisy'][last_loop_key])}")
    if metrics.get("rollouts"):
        lines.extend(["", "## Stochastic Rollouts", ""])
        for key, row in metrics["rollouts"].items():
            lines.append(
                f"- {key}: oracle_exact={row['oracle_label_exact']:.4f}, "
                f"oracle_solved={row['oracle_solved_valid_clue']:.4f}, "
                f"disagree={row['trajectory_token_disagreement']:.4f}"
            )
            lines.append(
                f"  confidence: {metric_line(row['selector_confidence'])}; "
                f"gap={row['selector_gap_confidence']:.4f}"
            )
            lines.append(
                f"  consistency: {metric_line(row['selector_consistency'])}; "
                f"gap={row['selector_gap_consistency']:.4f}"
            )
            lines.append(
                f"  residual: {metric_line(row['selector_residual'])}; "
                f"gap={row['selector_gap_residual']:.4f}"
            )
            lines.append(
                f"  majority: {metric_line(row['selector_majority'])}; "
                f"gap={row['selector_gap_majority']:.4f}"
            )
    if metrics.get("rollouts_by_loop"):
        lines.extend(["", "## Compute Scaling", ""])
        for loop_key, by_k in metrics["rollouts_by_loop"].items():
            lines.append(f"### {loop_key}")
            for key, row in by_k.items():
                lines.append(
                    f"- {key}: oracle_exact={row['oracle_label_exact']:.4f}, "
                    f"confidence={row['selector_confidence']['label_exact']:.4f}, "
                    f"residual={row['selector_residual']['label_exact']:.4f}, "
                    f"disagree={row['trajectory_token_disagreement']:.4f}"
                )
    if metrics.get("case_bank"):
        lines.extend(["", "## Case Bank", ""])
        case_bank = metrics["case_bank"]
        for hole_key, row in case_bank.get("holes", {}).items():
            summary = row.get("summary", {})
            lines.append(
                f"- {hole_key}: index={row.get('index_html', '')}; "
                f"cases={row.get('cases_json', '')}; "
                f"final_loop={summary.get('final_loop')}, "
                f"exact={summary.get('final_exact', 0.0):.4f}, "
                f"blank_acc={summary.get('final_blank_acc', 0.0):.4f}, "
                f"selected={summary.get('selected_counts', {})}"
            )
    if len(metrics.get("eval_by_holes", {})) > 1:
        lines.extend(["", "## Hole Transfer", ""])
        for hole_key, hole_metrics in metrics["eval_by_holes"].items():
            holes = int(hole_key.removeprefix("holes"))
            loop3 = hole_metrics["eval_clean"][f"loop{task.get('max_loops', 3)}"]
            lines.append(f"- {hole_key}: {metric_line(loop3)}; {coupling_line(loop3, holes)}")
    lines.extend(["", "## Decision", "", metrics["decision"], "", "## Artifacts", ""])
    for name, value in artifacts.items():
        lines.append(f"- {name}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.d_model != args.heads * args.head_dim:
        raise ValueError("--d_model must equal --heads * --head_dim")
    if args.future_seed_scale < 0:
        raise ValueError("--future_seed_scale must be non-negative")
    if not (0.0 <= args.future_seed_decay < 1.0):
        raise ValueError("--future_seed_decay must be in [0, 1)")
    if args.loop_feedback_scale < 0:
        raise ValueError("--loop_feedback_scale must be non-negative")
    if args.loop_time_scale < 0:
        raise ValueError("--loop_time_scale must be non-negative")
    if args.scratch_scale < 0:
        raise ValueError("--scratch_scale must be non-negative")
    if args.scratch_noise_scale < 0:
        raise ValueError("--scratch_noise_scale must be non-negative")
    if args.scratch_gauss_weight < 0:
        raise ValueError("--scratch_gauss_weight must be non-negative")
    if args.scratch_gauss_projections < 0:
        raise ValueError("--scratch_gauss_projections must be non-negative")
    if args.scratch_gauss_weight > 0 and args.scratch_mode == "none":
        raise ValueError("--scratch_gauss_weight requires --scratch_mode gated")
    if args.scratch_gauss_weight > 0 and args.scratch_gauss_projections == 0:
        raise ValueError("--scratch_gauss_weight requires positive --scratch_gauss_projections")
    if args.forward_dtype not in {"float32", "bfloat16"}:
        raise ValueError("--forward_dtype must be 'float32' or 'bfloat16'")
    configure_sudoku(args.size, args.box_rows, args.box_cols)
    if args.layers < 2:
        raise ValueError("--layers must be at least 2 for FutureSeed.")
    if args.rwkv_kernel in {"cuda", "statepassing"}:
        ok, reason = statepassing_available(args.head_dim)
        if not ok:
            raise ValueError(f"--rwkv_kernel {args.rwkv_kernel} is unavailable: {reason}")
    if args.rwkv_kernel == "wind":
        ok, reason = wind_available(args.head_dim)
        if not ok:
            raise ValueError(f"--rwkv_kernel wind is unavailable: {reason}")
    if args.hole_pattern != "random":
        raise ValueError("--hole_pattern now supports only 'random'; structured hole probes were removed from the clean mainline.")

    device = choose_device(args.cpu)
    if args.forward_dtype == "bfloat16" and device.type != "cuda":
        raise ValueError("--forward_dtype=bfloat16 requires CUDA; CPU/MPS fallback is disabled for this run")
    print(
        f"device={device} torch={torch.__version__} board={N}x{N} box={BOX_ROWS}x{BOX_COLS} "
        f"mainline=future_seed_loop rwkv_kernel={args.rwkv_kernel} forward_dtype={args.forward_dtype}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, train_stats = train_model(args, device=device)
    checkpoint_evals = train_stats.pop("checkpoint_evals", {})
    batch = make_batch(
        args.eval_n,
        args.eval_holes,
        args.eval_holes,
        args.hole_pattern,
        random.Random(args.seed + 999),
        device=device,
    )
    inputs, labels, clue_mask = batch
    clean, loop_preds = evaluate_model(
        model,
        batch,
        max_loops=args.max_loops,
        noise_scale=0.0,
        seed=args.seed + 4000,
        forward_dtype=args.forward_dtype,
    )
    metrics: Dict[str, Any] = {
        "train": train_stats,
        "eval_clean": clean,
        "eval_by_holes": {},
        "checkpoint_evals": checkpoint_evals,
    }
    if args.noise_scale > 0:
        noisy, _noisy_preds = evaluate_model(
            model,
            batch,
            max_loops=args.max_loops,
            noise_scale=args.noise_scale,
            seed=args.seed + 5000,
            forward_dtype=args.forward_dtype,
        )
        metrics["eval_noisy"] = noisy

    rollout_ks = parse_rollout_ks(args)
    if rollout_ks:
        rollout_noise = args.noise_scale if args.rollout_noise_scale < 0 else args.rollout_noise_scale
        metrics["rollouts"] = evaluate_rollouts(
            model,
            batch,
            max_loops=args.max_loops,
            noise_scale=rollout_noise,
            ks=rollout_ks,
            seed=args.seed + 6000,
            forward_dtype=args.forward_dtype,
        )
        rollout_loop_values = parse_rollout_loop_values(args)
        if rollout_loop_values:
            metrics["rollouts_by_loop"] = evaluate_rollouts_by_loop(
                model,
                batch,
                loop_values=rollout_loop_values,
                noise_scale=rollout_noise,
                ks=rollout_ks,
                seed=args.seed + 6500,
                forward_dtype=args.forward_dtype,
            )

    eval_holes_values = parse_eval_holes(args)
    for holes in eval_holes_values:
        if holes == args.eval_holes:
            metrics["eval_by_holes"][f"holes{holes}"] = {"eval_clean": clean}
            continue
        extra_batch = make_batch(
            args.eval_n,
            holes,
            holes,
            args.hole_pattern,
            random.Random(args.seed + 999 + holes * 17),
            device=device,
        )
        hole_clean, _hole_preds = evaluate_model(
            model,
            extra_batch,
            max_loops=args.max_loops,
            noise_scale=0.0,
            seed=args.seed + 4000 + holes,
            forward_dtype=args.forward_dtype,
        )
        metrics["eval_by_holes"][f"holes{holes}"] = {"eval_clean": hole_clean}

    l1 = metrics["eval_clean"]["loop1"]["label_exact"]
    last_key = f"loop{args.max_loops}"
    l_last = metrics["eval_clean"][last_key]["label_exact"]
    if l_last > l1:
        decision = "Depth loop adds useful full-board refinement."
    elif l_last == l1:
        decision = "Depth loop is neutral on full-board exact in this run."
    else:
        decision = "Depth loop is not yet stable; later loops reduce full-board exact."
    metrics["decision"] = decision
    metrics["task"] = {
        "size": N,
        "box_rows": BOX_ROWS,
        "box_cols": BOX_COLS,
        "hole_pattern": args.hole_pattern,
        "hole_stages": parse_hole_stages(args),
        "max_loops": args.max_loops,
        "loop_loss": args.loop_loss,
        "loop_loss_start": args.loop_loss_start,
        "loop_loss_power": args.loop_loss_power,
        "loop_loss_min_weight": args.loop_loss_min_weight,
        "noise_mode": "feature_diff",
        "feature_buffer_size": args.feature_buffer_size,
        "future_seed_scale": args.future_seed_scale,
        "future_seed_decay": args.future_seed_decay,
        "future_seed_update": args.future_seed_update,
        "loop_feedback_scale": args.loop_feedback_scale,
        "loop_time_scale": args.loop_time_scale,
        "scratch_mode": args.scratch_mode,
        "scratch_scale": args.scratch_scale,
        "scratch_noise_scale": args.scratch_noise_scale,
        "scratch_gauss_weight": args.scratch_gauss_weight,
        "scratch_gauss_projections": args.scratch_gauss_projections,
        "scratch_gate_bias": args.scratch_gate_bias,
        "scratch_decay_bias": args.scratch_decay_bias,
        "rwkv_kernel": args.rwkv_kernel,
        "forward_dtype": args.forward_dtype,
        "rollout_ks": rollout_ks,
        "rollout_loop_values": parse_rollout_loop_values(args),
        "rollout_noise_scale": args.noise_scale if args.rollout_noise_scale < 0 else args.rollout_noise_scale,
        "case_bank_holes": args.case_bank_holes,
        "case_bank_n": args.case_bank_n,
        "case_bank_eval_n": args.case_bank_eval_n,
        "case_bank_loop_values": args.case_bank_loop_values,
        "eval_checkpoint_steps": train_stats.get("eval_checkpoint_steps", []),
        "eval_checkpoint_holes": train_stats.get("eval_checkpoint_holes", []),
        "mainline": "future_seed_loop",
    }

    case_index = min(max(int(args.case_index), 0), inputs.shape[0] - 1)
    html_path = out_dir / f"futureseed_loop_case_seed{args.seed}.html"
    write_case_html(
        html_path,
        title=f"{N}x{N} FutureSeed loop case",
        inputs=inputs[case_index],
        labels=labels[case_index],
        clue_mask=clue_mask[case_index],
        loop_preds=[pred[case_index] for pred in loop_preds],
    )
    artifacts = {"case_html": str(html_path.resolve())}
    case_bank_holes = parse_int_csv(args.case_bank_holes, name="--case_bank_holes", low=1, high=CELLS)
    case_bank_loop_values = parse_int_csv(
        args.case_bank_loop_values,
        name="--case_bank_loop_values",
        low=1,
        high=args.max_loops,
    )
    if not case_bank_loop_values:
        case_bank_loop_values = sorted({1, min(2, args.max_loops), min(3, args.max_loops), args.max_loops})
    elif args.max_loops not in case_bank_loop_values:
        case_bank_loop_values = sorted(set(case_bank_loop_values + [args.max_loops]))
    if args.case_bank_n > 0 and case_bank_holes:
        case_bank = export_case_bank(
            model,
            out_dir,
            holes_values=case_bank_holes,
            eval_n=args.case_bank_eval_n,
            cases_per_kind=args.case_bank_n,
            loop_values=case_bank_loop_values,
            seed=args.seed,
            forward_dtype=args.forward_dtype,
        )
        if case_bank:
            metrics["case_bank"] = case_bank
            artifacts["case_bank_root"] = case_bank["case_bank_root"]
    metrics["artifacts"] = artifacts

    json_path = out_dir / f"futureseed_loop_seed{args.seed}.json"
    md_path = out_dir / f"futureseed_loop_seed{args.seed}.md"
    json_path.write_text(json.dumps({"args": vars(args), "metrics": metrics}, indent=2), encoding="utf-8")
    write_report(md_path, metrics, artifacts)

    print("\nfull-board metrics")
    for loop in range(1, args.max_loops + 1):
        loop_key = f"loop{loop}"
        print(f"loop={loop} {metric_line(metrics['eval_clean'][loop_key])}")
        print(f"        {fs_line(metrics['eval_clean'][f'{loop_key}/future_seed'])}")
    if "eval_noisy" in metrics:
        print(f"eval_noise loop={args.max_loops} {metric_line(metrics['eval_noisy'][last_key])}")
    if metrics.get("rollouts"):
        print("\nstochastic rollout metrics")
        for key, row in metrics["rollouts"].items():
            print(
                f"{key} oracle_exact={row['oracle_label_exact']:.4f} "
                f"oracle_solved={row['oracle_solved_valid_clue']:.4f} "
                f"disagree={row['trajectory_token_disagreement']:.4f}"
            )
            print(f"  confidence {metric_line(row['selector_confidence'])}; gap={row['selector_gap_confidence']:.4f}")
            print(f"  consistency {metric_line(row['selector_consistency'])}; gap={row['selector_gap_consistency']:.4f}")
            print(f"  residual    {metric_line(row['selector_residual'])}; gap={row['selector_gap_residual']:.4f}")
            print(f"  majority    {metric_line(row['selector_majority'])}; gap={row['selector_gap_majority']:.4f}")
    if metrics.get("rollouts_by_loop"):
        print("\ncompute-scaling rollout metrics")
        for loop_key, by_k in metrics["rollouts_by_loop"].items():
            print(loop_key)
            for key, row in by_k.items():
                print(
                    f"  {key} oracle_exact={row['oracle_label_exact']:.4f} "
                    f"confidence_exact={row['selector_confidence']['label_exact']:.4f} "
                    f"residual_exact={row['selector_residual']['label_exact']:.4f} "
                    f"disagree={row['trajectory_token_disagreement']:.4f}"
                )
    if len(eval_holes_values) > 1:
        print("\nhole-transfer metrics")
        for holes in eval_holes_values:
            loop_last = metrics["eval_by_holes"][f"holes{holes}"]["eval_clean"][last_key]
            print(f"holes={holes:<2d} {metric_line(loop_last)}; {coupling_line(loop_last, holes)}")
    if metrics.get("case_bank"):
        print("\ncase-bank artifacts")
        for hole_key, row in metrics["case_bank"]["holes"].items():
            summary = row["summary"]
            print(
                f"{hole_key} exact={summary['final_exact']:.4f} "
                f"blank_acc={summary['final_blank_acc']:.4f} selected={summary['selected_counts']}"
            )
            print(f"  index={row['index_html']}")
            print(f"  cases={row['cases_json']}")
    print(metrics["decision"])
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--box_rows", type=int, default=0)
    p.add_argument("--box_cols", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d_model", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=12)
    p.add_argument("--channel_mult", type=int, default=2)
    p.add_argument("--l_cycles", type=int, default=2)
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--future_seed_scale", type=float, default=1.0)
    p.add_argument("--future_seed_decay", type=float, default=0.0)
    p.add_argument("--future_seed_update", choices=("fixed", "learned", "loop_residual"), default="fixed")
    p.add_argument("--loop_feedback_scale", type=float, default=0.0)
    p.add_argument("--loop_time_scale", type=float, default=0.0)
    p.add_argument("--scratch_mode", choices=("none", "gated"), default="none")
    p.add_argument("--scratch_scale", type=float, default=1.0)
    p.add_argument("--scratch_noise_scale", type=float, default=0.0)
    p.add_argument("--scratch_gauss_weight", type=float, default=0.0)
    p.add_argument("--scratch_gauss_projections", type=int, default=0)
    p.add_argument("--scratch_gate_bias", type=float, default=-2.0)
    p.add_argument("--scratch_decay_bias", type=float, default=2.0)
    p.add_argument("--activation_checkpoint", action="store_true")
    p.add_argument("--resume_train_checkpoint", default="")
    p.add_argument("--train_checkpoint_dir", default="")
    p.add_argument("--save_train_checkpoint_every", type=int, default=0)
    p.add_argument("--forward_dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--rwkv_kernel", choices=("auto", "torch", "cuda", "statepassing", "wind"), default="auto")
    p.add_argument("--loop_loss", choices=("final", "all", "shaped", "delayed"), default="final")
    p.add_argument("--loop_loss_start", type=int, default=1)
    p.add_argument("--loop_loss_power", type=float, default=2.0)
    p.add_argument("--loop_loss_min_weight", type=float, default=0.05)
    p.add_argument("--noise_scale", type=float, default=0.0)
    p.add_argument("--feature_buffer_size", type=int, default=8192)
    p.add_argument("--feature_buffer_add", type=int, default=2048)
    p.add_argument("--rollout_ks", default="")
    p.add_argument("--rollout_loop_values", default="")
    p.add_argument("--rollout_noise_scale", type=float, default=-1.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--holes_min", type=int, default=4)
    p.add_argument("--holes_max", type=int, default=12)
    p.add_argument("--hole_stages", default="")
    p.add_argument("--hole_pattern", choices=("random",), default="random")
    p.add_argument("--eval_holes", type=int, default=8)
    p.add_argument("--eval_holes_list", default="")
    p.add_argument("--eval_n", type=int, default=128)
    p.add_argument("--eval_checkpoint_steps", default="")
    p.add_argument("--eval_checkpoint_stage_offsets", default="")
    p.add_argument("--eval_checkpoint_holes_list", default="")
    p.add_argument("--blank_loss_weight", type=float, default=8.0)
    p.add_argument("--case_index", type=int, default=0)
    p.add_argument("--case_bank_holes", default="")
    p.add_argument("--case_bank_n", type=int, default=0)
    p.add_argument("--case_bank_eval_n", type=int, default=256)
    p.add_argument("--case_bank_loop_values", default="")
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", default="runs/mainline")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
