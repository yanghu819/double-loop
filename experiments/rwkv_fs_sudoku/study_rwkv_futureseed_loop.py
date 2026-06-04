#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    pools: Dict[str, List[List[int]]] = {
        "row": ROWS,
        "col": COLS,
        "box": BOXES,
        "unit": UNITS,
    }
    if hole_pattern not in pools:
        raise ValueError(f"Unknown --hole_pattern {hole_pattern!r}")
    candidates = rng.choice(pools[hole_pattern])
    if holes > len(candidates):
        raise ValueError(f"--hole_pattern {hole_pattern} supports at most {len(candidates)} holes for {N}x{N}")
    return rng.sample(candidates, holes)


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
        rwkv_kernel: str = "auto",
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("FutureSeed needs at least two layers.")
        self.future_seed_scale = float(future_seed_scale)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        previous_state: Optional[torch.Tensor] = None
        gates = []
        state_norms = []
        for layer_idx, block in enumerate(self.blocks):
            initial_state = None
            if layer_idx > 0:
                assert previous_state is not None
                gate = torch.sigmoid(block.future_seed_logit) * self.future_seed_scale
                gates.append(gate.mean())
                if self.future_seed_scale > 0:
                    denom = previous_state.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp(min=1e-6)
                    normalized_state = previous_state / denom
                    initial_state = normalized_state * gate
                    state_norms.append(initial_state.norm(dim=(-1, -2)).mean())
                else:
                    state_norms.append(x.new_zeros(()))
            x, previous_state = block(x, initial_state=initial_state)

        if gates:
            return x, {
                "fs_gate_mean": torch.stack(gates).mean(),
                "fs_state_norm": torch.stack(state_norms).mean(),
            }
        zero = x.new_zeros(())
        return x, {"fs_gate_mean": zero, "fs_state_norm": zero}


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
        rwkv_kernel: str,
    ) -> None:
        super().__init__()
        self.l_cycles = int(l_cycles)
        self.lambda_ = float(lambda_)
        self.embed = nn.Embedding(VOCAB, d_model)
        self.position = nn.Embedding(CELLS, d_model)
        self.reasoner = FutureSeedRWKV(
            d_model,
            layers,
            heads,
            head_dim,
            channel_mult,
            future_seed_scale=future_seed_scale,
            rwkv_kernel=rwkv_kernel,
        )
        self.h_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.l_init = nn.Parameter(torch.zeros(1, 1, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N, bias=False)

    def input_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(CELLS, dtype=torch.long, device=inputs.device)
        return self.embed(inputs) + self.position(positions).unsqueeze(0)

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        updated, diag = self.reasoner(hidden + injection)
        out = (1.0 - self.lambda_) * hidden + self.lambda_ * updated
        if allow_noise and noise_scale > 0:
            if feature_buffer is None:
                raise ValueError("feature-difference noise requires a FeatureNoiseBuffer")
            out = out + feature_buffer.sample_diff_like(out) * float(noise_scale)
        if allow_noise and update_feature_buffer and feature_buffer is not None:
            feature_buffer.add(out, max_items=feature_buffer_add)
        return out, diag

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

        loop_logits: List[torch.Tensor] = []
        fs_trace: List[Dict[str, torch.Tensor]] = []
        for _ in range(int(loops)):
            for _ in range(self.l_cycles):
                z_l, _diag = self.depth_update(
                    z_l,
                    z_h + x,
                    noise_scale,
                    feature_buffer=feature_buffer,
                    update_feature_buffer=update_feature_buffer,
                    feature_buffer_add=feature_buffer_add,
                    allow_noise=True,
                )
            z_h, fs_diag = self.depth_update(
                z_h,
                z_l,
                noise_scale,
                feature_buffer=feature_buffer,
                update_feature_buffer=update_feature_buffer,
                feature_buffer_add=feature_buffer_add,
                allow_noise=True,
            )
            board_h = self.out_norm(z_h[:, :CELLS])
            loop_logits.append(self.head(board_h))
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
    loss = F.cross_entropy(logits.reshape(-1, N), labels.reshape(-1), reduction="none").view_as(labels)
    if blank_weight == 1.0:
        return loss.mean()
    weights = torch.where(clue_mask, torch.ones_like(loss), torch.full_like(loss, float(blank_weight)))
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


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
    return f"fs_gate={m['fs_gate_mean']:.3f}, fs_state_norm={m['fs_state_norm']:.3f}"


def train_model(args: argparse.Namespace, *, device: torch.device) -> Tuple[FutureSeedLoopSudoku, Dict[str, float]]:
    torch.manual_seed(args.seed)
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
    last_ce_loss = 0.0
    last_total_loss = 0.0
    last_loop1_loss = 0.0
    last_loop_last_loss = 0.0

    global_step = 0
    for stage_idx, (holes_min, holes_max, stage_steps) in enumerate(parse_hole_stages(args), start=1):
        for _ in range(stage_steps):
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
            if args.loop_loss == "all":
                loss = torch.stack(loop_losses).mean()
            else:
                loss = ce_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_ce_loss = float(ce_loss.detach().cpu())
            last_total_loss = float(loss.detach().cpu())
            last_loop1_loss = float(loop_losses[0].detach().cpu())
            last_loop_last_loss = float(loop_losses[-1].detach().cpu())
            if args.log_every and global_step % args.log_every == 0:
                print(
                    f"[future_seed_loop stage={stage_idx}:{holes_min}-{holes_max}] "
                    f"step={global_step:04d} ce={last_ce_loss:.4f} total={last_total_loss:.4f} "
                    f"loop1={last_loop1_loss:.4f} loop_last={last_loop_last_loss:.4f}",
                    flush=True,
                )

    train_stats = {
        "train_ce_loss": last_ce_loss,
        "train_total_loss": last_total_loss,
        "train_loop1_loss": last_loop1_loss,
        "train_loop_last_loss": last_loop_last_loss,
        "loop_loss_mode": args.loop_loss,
        "noise_mode": "feature_diff",
        "feature_buffer_count": feature_buffer.count,
        "train_sec": time.time() - t0,
        "rwkv_kernel": args.rwkv_kernel,
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
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    if noise_scale > 0:
        torch.manual_seed(seed)
    inputs, labels, clue_mask = batch
    feature_buffer = getattr(model, "feature_noise_buffer", None)
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


@torch.no_grad()
def evaluate_pattern(
    model: FutureSeedLoopSudoku,
    *,
    eval_n: int,
    holes: int,
    hole_pattern: str,
    max_loops: int,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    batch = make_batch(eval_n, holes, holes, hole_pattern, random.Random(seed), device=device)
    clean, _loop_preds = evaluate_model(
        model,
        batch,
        max_loops=max_loops,
        noise_scale=0.0,
        seed=seed + 3000,
    )
    return {"eval_clean": clean}


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

    probs = last_logits.softmax(dim=-1)
    max_prob = probs.max(dim=-1).values
    confidence_scores = (max_prob * blank_k).sum(dim=-1) / blank_k.sum(dim=-1).clamp_min(1)
    residual = (last_logits.softmax(dim=-1) - prev_logits.softmax(dim=-1)).square().mean(dim=-1)
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
        )
    return out


def board_list(x: torch.Tensor) -> List[int]:
    return [int(v) for v in x.detach().cpu().view(-1).tolist()]


def digit(v: int) -> str:
    return "." if int(v) == BLANK else str(int(v) + 1)


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


def parse_eval_patterns(args: argparse.Namespace) -> List[str]:
    raw = str(args.eval_hole_patterns).strip()
    patterns = [args.hole_pattern] if not raw else [x.strip() for x in raw.split(",") if x.strip()]
    allowed = {"random", "row", "col", "box", "unit"}
    for pattern_name in patterns:
        if pattern_name not in allowed:
            raise ValueError(f"--eval_hole_patterns contains unsupported pattern {pattern_name!r}")
    deduped = []
    for pattern_name in patterns:
        if pattern_name not in deduped:
            deduped.append(pattern_name)
    return deduped


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
    if len(metrics.get("eval_by_holes", {})) > 1:
        lines.extend(["", "## Hole Transfer", ""])
        for hole_key, hole_metrics in metrics["eval_by_holes"].items():
            holes = int(hole_key.removeprefix("holes"))
            loop3 = hole_metrics["eval_clean"][f"loop{task.get('max_loops', 3)}"]
            lines.append(f"- {hole_key}: {metric_line(loop3)}; {coupling_line(loop3, holes)}")
    if len(metrics.get("eval_by_pattern", {})) > 1:
        lines.extend(["", "## Pattern Transfer", ""])
        for pattern_name, by_holes in metrics["eval_by_pattern"].items():
            lines.append(f"### {pattern_name}")
            for hole_key, hole_metrics in by_holes.items():
                holes = int(hole_key.removeprefix("holes"))
                loop3 = hole_metrics["eval_clean"][f"loop{task.get('max_loops', 3)}"]
                lines.append(f"- {hole_key}: {metric_line(loop3)}; {coupling_line(loop3, holes)}")
            lines.append("")
    lines.extend(["", "## Decision", "", metrics["decision"], "", "## Artifacts", ""])
    for name, value in artifacts.items():
        lines.append(f"- {name}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.d_model != args.heads * args.head_dim:
        raise ValueError("--d_model must equal --heads * --head_dim")
    if args.future_seed_scale < 0:
        raise ValueError("--future_seed_scale must be non-negative")
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
    max_eval_holes = max(parse_eval_holes(args) + [hi for _lo, hi, _steps in parse_hole_stages(args)])
    if args.hole_pattern != "random" and max_eval_holes > N:
        raise ValueError(f"--hole_pattern {args.hole_pattern} supports at most {N} holes for {N}x{N}")
    eval_patterns = parse_eval_patterns(args)
    if any(pattern_name != "random" for pattern_name in eval_patterns) and max_eval_holes > N:
        raise ValueError(f"linked eval patterns support at most {N} holes for {N}x{N}")

    device = choose_device(args.cpu)
    print(
        f"device={device} torch={torch.__version__} board={N}x{N} box={BOX_ROWS}x{BOX_COLS} "
        f"mainline=future_seed_loop rwkv_kernel={args.rwkv_kernel}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, train_stats = train_model(args, device=device)
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
    )
    metrics: Dict[str, Any] = {
        "train": train_stats,
        "eval_clean": clean,
        "eval_by_holes": {},
        "eval_by_pattern": {},
    }
    if args.noise_scale > 0:
        noisy, _noisy_preds = evaluate_model(
            model,
            batch,
            max_loops=args.max_loops,
            noise_scale=args.noise_scale,
            seed=args.seed + 5000,
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
        )
        metrics["eval_by_holes"][f"holes{holes}"] = {"eval_clean": hole_clean}

    for pattern_name in eval_patterns:
        by_holes: Dict[str, Any] = {}
        for holes in eval_holes_values:
            if pattern_name == args.hole_pattern and f"holes{holes}" in metrics["eval_by_holes"]:
                by_holes[f"holes{holes}"] = metrics["eval_by_holes"][f"holes{holes}"]
            else:
                by_holes[f"holes{holes}"] = evaluate_pattern(
                    model,
                    eval_n=args.eval_n,
                    holes=holes,
                    hole_pattern=pattern_name,
                    max_loops=args.max_loops,
                    device=device,
                    seed=args.seed + 7000 + holes * 17 + sum(ord(c) for c in pattern_name),
                )
        metrics["eval_by_pattern"][pattern_name] = by_holes

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
        "eval_hole_patterns": eval_patterns,
        "hole_stages": parse_hole_stages(args),
        "max_loops": args.max_loops,
        "loop_loss": args.loop_loss,
        "noise_mode": "feature_diff",
        "feature_buffer_size": args.feature_buffer_size,
        "future_seed_scale": args.future_seed_scale,
        "rwkv_kernel": args.rwkv_kernel,
        "rollout_ks": rollout_ks,
        "rollout_loop_values": parse_rollout_loop_values(args),
        "rollout_noise_scale": args.noise_scale if args.rollout_noise_scale < 0 else args.rollout_noise_scale,
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
    if len(eval_patterns) > 1:
        print("\npattern-transfer metrics")
        for pattern_name in eval_patterns:
            print(f"pattern={pattern_name}")
            for holes in eval_holes_values:
                loop_last = metrics["eval_by_pattern"][pattern_name][f"holes{holes}"]["eval_clean"][last_key]
                print(f"  holes={holes:<2d} {metric_line(loop_last)}; {coupling_line(loop_last, holes)}")
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
    p.add_argument("--rwkv_kernel", choices=("auto", "torch", "cuda", "statepassing", "wind"), default="auto")
    p.add_argument("--loop_loss", choices=("final", "all"), default="final")
    p.add_argument("--noise_scale", type=float, default=0.01)
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
    p.add_argument("--hole_pattern", choices=("random", "row", "col", "box", "unit"), default="random")
    p.add_argument("--eval_hole_patterns", default="")
    p.add_argument("--eval_holes", type=int, default=8)
    p.add_argument("--eval_holes_list", default="")
    p.add_argument("--eval_n", type=int, default=128)
    p.add_argument("--blank_loss_weight", type=float, default=8.0)
    p.add_argument("--case_index", type=int, default=0)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out_dir", default="runs/mainline")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
