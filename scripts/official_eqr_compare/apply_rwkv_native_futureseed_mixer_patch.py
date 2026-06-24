#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


RWKV_NATIVE_FUTURESEED_MIXER = r'''

import torch.nn.functional as F

try:
    from rwkv7_cuda import StatePassingRWKV7, WindRWKV7, statepassing_available, wind_available
except Exception:
    StatePassingRWKV7 = None
    WindRWKV7 = None

    def statepassing_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"

    def wind_available(head_dim: int) -> Tuple[bool, str]:
        return False, "rwkv7_cuda import failed"


class NativeRWKVTimeMix(nn.Module):
    """Forward recurrent RWKV7 token mixer with optional CUDA kernels.

    This module is intentionally causal/left-to-right. FutureSeed is not a
    reverse scan; the FutureSeed variant only uses the previous RWKV layer's
    terminal recurrent state to initialize the next RWKV layer.
    """

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
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.rwkv_kernel = str(rwkv_kernel).lower()
        if self.rwkv_kernel not in {"auto", "statepassing", "wind", "torch"}:
            raise ValueError(f"Unknown rwkv_mixer_kernel '{rwkv_kernel}'")
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
        self.decay_delta = CastedLinear(d_model, d_model, bias=False)
        self.state_lr = CastedLinear(d_model, d_model, bias=True)
        self.key_scale = nn.Parameter((0.71 - linear * 0.1).view(1, 1, d_model))
        self.key_lr_mix = nn.Parameter(torch.full((1, 1, d_model), 1.02))

        self.receptance = CastedLinear(d_model, d_model, bias=False)
        self.key = CastedLinear(d_model, d_model, bias=False)
        self.value = CastedLinear(d_model, d_model, bias=False)
        self.gate = CastedLinear(d_model, d_model, bias=False)
        self.out = CastedLinear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(heads, d_model, eps=64e-5)

        scale = d_model**0.5
        with torch.no_grad():
            self.receptance.weight.uniform_(-0.5 / scale, 0.5 / scale)
            self.key.weight.uniform_(-0.05 / scale, 0.05 / scale)
            self.value.weight.uniform_(-0.5 / scale, 0.5 / scale)
            self.decay_delta.weight.zero_()
            self.state_lr.weight.zero_()
            if self.state_lr.bias is not None:
                self.state_lr.bias.copy_(-0.19 + zigzag * 0.3 + linear * 0.4)
            self.out.weight.zero_()

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

        xr = x + delta * self.mix_r.to(device=x.device, dtype=x.dtype)
        xw = x + delta * self.mix_w.to(device=x.device, dtype=x.dtype)
        xk = x + delta * self.mix_k.to(device=x.device, dtype=x.dtype)
        xv = x + delta * self.mix_v.to(device=x.device, dtype=x.dtype)
        xa = x + delta * self.mix_a.to(device=x.device, dtype=x.dtype)
        xg = x + delta * self.mix_g.to(device=x.device, dtype=x.dtype)

        r = torch.sigmoid(self.receptance(xr)).view(batch_size, seq_len, heads, head_dim)
        w_raw = self.time_decay.to(device=x.device, dtype=x.dtype) + torch.tanh(self.decay_delta(xw))
        w_log = -F.softplus(-w_raw.to(torch.float32)) - 0.5
        state_lr = torch.sigmoid(self.state_lr(xa)).view(batch_size, seq_len, heads, head_dim)
        k_flat = self.key(xk)
        kk = F.normalize((k_flat * self.key_scale.to(device=x.device, dtype=x.dtype)).view(batch_size, seq_len, heads, head_dim), dim=-1, p=2.0)
        key_lr_mix = self.key_lr_mix.to(device=x.device, dtype=x.dtype)
        k = (k_flat * (1.0 + (state_lr.reshape(batch_size, seq_len, channels) - 1.0) * key_lr_mix)).view(
            batch_size, seq_len, heads, head_dim
        )
        v = self.value(xv).view(batch_size, seq_len, heads, head_dim)
        g = torch.sigmoid(self.gate(xg))

        if self.rwkv_kernel in {"auto", "statepassing"}:
            if self._can_use_statepassing(x, initial_state):
                return self._forward_statepassing(
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
            if self.rwkv_kernel == "statepassing":
                ok, reason = statepassing_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV statepassing kernel unavailable: {reason}")
                raise RuntimeError("RWKV statepassing kernel requested but inputs are not CUDA tensors")

        if self.rwkv_kernel in {"auto", "wind"}:
            if self._can_use_wind(x, initial_state):
                return self._forward_wind(
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
            if self.rwkv_kernel == "wind":
                ok, reason = wind_available(self.head_dim)
                if not ok:
                    raise RuntimeError(f"RWKV wind kernel unavailable: {reason}")
                raise RuntimeError("RWKV wind kernel requested but inputs are not CUDA tensors")

        if self.rwkv_kernel == "auto" and not self._fallback_warned:
            sp_ok, sp_reason = statepassing_available(self.head_dim)
            wind_ok, wind_reason = wind_available(self.head_dim)
            print(
                f"NativeRWKVTimeMix auto using torch fallback: statepassing={sp_ok}:{sp_reason}; wind={wind_ok}:{wind_reason}",
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
        return self.out(y * gate).to(original_dtype), terminal_state.to(original_dtype)

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
            expected = (batch_size, heads, head_dim, head_dim)
            if tuple(initial_state.shape) != expected:
                raise ValueError(f"initial_state shape {tuple(initial_state.shape)} does not match {expected}")
            s0 = initial_state.to(device=r.device, dtype=torch.bfloat16)
        y_bf16, terminal_state_bf16 = WindRWKV7.apply(w_bf16, q_bf16, k_bf16, v_bf16, z_bf16, a_bf16, s0)
        y = y_bf16[:, :seq_len].reshape(batch_size, seq_len, heads * head_dim).to(original_dtype)
        y = self.group_norm(y.reshape(batch_size * seq_len, heads * head_dim)).reshape(batch_size, seq_len, heads * head_dim)
        return self.out(y * gate).to(original_dtype), terminal_state_bf16.to(original_dtype)

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
        w = torch.exp(-torch.exp(w_log.to(torch.float32))).to(gate.dtype).view(batch_size, seq_len, heads, head_dim)
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
            outputs.append(torch.einsum("bhij,bhj->bhi", memory, r[:, t]).reshape(batch_size, channels))
        y = torch.stack(outputs, dim=1)
        y = self.group_norm(y.reshape(batch_size * seq_len, channels)).reshape(batch_size, seq_len, channels)
        return self.out(y * gate), memory


class NativeRWKVRecurrentLayer(nn.Module):
    def __init__(self, config: EqRConfig, *, layer_id: int, layers: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        heads = int(config.num_heads)
        if hidden_size % heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {heads}")
        self.norm_eps = float(config.rms_norm_eps)
        self.future_seed_logit = nn.Parameter(
            torch.full((1, heads, 1, 1), float(config.rwkv_future_seed_gate_bias))
        )
        self.time_mix = NativeRWKVTimeMix(
            hidden_size,
            heads,
            hidden_size // heads,
            layer_id=layer_id,
            layers=layers,
            rwkv_kernel=config.rwkv_mixer_kernel,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        mixed, terminal_state = self.time_mix(normalized, initial_state=initial_state)
        return rms_norm(hidden_states + mixed, variance_epsilon=self.norm_eps), terminal_state


class NativeFutureSeedRWKVMixer(nn.Module):
    def __init__(self, config: EqRConfig) -> None:
        super().__init__()
        self.mode = str(config.mixer_replacement_mode).lower()
        if self.mode not in {"rwkv", "rwkv_native_futureseed"}:
            raise ValueError(f"Unknown RWKV mixer mode '{config.mixer_replacement_mode}'")
        self.future_seed_scale = float(config.rwkv_future_seed_scale)
        self.blocks = nn.ModuleList(
            [
                NativeRWKVRecurrentLayer(config, layer_id=layer_id, layers=int(config.rwkv_mixer_layers))
                for layer_id in range(int(config.rwkv_mixer_layers))
            ]
        )
        if len(self.blocks) < 1:
            raise ValueError("rwkv_mixer_layers must be >= 1")
        self.latest_future_seed_stats: Dict[str, float] = {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        previous_state: Optional[torch.Tensor] = None
        gate_values: List[torch.Tensor] = []
        seed_norms: List[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            initial_state = None
            if (
                self.mode == "rwkv_native_futureseed"
                and self.future_seed_scale != 0.0
                and layer_idx > 0
                and previous_state is not None
            ):
                denom = previous_state.to(torch.float32).square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
                normalized_state = (previous_state.to(torch.float32) / denom).to(previous_state.dtype)
                gate = torch.sigmoid(block.future_seed_logit.to(device=x.device, dtype=torch.float32)).to(previous_state.dtype)
                gate = gate * self.future_seed_scale
                initial_state = normalized_state * gate
                gate_values.append(gate.mean())
                seed_norms.append(initial_state.to(torch.float32).norm(dim=(-1, -2)).mean())
            x, previous_state = block(x, initial_state=initial_state)
        with torch.no_grad():
            if gate_values:
                self.latest_future_seed_stats = {
                    "rwkv_native_fs_gate_mean": float(torch.stack(gate_values).mean().detach().cpu()),
                    "rwkv_native_fs_state_norm": float(torch.stack(seed_norms).mean().detach().cpu()),
                }
            else:
                self.latest_future_seed_stats = {
                    "rwkv_native_fs_gate_mean": 0.0,
                    "rwkv_native_fs_state_norm": 0.0,
                }
        return x
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def copy_rwkv_cuda(eqr_dir: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "experiments" / "rwkv_fs_sudoku" / "rwkv7_cuda"
    if not source.exists():
        raise FileNotFoundError(f"vendored RWKV7 CUDA package not found: {source}")
    dest = eqr_dir / "rwkv7_cuda"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "build"),
    )


def patch_eqr_model(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "eqr.py"
    text = path.read_text(encoding="utf-8")
    if "class NativeFutureSeedRWKVMixer" in text:
        text = text.replace(
            "        self.norm = nn.LayerNorm(hidden_size)\n        self.norm_eps = float(config.rms_norm_eps)\n",
            "        self.norm_eps = float(config.rms_norm_eps)\n",
        )
        text = text.replace(
            "        mixed, terminal_state = self.time_mix(self.norm(hidden_states), initial_state=initial_state)\n        return rms_norm(hidden_states + mixed, variance_epsilon=self.norm_eps), terminal_state\n",
            "        normalized = rms_norm(hidden_states, variance_epsilon=self.norm_eps)\n        mixed, terminal_state = self.time_mix(normalized, initial_state=initial_state)\n        return rms_norm(hidden_states + mixed, variance_epsilon=self.norm_eps), terminal_state\n",
        )
        path.write_text(text, encoding="utf-8")
        return

    text = replace_once(
        text,
        "    noise_scale: float = 0.01\n    H_init_std: float = 1.0\n",
        "    noise_scale: float = 0.01\n    mixer_replacement_mode: str = \"none\"\n    rwkv_mixer_layers: int = 2\n    rwkv_mixer_kernel: str = \"auto\"\n    rwkv_future_seed_scale: float = 0.0\n    rwkv_future_seed_gate_bias: float = 0.0\n    H_init_std: float = 1.0\n",
        "EqRConfig native RWKV/FutureSeed fields",
    )

    text = replace_once(
        text,
        "\n\nclass ReasoningBlock(nn.Module):\n",
        RWKV_NATIVE_FUTURESEED_MIXER + "\n\nclass ReasoningBlock(nn.Module):\n",
        "NativeFutureSeedRWKVMixer insertion",
    )

    text = replace_once(
        text,
        "        if config.mlp_t:\n            self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)\n        else:\n            self.self_attn = Attention(\n",
        "        self.mixer_replacement_mode = str(config.mixer_replacement_mode).lower()\n        if self.mixer_replacement_mode in {\"rwkv\", \"rwkv_native_futureseed\"}:\n            self.rwkv_token_mixer = NativeFutureSeedRWKVMixer(config)\n        elif self.mixer_replacement_mode != \"none\":\n            raise ValueError(f\"Unknown mixer_replacement_mode '{config.mixer_replacement_mode}'\")\n        elif config.mlp_t:\n            self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)\n        else:\n            self.self_attn = Attention(\n",
        "ReasoningBlock native RWKV init replacement",
    )

    text = replace_once(
        text,
        "        if self.config.mlp_t:\n            hidden_states = hidden_states.transpose(1, 2)\n            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)\n            hidden_states = hidden_states.transpose(1, 2)\n        else:\n            hidden_states = rms_norm(\n                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),\n                variance_epsilon=self.norm_eps,\n            )\n",
        "        if self.mixer_replacement_mode in {\"rwkv\", \"rwkv_native_futureseed\"}:\n            hidden_states = self.rwkv_token_mixer(hidden_states)\n        elif self.config.mlp_t:\n            hidden_states = hidden_states.transpose(1, 2)\n            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)\n            hidden_states = hidden_states.transpose(1, 2)\n        else:\n            hidden_states = rms_norm(\n                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),\n                variance_epsilon=self.norm_eps,\n            )\n",
        "ReasoningBlock native RWKV forward replacement",
    )

    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "rwkv_mixer_layers:" in text:
        return
    text = replace_once(
        text,
        "noise_scale: 0.01\nH_init_std: 1.0\n",
        "noise_scale: 0.01\nmixer_replacement_mode: none\nrwkv_mixer_layers: 2\nrwkv_mixer_kernel: auto\nrwkv_future_seed_scale: 0.0\nrwkv_future_seed_gate_bias: 0.0\nH_init_std: 1.0\n",
        "arch native RWKV/FutureSeed defaults",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    copy_rwkv_cuda(eqr_dir)
    patch_eqr_model(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"Native RWKV/FutureSeed mixer patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
