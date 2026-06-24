#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


FUTURESEED_SCAN_MIXER = '''

import os as _futureseed_os

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _futureseed_scan_fwd_kernel(impulse, decay, out, B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
        b = tl.program_id(0)
        d_block = tl.program_id(1)
        offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < D
        dec = tl.load(decay + offsets, mask=mask, other=0.0).to(tl.float32)
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)
        base = b * L * D + offsets
        for t in range(0, L):
            impulse_t = tl.load(impulse + base + t * D, mask=mask, other=0.0).to(tl.float32)
            state = dec * state + impulse_t
            tl.store(out + base + t * D, state, mask=mask)


    @triton.jit
    def _futureseed_scan_bwd_kernel(
        grad_out,
        state,
        decay,
        grad_impulse,
        grad_decay_parts,
        B: tl.constexpr,
        L: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b = tl.program_id(0)
        d_block = tl.program_id(1)
        offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offsets < D
        dec = tl.load(decay + offsets, mask=mask, other=0.0).to(tl.float32)
        adjoint = tl.zeros((BLOCK_D,), dtype=tl.float32)
        grad_decay = tl.zeros((BLOCK_D,), dtype=tl.float32)
        base = b * L * D + offsets
        for rev_t in range(0, L):
            t = L - 1 - rev_t
            adjoint += tl.load(grad_out + base + t * D, mask=mask, other=0.0).to(tl.float32)
            tl.store(grad_impulse + base + t * D, adjoint, mask=mask)
            prev_state = tl.zeros((BLOCK_D,), dtype=tl.float32)
            if t > 0:
                prev_state = tl.load(state + base + (t - 1) * D, mask=mask, other=0.0).to(tl.float32)
            grad_decay += adjoint * prev_state
            adjoint = adjoint * dec
        tl.store(grad_decay_parts + b * D + offsets, grad_decay, mask=mask)


    class _FutureSeedScanFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, impulse: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
            if impulse.ndim != 3:
                raise ValueError(f"FutureSeed scan expects [B, L, D], got {tuple(impulse.shape)}")
            B, L, D = impulse.shape
            impulse32 = impulse.contiguous().to(torch.float32)
            decay32 = decay.contiguous().to(torch.float32)
            state = torch.empty_like(impulse32)
            block_d = 64
            grid = (B, triton.cdiv(D, block_d))
            _futureseed_scan_fwd_kernel[grid](impulse32, decay32, state, B, L, D, BLOCK_D=block_d, num_warps=2)
            ctx.save_for_backward(state, decay32)
            return state

        @staticmethod
        def backward(ctx, grad_state: torch.Tensor):
            state, decay = ctx.saved_tensors
            B, L, D = state.shape
            grad_out = grad_state.contiguous().to(torch.float32)
            grad_impulse = torch.empty_like(grad_out)
            grad_decay_parts = torch.empty((B, D), device=grad_out.device, dtype=torch.float32)
            block_d = 64
            grid = (B, triton.cdiv(D, block_d))
            _futureseed_scan_bwd_kernel[grid](
                grad_out,
                state,
                decay,
                grad_impulse,
                grad_decay_parts,
                B,
                L,
                D,
                BLOCK_D=block_d,
                num_warps=2,
            )
            return grad_impulse, grad_decay_parts.sum(dim=0)


class FutureSeedScanMixer(nn.Module):
    def __init__(self, config: EqRConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.norm_eps = float(config.rms_norm_eps)
        self.forward_proj = CastedLinear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.reverse_proj = CastedLinear(config.hidden_size, config.hidden_size * 2, bias=False)
        self.out_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        decay = float(config.future_seed_scan_decay_init)
        decay = min(max(decay, 1e-4), 1.0 - 1e-4)
        decay_logit = math.log(decay / (1.0 - decay))
        self.forward_decay_logit = nn.Parameter(torch.full((config.hidden_size,), decay_logit))
        self.reverse_decay_logit = nn.Parameter(torch.full((config.hidden_size,), decay_logit))
        self.mix_gate_logit = nn.Parameter(torch.tensor(float(config.future_seed_mix_gate_bias)))

    def _shift_right(self, tensor: torch.Tensor, offset: int, fill: float) -> torch.Tensor:
        pad_shape = list(tensor.shape)
        pad_shape[1] = offset
        pad = tensor.new_full(pad_shape, fill)
        return torch.cat((pad, tensor[:, :-offset]), dim=1)

    def _scan(self, x: torch.Tensor, proj: CastedLinear, decay_logit: torch.Tensor) -> torch.Tensor:
        projected = proj(x)
        write_gate, value = projected.chunk(2, dim=-1)
        write_gate = torch.sigmoid(write_gate.to(torch.float32))
        value = torch.tanh(value.to(torch.float32))
        decay_flat = torch.sigmoid(decay_logit.to(torch.float32))
        decay = decay_flat.view(1, 1, -1)
        impulse = (1.0 - decay) * write_gate * value

        backend = _futureseed_os.environ.get("FUTURESEED_SCAN_BACKEND", "triton").lower()
        if backend == "triton" and triton is not None and impulse.is_cuda:
            return _FutureSeedScanFunction.apply(impulse, decay_flat).to(x.dtype)

        # Parallel prefix form of y[t] = decay * y[t - 1] + impulse[t].
        # This keeps the mechanism recurrent while avoiding a Python loop over
        # sequence positions, which otherwise makes official EqR eval infeasible.
        coeff = decay.expand(1, x.shape[1], x.shape[-1])
        state = impulse
        offset = 1
        while offset < x.shape[1]:
            coeff_prev = self._shift_right(coeff, offset, 1.0)
            state_prev = self._shift_right(state, offset, 0.0)
            valid = (torch.arange(x.shape[1], device=x.device).view(1, -1, 1) >= offset)
            state = torch.where(valid, state + coeff * state_prev, state)
            coeff = torch.where(valid, coeff * coeff_prev, coeff)
            offset *= 2
        return state.to(x.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        forward_state = self._scan(normalized, self.forward_proj, self.forward_decay_logit)
        reverse_state = torch.flip(
            self._scan(torch.flip(normalized, dims=(1,)), self.reverse_proj, self.reverse_decay_logit),
            dims=(1,),
        )
        mixed = forward_state + reverse_state
        gate = torch.sigmoid(self.mix_gate_logit.to(torch.float32)).to(hidden_states.dtype)
        return gate * self.out_proj(mixed).to(hidden_states.dtype)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def patch_eqr_model(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "eqr.py"
    text = path.read_text(encoding="utf-8")
    if "class FutureSeedScanMixer" in text:
        return

    text = replace_once(
        text,
        "    noise_scale: float = 0.01\n    H_init_std: float = 1.0\n",
        "    noise_scale: float = 0.01\n    mixer_replacement_mode: str = \"none\"\n    future_seed_scan_decay_init: float = 0.9\n    future_seed_mix_gate_bias: float = 0.0\n    H_init_std: float = 1.0\n",
        "EqRConfig mixer replacement fields",
    )

    text = replace_once(
        text,
        "\n\nclass ReasoningBlock(nn.Module):\n",
        FUTURESEED_SCAN_MIXER + "\n\nclass ReasoningBlock(nn.Module):\n",
        "FutureSeedScanMixer insertion",
    )

    text = replace_once(
        text,
        "        if config.mlp_t:\n            self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)\n        else:\n            self.self_attn = Attention(\n",
        "        self.mixer_replacement_mode = str(config.mixer_replacement_mode).lower()\n        if self.mixer_replacement_mode == \"future_seed_scan\":\n            self.future_seed_scan_mixer = FutureSeedScanMixer(config)\n        elif self.mixer_replacement_mode != \"none\":\n            raise ValueError(f\"Unknown mixer_replacement_mode '{config.mixer_replacement_mode}'\")\n        elif config.mlp_t:\n            self.mlp_t = SwiGLU(hidden_size=config.seq_len, expansion=config.expansion)\n        else:\n            self.self_attn = Attention(\n",
        "ReasoningBlock mixer init replacement",
    )

    text = replace_once(
        text,
        "        if self.config.mlp_t:\n            hidden_states = hidden_states.transpose(1, 2)\n            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)\n            hidden_states = hidden_states.transpose(1, 2)\n        else:\n            hidden_states = rms_norm(\n                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),\n                variance_epsilon=self.norm_eps,\n            )\n",
        "        if self.mixer_replacement_mode == \"future_seed_scan\":\n            hidden_states = rms_norm(\n                hidden_states + self.future_seed_scan_mixer(hidden_states),\n                variance_epsilon=self.norm_eps,\n            )\n        elif self.config.mlp_t:\n            hidden_states = hidden_states.transpose(1, 2)\n            hidden_states = rms_norm(hidden_states + self.mlp_t(hidden_states), variance_epsilon=self.norm_eps)\n            hidden_states = hidden_states.transpose(1, 2)\n        else:\n            hidden_states = rms_norm(\n                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),\n                variance_epsilon=self.norm_eps,\n            )\n",
        "ReasoningBlock forward mixer replacement",
    )

    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "mixer_replacement_mode:" in text:
        return
    text = replace_once(
        text,
        "noise_scale: 0.01\nH_init_std: 1.0\n",
        "noise_scale: 0.01\nmixer_replacement_mode: none\nfuture_seed_scan_decay_init: 0.9\nfuture_seed_mix_gate_bias: 0.0\nH_init_std: 1.0\n",
        "arch mixer replacement defaults",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    patch_eqr_model(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"FutureSeed mixer replacement patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
