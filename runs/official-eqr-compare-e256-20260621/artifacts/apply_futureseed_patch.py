#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


FUTURESEED_CLASS = '''

class FutureSeedMixer(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float, scale: float, gate_bias: float) -> None:
        super().__init__()
        self.future_seed_scale = float(scale)
        self.norm_eps = float(rms_norm_eps)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias)))
        self.proj = CastedLinear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            idx = torch.arange(hidden_size)
            self.proj.weight[idx, idx] = 1.0

    def forward(self, target: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        scale = float(self.future_seed_scale)
        if scale == 0.0:
            return target
        seeded = self.proj(rms_norm(seed, variance_epsilon=self.norm_eps)).to(target.dtype)
        gate = torch.sigmoid(self.gate_logit.to(torch.float32)).to(target.dtype) * scale
        return target + gate * seeded
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def patch_eqr_model(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "eqr.py"
    text = path.read_text(encoding="utf-8")
    if "class FutureSeedMixer" in text:
        return

    text = replace_once(
        text,
        "    noise_scale: float = 0.01\n    H_init_std: float = 1.0\n",
        "    noise_scale: float = 0.01\n    future_seed_scale: float = 0.0\n    future_seed_gate_bias: float = -2.0\n    H_init_std: float = 1.0\n",
        "EqRConfig future_seed fields",
    )

    text = replace_once(
        text,
        "\n\nclass NoisyReasoningModule(nn.Module):\n",
        FUTURESEED_CLASS + "\n\nclass NoisyReasoningModule(nn.Module):\n",
        "FutureSeedMixer insertion",
    )

    text = replace_once(
        text,
        "        self.L_level = NoisyReasoningModule(config, [ReasoningBlock(config) for _ in range(config.L_layers)])\n        self.H_init = nn.Buffer",
        "        self.L_level = NoisyReasoningModule(config, [ReasoningBlock(config) for _ in range(config.L_layers)])\n        if float(config.future_seed_scale) != 0.0:\n            self.future_seed_l_from_h = FutureSeedMixer(\n                config.hidden_size,\n                config.rms_norm_eps,\n                config.future_seed_scale,\n                config.future_seed_gate_bias,\n            )\n            self.future_seed_h_from_l = FutureSeedMixer(\n                config.hidden_size,\n                config.rms_norm_eps,\n                config.future_seed_scale,\n                config.future_seed_gate_bias,\n            )\n        else:\n            self.future_seed_l_from_h = None\n            self.future_seed_h_from_l = None\n        self.H_init = nn.Buffer",
        "FutureSeed module init",
    )

    text = replace_once(
        text,
        "    def latent_recursion(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, seq: Dict[str, CosSin]) -> Tuple[torch.Tensor, torch.Tensor]:\n        for _ in range(self.config.L_cycles):\n            z_L = self.L_level(z_L, z_H + x, **seq)\n        z_H = self.L_level(z_H, z_L, **seq)\n        return z_H, z_L\n",
        "    def latent_recursion(self, z_H: torch.Tensor, z_L: torch.Tensor, x: torch.Tensor, seq: Dict[str, CosSin]) -> Tuple[torch.Tensor, torch.Tensor]:\n        if self.future_seed_l_from_h is not None:\n            z_L = self.future_seed_l_from_h(z_L, z_H)\n        for _ in range(self.config.L_cycles):\n            z_L = self.L_level(z_L, z_H + x, **seq)\n        if self.future_seed_h_from_l is not None:\n            z_H = self.future_seed_h_from_l(z_H, z_L)\n        z_H = self.L_level(z_H, z_L, **seq)\n        return z_H, z_L\n",
        "latent_recursion FutureSeed hooks",
    )

    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "future_seed_scale:" in text:
        return
    text = replace_once(
        text,
        "noise_scale: 0.01\nH_init_std: 1.0\n",
        "noise_scale: 0.01\nfuture_seed_scale: 0.0\nfuture_seed_gate_bias: -2.0\nH_init_std: 1.0\n",
        "arch future_seed defaults",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    patch_eqr_model(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"FutureSeed patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
