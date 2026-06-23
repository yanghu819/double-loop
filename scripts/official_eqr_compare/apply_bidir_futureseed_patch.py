#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


BIDIR_FUTURESEED_CLASS = '''

class ReverseCausalFutureSeed(nn.Module):
    def __init__(self, config: EqRConfig) -> None:
        super().__init__()
        self.scale = float(config.future_seed_scale)
        self.norm_eps = float(config.rms_norm_eps)
        self.puzzle_emb_len = int(config.puzzle_emb_len)
        self.gate_logit = nn.Parameter(torch.tensor(float(config.future_seed_gate_bias)))
        self.attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=True,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.zero_()
            idx = torch.arange(config.hidden_size)
            self.proj.weight[idx, idx] = 1.0

    def _body_cos_sin(self, cos_sin: CosSin) -> CosSin:
        prefix_len = self.puzzle_emb_len
        if prefix_len <= 0 or cos_sin is None:
            return cos_sin
        cos, sin = cos_sin
        if isinstance(cos, tuple) or isinstance(sin, tuple):
            raise ValueError("ReverseCausalFutureSeed does not support puzzle prefix with 2D RoPE.")
        return cos[prefix_len:], sin[prefix_len:]

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        if self.scale == 0.0:
            return torch.zeros_like(hidden_states)
        prefix_len = self.puzzle_emb_len
        if prefix_len > 0:
            prefix, body = hidden_states[:, :prefix_len], hidden_states[:, prefix_len:]
        else:
            prefix, body = None, hidden_states
        seed_cos_sin = self._body_cos_sin(cos_sin)
        reversed_states = torch.flip(body, dims=(1,))
        seed = rms_norm(
            reversed_states + self.attn(cos_sin=seed_cos_sin, hidden_states=reversed_states),
            variance_epsilon=self.norm_eps,
        )
        seed = rms_norm(seed + self.mlp(seed), variance_epsilon=self.norm_eps)
        seed = torch.flip(seed, dims=(1,))
        if prefix is not None:
            seed = torch.cat((torch.zeros_like(prefix), seed), dim=1)
        gate = torch.sigmoid(self.gate_logit.to(torch.float32)).to(seed.dtype) * self.scale
        return gate * self.proj(seed).to(hidden_states.dtype)
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def patch_eqr_model(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "eqr.py"
    text = path.read_text(encoding="utf-8")
    if "class ReverseCausalFutureSeed" in text:
        return

    if "future_seed_mode:" not in text:
        text = replace_once(
            text,
            "    noise_scale: float = 0.01\n    H_init_std: float = 1.0\n",
            "    noise_scale: float = 0.01\n    future_seed_mode: str = \"none\"\n    future_seed_scale: float = 0.0\n    future_seed_gate_bias: float = -2.0\n    H_init_std: float = 1.0\n",
            "EqRConfig bidirectional FutureSeed fields",
        )

    text = replace_once(
        text,
        "\n\nclass NoisyReasoningModule(nn.Module):\n",
        BIDIR_FUTURESEED_CLASS + "\n\nclass NoisyReasoningModule(nn.Module):\n",
        "ReverseCausalFutureSeed insertion",
    )

    text = replace_once(
        text,
        "        self._init_pos()\n        self.L_level = NoisyReasoningModule(config, [ReasoningBlock(config) for _ in range(config.L_layers)])\n        self.H_init = nn.Buffer",
        "        self._init_pos()\n        self.future_seed_mode = str(config.future_seed_mode).lower()\n        if self.future_seed_mode not in {\"none\", \"reverse_causal\"}:\n            raise ValueError(f\"Unknown future_seed_mode '{config.future_seed_mode}'\")\n        self.future_seed = (\n            ReverseCausalFutureSeed(config)\n            if self.future_seed_mode == \"reverse_causal\" and float(config.future_seed_scale) != 0.0\n            else None\n        )\n        self.L_level = NoisyReasoningModule(config, [ReasoningBlock(config) for _ in range(config.L_layers)])\n        self.H_init = nn.Buffer",
        "InnerNetwork bidirectional FutureSeed init",
    )

    text = replace_once(
        text,
        "    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, torch.Tensor]:\n        z_H, z_L = self.deep_recursion(\n            carry.z_H,\n            carry.z_L,\n            self._input_embeddings(batch[\"inputs\"], batch[\"puzzle_identifiers\"]),\n            {\"cos_sin\": self._cos_sin()},\n        )\n",
        "    def forward(self, carry: LatentCarry, batch: Dict[str, torch.Tensor]) -> Tuple[LatentCarry, torch.Tensor, torch.Tensor]:\n        seq = {\"cos_sin\": self._cos_sin()}\n        input_embeddings = self._input_embeddings(batch[\"inputs\"], batch[\"puzzle_identifiers\"])\n        if self.future_seed is not None:\n            input_embeddings = input_embeddings + self.future_seed(input_embeddings, **seq)\n        z_H, z_L = self.deep_recursion(\n            carry.z_H,\n            carry.z_L,\n            input_embeddings,\n            seq,\n        )\n",
        "InnerNetwork forward bidirectional FutureSeed hook",
    )

    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "future_seed_mode:" in text:
        return
    text = replace_once(
        text,
        "noise_scale: 0.01\nH_init_std: 1.0\n",
        "noise_scale: 0.01\nfuture_seed_mode: none\nfuture_seed_scale: 0.0\nfuture_seed_gate_bias: -2.0\nH_init_std: 1.0\n",
        "arch bidirectional FutureSeed defaults",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    patch_eqr_model(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"Bidirectional FutureSeed patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
