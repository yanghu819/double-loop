#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def patch_eqr_model(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "eqr.py"
    text = path.read_text(encoding="utf-8")
    if "attention_causal: bool = False" not in text:
        text = replace_once(
            text,
            "    mlp_t: bool = False\n    puzzle_emb_len: int = 0\n",
            "    mlp_t: bool = False\n    attention_causal: bool = False\n    puzzle_emb_len: int = 0\n",
            "EqRConfig attention_causal field",
        )
    if "causal=config.attention_causal" not in text:
        text = replace_once(
            text,
            "                causal=False,\n",
            "                causal=config.attention_causal,\n",
            "Attention causal config hook",
        )
    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "attention_causal:" not in text:
        text = replace_once(
            text,
            "mlp_t: false\n\nlambda_: 0.95\n",
            "mlp_t: false\nattention_causal: false\n\nlambda_: 0.95\n",
            "arch attention_causal default",
        )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    patch_eqr_model(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"Causal attention patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
