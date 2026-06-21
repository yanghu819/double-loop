#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} match, found {count}")
    return text.replace(old, new, 1)


def replace_regex_once(text: str, pattern: str, new: str, label: str) -> str:
    updated, count = re.subn(pattern, new, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected one {label} regex match, found {count}")
    return updated


def patch_loss_heads(eqr_dir: Path) -> None:
    path = eqr_dir / "models" / "losses" / "loss_heads.py"
    text = path.read_text(encoding="utf-8")
    if "path_token_weight" in text:
        return

    text = replace_regex_once(
        text,
        r"    def __init__\(self, model: nn\.Module, loss_type: str, \*\*kwargs: Any\) -> None:\s+        del kwargs\s+        super\(\).__init__\(\)\s+        self\.model = model\s+        self\.loss_fn = globals\(\)\[loss_type\]\n(?:[ \t]*\n)*(?=    def initial_carry)",
        "    def __init__(\n        self,\n        model: nn.Module,\n        loss_type: str,\n        path_token_id: int = 5,\n        path_token_weight: float = 1.0,\n        **kwargs: Any,\n    ) -> None:\n\n\n\n        del kwargs\n\n        super().__init__()\n\n        self.model = model\n\n        self.loss_fn = globals()[loss_type]\n        self.path_token_id = int(path_token_id)\n        self.path_token_weight = float(path_token_weight)\n",
        "ACTLossHead init",
    )

    text = replace_regex_once(
        text,
        r"        lm_loss = \(\s+            self\.loss_fn\(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid_mask\) / loss_divisor\s+        \)\.sum\(\)\s+        return \{\"lm_loss\": lm_loss\}\n(?:[ \t]*\n)*(?=    def forward)",
        "        token_loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=valid_mask)\n        if self.path_token_weight != 1.0:\n            token_weights = torch.where(\n                valid_mask & (labels == self.path_token_id),\n                torch.full_like(token_loss, self.path_token_weight, dtype=token_loss.dtype),\n                torch.ones_like(token_loss),\n            )\n            weighted_divisor = token_weights.masked_fill(~valid_mask, 0).sum(-1).clamp_min(1).unsqueeze(-1)\n        else:\n            token_weights = torch.ones_like(token_loss)\n            weighted_divisor = loss_divisor\n        weighted_token_loss = token_loss * token_weights\n        lm_loss = (weighted_token_loss / weighted_divisor).sum()\n\n        result = {\"lm_loss\": lm_loss}\n        if self.path_token_weight != 1.0:\n            with torch.no_grad():\n                path_mask = valid_mask & (labels == self.path_token_id)\n                non_path_mask = valid_mask & ~path_mask\n                path_count = path_mask.sum().clamp_min(1)\n                non_path_count = non_path_mask.sum().clamp_min(1)\n                result[\"path_ce\"] = torch.where(path_mask, token_loss, torch.zeros_like(token_loss)).sum() / path_count\n                result[\"non_path_ce\"] = torch.where(non_path_mask, token_loss, torch.zeros_like(token_loss)).sum() / non_path_count\n                result[\"path_token_frac\"] = path_mask.to(torch.float32).sum() / valid_mask.to(torch.float32).sum().clamp_min(1)\n        return result\n",
        "ACTLossHead weighted compute_lm_loss",
    )

    text = replace_regex_once(
        text,
        r"        lm_loss = \(\s+            self\.loss_fn\(outputs\[\"logits\"\], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask\) / loss_divisor\s+        \)\.sum\(\)\n(?:[ \t]*\n)*(?=        q_halt_loss)",
        "        token_loss = self.loss_fn(outputs[\"logits\"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)\n        if self.path_token_weight != 1.0:\n            token_weights = torch.where(\n                mask & (labels == self.path_token_id),\n                torch.full_like(token_loss, self.path_token_weight, dtype=token_loss.dtype),\n                torch.ones_like(token_loss),\n            )\n            weighted_divisor = token_weights.masked_fill(~mask, 0).sum(-1).clamp_min(1).unsqueeze(-1)\n        else:\n            token_weights = torch.ones_like(token_loss)\n            weighted_divisor = loss_divisor\n        lm_loss = ((token_loss * token_weights) / weighted_divisor).sum()\n",
        "InferenceLossHead weighted loss",
    )
    path.write_text(text, encoding="utf-8")


def patch_arch_config(eqr_dir: Path) -> None:
    path = eqr_dir / "config" / "arch" / "eqr.yaml"
    text = path.read_text(encoding="utf-8")
    if "path_token_weight:" in text:
        return
    text = replace_once(
        text,
        "  loss_type: stablemax_cross_entropy\n",
        "  loss_type: stablemax_cross_entropy\n  path_token_id: 5\n  path_token_weight: 1.0\n",
        "loss path-weight defaults",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eqr_dir", type=Path)
    args = parser.parse_args()
    eqr_dir = args.eqr_dir.resolve()
    patch_loss_heads(eqr_dir)
    patch_arch_config(eqr_dir)
    print(f"Path-loss patch applied to {eqr_dir}")


if __name__ == "__main__":
    main()
