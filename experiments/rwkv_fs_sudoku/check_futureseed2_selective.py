#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from futureseed2_selective import FutureSeedSelectiveGate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FutureSeed2 preflight is CUDA-only; CPU smoke is forbidden")

    torch.manual_seed(52)
    state = torch.randn(
        3,
        2,
        8,
        6,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    base_logit = torch.randn(
        1,
        2,
        1,
        1,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    module = FutureSeedSelectiveGate(
        mode="state",
        layers=4,
        heads=2,
        row_dim=8,
        col_dim=6,
    ).to(device)

    identity_gate, identity_diag = module(
        state,
        base_logit=base_logit,
        layer_idx=1,
    )
    baseline_gate = torch.sigmoid(base_logit)
    if not torch.equal(identity_gate, baseline_gate):
        raise AssertionError("Zero-initialized FutureSeed2 gate is not exact FutureSeed1")
    if float(identity_diag["fs2_seed_relative_change"]) != 0.0:
        raise AssertionError("Zero-initialized selective gate reports nonzero seed change")

    with torch.no_grad():
        assert module.gate_delta is not None
        module.gate_delta[0, 0, 1, 2] = 0.20
        module.gate_delta[0, 1, 3, 4] = -0.15

    gate, diagnostics = module(
        state,
        base_logit=base_logit,
        layer_idx=1,
    )
    reference = torch.sigmoid(base_logit + module.gate_delta[0].unsqueeze(0))
    reference_max_abs = float((gate - reference).float().abs().max())
    if reference_max_abs != 0.0:
        raise AssertionError(
            f"FutureSeed2 selective gate differs from Torch reference: {reference_max_abs}"
        )
    if torch.equal(gate, baseline_gate):
        raise AssertionError("Activated FutureSeed2 gate did not change the seed")

    selected = state * gate
    loss = selected.float().square().mean()
    loss.backward()
    gradients = {
        "state": state.grad,
        "base_logit": base_logit.grad,
        "gate_delta": module.gate_delta.grad,
    }
    gradient_norms = {}
    for name, gradient in gradients.items():
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"Missing or non-finite gradient for {name}")
        norm = float(gradient.float().norm())
        if norm <= 0:
            raise AssertionError(f"Zero gradient for {name}")
        gradient_norms[name] = norm

    print(
        json.dumps(
            {
                "device": str(device),
                "identity_exact": True,
                "reference_max_abs": reference_max_abs,
                "activated_gate_delta_rms": float(
                    diagnostics["fs2_gate_delta_rms"]
                ),
                "activated_gate_std": float(diagnostics["fs2_gate_std"]),
                "activated_seed_relative_change": float(
                    diagnostics["fs2_seed_relative_change"]
                ),
                "gradient_norms": gradient_norms,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
