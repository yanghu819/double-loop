from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2

from experiments.zoology_mqar.gdn2_chunk_local_biaxis import (
    ChunkLocalBiAxisGatedDeltaNet2,
    parent_parameter_hash,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    make_model,
    parameter_hash,
)
from zoology.utils import set_determinism


EXPECTED_FLA_SHA = "9c8e42e762fce087c27b673af4922795d9edb85e"


def gpu_uuid() -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {output}")
    index, uuid = (part.strip() for part in output[0].split(",", maxsplit=1))
    if index != "0":
        raise RuntimeError(f"Expected CUDA index0, got {index}")
    return uuid


def graph_names(tensor: torch.Tensor) -> list[str]:
    names = []
    queue = [tensor.grad_fn]
    seen = set()
    while queue:
        node = queue.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        queue.extend(next_node for next_node, _index in node.next_functions)
    return names


def direct_biaxis(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(q.float(), dim=-1) * (q.shape[-1] ** -0.5)
    k = F.normalize(k.float(), dim=-1)
    state = initial_state.float()
    outputs = []
    for index in range(q.shape[1]):
        state = state * gk[:, index].float().exp().unsqueeze(-1)
        state = state * gv[:, index].float().exp().unsqueeze(-2)
        erase = (
            (b[:, index].float() * k[:, index]).unsqueeze(-1) * state
        ).sum(dim=-2)
        update = w[:, index].float() * v[:, index].float() - erase
        state = state + k[:, index].unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append((q[:, index].unsqueeze(-1) * state).sum(dim=-2))
    return torch.stack(outputs, dim=1), state


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != EXPECTED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA was not asserted")
    if gpu_uuid() != args.expected_gpu_uuid:
        raise RuntimeError("GPU UUID mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one CUDA device")
    source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if "/huyang2/double-loop/.cache/fla-versions/" not in str(source):
        raise RuntimeError(f"Unexpected GDN2 source: {source}")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=128,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=2,
    )
    candidate_config = build_config(
        arm="future_seed_chunk_local_biaxis_gdn2",
        sequence_length=128,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=2,
    )
    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_chunk_local_biaxis_gdn2",
    )
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Candidate parent initialization differs from control")
    parameter_delta = sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in control.parameters()
    )
    if parameter_delta != 8_192:
        raise RuntimeError(f"Unexpected parameter delta {parameter_delta}")

    inputs = torch.randint(0, 256, (2, 128), device="cuda")
    control = control.cuda().eval()
    candidate = candidate.cuda().eval()
    with torch.no_grad():
        control_logits = control(inputs)
        candidate_logits = candidate(inputs)
    identity_max_abs = float(
        (control_logits.float() - candidate_logits.float()).abs().max().item()
    )
    identity_relative_rms = float(
        (
            (control_logits.float() - candidate_logits.float()).square().mean().sqrt()
            / control_logits.float().square().mean().sqrt().clamp_min(1e-8)
        ).item()
    )
    if identity_max_abs > 0.04 or identity_relative_rms > 0.01:
        raise RuntimeError(
            "Zero-decay split path does not preserve the parent within BF16 tolerance"
        )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    loss = candidate(inputs).float().square().mean()
    loss.backward()
    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    projection_grad_norms = [
        float(mixer.layer.value_decay_proj.weight.grad.float().norm().item())
        for mixer in mixers
    ]
    if not all(value > 0.0 and math_isfinite(value) for value in projection_grad_norms):
        raise RuntimeError(f"Value-decay gradients are invalid: {projection_grad_norms}")

    torch.manual_seed(9)
    B, T, H, K, V = 1, 81, 1, 32, 32
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    gk = -0.02 * torch.rand(B, T, H, K, device="cuda")
    raw_gv = torch.rand(B, T, H, V, device="cuda")
    gv = -(math.log(4.0) / 64.0) * raw_gv
    b = torch.sigmoid(torch.randn(B, T, H, K, device="cuda"))
    w = torch.sigmoid(torch.randn(B, T, H, V, device="cuda"))
    initial_state = torch.randn(B, H, K, V, device="cuda") * 0.05
    reference_output, reference_state = direct_biaxis(
        q,
        k,
        v,
        gk,
        gv,
        b,
        w,
        initial_state,
    )
    shell = ChunkLocalBiAxisGatedDeltaNet2(mixers[0].layer.base).cuda()
    actual_output, actual_state = shell.run_chunk_local_recurrence(
        q=q,
        k=k,
        v=v,
        g=gk,
        b=b,
        w=w,
        value_log_decay=gv,
        recurrent_state=initial_state,
    )
    output_max_abs = float(
        (actual_output.float() - reference_output).abs().max().item()
    )
    state_max_abs = float((actual_state - reference_state).abs().max().item())
    if output_max_abs > 0.08 or state_max_abs > 0.08:
        raise RuntimeError(
            f"Direct recurrence mismatch: output={output_max_abs}, state={state_max_abs}"
        )

    q_leaf = q.detach().requires_grad_(True)
    v_leaf = v.detach().requires_grad_(True)
    gk_leaf = gk.detach().requires_grad_(True)
    gv_leaf = gv.detach().requires_grad_(True)
    output, terminal = shell.run_chunk_local_recurrence(
        q=q_leaf,
        k=k,
        v=v_leaf,
        g=gk_leaf,
        b=b,
        w=w,
        value_log_decay=gv_leaf,
        recurrent_state=initial_state,
    )
    cross_chunk_loss = output[:, 64:].float().square().mean() + terminal.square().mean()
    cross_chunk_loss.backward()
    cross_chunk_gradients = {
        "v_first_chunk": float(v_leaf.grad[:, :64].float().norm().item()),
        "gk_first_chunk": float(gk_leaf.grad[:, :64].float().norm().item()),
        "gv_first_chunk": float(gv_leaf.grad[:, :64].float().norm().item()),
    }
    if not all(value > 0.0 for value in cross_chunk_gradients.values()):
        raise RuntimeError(f"Cross-chunk VJP is missing: {cross_chunk_gradients}")
    graph_chunk_count = graph_names(terminal).count("ChunkGDN2FunctionBackward")
    if graph_chunk_count != 2:
        raise RuntimeError(f"Expected two chained official chunk nodes, got {graph_chunk_count}")

    diagnostics = shell.last_diagnostics
    if diagnostics is None:
        raise RuntimeError("Bi-Axis diagnostics missing")
    if diagnostics["local_scale_min"] < 0.249 or diagnostics["local_inverse_max"] > 4.01:
        raise RuntimeError(f"Local frame bound failed: {diagnostics}")
    result = {
        "status": "passed",
        "gpu_uuid": args.expected_gpu_uuid,
        "fla_source": str(source),
        "fla_expected_sha": EXPECTED_FLA_SHA,
        "parameter_delta": parameter_delta,
        "parent_parameter_hash": parent_parameter_hash(candidate),
        "control_parameter_hash": parameter_hash(control),
        "zero_decay_identity_max_abs": identity_max_abs,
        "zero_decay_identity_relative_rms": identity_relative_rms,
        "projection_grad_norms": projection_grad_norms,
        "direct_reference_output_max_abs": output_max_abs,
        "direct_reference_state_max_abs": state_max_abs,
        "cross_chunk_gradients": cross_chunk_gradients,
        "official_chunk_backward_nodes": graph_chunk_count,
        "diagnostics": diagnostics,
        "input_sha256": tensor_sha256(inputs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def math_isfinite(value: float) -> bool:
    return value == value and abs(value) < float("inf")


if __name__ == "__main__":
    main()
