from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from fla.ops.mesa_net import chunk_mesa_net
from fla.ops.mesa_net.naive import naive_mesa_net_exact
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.length_scaling import build_config, make_model
from experiments.zoology_mqar.mesa_futureseed import (
    EXPECTED_MESA_SOURCE_SHA256,
    EXPECTED_MIXER_PARAMETERS,
    EXPECTED_STATE_VALUES_PER_LAYER,
    HEAD_DIM,
    LAMBDA_LOWER_BOUND,
    MAX_CG_ITERATIONS,
    NUM_HEADS,
    ZoologyMesaFutureSeedMixer,
    load_matched_parent_state,
    mesa_futureseed_diagnostics,
    official_mesa_provenance,
)


EXPECTED_MATCHED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)
EXPECTED_TRAIN_HASH = (
    "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
)
EXPECTED_TEST_HASH = (
    "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
)


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def _visible_gpu() -> tuple[str, str]:
    row = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        text=True,
    ).strip()
    uuid, name = [field.strip() for field in row.split(",", maxsplit=1)]
    return uuid, name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("+", maxsplit=1)[0].split(".")[:3])


def backward_names(loss: torch.Tensor) -> list[str]:
    names: list[str] = []
    visited = set()
    queue = [loss.grad_fn]
    while queue:
        node = queue.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        names.append(type(node).__name__)
        queue.extend(next_node for next_node, _index in node.next_functions)
    return names


def _gradient(parameter: torch.Tensor, name: str) -> dict[str, float | bool]:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"Missing or nonfinite gradient: {name}")
    rms = float(gradient.float().square().mean().sqrt())
    nonzero = bool((gradient != 0).any())
    if not math.isfinite(rms) or rms <= 0 or not nonzero:
        raise RuntimeError(f"Inactive gradient: {name}")
    return {"rms": rms, "nonzero": nonzero}


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(numerator / denominator)


def _mesa_reference_contract() -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(66066)
    shape = (2, 64, NUM_HEADS, HEAD_DIM)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    g_logits = torch.randn(
        shape[:-1], generator=generator, device="cuda", dtype=torch.float32
    ).requires_grad_()
    beta_logits = torch.randn(
        shape[:-1], generator=generator, device="cuda", dtype=torch.float32
    ).requires_grad_()
    lambda_logits = torch.randn(
        NUM_HEADS, HEAD_DIM, generator=generator, device="cuda", dtype=torch.float32
    ).requires_grad_()
    factor = 0.03 * torch.randn(
        2,
        NUM_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        generator=generator,
        device="cuda",
    )
    h_kk_init = (factor.transpose(-1, -2) @ factor).detach().requires_grad_()
    h_kv_init = (
        0.02
        * torch.randn(
            2,
            NUM_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        )
    ).requires_grad_()
    g = F.logsigmoid(g_logits)
    beta = torch.sigmoid(beta_logits)
    lamb = F.softplus(lambda_logits) + LAMBDA_LOWER_BOUND
    output, h_kk, h_kv = chunk_mesa_net(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        lamb=lamb,
        h_kk_init=h_kk_init,
        h_kv_init=h_kv_init,
        output_final_state=True,
        max_CG_iteration=MAX_CG_ITERATIONS,
        use_qk_l2norm_in_kernel=True,
    )
    exact_output, exact_h_kk, exact_h_kv = naive_mesa_net_exact(
        F.normalize(q.float(), dim=-1),
        F.normalize(k.float(), dim=-1),
        v.float(),
        g,
        lamb,
        beta,
        h_kk_init=h_kk_init,
        h_kv_init=h_kv_init,
    )
    parity = {
        "output_relative_rms": _relative_rms(output, exact_output),
        "hkk_relative_rms": _relative_rms(h_kk, exact_h_kk),
        "hkv_relative_rms": _relative_rms(h_kv, exact_h_kv),
    }
    if (
        parity["output_relative_rms"] > 0.05
        or parity["hkk_relative_rms"] > 0.01
        or parity["hkv_relative_rms"] > 0.01
    ):
        raise RuntimeError(f"Official Mesa/reference parity failed: {parity}")

    loss = output.float().square().mean()
    graph_names = backward_names(loss)
    if sum("ChunkMesaNetFunctionBackward" in name for name in graph_names) != 1:
        raise RuntimeError(f"Official Mesa backward missing: {sorted(graph_names)}")
    loss.backward()
    gradients = {
        "q": _gradient(q, "reference.q"),
        "k": _gradient(k, "reference.k"),
        "v": _gradient(v, "reference.v"),
        "decay": _gradient(g_logits, "reference.decay"),
        "write": _gradient(beta_logits, "reference.write"),
        "lambda": _gradient(lambda_logits, "reference.lambda"),
        "hkk_init": _gradient(h_kk_init, "reference.hkk_init"),
        "hkv_init": _gradient(h_kv_init, "reference.hkv_init"),
    }

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        permuted_output, permuted_hkk, permuted_hkv = chunk_mesa_net(
            q=q[:, :, permutation],
            k=k[:, :, permutation],
            v=v[:, :, permutation],
            g=g[:, :, permutation],
            beta=beta[:, :, permutation],
            lamb=lamb[permutation],
            h_kk_init=h_kk_init[:, permutation],
            h_kv_init=h_kv_init[:, permutation],
            output_final_state=True,
            max_CG_iteration=MAX_CG_ITERATIONS,
            use_qk_l2norm_in_kernel=True,
        )
    equivariance = {
        "output_relative_rms": _relative_rms(permuted_output[:, :, inverse], output),
        "hkk_relative_rms": _relative_rms(permuted_hkk[:, inverse], h_kk),
        "hkv_relative_rms": _relative_rms(permuted_hkv[:, inverse], h_kv),
    }
    if max(equivariance.values()) > 0.005:
        raise RuntimeError(f"Mesa head equivariance failed: {equivariance}")
    return {
        "parity": parity,
        "gradients": gradients,
        "head_equivariance": equivariance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-init", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-066 requires exactly CUDA index 0")
    gpu_uuid, gpu_name = _visible_gpu()
    device = torch.cuda.get_device_properties(0)
    if (
        gpu_name != args.expected_gpu_name
        or gpu_uuid != args.expected_gpu_uuid
        or device.name != args.expected_gpu_name
        or normalized_uuid(str(getattr(device, "uuid", "unavailable")))
        != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {gpu_uuid}, {gpu_name}")
    if _version(torch.__version__)[:2] != (2, 8):
        raise RuntimeError(f"Official Mesa requires PyTorch 2.8, got {torch.__version__}")
    if _version(triton.__version__) < (3, 4):
        raise RuntimeError(f"Official Mesa requires Triton >=3.4, got {triton.__version__}")
    if (
        not args.matched_init.is_file()
        or _sha256(args.matched_init) != EXPECTED_MATCHED_INIT_SHA256
    ):
        raise RuntimeError("Matched initialization drifted")
    provenance = official_mesa_provenance()
    if provenance["source_sha256"] != EXPECTED_MESA_SOURCE_SHA256:
        raise RuntimeError("Official Mesa provenance drifted")

    config = build_config(
        arm="future_seed_mesa",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    model = make_model(config, "future_seed_mesa")
    load_matched_parent_state(model, parent_state)
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyMesaFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Expected two exact official Mesa mixers")
    mixer_parameter_counts = [sum(p.numel() for p in mixer.parameters()) for mixer in mixers]
    if mixer_parameter_counts != [EXPECTED_MIXER_PARAMETERS, EXPECTED_MIXER_PARAMETERS]:
        raise RuntimeError(f"Mesa parameter accounting drifted: {mixer_parameter_counts}")
    if any(mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER for mixer in mixers):
        raise RuntimeError("Mesa state accounting drifted")

    reference = _mesa_reference_contract()

    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:4].cuda(), targets[:4].cuda()
    model = model.cuda().train()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = sum(
        "ChunkMesaNetFunctionBackward" in name for name in graph_names
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official Mesa backward paths, got {official_backward_count}"
        )
    producer_terminal_autograd = {
        "requires_grad": mixers[0].last_terminal_requires_grad,
        "grad_fn": mixers[0].last_terminal_grad_fn,
    }
    loss.backward()
    gradients = []
    for index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": index,
            "q": _gradient(layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(layer.v_proj.weight, f"layer{index}.v"),
            "q_conv": _gradient(layer.q_conv1d.weight, f"layer{index}.q_conv"),
            "k_conv": _gradient(layer.k_conv1d.weight, f"layer{index}.k_conv"),
            "decay": _gradient(layer.a_proj.weight, f"layer{index}.decay"),
            "write": _gradient(layer.b_proj.weight, f"layer{index}.write"),
            "lambda": _gradient(layer.lambda_params, f"layer{index}.lambda"),
            "output": _gradient(layer.o_proj.weight, f"layer{index}.output"),
            "output_gate": _gradient(layer.g_proj.weight, f"layer{index}.output_gate"),
        }
        if index == 1:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit,
                "layer1.future_seed",
            )
        gradients.append(row)

    generator = torch.Generator(device="cuda").manual_seed(66166)
    hidden = torch.randn(2, 64, 128, generator=generator, device="cuda")
    factor = 0.02 * torch.randn(
        2, NUM_HEADS, HEAD_DIM, HEAD_DIM, generator=generator, device="cuda"
    )
    incoming_hkk = factor.transpose(-1, -2) @ factor
    incoming_hkv = 0.02 * torch.randn(
        2, NUM_HEADS, HEAD_DIM, HEAD_DIM, generator=generator, device="cuda"
    )
    receiving = mixers[1].eval()
    with torch.no_grad():
        zero_output, zero_terminal = receiving.forward_with_state(
            hidden,
            initial_state=torch.zeros(
                2,
                2,
                NUM_HEADS,
                HEAD_DIM,
                HEAD_DIM,
                device="cuda",
            ),
        )
        seeded_output, seeded_terminal = receiving.forward_with_state(
            hidden,
            initial_state=torch.stack((incoming_hkk, incoming_hkv), dim=0),
        )
    incoming_dependency = {
        "output_relative_rms": _relative_rms(seeded_output, zero_output),
        "terminal_relative_rms": _relative_rms(seeded_terminal, zero_terminal),
        "finite": bool(
            torch.isfinite(seeded_output).all()
            and torch.isfinite(seeded_terminal).all()
        ),
    }
    if (
        incoming_dependency["output_relative_rms"] <= 1e-4
        or incoming_dependency["terminal_relative_rms"] <= 1e-4
        or not incoming_dependency["finite"]
    ):
        raise RuntimeError(f"Incoming Mesa state dependency failed: {incoming_dependency}")

    diagnostics = mesa_futureseed_diagnostics(model, inputs[:2])
    rows = diagnostics["per_layer"]
    if (
        diagnostics["active_layers"] != 2
        or diagnostics["active_futureseed_routes"] != 1
        or diagnostics["state_values_per_layer"] != EXPECTED_STATE_VALUES_PER_LAYER
        or diagnostics["max_cg_iterations"] != MAX_CG_ITERATIONS
        or diagnostics["lambda_lower_bound"] != LAMBDA_LOWER_BOUND
    ):
        raise RuntimeError(f"Mesa diagnostic geometry failed: {diagnostics}")
    for row in rows:
        if (
            row["state_rms"] <= 1e-4
            or row["state_board_std"] <= 0
            or row["hkk_rms"] <= 1e-4
            or row["hkv_rms"] <= 1e-4
            or row["hkk_symmetry_max_abs"] > 0.01
            or row["hkk_min_eigenvalue"] < -0.01
            or row["hkk_effective_rank"] <= 1
            or row["cg_output_relative_rms"] > 0.05
        ):
            raise RuntimeError(f"Mesa activation/stability failed: {row}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-066",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "runtime": {"torch": torch.__version__, "triton": triton.__version__},
        "external": provenance,
        "data_hashes": data_hashes,
        "parameters": {
            "model": sum(parameter.numel() for parameter in model.parameters()),
            "per_mixer": mixer_parameter_counts,
            "expected_per_mixer": EXPECTED_MIXER_PARAMETERS,
        },
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "official_backward_count": official_backward_count,
        "gradients": gradients,
        "producer_terminal_autograd": producer_terminal_autograd,
        "incoming_state_dependency": incoming_dependency,
        "reference": reference,
        "diagnostics": diagnostics,
        "fallback_operator_count": 0,
        "external_source_redistributed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
