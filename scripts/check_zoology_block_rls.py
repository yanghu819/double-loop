from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_block_rls import (
    BLOCK_SIZE,
    EXPECTED_PARAMETER_DELTA,
    TRANSIENT_GEOMETRY_VALUES_PER_LAYER,
    BlockRLSGatedDeltaNet2,
    ZoologyBlockRLSGDN2FutureSeedMixer,
    block_rls_precision_torch,
    block_rls_precision_triton,
    constrained_block_rls_address,
    load_matched_parent_state,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    finite_module_gradient_rms,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_DPLR_TREE_SHA256 = (
    "686ca42b82ba098f9187fcacfe8ab689b9038938b1edae2809a21b6f03ec8643"
)
EXPECTED_FLA_ROOT = Path(
    "/huyang2/double-loop/.cache/fla-versions/"
    "9c8e42e762fce087c27b673af4922795d9edb85e-0280db310981915e"
)
EXPECTED_PARENT_PARAMETERS = 661_584
EXPECTED_PARENT_PARAMETER_HASH = (
    "3e8fe038f1401735168783c2de1d9217a8807e88685ee12010142fc7b05e4c44"
)
MODEL_LAYERS = 2
MODEL_HEADS = 4
HEAD_DIM = 32


def _visible_gpu() -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU row, got {rows}")
    uuid, name = (part.strip() for part in rows[0].split(",", maxsplit=1))
    return uuid, name


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((delta / scale).item())


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _finite_gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(gradient.square().mean().sqrt().item())


def _kernel_contract() -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(54054)
    key = F.normalize(
        torch.randn(
            2,
            17,
            MODEL_HEADS,
            HEAD_DIM,
            generator=generator,
            device="cuda",
        ),
        dim=-1,
    ).to(torch.bfloat16)
    log_decay = (
        -0.01
        - 0.08
        * torch.rand(
            key.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    )
    triton_rows = block_rls_precision_triton(key, log_decay)
    torch_rows = block_rls_precision_torch(key, log_decay)
    errors = {
        name: _max_abs(triton_value, torch_value)
        for name, triton_value, torch_value in zip(
            ("transported", "terminal", "minimum_denominator"),
            triton_rows,
            torch_rows,
        )
    }
    if max(errors.values()) > 5e-5:
        raise RuntimeError(f"Block-RLS Triton/reference parity failed: {errors}")

    transported, terminal, minimum = triton_rows
    erase = torch.sigmoid(
        torch.randn(key.shape, generator=generator, device="cuda")
    ).to(torch.bfloat16)
    mix = torch.full((MODEL_HEADS,), 0.25, device="cuda")
    address, diagnostics = constrained_block_rls_address(
        key,
        erase,
        transported,
        torch.atanh(mix),
    )
    erase_address = diagnostics["erase_address"]
    constraint_error = float(
        (
            (erase_address * address).sum(dim=-1)
            - (erase_address * key.float()).sum(dim=-1)
        )
        .abs()
        .max()
        .item()
    )
    eigenvalues = torch.linalg.eigvalsh(
        0.5 * (terminal + terminal.transpose(-1, -2))
    )
    if constraint_error > 1e-5:
        raise RuntimeError(f"Committed-response constraint failed: {constraint_error}")
    if float(minimum.min()) < 1e-4:
        raise RuntimeError("Block-RLS denominator lost its positive bound")
    if float(eigenvalues.min()) < 1e-4 or float(eigenvalues.max()) > 1.001:
        raise RuntimeError("Block-RLS terminal precision escaped its SPD bounds")

    differentiable_key = key.detach().clone().requires_grad_(True)
    differentiable_address, _ = constrained_block_rls_address(
        differentiable_key,
        erase,
        transported.detach(),
        torch.atanh(mix),
    )
    differentiable_address.square().mean().backward()
    if differentiable_key.grad is None or not torch.isfinite(
        differentiable_key.grad
    ).all():
        raise RuntimeError("Current-key local Jacobian is missing or nonfinite")
    current_key_gradient_rms = float(
        differentiable_key.grad.float().square().mean().sqrt().item()
    )
    if current_key_gradient_rms == 0.0:
        raise RuntimeError("Current-key local Jacobian is inactive")

    head_permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted = block_rls_precision_triton(
        key[:, :, head_permutation],
        log_decay[:, :, head_permutation],
    )
    head_errors = {
        "transported": _max_abs(
            transported[:, :, head_permutation], permuted[0]
        ),
        "terminal": _max_abs(terminal[:, head_permutation], permuted[1]),
        "minimum_denominator": _max_abs(
            minimum[:, head_permutation], permuted[2]
        ),
    }
    block_permutation = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4], device="cuda")
    key_blocks = key.reshape(2, 17, MODEL_HEADS, -1, BLOCK_SIZE)
    decay_blocks = log_decay.reshape_as(key_blocks)
    block_rows = block_rls_precision_triton(
        key_blocks[:, :, :, block_permutation].reshape_as(key),
        decay_blocks[:, :, :, block_permutation].reshape_as(log_decay),
    )
    block_errors = {
        "transported": _max_abs(
            transported[:, :, :, block_permutation], block_rows[0]
        ),
        "terminal": _max_abs(terminal[:, :, block_permutation], block_rows[1]),
        "minimum_denominator": _max_abs(
            minimum[:, :, block_permutation], block_rows[2]
        ),
    }
    if max((*head_errors.values(), *block_errors.values())) > 5e-5:
        raise RuntimeError(
            f"Block/head permutation equivariance failed: {head_errors}, {block_errors}"
        )

    changed_key = key.clone()
    changed_key[:, 0] = F.normalize(
        torch.randn(
            changed_key[:, 0].shape,
            generator=generator,
            device="cuda",
        ),
        dim=-1,
    ).to(changed_key.dtype)
    changed = block_rls_precision_triton(changed_key, log_decay)[0]
    prefix_dependency = _relative_rms(changed[:, -1], transported[:, -1])
    if prefix_dependency <= 1e-4:
        raise RuntimeError("Block-RLS address does not depend on causal prefix keys")
    return {
        "reference_max_errors": errors,
        "head_permutation_max_errors": head_errors,
        "block_permutation_max_errors": block_errors,
        "prefix_dependency_relative_rms": prefix_dependency,
        "constraint_error_max": constraint_error,
        "minimum_denominator": float(minimum.min().item()),
        "terminal_eigen_min": float(eigenvalues.min().item()),
        "terminal_eigen_max": float(eigenvalues.max().item()),
        "current_key_gradient_rms": current_key_gradient_rms,
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
        raise RuntimeError("P-GDN3-054 requires exactly CUDA index 0")
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
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if fla_root != EXPECTED_FLA_ROOT:
        raise RuntimeError(f"Unexpected pinned FLA root: {fla_root}")
    dplr_root = fla_root / "fla" / "ops" / "generalized_delta_rule" / "dplr"
    dplr_module = importlib.import_module("fla.ops.generalized_delta_rule.dplr.chunk")
    if getattr(dplr_module, "chunk_dplr_delta_rule") is not chunk_dplr_delta_rule:
        raise RuntimeError("Exported DPLR chunk identity changed")
    dplr_source = Path(dplr_module.__file__).resolve()
    if dplr_source != dplr_root / "chunk.py":
        raise RuntimeError(f"Unexpected DPLR source: {dplr_source}")
    dplr_tree_sha256, dplr_source_hashes = python_tree_hash(dplr_root)
    if dplr_tree_sha256 != EXPECTED_DPLR_TREE_SHA256:
        raise RuntimeError(f"Official DPLR tree changed: {dplr_tree_sha256}")
    carrier_sources = {
        "gdn2": Path(inspect.getfile(GatedDeltaNet2)).resolve(),
        "short_convolution": Path(inspect.getfile(ShortConvolution)).resolve(),
    }
    if any(fla_root not in source.parents for source in carrier_sources.values()):
        raise RuntimeError(f"Carrier escaped pinned FLA: {carrier_sources}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_sha = git_head(zoology_root)
    if zoology_sha != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError(f"Unexpected Zoology SHA: {zoology_sha}")

    candidate_config = build_config(
        arm="future_seed_block_rls_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(candidate_config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {
        "train": EXPECTED_TRAIN_HASH,
        "test": EXPECTED_TEST_HASH,
    }:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_block_rls_gdn2")
    load_matched_parent_state(candidate, parent_state)
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    native.load_state_dict(parent_state, strict=True)
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    native_parameters = sum(p.numel() for p in native.parameters())
    if native_parameters != EXPECTED_PARENT_PARAMETERS:
        raise RuntimeError(f"Native parameter count drifted: {native_parameters}")
    if candidate_parameters - native_parameters != EXPECTED_PARAMETER_DELTA:
        raise RuntimeError("Block-RLS parameter delta changed")
    if parent_parameter_hash(candidate) != parameter_hash(native):
        raise RuntimeError("Candidate parent initialization differs from frozen native")
    if parent_parameter_hash(candidate) != EXPECTED_PARENT_PARAMETER_HASH:
        raise RuntimeError("Frozen parent parameter hash changed")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != MODEL_LAYERS or not all(
        isinstance(mixer, ZoologyBlockRLSGDN2FutureSeedMixer) for mixer in mixers
    ):
        raise RuntimeError("Contract requires two block-RLS layers")
    provenance = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        if not isinstance(layer, BlockRLSGatedDeltaNet2):
            raise RuntimeError("Block-RLS wrapper changed")
        if type(layer.base) is not GatedDeltaNet2:
            raise RuntimeError("Carrier is not exact official GatedDeltaNet2")
        convolutions = {
            name: module.backend
            for name, module in layer.named_modules()
            if isinstance(module, ShortConvolution)
        }
        if len(convolutions) != 3 or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"ShortConv fallback in layer {layer_index}: {convolutions}")
        provenance.append(
            {
                "layer": layer_index,
                "carrier": type(layer.base).__qualname__,
                "operator": "ChunkDPLRDeltaRuleFunctionBackward",
                "convolution_backends": convolutions,
            }
        )

    kernel_contract = _kernel_contract()
    candidate = candidate.cuda().train()
    native = native.cuda().eval()
    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate.eval()(inputs)
    model_parity = _relative_rms(candidate_logits, native_logits)
    if model_parity > 0.10:
        raise RuntimeError(f"Zero-mix model parity failed: {model_parity}")

    hidden = torch.randn(2, 64, 128, device="cuda")
    incoming = 0.05 * torch.randn(2, MODEL_HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
    with torch.no_grad():
        native_output, native_state = native.backbone.layers[0].sequence_mixer.forward_with_state(
            hidden,
            initial_state=incoming,
        )
        candidate_output, candidate_state = mixers[0].forward_with_state(
            hidden,
            initial_state=incoming,
        )
    incoming_parity = {
        "output_relative_rms": _relative_rms(candidate_output, native_output),
        "state_relative_rms": _relative_rms(candidate_state, native_state),
    }
    if max(incoming_parity.values()) > 0.10:
        raise RuntimeError(f"Nonzero-state parity failed: {incoming_parity}")

    candidate.train().zero_grad(set_to_none=True)
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkDPLRDeltaRuleFunctionBackward")
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(f"Expected two official DPLR paths, got {official_backward_count}")
    loss.backward()
    gradient_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": layer_index,
            "mix": _finite_gradient(
                layer.block_rls_mix_logit,
                f"layer{layer_index}.block_rls_mix_logit",
            ),
        }
        if not torch.all(layer.block_rls_mix_logit.grad.detach() != 0):
            raise RuntimeError(f"Not every mix path has a gradient in layer {layer_index}")
        for name in ("q", "k", "v", "f", "b", "w"):
            row[name] = finite_module_gradient_rms(
                getattr(layer.base, f"{name}_proj"),
                f"layer{layer_index}.{name}_proj",
            )
        gradient_rows.append(row)

    with torch.no_grad():
        baseline_output, baseline_state = mixers[0].forward_with_state(
            hidden,
            initial_state=incoming,
        )
        mixers[0].layer.block_rls_mix_logit.fill_(math.atanh(0.25))
        opened_output, opened_state = mixers[0].forward_with_state(
            hidden,
            initial_state=incoming,
        )
        mixers[0].layer.block_rls_mix_logit.zero_()
    opened_dependency = {
        "output_relative_rms": _relative_rms(opened_output, baseline_output),
        "state_relative_rms": _relative_rms(opened_state, baseline_state),
    }
    if min(opened_dependency.values()) <= 1e-4:
        raise RuntimeError(f"Opened block-RLS path is inactive: {opened_dependency}")

    result = {
        "status": "passed",
        "plan": "P-GDN3-054",
        "gpu": {"name": gpu_name, "uuid": gpu_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "official_dplr_root": str(dplr_root),
        "official_dplr_tree_sha256": dplr_tree_sha256,
        "official_dplr_source_sha256": dplr_source_hashes,
        "official_carrier_sources": {
            name: str(source) for name, source in carrier_sources.items()
        },
        "zoology_sha": zoology_sha,
        "data_hashes": data_hashes,
        "candidate_parameters": candidate_parameters,
        "native_parameters": native_parameters,
        "parameter_delta": candidate_parameters - native_parameters,
        "persistent_state_delta": 0,
        "transient_geometry_values_per_layer": TRANSIENT_GEOMETRY_VALUES_PER_LAYER,
        "official_scans_per_layer": 1,
        "parent_parameter_hash": parent_parameter_hash(candidate.cpu()),
        "native_parameter_hash": parameter_hash(native.cpu()),
        "official_module_provenance": provenance,
        "official_backward_count": official_backward_count,
        "kernel_contract": kernel_contract,
        "zero_mix_model_output_relative_rms": model_parity,
        "zero_mix_nonzero_incoming_parity": incoming_parity,
        "gradient_rms": gradient_rows,
        "opened_path_dependency": opened_dependency,
        "current_key_local_jacobian": True,
        "history_gradient": "stopped",
        "fallback": False,
    }
    if result["parent_parameter_hash"] != result["native_parameter_hash"]:
        raise RuntimeError("Parent hash changed while serializing contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
