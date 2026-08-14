from __future__ import annotations

import argparse
import copy
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
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.gdn2_linear_product_state import (
    FACTOR_DIM,
    LINEAR_DIM,
    PRODUCT_DIM,
    STATE_VALUES_PER_LAYER,
    TOTAL_KEY_DIM,
    LinearProductStateGatedDeltaNet2,
    ZoologyLinearProductStateFutureSeedMixer,
    linear_product_state_diagnostics,
    load_matched_parent_state,
    parent_parameter_hash,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)
from scripts.check_zoology_gdn2_log_spd import (
    EXPECTED_TEST_HASH,
    EXPECTED_TRAIN_HASH,
    EXPECTED_ZOOLOGY_SHA,
    backward_names,
    finite_max_abs_difference,
    git_head,
    normalized_uuid,
    python_tree_hash,
)


def _mixers(
    model: torch.nn.Module,
) -> list[ZoologyLinearProductStateFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        isinstance(mixer, ZoologyLinearProductStateFutureSeedMixer)
        for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two linear-product state mixers")
    return mixers


def _mapped_parent_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    return {
        name.replace(
            ".sequence_mixer.layer.base.",
            ".sequence_mixer.layer.",
        ): parameter
        for name, parameter in model.named_parameters()
    }


def _gradient_abs_max(parameter: torch.nn.Parameter, label: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {label}")
    gradient = parameter.grad.float()
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"Non-finite gradient: {label}")
    value = float(gradient.abs().max().item())
    if value <= 0:
        raise RuntimeError(f"Zero gradient: {label}")
    return value


def _projection_gradient_blocks(
    parameter: torch.nn.Parameter,
    label: str,
) -> dict[str, float]:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {label}")
    gradient = parameter.grad.float().reshape(4, 32, -1)
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"Non-finite gradient: {label}")
    result = {
        "factor_rows": float(gradient[:, : 2 * FACTOR_DIM].abs().max().item()),
        "native_only_rows": float(gradient[:, 2 * FACTOR_DIM :].abs().max().item()),
    }
    if min(result.values()) <= 0:
        raise RuntimeError(f"Inactive projection gradient block {label}: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-040 requires one visible CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if (
        os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1"
        or os.environ.get("FLA_CONV_BACKEND") != "triton"
    ):
        raise RuntimeError("Official backend dispatch/Triton contract failed")
    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if fla_root not in gdn2_source.parents:
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 source hash: {source_hash}")
    ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError(f"Unexpected GDN2 ops hash: {ops_hash}")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")

    control_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_linear_product_state_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(control_config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    control = make_model(control_config, "future_seed_gdn2")
    control_state = copy.deepcopy(control.state_dict())
    set_determinism(123)
    candidate = make_model(
        candidate_config,
        "future_seed_linear_product_state_gdn2",
    )
    load_matched_parent_state(candidate, control_state)
    if parent_parameter_hash(candidate) != parameter_hash(control):
        raise RuntimeError("Linear-product insertion changed parent initialization")
    parameter_counts = {
        "control": sum(parameter.numel() for parameter in control.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if parameter_counts["candidate"] != parameter_counts["control"]:
        raise RuntimeError(f"Linear-product state added parameters: {parameter_counts}")
    control_parameters = dict(control.named_parameters())
    candidate_parameters = _mapped_parent_parameters(candidate)
    if set(control_parameters) != set(candidate_parameters):
        raise RuntimeError("Candidate parent parameter names differ from control")
    parent_init_max_diff = max(
        finite_max_abs_difference(
            control_parameters[name],
            candidate_parameters[name],
            f"parent initialization {name}",
        )
        for name in control_parameters
    )
    if parent_init_max_diff != 0.0:
        raise RuntimeError("Candidate parent tensors differ from control")

    main_layers = [
        module for module in candidate.modules() if isinstance(module, GatedDeltaNet2)
    ]
    convolutions = [
        module for module in candidate.modules() if isinstance(module, ShortConvolution)
    ]
    if len(main_layers) != 2 or len(convolutions) != 6:
        raise RuntimeError("Official GDN2/short-convolution module counts drifted")

    generator = torch.Generator(device="cuda").manual_seed(52040)
    raw_q = torch.randn(
        3, 19, 4, LINEAR_DIM, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    raw_k = torch.randn(
        3, 19, 4, LINEAR_DIM, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    q, q_native, q_product, _q_axes = (
        LinearProductStateGatedDeltaNet2.address_features(raw_q)
    )
    k, k_native, k_product, _k_axes = (
        LinearProductStateGatedDeltaNet2.address_features(raw_k)
    )
    native_q_max_diff = finite_max_abs_difference(
        q[..., :LINEAR_DIM],
        q_native.to(q.dtype),
        "native Q block",
    )
    native_k_max_diff = finite_max_abs_difference(
        k[..., :LINEAR_DIM],
        k_native.to(k.dtype),
        "native K block",
    )
    q_block_norm_error = max(
        float((q_native.norm(dim=-1) - 1).abs().max().item()),
        float((q_product.norm(dim=-1) - 1).abs().max().item()),
    )
    k_block_norm_error = max(
        float((k_native.norm(dim=-1) - 1).abs().max().item()),
        float((k_product.norm(dim=-1) - 1).abs().max().item()),
    )
    similarity_error = finite_max_abs_difference(
        (q.float() * k.float()).sum(dim=-1),
        (q_native * k_native).sum(dim=-1)
        + (q_product * k_product).sum(dim=-1),
        "direct-sum similarity",
    )
    if (
        native_q_max_diff != 0.0
        or native_k_max_diff != 0.0
        or q_block_norm_error > 1e-5
        or k_block_norm_error > 1e-5
        or similarity_error > 0.02
    ):
        raise RuntimeError("Linear/product address algebra failed")

    raw_decay = -torch.rand(
        3, 19, 4, LINEAR_DIM, generator=generator, device="cuda"
    )
    raw_erase = torch.rand(
        3, 19, 4, LINEAR_DIM, generator=generator, device="cuda"
    )
    lifted_decay, lifted_erase = LinearProductStateGatedDeltaNet2.transition_features(
        raw_decay,
        raw_erase,
    )
    native_decay_max_diff = finite_max_abs_difference(
        lifted_decay[..., :LINEAR_DIM], raw_decay, "native decay block"
    )
    native_erase_max_diff = finite_max_abs_difference(
        lifted_erase[..., :LINEAR_DIM], raw_erase, "native erase block"
    )
    if (
        native_decay_max_diff != 0.0
        or native_erase_max_diff != 0.0
        or float(lifted_decay.max().item()) > 0.0
        or float(lifted_erase.min().item()) < 0.0
        or float(lifted_erase.max().item()) > 1.0
    ):
        raise RuntimeError("Linear/product transition algebra failed")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted, *_ = LinearProductStateGatedDeltaNet2.address_features(
        raw_q[:, :, permutation]
    )
    head_permutation_error = finite_max_abs_difference(
        q[:, :, permutation], permuted, "head permutation"
    )
    if head_permutation_error != 0.0:
        raise RuntimeError("Head permutation equivariance failed")
    changed_raw_q = raw_q.clone()
    changed_raw_q[..., 0] += 0.5
    _changed_q, _changed_native, changed_product, _changed_axes = (
        LinearProductStateGatedDeltaNet2.address_features(changed_raw_q)
    )
    product_dependency = float(
        (changed_product - q_product).abs().mean().item()
    )
    if product_dependency <= 1e-5:
        raise RuntimeError("Exact product block is independent of its factors")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    candidate = candidate.cuda().eval()
    mixers = _mixers(candidate)
    with torch.no_grad():
        logits = candidate(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Linear-product full model produced non-finite logits")
    diagnostics = linear_product_state_diagnostics(candidate)
    if (
        diagnostics["state_values_per_layer"] != STATE_VALUES_PER_LAYER
        or diagnostics["total_key_dim"] != TOTAL_KEY_DIM
        or diagnostics["official_scans_per_layer"] != 1
        or diagnostics["new_parameters"] != 0
    ):
        raise RuntimeError("Linear-product topology diagnostics drifted")

    hidden = torch.randn(2, 128, 128, generator=generator, device="cuda")
    zero_state = torch.zeros(2, 4, TOTAL_KEY_DIM, 32, device="cuda")
    product_state = zero_state.clone()
    product_state[..., LINEAR_DIM:, :] = 0.05 * torch.randn(
        2, 4, PRODUCT_DIM, 32, generator=generator, device="cuda"
    )
    with torch.no_grad():
        zero_output, zero_terminal = mixers[1].forward_with_state(
            hidden, initial_state=zero_state
        )
        product_output, product_terminal = mixers[1].forward_with_state(
            hidden, initial_state=product_state
        )
    incoming_output_dependency = float(
        (product_output - zero_output).float().square().mean().sqrt().item()
    )
    incoming_state_dependency = float(
        (product_terminal - zero_terminal).float().square().mean().sqrt().item()
    )
    if incoming_output_dependency <= 1e-5 or incoming_state_dependency <= 1e-5:
        raise RuntimeError("Product-state incoming dependency is inactive")
    if tuple(product_terminal.shape) != (2, 4, TOTAL_KEY_DIM, 32):
        raise RuntimeError(f"Unexpected terminal shape: {tuple(product_terminal.shape)}")

    candidate.train().zero_grad(set_to_none=True)
    train_logits = candidate(inputs)
    graph = backward_names(train_logits)
    mask = targets != -100
    F.cross_entropy(train_logits[mask], targets[mask]).backward()
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official GDN2 backwards, got {official_backward_count}"
        )
    gradient_blocks = []
    for layer_index, mixer in enumerate(mixers):
        base = mixer.layer.base
        gradient_blocks.append(
            {
                "layer": layer_index,
                "q": _projection_gradient_blocks(base.q_proj.weight, "q_proj"),
                "k": _projection_gradient_blocks(base.k_proj.weight, "k_proj"),
                "decay": _projection_gradient_blocks(
                    base.f_proj[1].weight,
                    "f_proj[1]",
                ),
                "erase": _projection_gradient_blocks(base.b_proj.weight, "b_proj"),
                "value": _gradient_abs_max(base.v_proj.weight, "v_proj"),
                "write": _gradient_abs_max(base.w_proj.weight, "w_proj"),
            }
        )

    conv_backends = [mixer.layer.base.q_conv1d.backend for mixer in mixers]
    if any(backend != "triton" for backend in conv_backends):
        raise RuntimeError(f"Short convolution fallback detected: {conv_backends}")
    result = {
        "device": device.name,
        "device_uuid": device_uuid,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "fla_sha": PINNED_FLA_SHA,
        "fla_gdn2_source_sha256": source_hash,
        "fla_gdn2_ops_sha256": ops_hash,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_delta": 0,
        "parent_init_parameter_hash": parent_parameter_hash(candidate),
        "parent_init_max_diff": parent_init_max_diff,
        "linear_dim": LINEAR_DIM,
        "factor_dim": FACTOR_DIM,
        "product_dim": PRODUCT_DIM,
        "total_key_dim": TOTAL_KEY_DIM,
        "state_values_per_layer": STATE_VALUES_PER_LAYER,
        "native_q_max_diff": native_q_max_diff,
        "native_k_max_diff": native_k_max_diff,
        "native_decay_max_diff": native_decay_max_diff,
        "native_erase_max_diff": native_erase_max_diff,
        "q_block_norm_error": q_block_norm_error,
        "k_block_norm_error": k_block_norm_error,
        "similarity_error": similarity_error,
        "head_permutation_error": head_permutation_error,
        "product_dependency": product_dependency,
        "incoming_output_dependency": incoming_output_dependency,
        "incoming_state_dependency": incoming_state_dependency,
        "official_backward_count": official_backward_count,
        "gradient_blocks": gradient_blocks,
        "diagnostics": diagnostics,
        "conv_backends": conv_backends,
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
    }
    if not math.isfinite(result["incoming_output_dependency"]):
        raise RuntimeError("Incoming product-state dependency is non-finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
