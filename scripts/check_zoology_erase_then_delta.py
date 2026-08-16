from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.modules.convolution import ShortConvolution
from fla.ops.gated_delta_product import chunk_gated_delta_product
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.erase_then_delta_futureseed import (
    HEAD_DIM,
    NUM_HEADS,
    NUM_MICROSTEPS,
    STATE_VALUES_PER_LAYER,
    EraseThenDelta,
    ZoologyEraseThenDeltaFutureSeedMixer,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_LAYER_SHA = "5bfc10f4a8b51d5fe12236e00080ba5a1340dcd5488c6dc06b03a91d47a52199"
EXPECTED_CHUNK_SHA = "2f01e53c68c83d43def22845950fdd86b39d5cb8e2559df6db3fa4958fe9f032"
EXPECTED_PARAMETERS = 632_408
EXPECTED_MIXER_PARAMETERS = 101_932
MODEL_LAYERS = 2


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((numerator / denominator).item())


def backward_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    seen = set()
    stack = [tensor.grad_fn]
    while stack:
        function = stack.pop()
        if function is None or function in seen:
            continue
        seen.add(function)
        names.append(type(function).__name__)
        stack.extend(next_function for next_function, _ in function.next_functions)
    return names


def finite_gradient_rms(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    rms = float(gradient.float().square().mean().sqrt().item())
    if rms <= 1e-9:
        raise RuntimeError(f"{label} gradient is inactive: {rms}")
    return rms


def split_gradient_rms(
    parameter: torch.nn.Parameter,
    *,
    chunks: int,
    label: str,
) -> list[float]:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    rows = []
    for index, chunk in enumerate(gradient.chunk(chunks, dim=0)):
        rms = float(chunk.float().square().mean().sqrt().item())
        if rms <= 1e-9:
            raise RuntimeError(f"{label}[{index}] gradient is inactive: {rms}")
        rows.append(rms)
    return rows


def swap_microsteps(tensor: torch.Tensor) -> torch.Tensor:
    batch, expanded_length = tensor.shape[:2]
    if expanded_length % NUM_MICROSTEPS:
        raise ValueError("Expanded transform length is not divisible by two")
    return (
        tensor.reshape(
            batch,
            expanded_length // NUM_MICROSTEPS,
            NUM_MICROSTEPS,
            *tensor.shape[2:],
        )
        .flip(2)
        .reshape_as(tensor)
    )


def explicit_recurrence(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = F.normalize(q.float(), dim=-1)
    k = F.normalize(k.float(), dim=-1).reshape(
        k.shape[0], q.shape[1], NUM_MICROSTEPS, *k.shape[2:]
    )
    v = v.float().reshape(
        v.shape[0], q.shape[1], NUM_MICROSTEPS, *v.shape[2:]
    )
    beta = beta.float().reshape(
        beta.shape[0], q.shape[1], NUM_MICROSTEPS, *beta.shape[2:]
    )
    state = initial_state.float().clone()
    outputs = []
    for token in range(q.shape[1]):
        state = state * g[:, token].float().exp()[:, :, None, None]
        for microstep in range(NUM_MICROSTEPS):
            key = k[:, token, microstep]
            value = v[:, token, microstep]
            prediction = torch.einsum("bhk,bhkv->bhv", key, state)
            correction = value - prediction
            state = state + (
                beta[:, token, microstep, :, None, None]
                * key[:, :, :, None]
                * correction[:, :, None, :]
            )
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q[:, token], state)
            * (q.shape[-1] ** -0.5)
        )
    return torch.stack(outputs, dim=1), state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-067 requires exactly CUDA index 0")
    device = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(device, "uuid", "unavailable"))
    if (
        device.name != args.expected_gpu_name
        or normalized_uuid(device_uuid) != normalized_uuid(args.expected_gpu_uuid)
    ):
        raise RuntimeError(f"Unexpected GPU: {device.name} {device_uuid}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA is missing")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("Short convolution must use Triton")

    fla_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    official_layer_module = importlib.import_module("fla.layers.gated_deltaproduct")
    chunk_module = importlib.import_module("fla.ops.gated_delta_product.chunk")
    layer_source = Path(official_layer_module.__file__).resolve()
    chunk_source = Path(chunk_module.__file__).resolve()
    if chunk_module.chunk_gated_delta_product is not chunk_gated_delta_product:
        raise RuntimeError("Exported official product function identity changed")
    if layer_source != fla_root / "fla/layers/gated_deltaproduct.py":
        raise RuntimeError(f"Unexpected official layer source: {layer_source}")
    if chunk_source != fla_root / "fla/ops/gated_delta_product/chunk.py":
        raise RuntimeError(f"Unexpected official chunk source: {chunk_source}")
    if sha256(layer_source) != EXPECTED_LAYER_SHA:
        raise RuntimeError("Official GatedDeltaProduct layer source changed")
    if sha256(chunk_source) != EXPECTED_CHUNK_SHA:
        raise RuntimeError("Official GatedDeltaProduct chunk source changed")
    if os.environ["FLA_GATED_DELTA_PRODUCT_SOURCE_SHA256"] != EXPECTED_LAYER_SHA:
        raise RuntimeError("Registered layer source hash changed")
    if os.environ["FLA_GATED_DELTA_PRODUCT_CHUNK_SHA256"] != EXPECTED_CHUNK_SHA:
        raise RuntimeError("Registered chunk source hash changed")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    config = build_config(
        arm="future_seed_erase_then_delta",
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
    if data_hashes != {
        "train": EXPECTED_TRAIN_HASH,
        "test": EXPECTED_TEST_HASH,
    }:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    set_determinism(123)
    model = make_model(config, "future_seed_erase_then_delta")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count changed: {parameter_count}")

    mixers: list[ZoologyEraseThenDeltaFutureSeedMixer] = []
    provenance = []
    for layer_index, block in enumerate(model.backbone.layers):
        mixer = block.sequence_mixer
        if type(mixer) is not ZoologyEraseThenDeltaFutureSeedMixer:
            raise RuntimeError(f"Layer {layer_index} mixer type changed: {type(mixer)}")
        if type(mixer.layer) is not EraseThenDelta:
            raise RuntimeError(f"Layer {layer_index} recurrence type changed")
        layer = mixer.layer
        if (
            layer.mode != "chunk"
            or layer.num_householder != NUM_MICROSTEPS
            or not layer.use_forget_gate
            or layer.allow_neg_eigval
            or layer.num_heads != NUM_HEADS
            or layer.num_v_heads != NUM_HEADS
            or layer.head_k_dim != HEAD_DIM
            or layer.head_v_dim != HEAD_DIM
        ):
            raise RuntimeError(f"Layer {layer_index} recurrence configuration drifted")
        mixer_parameters = sum(parameter.numel() for parameter in mixer.parameters())
        if mixer_parameters != EXPECTED_MIXER_PARAMETERS:
            raise RuntimeError(
                f"Layer {layer_index} mixer parameter count changed: {mixer_parameters}"
            )
        if mixer.state_size() != STATE_VALUES_PER_LAYER:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        convolutions = {
            name: getattr(layer, name).backend
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        if not all(
            isinstance(getattr(layer, name), ShortConvolution)
            for name in convolutions
        ) or set(convolutions.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution fallback: {convolutions}")
        provenance.append(
            {
                "layer": layer_index,
                "mixer_type": type(mixer).__qualname__,
                "recurrence_type": type(layer).__qualname__,
                "official_operator": "chunk_gated_delta_product",
                "microsteps": NUM_MICROSTEPS,
                "state_values": mixer.state_size(),
                "convolution_backends": convolutions,
            }
        )
        mixers.append(mixer)
    if len(mixers) != MODEL_LAYERS:
        raise RuntimeError(f"Expected two EDA layers, got {len(mixers)}")

    model = model.cuda().train()
    inputs, targets, _slices = next(iter(test_loader))
    inputs, targets = inputs[:2].cuda(), targets[:2].cuda()
    for mixer in mixers:
        mixer.layer.set_capture(True)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(inputs)
    finally:
        for mixer in mixers:
            mixer.layer.set_capture(False)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Full P-GDN3-067 logits are non-finite")
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count(
        "ChunkGatedDeltaProductFunctionBackward"
    )
    if official_backward_count != MODEL_LAYERS:
        raise RuntimeError(
            f"Expected two official product backward paths, got {official_backward_count}"
        )
    loss.backward()

    gradient_rows = []
    capture_rows = []
    for layer_index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": layer_index,
            "erase_write_key_rms": split_gradient_rms(
                layer.k_proj.weight,
                chunks=NUM_MICROSTEPS,
                label=f"layer{layer_index}.k_proj",
            ),
            "erase_write_gate_rms": split_gradient_rms(
                layer.b_proj.weight,
                chunks=NUM_MICROSTEPS,
                label=f"layer{layer_index}.b_proj",
            ),
            "query_rms": finite_gradient_rms(
                layer.q_proj.weight, f"layer{layer_index}.q_proj"
            ),
            "value_rms": finite_gradient_rms(
                layer.v_proj.weight, f"layer{layer_index}.v_proj"
            ),
            "decay_rms": finite_gradient_rms(
                layer.a_proj.weight, f"layer{layer_index}.a_proj"
            ),
            "output_gate_rms": finite_gradient_rms(
                layer.g_proj.weight, f"layer{layer_index}.g_proj"
            ),
            "output_projection_rms": finite_gradient_rms(
                layer.o_proj.weight, f"layer{layer_index}.o_proj"
            ),
        }
        if layer_index == 1:
            row["future_seed_logit_rms"] = finite_gradient_rms(
                mixer.future_seed_logit, "layer1.future_seed_logit"
            )
        gradient_rows.append(row)
        captures = layer.captures
        if set(captures) != {
            "initial_state",
            "pair_key",
            "pair_value",
            "pair_beta",
            "decay",
            "terminal_state",
        }:
            raise RuntimeError(f"Layer {layer_index} capture changed: {set(captures)}")
        first_payload_max_abs = float(captures["pair_value"][:, :, 0].abs().max())
        if first_payload_max_abs != 0.0:
            raise RuntimeError("The erase microstep must have exactly zero payload")
        capture_rows.append(
            {
                "layer": layer_index,
                "first_payload_max_abs": first_payload_max_abs,
                "terminal_state_finite": bool(
                    torch.isfinite(captures["terminal_state"]).all()
                ),
            }
        )

    generator = torch.Generator(device="cuda").manual_seed(670)
    q = torch.randn(
        2, 64, NUM_HEADS, HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    erase_key = torch.randn(
        2, 64, NUM_HEADS, HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    write_key = torch.randn_like(erase_key)
    key = torch.stack((erase_key, write_key), dim=2).flatten(1, 2).requires_grad_()
    payload = torch.randn(
        2, 64, NUM_HEADS, HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.stack((torch.zeros_like(payload), payload), dim=2).flatten(1, 2)
    value.requires_grad_()
    g = (
        -0.01
        * torch.sigmoid(
            torch.randn(2, 64, NUM_HEADS, generator=generator, device="cuda")
        )
    ).requires_grad_()
    beta = torch.sigmoid(
        torch.randn(
            2,
            128,
            NUM_HEADS,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
    ).requires_grad_()
    initial_state = torch.randn(
        2,
        NUM_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, terminal_state = chunk_gated_delta_product(
            q=q,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            num_householder=NUM_MICROSTEPS,
            use_qk_l2norm_in_kernel=True,
        )
        zero_output, zero_terminal = chunk_gated_delta_product(
            q=q.detach(),
            k=key.detach(),
            v=value.detach(),
            g=g.detach(),
            beta=beta.detach(),
            initial_state=torch.zeros_like(initial_state),
            output_final_state=True,
            num_householder=NUM_MICROSTEPS,
            use_qk_l2norm_in_kernel=True,
        )
        swapped_output, swapped_terminal = chunk_gated_delta_product(
            q=q.detach(),
            k=swap_microsteps(key.detach()),
            v=swap_microsteps(value.detach()),
            g=g.detach(),
            beta=swap_microsteps(beta.detach()),
            initial_state=initial_state.detach(),
            output_final_state=True,
            num_householder=NUM_MICROSTEPS,
            use_qk_l2norm_in_kernel=True,
        )
    reference_output, reference_terminal = explicit_recurrence(
        q=q.detach(),
        k=key.detach(),
        v=value.detach(),
        g=g.detach(),
        beta=beta.detach(),
        initial_state=initial_state.detach(),
    )
    output_parity = relative_rms(output, reference_output)
    terminal_parity = relative_rms(terminal_state, reference_terminal)
    if output_parity > 0.03 or terminal_parity > 0.03:
        raise RuntimeError(
            f"Explicit EDA recurrence parity failed: {output_parity} {terminal_parity}"
        )
    incoming_output_dependency = relative_rms(output, zero_output)
    incoming_state_dependency = relative_rms(terminal_state, zero_terminal)
    order_output_dependency = relative_rms(output, swapped_output)
    order_state_dependency = relative_rms(terminal_state, swapped_terminal)
    if min(incoming_output_dependency, incoming_state_dependency) <= 1e-4:
        raise RuntimeError("EDA recurrence does not depend on incoming state")
    if min(order_output_dependency, order_state_dependency) <= 1e-4:
        raise RuntimeError("EDA recurrence is insensitive to microstep order")

    synthetic_loss = output.float().square().mean() + terminal_state.square().mean()
    synthetic_loss.backward()
    synthetic_gradient_rms = {}
    for name, tensor in {
        "q": q,
        "key": key,
        "value": value,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
    }.items():
        gradient = tensor.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Synthetic {name} gradient is not finite")
        rms = float(gradient.float().square().mean().sqrt().item())
        if rms <= 1e-9:
            raise RuntimeError(f"Synthetic {name} gradient is inactive: {rms}")
        synthetic_gradient_rms[name] = rms

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        permuted_output, permuted_state = chunk_gated_delta_product(
            q=q.detach().index_select(2, permutation),
            k=key.detach().index_select(2, permutation),
            v=value.detach().index_select(2, permutation),
            g=g.detach().index_select(2, permutation),
            beta=beta.detach().index_select(2, permutation),
            initial_state=initial_state.detach().index_select(1, permutation),
            output_final_state=True,
            num_householder=NUM_MICROSTEPS,
            use_qk_l2norm_in_kernel=True,
        )
    head_output_error = relative_rms(
        permuted_output, output.detach().index_select(2, permutation)
    )
    head_state_error = relative_rms(
        permuted_state, terminal_state.detach().index_select(1, permutation)
    )
    if max(head_output_error, head_state_error) > 0.005:
        raise RuntimeError(
            f"Head permutation equivariance failed: {head_output_error} {head_state_error}"
        )

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "official_layer_source": str(layer_source),
        "official_layer_sha256": sha256(layer_source),
        "official_chunk_source": str(chunk_source),
        "official_chunk_sha256": sha256(chunk_source),
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": data_hashes,
        "parameter_count": parameter_count,
        "mixer_parameter_count": EXPECTED_MIXER_PARAMETERS,
        "recurrent_state_values_per_layer": STATE_VALUES_PER_LAYER,
        "microsteps_per_token": NUM_MICROSTEPS,
        "persistent_state_delta": 0,
        "official_module_provenance": provenance,
        "official_chunk_backward_count": official_backward_count,
        "gradient_rows": gradient_rows,
        "production_capture_rows": capture_rows,
        "explicit_output_relative_rms": output_parity,
        "explicit_terminal_relative_rms": terminal_parity,
        "incoming_state_output_relative_rms": incoming_output_dependency,
        "incoming_state_terminal_relative_rms": incoming_state_dependency,
        "microstep_order_output_relative_rms": order_output_dependency,
        "microstep_order_terminal_relative_rms": order_state_dependency,
        "head_permutation_output_relative_rms": head_output_error,
        "head_permutation_terminal_relative_rms": head_state_error,
        "synthetic_gradient_rms": synthetic_gradient_rms,
        "first_microstep_payload_max_abs": 0.0,
        "fallback": False,
    }
    if not all(math.isfinite(float(value)) for value in synthetic_gradient_rms.values()):
        raise RuntimeError("Non-finite synthetic gradient summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
