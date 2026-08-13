from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch

from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.modules.convolution import ShortConvolution
from fla.ops.gated_delta_product import chunk_gated_delta_product
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gated_delta_product_futureseed import (
    ZoologyGatedDeltaProductFutureSeedMixer,
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
EXPECTED_PARAMETERS = 666_200
EXPECTED_MIXER_PARAMETERS = 118_828
EXPECTED_STATE_VALUES = 4_096


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
        stack.extend(next_function for next_function, _index in function.next_functions)
    return names


def finite_gradient_rms(parameter: torch.nn.Parameter, label: str) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError(f"{label} has no finite gradient")
    rms = float(gradient.float().square().mean().sqrt().item())
    if rms <= 1e-8:
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
        if rms <= 1e-8:
            raise RuntimeError(f"{label}[{index}] gradient is inactive: {rms}")
        rows.append(rms)
    return rows


def swap_transform_axis(tensor: torch.Tensor) -> torch.Tensor:
    batch, expanded_length = tensor.shape[:2]
    if expanded_length % 2:
        raise ValueError("Expanded transform length must be even")
    return tensor.reshape(batch, expanded_length // 2, 2, *tensor.shape[2:]).flip(2).reshape_as(tensor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-029 requires exactly CUDA index 0")
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
    layer_source = Path(inspect.getfile(GatedDeltaProduct)).resolve()
    chunk_source = Path(inspect.getfile(chunk_gated_delta_product)).resolve()
    expected_layer_source = fla_root / "fla" / "layers" / "gated_deltaproduct.py"
    expected_chunk_source = fla_root / "fla" / "ops" / "gated_delta_product" / "chunk.py"
    if layer_source != expected_layer_source or chunk_source != expected_chunk_source:
        raise RuntimeError(f"Unexpected official FLA sources: {layer_source} {chunk_source}")
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
        arm="future_seed_gated_delta_product_n2",
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

    set_determinism(123)
    model = make_model(config, "future_seed_gated_delta_product_n2")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count changed: {parameter_count}")

    provenance = []
    for layer_index, block in enumerate(model.backbone.layers):
        mixer = block.sequence_mixer
        if not isinstance(mixer, ZoologyGatedDeltaProductFutureSeedMixer):
            raise RuntimeError(f"Layer {layer_index} mixer type changed")
        layer = mixer.layer
        if type(layer) is not GatedDeltaProduct:
            raise RuntimeError(f"Layer {layer_index} is not exact official GatedDeltaProduct")
        if (
            layer.num_householder != 2
            or not layer.use_forget_gate
            or layer.allow_neg_eigval
            or layer.mode != "chunk"
        ):
            raise RuntimeError(f"Layer {layer_index} operator configuration drifted")
        if (
            layer.num_heads != 4
            or layer.num_v_heads != 4
            or layer.head_k_dim != 32
            or layer.head_v_dim != 32
        ):
            raise RuntimeError(f"Layer {layer_index} state geometry drifted")
        mixer_parameters = sum(parameter.numel() for parameter in mixer.parameters())
        if mixer_parameters != EXPECTED_MIXER_PARAMETERS:
            raise RuntimeError(
                f"Layer {layer_index} mixer parameter count changed: {mixer_parameters}"
            )
        if mixer.state_size() != EXPECTED_STATE_VALUES:
            raise RuntimeError(f"Layer {layer_index} state size changed")
        conv_backends = {
            name: getattr(getattr(layer, name), "backend", None)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        if not all(
            isinstance(getattr(layer, name), ShortConvolution)
            for name in conv_backends
        ) or set(conv_backends.values()) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} short-conv fallback: {conv_backends}")
        provenance.append(
            {
                "layer": layer_index,
                "type": type(layer).__qualname__,
                "num_householder": layer.num_householder,
                "state_values": mixer.state_size(),
                "conv_backends": conv_backends,
            }
        )

    model = model.cuda().train()
    inputs, _targets, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Full GatedDeltaProduct logits are non-finite")
    loss = logits.float().square().mean()
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count(
        "ChunkGatedDeltaProductFunctionBackward"
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official product backward paths, got {official_backward_count}"
        )
    loss.backward()

    gradient_rows = []
    for layer_index, block in enumerate(model.backbone.layers):
        layer = block.sequence_mixer.layer
        row = {
            "layer": layer_index,
            "key_transform_rms": split_gradient_rms(
                layer.k_proj.weight,
                chunks=2,
                label=f"layer{layer_index}.k_proj",
            ),
            "value_transform_rms": split_gradient_rms(
                layer.v_proj.weight,
                chunks=2,
                label=f"layer{layer_index}.v_proj",
            ),
            "beta_transform_rms": split_gradient_rms(
                layer.b_proj.weight,
                chunks=2,
                label=f"layer{layer_index}.b_proj",
            ),
            "query_rms": finite_gradient_rms(
                layer.q_proj.weight,
                f"layer{layer_index}.q_proj",
            ),
            "forget_rms": finite_gradient_rms(
                layer.a_proj.weight,
                f"layer{layer_index}.a_proj",
            ),
        }
        if layer_index == 1:
            row["future_seed_logit_rms"] = finite_gradient_rms(
                block.sequence_mixer.future_seed_logit,
                "layer1.future_seed_logit",
            )
        gradient_rows.append(row)

    generator = torch.Generator(device="cuda").manual_seed(529)
    q = torch.randn(
        2, 64, 4, 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        2, 128, 4, 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn(
        2, 128, 4, 32, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    g = -0.001 * torch.sigmoid(
        torch.randn(2, 64, 4, generator=generator, device="cuda")
    )
    beta = torch.sigmoid(
        torch.randn(
            2, 128, 4, generator=generator, device="cuda", dtype=torch.bfloat16
        )
    )
    initial_state = torch.randn(
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, state = chunk_gated_delta_product(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_householder=2,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        zero_output, zero_state = chunk_gated_delta_product(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_householder=2,
            initial_state=torch.zeros_like(initial_state),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        swapped_output, swapped_state = chunk_gated_delta_product(
            q=q,
            k=swap_transform_axis(k),
            v=swap_transform_axis(v),
            g=g,
            beta=swap_transform_axis(beta),
            num_householder=2,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    for label, tensor in {
        "output": output,
        "state": state,
        "zero_output": zero_output,
        "zero_state": zero_state,
        "swapped_output": swapped_output,
        "swapped_state": swapped_state,
    }.items():
        if tensor is None or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Official synthetic {label} is non-finite")
    if tuple(state.shape) != (2, 4, 32, 32):
        raise RuntimeError(f"Official final state shape changed: {tuple(state.shape)}")
    initial_state_output_dependency = relative_rms(output, zero_output)
    initial_state_final_dependency = relative_rms(state, zero_state)
    order_output_dependency = relative_rms(output, swapped_output)
    order_state_dependency = relative_rms(state, swapped_state)
    if min(initial_state_output_dependency, initial_state_final_dependency) <= 1e-4:
        raise RuntimeError("Official product does not depend on incoming state")
    if min(order_output_dependency, order_state_dependency) <= 1e-4:
        raise RuntimeError("The two official product transforms are order invariant")

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
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES,
        "official_module_provenance": provenance,
        "official_chunk_backward_count": official_backward_count,
        "gradient_rows": gradient_rows,
        "incoming_state_output_relative_rms": initial_state_output_dependency,
        "incoming_state_final_relative_rms": initial_state_final_dependency,
        "transform_order_output_relative_rms": order_output_dependency,
        "transform_order_state_relative_rms": order_state_dependency,
        "num_householder": 2,
        "persistent_state_delta": 0,
        "fallback": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
