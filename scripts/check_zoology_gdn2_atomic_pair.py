from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from fla.modules.convolution import ShortConvolution
from fla.ops.gdn2 import chunk_gdn2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_atomic_pair import (
    AtomicPairGatedDeltaNet2,
    ZoologyAtomicPairGDN2FutureSeedMixer,
    parent_parameter_hash,
    symmetric_weighted_pair,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
    parameter_hash,
)


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"


def normalized_uuid(value: str) -> str:
    return value.removeprefix("GPU-").lower()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def python_tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left.float() - right.float()).square().mean().sqrt()
    denominator = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float((numerator / denominator).item())


def backward_names(tensor: torch.Tensor) -> list[str]:
    names = []
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


def interleave(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    batch, length = first.shape[:2]
    return torch.stack((first, second), dim=2).reshape(
        batch, 2 * length, *first.shape[2:]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-028 requires exactly CUDA index 0")
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
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if fla_root not in gdn2_source.parents:
        raise RuntimeError(f"Unexpected GDN2 source: {gdn2_source}")
    if hashlib.sha256(gdn2_source.read_bytes()).hexdigest() != os.environ[
        "FLA_GDN2_SOURCE_SHA256"
    ]:
        raise RuntimeError("Official GDN2 layer source hash changed")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Official GDN2 recurrence source hash changed")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    arms = ("future_seed_gdn2", "future_seed_gdn2_atomic_pair")
    configs = {
        arm: build_config(
            arm=arm,
            sequence_length=1024,
            num_kv_pairs=4,
            max_epochs=10,
            batch_size=32,
        )
        for arm in arms
    }
    train_loader, test_loader = prepare_data(configs[arms[0]].data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    models = {}
    for arm in arms:
        set_determinism(123)
        models[arm] = make_model(configs[arm], arm)
    control, candidate = models[arms[0]], models[arms[1]]
    parameter_counts = {
        arm: sum(parameter.numel() for parameter in model.parameters())
        for arm, model in models.items()
    }
    if parameter_counts != {
        "future_seed_gdn2": 661_584,
        "future_seed_gdn2_atomic_pair": 727_120,
    }:
        raise RuntimeError(f"Parameter counts changed: {parameter_counts}")
    control_parameter_hash = parameter_hash(control)
    candidate_parent_hash = parent_parameter_hash(candidate)
    if candidate_parent_hash != control_parameter_hash:
        raise RuntimeError("Atomic pair changed parent initialization")

    provenance = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        mixer = block.sequence_mixer
        if not isinstance(mixer, ZoologyAtomicPairGDN2FutureSeedMixer):
            raise RuntimeError(f"Layer {layer_index} atomic mixer type changed")
        wrapper = mixer.layer
        if not isinstance(wrapper, AtomicPairGatedDeltaNet2):
            raise RuntimeError(f"Layer {layer_index} atomic wrapper type changed")
        if type(wrapper.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} is not exact official GDN2")
        conv_backends = {
            name: getattr(getattr(wrapper.base, name), "backend", None)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        if not all(
            isinstance(getattr(wrapper.base, name), ShortConvolution)
            for name in conv_backends
        ) or set(conv_backends.values()) != {"triton"}:
            raise RuntimeError(f"Short-conv fallback: {conv_backends}")
        provenance.append(
            {
                "layer": layer_index,
                "base_type": type(wrapper.base).__qualname__,
                "conv_backends": conv_backends,
            }
        )

    generator = torch.Generator(device="cuda").manual_seed(528)
    shape = (2, 64, 4, 32)
    raw_first = torch.randn(*shape, generator=generator, device="cuda")
    raw_second = torch.randn(*shape, generator=generator, device="cuda")
    erase = torch.sigmoid(torch.randn(*shape, generator=generator, device="cuda"))
    first, second, pair_diag = symmetric_weighted_pair(
        raw_first, raw_second, erase
    )
    swapped_first, swapped_second, _swapped_diag = symmetric_weighted_pair(
        raw_second, raw_first, erase
    )
    polar_swap_error = max(
        relative_rms(first, swapped_second),
        relative_rms(second, swapped_first),
    )
    weighted_gram = torch.einsum(
        "...ki,...kj->...ij",
        torch.stack((first.float(), second.float()), dim=-1)
        * erase.float().unsqueeze(-1),
        torch.stack((first.float(), second.float()), dim=-1),
    ) / 32
    identity = torch.eye(2, device="cuda")
    polar_identity_error = float((weighted_gram - identity).abs().amax().item())
    if polar_swap_error > 2e-4 or polar_identity_error > 2e-4:
        raise RuntimeError(
            f"Symmetric weighted polar failed: {polar_swap_error=} {polar_identity_error=}"
        )

    query_main = torch.randn(*shape, generator=generator, device="cuda")
    query_aux = torch.randn(*shape, generator=generator, device="cuda")
    query = (query_main + query_aux) * (2.0**-0.5)
    value = torch.randn(*shape, generator=generator, device="cuda")
    write = torch.sigmoid(torch.randn(*shape, generator=generator, device="cuda"))
    shared_value = value * (2.0**-0.5)
    decay = -F.softplus(torch.randn(*shape, generator=generator, device="cuda"))
    initial_state = torch.randn(
        2, 4, 32, 32, generator=generator, device="cuda", dtype=torch.float32
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, state = chunk_gdn2(
            q=interleave(query, query),
            k=interleave(first, second),
            v=interleave(shared_value, shared_value),
            g=interleave(decay, torch.zeros_like(decay)),
            b=interleave(erase, erase),
            w=interleave(write, write),
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        swapped_output, swapped_state = chunk_gdn2(
            q=interleave(query, query),
            k=interleave(second, first),
            v=interleave(shared_value, shared_value),
            g=interleave(decay, torch.zeros_like(decay)),
            b=interleave(erase, erase),
            w=interleave(write, write),
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    output_swap_relative_rms = relative_rms(
        output[:, 1::2], swapped_output[:, 1::2]
    )
    state_swap_relative_rms = relative_rms(state, swapped_state)
    if output_swap_relative_rms > 5e-3 or state_swap_relative_rms > 5e-3:
        raise RuntimeError(
            "Official scan is not column-swap invariant: "
            f"output={output_swap_relative_rms} state={state_swap_relative_rms}"
        )

    candidate = candidate.cuda().train()
    inputs, _targets, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    candidate.zero_grad(set_to_none=True)
    logits = candidate(inputs)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Full atomic-pair logits are non-finite")
    loss = logits.float().square().mean()
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkGDN2FunctionBackward")
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official chunk backward paths, got {official_backward_count}"
        )
    loss.backward()
    gradient_rows = []
    for layer_index, block in enumerate(candidate.backbone.layers):
        wrapper = block.sequence_mixer.layer
        for name in ("aux_q_proj", "aux_k_proj"):
            gradient = getattr(wrapper, name).weight.grad
            gradient_rms = (
                0.0
                if gradient is None
                else float(gradient.float().square().mean().sqrt().item())
            )
            if (
                gradient is None
                or not torch.isfinite(gradient).all()
                or gradient_rms <= 1e-8
            ):
                raise RuntimeError(
                    f"Layer {layer_index} {name} has no finite gradient: {gradient_rms}"
                )
            gradient_rows.append(
                {"layer": layer_index, "projection": name, "rms": gradient_rms}
            )

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid, "count": 1},
        "pinned_fla_sha": PINNED_FLA_SHA,
        "zoology_sha": EXPECTED_ZOOLOGY_SHA,
        "data_hashes": data_hashes,
        "parameter_counts": parameter_counts,
        "parameter_delta": parameter_counts[arms[1]] - parameter_counts[arms[0]],
        "control_parameter_hash": control_parameter_hash,
        "candidate_parent_parameter_hash": candidate_parent_hash,
        "official_module_provenance": provenance,
        "raw_condition_mean": float(pair_diag["raw_condition"].mean().item()),
        "raw_condition_max": float(pair_diag["raw_condition"].amax().item()),
        "polar_identity_max_error": polar_identity_error,
        "polar_column_swap_relative_rms": polar_swap_error,
        "official_output_column_swap_relative_rms": output_swap_relative_rms,
        "official_state_column_swap_relative_rms": state_swap_relative_rms,
        "official_chunk_backward_count": official_backward_count,
        "auxiliary_gradient_rows": gradient_rows,
        "logical_block_rank": 2,
        "shared_payload": True,
        "aggregate_payload_energy_ratio": 1.0,
        "persistent_state_delta": 0,
        "fallback": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
