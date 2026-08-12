from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_dual_hash import (
    ZoologyDualHashGDN2FutureSeedMixer,
    dual_hash_diagnostics,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
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
        raise RuntimeError("Unexpected GDN2 source hash")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Unexpected GDN2 recurrence source hash")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    config = build_config(
        arm="future_seed_gdn2_dual_hash",
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
    model = make_model(config, "future_seed_gdn2_dual_hash").cuda()
    if sum(parameter.numel() for parameter in model.parameters()) != 661_584:
        raise RuntimeError("Dual-hash candidate changed the baseline parameter count")
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if not all(isinstance(mixer, ZoologyDualHashGDN2FutureSeedMixer) for mixer in mixers):
        raise RuntimeError("Dual-hash mixer is not active in both layers")
    if any(mixer.state_size() != 4096 for mixer in mixers):
        raise RuntimeError("Dual-hash candidate changed recurrent state size")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs.cuda()
    targets = targets.cuda()
    model.train().zero_grad(set_to_none=True)
    logits = model(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(logits)
    loss.backward()
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph_names
    )
    if official_backward_count != 2:
        raise RuntimeError(
            f"Expected two official GDN2 backwards, got {official_backward_count}"
        )
    gradients = [parameter.grad for parameter in model.parameters()]
    if not all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise RuntimeError("Dual-hash backward produced non-finite gradients")

    diagnostics = dual_hash_diagnostics(model)
    if diagnostics["parameter_delta"] != 0:
        raise RuntimeError("Dual-hash parameter delta is nonzero")
    if diagnostics["state_values_per_layer"] != 4096:
        raise RuntimeError("Dual-hash state size changed")
    if diagnostics["official_scans_per_layer"] != 1:
        raise RuntimeError("Dual-hash scan count changed")
    if not all(
        row["q_hash_token_std"] > 0
        and row["k_hash_token_std"] > 0
        and row["q_hash_norm_imbalance_max"] <= 1e-3
        and row["k_hash_norm_imbalance_max"] <= 1e-3
        for row in diagnostics["per_layer"]
    ):
        raise RuntimeError(f"Dual-hash activation or norm gate failed: {diagnostics}")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid},
        "fla_source_sha": PINNED_FLA_SHA,
        "fla_gdn2_source_sha256": hashlib.sha256(gdn2_source.read_bytes()).hexdigest(),
        "data_hashes": data_hashes,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_values_per_layer": 4096,
        "official_backward_count": official_backward_count,
        "loss": float(loss.item()),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
