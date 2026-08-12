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

from experiments.zoology_mqar.gdn2_committed_delta import (
    ZoologyClusteredCommittedDeltaFutureSeedMixer,
    ZoologySharedCommittedDeltaFutureSeedMixer,
    committed_delta_diagnostics,
    parent_parameter_hash,
)
from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.length_scaling import build_config, dataset_hash, make_model


EXPECTED_ZOOLOGY_SHA = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARENT_HASH = "3b0c133410eaed1135224cc0acb094705655cc7e97ea6beeba17b40beab1a368"


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


def exact_equal(first: torch.Tensor, second: torch.Tensor, name: str) -> None:
    if not torch.equal(first, second):
        delta = float((first.float() - second.float()).abs().max().item())
        raise RuntimeError(f"{name} lost exact identity; max delta={delta}")


def check_same_weight_parent_identity(
    mixer: ZoologySharedCommittedDeltaFutureSeedMixer
    | ZoologyClusteredCommittedDeltaFutureSeedMixer,
    hidden: torch.Tensor,
    incoming: torch.Tensor,
    name: str,
) -> None:
    wrapper = mixer.layer
    if torch.tanh(wrapper.correction_read_logit).count_nonzero().item() != 0:
        raise RuntimeError(f"{name} correction read gate is not exactly zero")

    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        base_output, _base_attentions, _base_cache = wrapper.base(hidden)
        wrapped_output, _wrapped_attentions, _wrapped_cache = wrapper(hidden)
    exact_equal(base_output, wrapped_output, f"{name} zero-state same-weight output")

    base_cache = mixer._new_cache(incoming.clone())
    wrapped_cache = mixer._new_cache(incoming.clone())
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        base_output, _base_attentions, base_cache = wrapper.base(
            hidden,
            past_key_values=base_cache,
            use_cache=True,
        )
        wrapped_output, _wrapped_attentions, wrapped_cache = wrapper(
            hidden,
            past_key_values=wrapped_cache,
            use_cache=True,
        )
    exact_equal(
        base_output,
        wrapped_output,
        f"{name} nonzero-state same-weight output",
    )
    exact_equal(
        base_cache[mixer.layer_idx]["recurrent_state"],
        wrapped_cache[mixer.layer_idx]["recurrent_state"],
        f"{name} nonzero-state same-weight terminal state",
    )


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
        raise RuntimeError("Unexpected GDN2 layer source hash")
    if python_tree_hash(fla_root / "fla" / "ops" / "gdn2") != os.environ[
        "FLA_GDN2_OPS_SHA256"
    ]:
        raise RuntimeError("Unexpected GDN2 recurrence source hash")
    chunk_source = (fla_root / "fla" / "ops" / "gdn2" / "chunk_fwd.py").read_text()
    if "h, v_new, final_state = chunk_gated_delta_rule_fwd_h" not in chunk_source:
        raise RuntimeError("Pinned recurrence no longer exposes the committed v_new edit")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA:
        raise RuntimeError("Unexpected Zoology SHA")

    configs = {
        "parent": build_config(
            arm="future_seed_gdn2", sequence_length=1024, num_kv_pairs=4,
            max_epochs=10, batch_size=32,
        ),
        "shared": build_config(
            arm="future_seed_gdn2_shared_committed_delta", sequence_length=1024,
            num_kv_pairs=4, max_epochs=10, batch_size=32,
        ),
        "clustered": build_config(
            arm="future_seed_gdn2_clustered_committed_delta", sequence_length=1024,
            num_kv_pairs=4, max_epochs=10, batch_size=32,
        ),
    }
    train_loader, test_loader = prepare_data(configs["parent"].data)
    data_hashes = {"train": dataset_hash(train_loader), "test": dataset_hash(test_loader)}
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    arm_names = {
        "shared": "future_seed_gdn2_shared_committed_delta",
        "clustered": "future_seed_gdn2_clustered_committed_delta",
    }
    models = {}
    for name, arm_name in arm_names.items():
        set_determinism(123)
        models[name] = make_model(configs[name], arm_name).cuda()
    shared, clustered = models["shared"], models["clustered"]
    if sum(p.numel() for p in shared.parameters()) != 661_592:
        raise RuntimeError("Shared correction parameter count changed")
    if sum(p.numel() for p in clustered.parameters()) != 665_560:
        raise RuntimeError("Clustered correction parameter count changed")
    if not all(
        isinstance(block.sequence_mixer, ZoologySharedCommittedDeltaFutureSeedMixer)
        for block in shared.backbone.layers
    ):
        raise RuntimeError("Shared committed-delta mixer is not active in both layers")
    if not all(
        isinstance(block.sequence_mixer, ZoologyClusteredCommittedDeltaFutureSeedMixer)
        for block in clustered.backbone.layers
    ):
        raise RuntimeError("Clustered committed-delta mixer is not active in both layers")
    if not (
        parent_parameter_hash(shared)
        == parent_parameter_hash(clustered)
        == EXPECTED_PARENT_HASH
    ):
        raise RuntimeError("Parent initialization hash mismatch")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    targets = targets[:2].cuda()
    shared.eval()
    clustered.eval()
    with torch.no_grad():
        shared_logits = shared(inputs)
        clustered_logits = clustered(inputs)
    exact_equal(
        shared_logits,
        clustered_logits,
        "fixed/learned zero-read full output",
    )

    hidden = torch.randn(2, 128, 128, device="cuda")
    incoming = torch.randn(2, 4, 32, 32, device="cuda")
    shared_mixer = shared.backbone.layers[1].sequence_mixer
    clustered_mixer = clustered.backbone.layers[1].sequence_mixer
    check_same_weight_parent_identity(shared_mixer, hidden, incoming, "shared")
    check_same_weight_parent_identity(clustered_mixer, hidden, incoming, "clustered")

    clustered.train().zero_grad(set_to_none=True)
    logits = clustered(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(logits)
    loss.backward()
    official_backward_count = sum(
        "ChunkGDN2FunctionBackward" in name for name in graph_names
    )
    if official_backward_count != 4:
        raise RuntimeError(f"Expected four official GDN2 backwards, got {official_backward_count}")
    read_gradients = [
        block.sequence_mixer.layer.correction_read_logit.grad
        for block in clustered.backbone.layers
    ]
    if not all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().max() > 0
        for gradient in read_gradients
    ):
        raise RuntimeError("Zero-read correction gates did not receive finite nonzero gradients")

    with torch.no_grad():
        for block in clustered.backbone.layers:
            block.sequence_mixer.layer.correction_read_logit.fill_(0.1)
    clustered.zero_grad(set_to_none=True)
    active_logits = clustered(inputs)
    active_loss = F.cross_entropy(active_logits[mask], targets[mask])
    active_loss.backward()
    basis_gradients = [
        block.sequence_mixer.layer.correction_subspace.raw.grad
        for block in clustered.backbone.layers
    ]
    if not all(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().max() > 0
        for gradient in basis_gradients
    ):
        raise RuntimeError("Active correction basis did not receive finite nonzero gradients")
    diagnostics = committed_delta_diagnostics(clustered)
    if diagnostics["correction_parameters"] != 3976:
        raise RuntimeError("Clustered correction parameter delta changed")
    if diagnostics["total_state_values_per_layer"] != 6144:
        raise RuntimeError("Clustered correction state size changed")
    if diagnostics["official_scans_per_layer"] != 2:
        raise RuntimeError("Clustered official scan count changed")
    if not all(
        row["committed_edit_rms"] > 0
        and row["correction_state_rms"] > 0
        and row["correction_q_token_std"] > 0
        and row["correction_k_token_std"] > 0
        and row["row_orthogonality_max_error"] <= 1e-5
        for row in diagnostics["per_layer"]
    ):
        raise RuntimeError(f"Committed-delta activation contract failed: {diagnostics}")

    result = {
        "status": "passed",
        "gpu": {"name": device.name, "uuid": device_uuid},
        "fla_source_sha": PINNED_FLA_SHA,
        "fla_gdn2_source_sha256": hashlib.sha256(gdn2_source.read_bytes()).hexdigest(),
        "data_hashes": data_hashes,
        "parameter_counts": {
            "parent": 661_584, "shared": 661_592, "clustered": 665_560,
        },
        "parent_parameter_hash": EXPECTED_PARENT_HASH,
        "state_values_per_layer": {"main": 4096, "correction": 2048, "total": 6144},
        "official_backward_count": official_backward_count,
        "fixed_learned_zero_read_full_output_exact": True,
        "same_weight_parent_zero_read_output_exact": True,
        "same_weight_parent_nonzero_state_output_and_terminal_exact": True,
        "two_stage_gradients": {"read_gate": True, "learned_basis": True},
        "loss": float(loss.item()),
        "active_loss": float(active_loss.item()),
        "diagnostics": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
