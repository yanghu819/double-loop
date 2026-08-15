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
from fla.modules.convolution import ShortConvolution
from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import PINNED_FLA_SHA
from experiments.zoology_mqar.lagged_commit_futureseed import (
    EXPECTED_NEW_PARAMETERS,
    EXPECTED_STATE_VALUES,
    LaggedCommitGDN2,
    ZoologyLaggedCommitFutureSeedMixer,
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


EXPECTED_NATIVE_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = EXPECTED_NATIVE_PARAMETERS + EXPECTED_NEW_PARAMETERS
EXPECTED_INIT_SHA256 = (
    "7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mixers(model: torch.nn.Module) -> list[ZoologyLaggedCommitFutureSeedMixer]:
    mixers = [block.sequence_mixer for block in model.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyLaggedCommitFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Expected exactly two lagged-commit mixers")
    return mixers


def _gradient_vector(parameter: torch.nn.Parameter, label: str) -> list[float]:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {label}")
    gradient = parameter.grad.detach().float()
    if not torch.isfinite(gradient).all() or not bool((gradient != 0).all()):
        raise RuntimeError(f"Non-finite or zero element in gradient: {label}")
    return [float(value) for value in gradient.cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-init-path", type=Path, required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise RuntimeError("P-GDN3-051 requires one visible CUDA index 0")
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
        raise RuntimeError("GDN2 was imported outside the pinned FLA root")
    gdn2_source_hash = hashlib.sha256(gdn2_source.read_bytes()).hexdigest()
    if gdn2_source_hash != os.environ["FLA_GDN2_SOURCE_SHA256"]:
        raise RuntimeError("Pinned GDN2 layer source hash drifted")
    gdn2_ops_hash = python_tree_hash(fla_root / "fla" / "ops" / "gdn2")
    if gdn2_ops_hash != os.environ["FLA_GDN2_OPS_SHA256"]:
        raise RuntimeError("Pinned GDN2 ops hash drifted")
    zoology_root = Path(os.environ["ZOOLOGY_ROOT"]).resolve()
    zoology_status = subprocess.run(
        ["git", "-C", str(zoology_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_head(zoology_root) != EXPECTED_ZOOLOGY_SHA or zoology_status:
        raise RuntimeError("Zoology source is not exact and clean")
    if _sha256(args.matched_init_path) != EXPECTED_INIT_SHA256:
        raise RuntimeError("Frozen matched initialization hash drifted")

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_lagged_commit_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    train_loader, test_loader = prepare_data(native_config.data)
    data_hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(
        args.matched_init_path,
        map_location="cpu",
        weights_only=True,
    )
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    native.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_lagged_commit_gdn2")
    load_matched_parent_state(candidate, parent_state)
    native_hash = parameter_hash(native)
    if parent_parameter_hash(candidate) != native_hash:
        raise RuntimeError("Candidate parent initialization changed")
    counts = {
        "native": sum(parameter.numel() for parameter in native.parameters()),
        "candidate": sum(parameter.numel() for parameter in candidate.parameters()),
    }
    if counts != {
        "native": EXPECTED_NATIVE_PARAMETERS,
        "candidate": EXPECTED_CANDIDATE_PARAMETERS,
    }:
        raise RuntimeError(f"Unexpected parameter counts: {counts}")

    mixers = _mixers(candidate)
    if any(mixer.state_size() != EXPECTED_STATE_VALUES for mixer in mixers):
        raise RuntimeError("Native recurrent state size drifted")
    if len([module for module in candidate.modules() if type(module) is GatedDeltaNet2]) != 2:
        raise RuntimeError("Expected exactly two pinned official GDN2 layers")
    for layer_index, mixer in enumerate(mixers):
        if type(mixer.layer) is not LaggedCommitGDN2:
            raise RuntimeError(f"Layer {layer_index} wrapper changed")
        if type(mixer.layer.base) is not GatedDeltaNet2:
            raise RuntimeError(f"Layer {layer_index} carrier changed")
        convolutions = [
            module.backend
            for module in mixer.layer.modules()
            if isinstance(module, ShortConvolution)
        ]
        if len(convolutions) != 3 or set(convolutions) != {"triton"}:
            raise RuntimeError(f"Layer {layer_index} convolution backend changed")
        if mixer.layer.lag_mix_logits.detach().abs().max().item() != 0:
            raise RuntimeError(f"Layer {layer_index} lag logits are not zero")

    native = native.cuda().eval()
    candidate = candidate.cuda().eval()
    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    with torch.no_grad():
        native_logits = native(inputs)
        candidate_logits = candidate(inputs)
    output_identity_error = finite_max_abs_difference(
        native_logits,
        candidate_logits,
        "zero-lag full output identity",
    )
    if output_identity_error != 0.0:
        raise RuntimeError("Zero lag changed the parent model")

    generator = torch.Generator(device="cuda").manual_seed(52051)
    incoming_rows = []
    opened_rows = []
    for layer_index, (native_mixer, candidate_mixer) in enumerate(
        zip(
            [block.sequence_mixer for block in native.backbone.layers],
            mixers,
        )
    ):
        hidden = torch.randn(2, 97, 128, generator=generator, device="cuda")
        incoming = torch.randn(
            2,
            4,
            32,
            32,
            generator=generator,
            device="cuda",
        )
        with torch.no_grad():
            native_output, native_state = native_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_output, candidate_state = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
        output_error = finite_max_abs_difference(
            native_output,
            candidate_output,
            f"layer-{layer_index} nonzero-state output identity",
        )
        state_error = finite_max_abs_difference(
            native_state,
            candidate_state,
            f"layer-{layer_index} nonzero-state terminal identity",
        )
        if output_error != 0.0 or state_error != 0.0:
            raise RuntimeError("Zero lag changed a nonzero incoming-state path")
        with torch.no_grad():
            candidate_mixer.layer.lag_mix_logits.fill_(
                torch.atanh(torch.tensor(0.25, device="cuda"))
            )
            opened_output, opened_state = candidate_mixer.forward_with_state(
                hidden,
                initial_state=incoming,
            )
            candidate_mixer.layer.lag_mix_logits.zero_()
        opened_output_rms = float(
            (opened_output.float() - candidate_output.float())
            .square()
            .mean()
            .sqrt()
            .item()
        )
        opened_state_rms = float(
            (opened_state.float() - candidate_state.float())
            .square()
            .mean()
            .sqrt()
            .item()
        )
        if opened_output_rms <= 1e-4 or opened_state_rms <= 1e-4:
            raise RuntimeError("Opened lag does not change output and state")
        incoming_rows.append(
            {
                "layer": layer_index,
                "output_max_error": output_error,
                "state_max_error": state_error,
            }
        )
        opened_rows.append(
            {
                "layer": layer_index,
                "output_change_rms": opened_output_rms,
                "state_change_rms": opened_state_rms,
            }
        )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        candidate(inputs).flatten(0, 1),
        targets.flatten(),
    )
    graph_names = backward_names(loss)
    official_backward_count = graph_names.count("ChunkGDN2FunctionBackward")
    if official_backward_count != 2:
        raise RuntimeError(f"Unexpected official backward count: {graph_names}")
    loss.backward()
    lag_gradients = {
        str(index): _gradient_vector(mixer.layer.lag_mix_logits, f"layer {index}")
        for index, mixer in enumerate(mixers)
    }

    native_key = torch.randn(
        3,
        19,
        4,
        32,
        generator=generator,
        device="cuda",
    )
    mix = torch.tensor([0.1, -0.2, 0.3, 0.05], device="cuda")
    committed, previous = LaggedCommitGDN2.lagged_key(native_key, mix)
    token0_error = finite_max_abs_difference(
        committed[:, 0],
        native_key[:, 0],
        "token-zero causal identity",
    )
    shift_error = finite_max_abs_difference(
        previous[:, 1:],
        native_key[:, :-1],
        "strict previous-token shift",
    )
    if token0_error != 0.0 or shift_error != 0.0:
        raise RuntimeError("Causal lag shift wraps or changes token zero")
    future_changed = native_key.clone()
    future_changed[:, -1].add_(7.0)
    changed_committed, _ = LaggedCommitGDN2.lagged_key(future_changed, mix)
    causal_prefix_error = finite_max_abs_difference(
        committed[:, :-1],
        changed_committed[:, :-1],
        "future perturbation causal prefix",
    )
    if causal_prefix_error != 0.0:
        raise RuntimeError("Future K leaked into an earlier commit")

    permutation = torch.tensor([2, 0, 3, 1], device="cuda")
    permuted_commit, _ = LaggedCommitGDN2.lagged_key(
        native_key[:, :, permutation],
        mix[permutation],
    )
    permutation_error = finite_max_abs_difference(
        committed[:, :, permutation],
        permuted_commit,
        "head permutation equivariance",
    )
    if permutation_error != 0.0:
        raise RuntimeError("Lagged commit is not head-equivariant")

    candidate_mixer = mixers[1]
    hidden = torch.randn(2, 73, 128, generator=generator, device="cuda")
    incoming = torch.randn(
        2,
        4,
        32,
        32,
        generator=generator,
        device="cuda",
    )
    with torch.no_grad():
        output, terminal = candidate_mixer.forward_with_state(
            hidden,
            initial_state=incoming,
        )
        shuffled_output, _ = candidate_mixer.forward_with_state(
            hidden,
            initial_state=incoming.flip(0),
        )
    state_dependency_rms = float(
        (output.float() - shuffled_output.float()).square().mean().sqrt().item()
    )
    terminal_rms = float(terminal.float().square().mean().sqrt().item())
    if state_dependency_rms <= 1e-4 or terminal_rms <= 0:
        raise RuntimeError("Official recurrent state dependency collapsed")

    source_text = Path(inspect.getfile(LaggedCommitGDN2)).read_text().lower()
    if "fallback" in source_text:
        raise RuntimeError("Lagged-commit source contains an alternate path")
    finite_values = [
        state_dependency_rms,
        terminal_rms,
        *[value for row in lag_gradients.values() for value in row],
        *[row["output_change_rms"] for row in opened_rows],
        *[row["state_change_rms"] for row in opened_rows],
    ]
    if not all(math.isfinite(value) for value in finite_values):
        raise RuntimeError("Contract produced non-finite values")

    result = {
        "status": "passed",
        "plan": "P-GDN3-051",
        "gpu": {"index": 0, "name": device.name, "uuid": device_uuid},
        "provenance": {
            "fla_sha": PINNED_FLA_SHA,
            "gdn2_source_sha256": gdn2_source_hash,
            "gdn2_ops_sha256": gdn2_ops_hash,
            "zoology_sha": EXPECTED_ZOOLOGY_SHA,
            "official_backward_count": official_backward_count,
            "triton_short_convolutions": 6,
        },
        "data_hashes": data_hashes,
        "matched_parent_parameter_hash": native_hash,
        "parameter_counts": counts,
        "parameter_delta": counts["candidate"] - counts["native"],
        "recurrent_state_values_per_layer": EXPECTED_STATE_VALUES,
        "recurrent_state_delta": 0,
        "logical_scans_per_layer": 1,
        "output_identity_max_error": output_identity_error,
        "incoming_state_identity": incoming_rows,
        "lag_gradient_per_layer": lag_gradients,
        "opened_mix_effect": opened_rows,
        "strict_causal_token0_error": token0_error,
        "strict_causal_shift_error": shift_error,
        "strict_causal_prefix_error": causal_prefix_error,
        "head_permutation_error": permutation_error,
        "erase_and_write_share_one_commit_key": True,
        "state_shuffle_output_rms": state_dependency_rms,
        "terminal_state_rms": terminal_rms,
        "all_values_finite": True,
        "no_fallback": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
