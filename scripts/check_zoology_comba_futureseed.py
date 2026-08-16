from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from zoology.data.utils import prepare_data
from zoology.utils import set_determinism

from experiments.zoology_mqar.comba_futureseed import (
    EXPECTED_CHUNK_SHA256,
    EXPECTED_COMPAT_WY_SHA256,
    EXPECTED_LAYER_SHA256,
    EXPECTED_MDN_SHA,
    EXPECTED_PARAMETER_DELTA_VS_GDN2,
    EXPECTED_RECURRENT_SHA256,
    EXPECTED_WY_SHA256,
    EXPECTED_STATE_VALUES_PER_LAYER,
    ZoologyCombaFutureSeedMixer,
    comba_compatibility_metadata,
    load_external_comba_layer,
    load_matched_parent_state,
)
from experiments.zoology_mqar.length_scaling import (
    build_config,
    dataset_hash,
    make_model,
)
from experiments.zoology_mqar.momentum_futureseed import (
    momentum_fla_compatibility,
)
from scripts.check_zoology_contractive_dplr import (
    backward_names,
    normalized_uuid,
)


EXPECTED_TRAIN_HASH = "647c64ece84984a23dfd817c4f277ea83840dbec57cc18bb6c9bf9eda7cc9a68"
EXPECTED_TEST_HASH = "4a8237ba8fe19aaff0d1d72de7b7f6505eaab59cd091442c2f463df34cce278f"
EXPECTED_PARENT_PARAMETERS = 661_584
EXPECTED_CANDIDATE_PARAMETERS = 597_608


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return tuple(part.strip() for part in rows[0].split(",", maxsplit=1))


def _gradient(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None:
        raise RuntimeError(f"Missing gradient: {name}")
    gradient = parameter.grad.detach().float()
    rms = gradient.square().mean().sqrt()
    if not torch.isfinite(gradient).all() or float(rms) == 0.0:
        raise RuntimeError(f"Inactive or nonfinite gradient: {name}")
    return float(rms)


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).square().mean().sqrt()
    scale = right.float().square().mean().sqrt().clamp_min(1e-8)
    return float(delta / scale)


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
        raise RuntimeError("P-GDN3-065 requires exactly CUDA index 0")
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

    layer_class = load_external_comba_layer()
    compatibility = momentum_fla_compatibility()
    repo_root = Path(os.environ["MDN_REPO_ROOT"]).resolve()
    fla_root = Path(os.environ["MDN_FLA_ROOT"]).resolve()
    if subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip() != EXPECTED_MDN_SHA:
        raise RuntimeError("External Comba checkout changed")
    source_hashes = {
        "layer": _sha256(Path(inspect.getfile(layer_class)).resolve()),
        "chunk": _sha256(fla_root / "fla" / "ops" / "comba" / "chunk.py"),
        "fused_recurrent": _sha256(
            fla_root / "fla" / "ops" / "comba" / "fused_recurrent.py"
        ),
        "wy_fast": _sha256(fla_root / "fla" / "ops" / "comba" / "wy_fast.py"),
    }
    if source_hashes != {
        "layer": EXPECTED_LAYER_SHA256,
        "chunk": EXPECTED_CHUNK_SHA256,
        "fused_recurrent": EXPECTED_RECURRENT_SHA256,
        "wy_fast": EXPECTED_WY_SHA256,
    }:
        raise RuntimeError(f"External Comba source drifted: {source_hashes}")
    compatibility_overlay = comba_compatibility_metadata()
    if (
        compatibility_overlay["source_wy_sha256"] != EXPECTED_WY_SHA256
        or compatibility_overlay["effective_wy_sha256"]
        != EXPECTED_COMPAT_WY_SHA256
        or compatibility_overlay["patch_count"] != 2
        or Path(compatibility_overlay["root"]).resolve()
        != Path(os.environ["COMBA_COMPAT_ROOT"]).resolve()
        or _sha256(Path(compatibility_overlay["effective_wy_path"]))
        != EXPECTED_COMPAT_WY_SHA256
    ):
        raise RuntimeError(
            f"Comba compatibility overlay drifted: {compatibility_overlay}"
        )

    native_config = build_config(
        arm="future_seed_gdn2",
        sequence_length=1024,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    candidate_config = build_config(
        arm="future_seed_comba",
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
    if data_hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional MQAR data drifted: {data_hashes}")

    parent_state = torch.load(args.matched_init, map_location="cpu", weights_only=True)
    set_determinism(123)
    native = make_model(native_config, "future_seed_gdn2")
    native.load_state_dict(parent_state, strict=True)
    set_determinism(123)
    candidate = make_model(candidate_config, "future_seed_comba")
    load_matched_parent_state(candidate, parent_state)
    native_parameters = sum(parameter.numel() for parameter in native.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    if native_parameters != EXPECTED_PARENT_PARAMETERS:
        raise RuntimeError(f"Native parameter count drifted: {native_parameters}")
    if candidate_parameters != EXPECTED_CANDIDATE_PARAMETERS:
        raise RuntimeError(f"Candidate parameter count drifted: {candidate_parameters}")
    if candidate_parameters - native_parameters != EXPECTED_PARAMETER_DELTA_VS_GDN2:
        raise RuntimeError("Comba parameter delta changed")
    metadata = candidate._comba_parent_metadata
    if (
        metadata["tensor_count"] < 20
        or metadata["numel"] < 300_000
        or metadata["source_hash"] != metadata["loaded_hash"]
    ):
        raise RuntimeError(f"Shared parent mapping failed: {metadata}")

    mixers = [block.sequence_mixer for block in candidate.backbone.layers]
    if len(mixers) != 2 or not all(
        type(mixer) is ZoologyCombaFutureSeedMixer for mixer in mixers
    ):
        raise RuntimeError("Contract requires two exact Comba mixers")
    for index, mixer in enumerate(mixers):
        if type(mixer.layer) is not layer_class:
            raise RuntimeError(f"Layer {index} escaped the pinned Comba class")
        if mixer.state_size() != EXPECTED_STATE_VALUES_PER_LAYER:
            raise RuntimeError(f"Layer {index} state geometry changed")

    inputs, targets, _slices = next(iter(test_loader))
    inputs = inputs[:4].cuda()
    targets = targets[:4].cuda()
    candidate = candidate.cuda().train()
    logits = candidate(inputs)
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    graph_names = backward_names(loss)
    chunk_backward_count = sum(
        "ChunkCombaFunctionBackward" in name for name in graph_names
    )
    if chunk_backward_count != 2:
        raise RuntimeError(
            f"Expected two Comba chunk backward paths, got {chunk_backward_count}"
        )
    loss.backward()
    gradients = []
    for index, mixer in enumerate(mixers):
        layer = mixer.layer
        row = {
            "layer": index,
            "q": _gradient(layer.q_proj.weight, f"layer{index}.q"),
            "k": _gradient(layer.k_proj.weight, f"layer{index}.k"),
            "v": _gradient(layer.v_proj.weight, f"layer{index}.v"),
            "decay_logits": _gradient(layer.a_proj.weight, f"layer{index}.a"),
            "write_gate": _gradient(layer.b_proj.weight, f"layer{index}.b"),
            "output_correction": _gradient(layer.D, f"layer{index}.D"),
        }
        if index > 0:
            row["future_seed"] = _gradient(
                mixer.future_seed_logit, f"layer{index}.future_seed"
            )
        gradients.append(row)
    stack_state = {
        "active_layers": sum(mixer.last_state_rms is not None for mixer in mixers),
        "seed_routes": sum(mixer.last_seed_gate is not None for mixer in mixers),
        "state_rms": [float(mixer.last_state_rms) for mixer in mixers],
        "state_board_std": [float(mixer.last_state_board_std) for mixer in mixers],
    }
    if (
        stack_state["active_layers"] != 2
        or stack_state["seed_routes"] != 1
        or min(stack_state["state_rms"]) <= 1e-4
        or min(stack_state["state_board_std"]) <= 0
    ):
        raise RuntimeError(f"Full-stack state activation failed: {stack_state}")

    ops = importlib.import_module("fla.ops.comba")
    generator = torch.Generator(device="cuda").manual_seed(65065)
    shape = (2, 128, 4)
    q = torch.randn(*shape, 32, generator=generator, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(q)
    value = torch.randn_like(q)
    predictor_scale = torch.sigmoid(
        torch.randn(4, generator=generator, device="cuda", dtype=torch.float32)
    )
    prediction_key = key * predictor_scale[None, None, :, None].to(key)
    g = -0.01 - 0.2 * torch.rand(*shape, generator=generator, device="cuda")
    beta = 0.2 + 0.6 * torch.rand(*shape, generator=generator, device="cuda")
    initial = 0.05 * torch.randn(
        2, 4, 32, 32, generator=generator, device="cuda", dtype=torch.float32
    )
    with torch.no_grad():
        chunk_output, chunk_state = ops.chunk_comba(
            q=q,
            k=key,
            v=value,
            p=prediction_key,
            g=g,
            beta=beta,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_output, recurrent_state = ops.fused_recurrent_comba(
            q=q,
            k=key,
            p=prediction_key,
            v=value,
            g=g,
            beta=beta,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        zero_output, _ = ops.fused_recurrent_comba(
            q=q,
            k=key,
            p=prediction_key,
            v=value,
            g=g,
            beta=beta,
            initial_state=torch.zeros_like(initial),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    parity = {
        "output_relative_rms": _relative_rms(chunk_output, recurrent_output),
        "state_relative_rms": _relative_rms(chunk_state, recurrent_state),
        "initial_state_dependency": _relative_rms(recurrent_output, zero_output),
    }
    if (
        parity["output_relative_rms"] > 0.05
        or parity["state_relative_rms"] > 0.05
        or parity["initial_state_dependency"] <= 1e-4
    ):
        raise RuntimeError(f"Chunk/recurrent contract failed: {parity}")

    normalized_key = F.normalize(key.float(), dim=-1)
    normalized_prediction = F.normalize(prediction_key.float(), dim=-1)
    cosine = F.cosine_similarity(normalized_key, normalized_prediction, dim=-1)
    owner_geometry = {
        "prediction_key_cosine_mean": float(cosine.mean()),
        "prediction_key_cosine_min": float(cosine.min()),
        "prediction_scale_min": float(predictor_scale.min()),
        "prediction_scale_max": float(predictor_scale.max()),
    }
    if (
        owner_geometry["prediction_key_cosine_min"] < 0.999
        or not 0 < owner_geometry["prediction_scale_min"]
        or owner_geometry["prediction_scale_max"] >= 1
    ):
        raise RuntimeError(f"Closed-loop owner geometry failed: {owner_geometry}")

    seed = mixers[1].make_initial_state(recurrent_state)
    seed_geometry = {
        "shape": list(seed.shape),
        "rms": float(seed.float().square().mean().sqrt()),
        "finite": bool(torch.isfinite(seed).all()),
    }
    if seed_geometry["shape"] != [2, 4, 32, 32] or seed_geometry["rms"] <= 1e-4:
        raise RuntimeError(f"FutureSeed state transport failed: {seed_geometry}")

    result = {
        "status": "passed",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "external": {
            "repo": str(repo_root),
            "sha": EXPECTED_MDN_SHA,
            "source_hashes": source_hashes,
            "compatibility_overlay": compatibility_overlay,
            "redistributed_source": False,
        },
        "host_fla_compatibility": compatibility,
        "data_hashes": data_hashes,
        "parameters": {
            "native": native_parameters,
            "candidate": candidate_parameters,
            "delta": candidate_parameters - native_parameters,
        },
        "matched_parent": metadata,
        "state_values_per_layer": EXPECTED_STATE_VALUES_PER_LAYER,
        "chunk_backward_count": chunk_backward_count,
        "gradients": gradients,
        "full_stack_state": stack_state,
        "chunk_recurrent_parity": parity,
        "closed_loop_owner_geometry": owner_geometry,
        "futureseed_state_transport": seed_geometry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
