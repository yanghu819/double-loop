from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.data.utils import prepare_data

from experiments.zoology_mqar.bidirectional_gdn2 import (
    ExplicitBidirectionalGDN2Mixer,
)
from experiments.zoology_mqar.bidirectional_gdn2_ceiling import (
    REFERENCE_GIT_SHA,
    REFERENCE_SCORE_SHA256,
    SEQUENCE_LENGTH,
    build_bidirectional_config,
    file_sha256,
)
from experiments.zoology_mqar.futureseed_directionality import dataset_hash
from experiments.zoology_mqar.gdn2_futureseed import (
    ZoologyGDN2FutureSeedMixer,
)
from experiments.zoology_mqar.length_scaling import build_config, make_model
from zoology.model import LanguageModel


EXPECTED_TRAIN_HASH = (
    "0bc802f155f02849f9220f5a3432115093b5a863da20df76db769889d2699eda"
)
EXPECTED_TEST_HASH = (
    "67101e7e04f9bfaac441f426d85a43ea11067ceecdc4fb5197b9ece8ed38b077"
)


def changed_future_values(inputs: torch.Tensor) -> torch.Tensor:
    altered = inputs.clone()
    quarter = SEQUENCE_LENGTH // 4
    region = altered[:, 3 * quarter :]
    values = region >= 160
    region[values] = 160 + ((region[values] - 160 + 1) % 96)
    return altered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one visible CUDA GPU")
    reference = args.reference_run.resolve()
    if (reference / "git_sha.txt").read_text().strip() != REFERENCE_GIT_SHA:
        raise RuntimeError("P-CAUSAL-016 source reference drifted")
    if file_sha256(reference / "score.json") != REFERENCE_SCORE_SHA256:
        raise RuntimeError("P-CAUSAL-016 score reference drifted")

    bidir_config = build_bidirectional_config()
    train_loader, test_loader = prepare_data(bidir_config.data)
    hashes = {
        "train": dataset_hash(train_loader),
        "test": dataset_hash(test_loader),
    }
    if hashes != {"train": EXPECTED_TRAIN_HASH, "test": EXPECTED_TEST_HASH}:
        raise RuntimeError(f"Directional L512 data hash drifted: {hashes}")

    torch.manual_seed(bidir_config.seed)
    bidir = LanguageModel(bidir_config.model).cuda().train()
    causal_config = build_config(
        arm="causal_gdn2",
        sequence_length=SEQUENCE_LENGTH,
        num_kv_pairs=4,
        max_epochs=10,
        batch_size=32,
    )
    torch.manual_seed(causal_config.seed)
    causal = make_model(causal_config, "causal_gdn2").cuda().eval()

    inputs, labels, _ = next(iter(test_loader))
    inputs = inputs[:2].cuda()
    labels = labels[:2].cuda()
    altered = changed_future_values(inputs)
    quarter = SEQUENCE_LENGTH // 4
    positions = torch.arange(SEQUENCE_LENGTH, device="cuda")
    future_mask = (labels != -100) & (positions[None, :] < quarter)

    with torch.no_grad():
        causal_logits = causal(inputs)
        causal_altered = causal(altered)
        bidir_logits = bidir(inputs)
        bidir_altered = bidir(altered)
    causal_dependency = float(
        (causal_logits[future_mask] - causal_altered[future_mask]).abs().max()
    )
    bidir_dependency = float(
        (bidir_logits[future_mask] - bidir_altered[future_mask]).abs().mean()
    )
    if causal_dependency != 0.0:
        raise RuntimeError("Strict causal GDN2 leaked future values")
    if bidir_dependency <= 1e-6:
        raise RuntimeError("Reverse GDN2 stream did not expose future values")

    bidir.zero_grad(set_to_none=True)
    logits = bidir(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Bidirectional GDN2 loss is non-finite")
    backward_grad = max(
        float(parameter.grad.detach().abs().max())
        for name, parameter in bidir.named_parameters()
        if ".backward_layer." in name and parameter.grad is not None
    )
    merge_grad = max(
        float(parameter.grad.detach().abs().max())
        for name, parameter in bidir.named_parameters()
        if ".merge." in name and parameter.grad is not None
    )
    if backward_grad <= 0 or merge_grad <= 0:
        raise RuntimeError("Reverse stream or fusion projection has zero gradient")

    mixers = [
        module
        for module in bidir.modules()
        if isinstance(module, ExplicitBidirectionalGDN2Mixer)
    ]
    official = [module for module in bidir.modules() if isinstance(module, GatedDeltaNet2)]
    if len(mixers) != 2 or len(official) != 4:
        raise RuntimeError("Expected two mixers and four official GDN2 streams")
    if any(isinstance(module, ZoologyGDN2FutureSeedMixer) for module in bidir.modules()):
        raise RuntimeError("Explicit bidirectional baseline unexpectedly contains FutureSeed")
    modes = [module.mode for module in official]
    backends = [module.q_conv1d.backend for module in official]
    if set(modes) != {"chunk"} or set(backends) != {"triton"}:
        raise RuntimeError(f"Official CUDA mode/backend drifted: {modes} {backends}")

    result = {
        "device": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "official_gdn2_source": str(Path(inspect.getfile(GatedDeltaNet2)).resolve()),
        "data_hashes": hashes,
        "reference_git_sha": REFERENCE_GIT_SHA,
        "reference_score_sha256": REFERENCE_SCORE_SHA256,
        "parameters": {
            "causal_gdn2": sum(p.numel() for p in causal.parameters()),
            "explicit_bidirectional_gdn2": sum(p.numel() for p in bidir.parameters()),
        },
        "official_gdn2_streams": len(official),
        "gdn2_modes": modes,
        "conv_backends": backends,
        "causal_future_dependency_max": causal_dependency,
        "bidirectional_future_dependency_mean": bidir_dependency,
        "backward_stream_gradient_max": backward_grad,
        "merge_gradient_max": merge_grad,
        "loss": float(loss.detach()),
        "peak_cuda_mem_bytes": torch.cuda.max_memory_allocated(),
        "future_seed_present": False,
        "reverse_scan_present": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
