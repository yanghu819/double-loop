from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
import fla
from torch import nn
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
)

from fla.layers.gdn2 import GatedDeltaNet2
from zoology.config import ModelConfig, ModuleConfig
from zoology.model import LMBackbone
from zoology.utils import set_determinism

from experiments.zoology_mqar.gdn2_futureseed import (
    PINNED_FLA_SHA,
    FutureSeedLMBackbone,
    ZoologyGDN2FutureSeedMixer,
    futureseed_diagnostics,
)


EXPECTED_GPU_UUID = "GPU-53e9f3b4-2966-65d3-6614-09c540921519"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_TRANSFORMERS_VERSION = "4.46.3"
EXPECTED_BERT_SOURCE_SHA256 = (
    "3493bff5da90fdcce98dad5c84aafe4d3ce1c550dcd93bc99289309953559eca"
)
EXPECTED_FLA_WHEEL_SHA256 = (
    "0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
)
EXPECTED_GDN2_SOURCE_SHA256 = (
    "4d001b6a8903cc7acb42b908ab3ade0f1edca0a17275630fc9c080c0232bf910"
)
VOCAB_SIZE = 30_522
HIDDEN_SIZE = 128
PLAN_ID = os.environ.get("WORDPIECE_EXPERIMENT_PLAN", "P-CAUSAL-019")
REGISTERED_LAYERS = {
    "P-CAUSAL-019": 2,
    "P-CAUSAL-020": 4,
}
if PLAN_ID not in REGISTERED_LAYERS:
    raise RuntimeError(f"Unregistered WordPiece experiment plan: {PLAN_ID}")
LAYERS = int(os.environ.get("WORDPIECE_MODEL_LAYERS", REGISTERED_LAYERS[PLAN_ID]))
if LAYERS != REGISTERED_LAYERS[PLAN_ID]:
    raise RuntimeError(
        f"{PLAN_ID} requires {REGISTERED_LAYERS[PLAN_ID]} layers, got {LAYERS}"
    )
HEADS = 4
HEAD_DIM = 32
EVAL_STEPS = (0, 250, 500, 750, 1000, 1250)
P019_L2_ACCURACY_DELTA = 0.002530577815267776
P019_L2_CE_ADVANTAGE = 0.08734900556199943


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(named_tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        digest.update(name.encode("utf-8"))
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    return tensor_digest(model.state_dict().items())


def parameter_hash(model: nn.Module) -> str:
    return tensor_digest(model.named_parameters())


def verify_fla_tree(wheel_path: Path) -> dict[str, Any]:
    wheel_path = wheel_path.resolve()
    if sha256(wheel_path) != EXPECTED_FLA_WHEEL_SHA256:
        raise RuntimeError("Pinned FLA wheel hash drifted")
    package_root = Path(fla.__file__).resolve().parent
    expected_root = Path(os.environ["FLA_SOURCE_ROOT"]).resolve()
    if package_root != expected_root / "fla":
        raise RuntimeError(
            f"FLA imported from {package_root}, expected {expected_root / 'fla'}"
        )
    wheel_tree = hashlib.sha256()
    installed_tree = hashlib.sha256()
    verified = 0
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("fla/") and not name.endswith("/")
        )
        for name in wheel_files:
            installed_path = expected_root / name
            if not installed_path.is_file():
                raise RuntimeError(f"Installed FLA file is missing: {installed_path}")
            wheel_hash = hashlib.sha256(archive.read(name)).hexdigest()
            installed_hash = sha256(installed_path)
            if wheel_hash != installed_hash:
                raise RuntimeError(f"Installed FLA file differs from wheel: {name}")
            encoded = name.encode("utf-8")
            for digest, value in (
                (wheel_tree, wheel_hash),
                (installed_tree, installed_hash),
            ):
                digest.update(encoded)
                digest.update(b"\0")
                digest.update(bytes.fromhex(value))
            verified += 1
        installed_python = {
            str(path.relative_to(expected_root))
            for path in package_root.rglob("*.py")
        }
        wheel_python = {name for name in wheel_files if name.endswith(".py")}
        extras = sorted(installed_python - wheel_python)
        if extras:
            raise RuntimeError(f"Installed FLA has extra Python sources: {extras}")
    gdn2_source = Path(inspect.getfile(GatedDeltaNet2)).resolve()
    if sha256(gdn2_source) != EXPECTED_GDN2_SOURCE_SHA256:
        raise RuntimeError("Official GDN2 source hash drifted")
    return {
        "wheel_path": str(wheel_path),
        "wheel_sha256": EXPECTED_FLA_WHEEL_SHA256,
        "package_root": str(package_root),
        "verified_files": verified,
        "wheel_tree_sha256": wheel_tree.hexdigest(),
        "installed_tree_sha256": installed_tree.hexdigest(),
        "gdn2_source": str(gdn2_source),
        "gdn2_source_sha256": EXPECTED_GDN2_SOURCE_SHA256,
    }


def load_grouped_examples(
    json_path: Path,
    tokenizer: Any,
    sequence_length: int,
    limit: int,
) -> list[dict[str, list[int]]]:
    texts = [
        json.loads(line)["text"]
        for line in json_path.read_text(encoding="utf-8").splitlines()
    ]
    tokenized: dict[str, list[list[int]]] = {}
    for start in range(0, len(texts), 256):
        encoded = tokenizer(
            texts[start : start + 256],
            return_special_tokens_mask=True,
        )
        for key, rows in encoded.items():
            tokenized.setdefault(key, []).extend(rows)
    concatenated = {
        key: list(chain.from_iterable(rows)) for key, rows in tokenized.items()
    }
    total = (len(concatenated["input_ids"]) // sequence_length) * sequence_length
    available = total // sequence_length
    if available < limit:
        raise RuntimeError(f"Only {available} grouped windows are available, need {limit}")
    windows = [
        {
            key: values[start : start + sequence_length]
            for key, values in concatenated.items()
        }
        for start in range(0, limit * sequence_length, sequence_length)
    ]
    if any(value != 1 for row in windows for value in row["attention_mask"]):
        raise RuntimeError("Grouped WordPiece windows contain padding")
    if any(value != 0 for row in windows for value in row["token_type_ids"]):
        raise RuntimeError("Grouped WordPiece windows contain nonzero token types")
    return windows


def corruption(
    examples: list[dict[str, list[int]]],
    tokenizer: Any,
    probability: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(seed)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=probability,
        return_tensors="pt",
    )
    batch = collator(examples)
    return batch["input_ids"].long().contiguous(), batch["labels"].long().contiguous()


def data_hash(
    train_epochs: list[tuple[torch.Tensor, torch.Tensor]],
    orders: list[torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
) -> str:
    rows = []
    for epoch, ((inputs, labels), order) in enumerate(zip(train_epochs, orders)):
        rows.extend(
            [
                (f"train/{epoch}/inputs", inputs),
                (f"train/{epoch}/labels", labels),
                (f"train/{epoch}/order", order),
            ]
        )
    rows.extend(
        [
            ("validation/inputs", validation[0]),
            ("validation/labels", validation[1]),
        ]
    )
    return tensor_digest(rows)


def build_config(*, future_seed_scale: float, direct: bool = False) -> ModelConfig:
    if direct:
        sequence_mixer = ModuleConfig(
            name="experiments.zoology_mqar.gdn2_mixer.ZoologyGDN2Mixer",
            kwargs={
                "num_heads": HEADS,
                "head_dim": HEAD_DIM,
                "expand_v": 1.0,
                "conv_size": 4,
            },
        )
    else:
        sequence_mixer = ModuleConfig(
            name=(
                "experiments.zoology_mqar.gdn2_futureseed."
                "ZoologyGDN2FutureSeedMixer"
            ),
            kwargs={
                "num_heads": HEADS,
                "head_dim": HEAD_DIM,
                "expand_v": 1.0,
                "conv_size": 4,
                "future_seed_scale": future_seed_scale,
            },
        )
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        pad_vocab_size_multiple=1,
        max_position_embeddings=128,
        d_model=HIDDEN_SIZE,
        n_layers=LAYERS,
        sequence_mixer=sequence_mixer,
    )


class FreshWordPieceMLMHead(nn.Module):
    def __init__(self, word_embeddings: nn.Embedding) -> None:
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.decoder = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense.bias)
        self.decoder.weight = word_embeddings.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states) + self.bias


class WordPieceGDN2ForMaskedLM(nn.Module):
    """Random GDN2 core with one shared frozen official lexical table."""

    def __init__(
        self,
        config: ModelConfig,
        pretrained: BertForMaskedLM,
        *,
        direct: bool = False,
    ) -> None:
        super().__init__()
        backbone_cls = LMBackbone if direct else FutureSeedLMBackbone
        self.backbone = backbone_cls(copy.deepcopy(config))
        with torch.no_grad():
            self.backbone.embeddings.word_embeddings.weight.copy_(
                pretrained.bert.embeddings.word_embeddings.weight
            )
        self.backbone.embeddings.word_embeddings.weight.requires_grad_(False)
        self.mlm_head = FreshWordPieceMLMHead(
            self.backbone.embeddings.word_embeddings
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids)

    def masked_logits(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = labels != -100
        hidden = self.encode(input_ids)
        return self.mlm_head(hidden[mask]), labels[mask]


def lexical_integrity(
    model: WordPieceGDN2ForMaskedLM,
    expected_hash: str,
) -> dict[str, Any]:
    embedding = model.backbone.embeddings.word_embeddings.weight
    decoder = model.mlm_head.decoder.weight
    actual_hash = tensor_digest([("word_embeddings", embedding.detach())])
    result = {
        "sha256": actual_hash,
        "frozen": not embedding.requires_grad,
        "input_output_tied": embedding.data_ptr() == decoder.data_ptr(),
        "pretrained_bert_module_present": any(
            isinstance(module, BertForMaskedLM) for module in model.modules()
        ),
    }
    if actual_hash != expected_hash:
        raise RuntimeError("Frozen lexical tensor differs from registered source")
    if not result["frozen"] or not result["input_output_tied"]:
        raise RuntimeError(f"Frozen/tied lexical contract failed: {result}")
    if result["pretrained_bert_module_present"]:
        raise RuntimeError("Pretrained BERT encoder is present in the GDN2 model")
    return result


def build_model(
    pretrained: BertForMaskedLM,
    *,
    future_seed_scale: float,
    direct: bool = False,
) -> WordPieceGDN2ForMaskedLM:
    set_determinism(123)
    return WordPieceGDN2ForMaskedLM(
        build_config(future_seed_scale=future_seed_scale, direct=direct),
        pretrained,
        direct=direct,
    )


def shared_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach()
        for name, tensor in model.state_dict().items()
        if not name.endswith("future_seed_logit")
    }


def visible_gpu() -> dict[str, str | int]:
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one visible GPU, got {len(rows)}")
    index, uuid, name, memory = [item.strip() for item in rows[0].split(",")]
    if index != "0" or uuid != EXPECTED_GPU_UUID or name != EXPECTED_GPU_NAME:
        raise RuntimeError(f"Unexpected GPU row: {rows[0]}")
    return {"index": int(index), "uuid": uuid, "name": name, "memory_mib": int(memory)}


def probe_positions(
    labels: torch.Tensor,
    count: int | None = 8,
) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    positions: list[int] = []
    for row in range(labels.shape[0]):
        candidates = torch.nonzero(labels[row] != -100).flatten()
        candidates = candidates[candidates < labels.shape[1] // 2]
        if len(candidates):
            rows.append(row)
            positions.append(int(candidates[0]))
        if count is not None and len(rows) == count:
            break
    if count is not None and len(rows) != count:
        raise RuntimeError("Insufficient early masked positions for dependency probe")
    if not rows:
        raise RuntimeError("No early masked positions for dependency probe")
    return rows, positions


@torch.no_grad()
def future_dependency(
    model: WordPieceGDN2ForMaskedLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    replacement_id: int,
) -> float:
    rows, positions = probe_positions(labels)
    original = inputs[rows].cuda()
    altered = original.clone()
    for local_row, position in enumerate(positions):
        altered[local_row, position + 1 :] = replacement_id
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        original_hidden = model.encode(original)
        altered_hidden = model.encode(altered)
        original_logits = model.mlm_head(
            torch.stack(
                [original_hidden[row, pos] for row, pos in enumerate(positions)]
            )
        ).float()
        altered_logits = model.mlm_head(
            torch.stack(
                [altered_hidden[row, pos] for row, pos in enumerate(positions)]
            )
        ).float()
    return float((original_logits - altered_logits).abs().mean().item())


@torch.no_grad()
def suffix_utility(
    model: WordPieceGDN2ForMaskedLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    replacement_id: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    rows, positions = probe_positions(labels, count=None)
    differences = []
    original_nll = []
    altered_nll = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_positions = positions[start : start + batch_size]
        original = inputs[batch_rows].cuda()
        altered = original.clone()
        targets = torch.tensor(
            [int(labels[row, pos]) for row, pos in zip(batch_rows, batch_positions)],
            device="cuda",
        )
        for local_row, position in enumerate(batch_positions):
            altered[local_row, position + 1 :] = replacement_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            original_hidden = model.encode(original)
            altered_hidden = model.encode(altered)
            original_logits = model.mlm_head(
                torch.stack(
                    [
                        original_hidden[row, position]
                        for row, position in enumerate(batch_positions)
                    ]
                )
            ).float()
            altered_logits = model.mlm_head(
                torch.stack(
                    [
                        altered_hidden[row, position]
                        for row, position in enumerate(batch_positions)
                    ]
                )
            ).float()
        original_loss = F.cross_entropy(original_logits, targets, reduction="none")
        altered_loss = F.cross_entropy(altered_logits, targets, reduction="none")
        original_nll.extend(original_loss.cpu().tolist())
        altered_nll.extend(altered_loss.cpu().tolist())
        differences.extend((altered_loss - original_loss).cpu().tolist())
    lower, upper = bootstrap_mean_interval(differences, seed=31_337)
    return {
        "probe_windows": len(differences),
        "original_ce": sum(original_nll) / len(original_nll),
        "suffix_replaced_ce": sum(altered_nll) / len(altered_nll),
        "ce_degradation_when_suffix_removed": sum(differences) / len(differences),
        "bootstrap_95_lower": lower,
        "bootstrap_95_upper": upper,
        "max_abs_change": max(abs(value) for value in differences),
    }


def bootstrap_mean_interval(
    values: list[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        len(tensor),
        (samples, len(tensor)),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    lower, upper = torch.quantile(
        means, torch.tensor([0.025, 0.975], dtype=means.dtype)
    ).tolist()
    return float(lower), float(upper)


def autograd_graph_names(tensor: torch.Tensor) -> list[str]:
    names: list[str] = []
    queue = [tensor.grad_fn]
    seen: set[int] = set()
    keepalive = []
    while queue:
        fn = queue.pop(0)
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keepalive.append(fn)
        names.append(type(fn).__name__)
        queue.extend(
            next_fn
            for next_fn, _index in fn.next_functions
            if next_fn is not None
        )
    return names


def finite_backward(
    model: WordPieceGDN2ForMaskedLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, targets = model.masked_logits(inputs.cuda(), labels.cuda())
        loss = F.cross_entropy(logits.float(), targets)
    graph_names = autograd_graph_names(loss)
    expected_node = "ChunkGDN2FunctionBackward"
    if expected_node not in graph_names:
        raise RuntimeError(
            "Masked-token loss did not traverse the official FLA GDN2 chunk "
            f"autograd node {expected_node}: {sorted(set(graph_names))}"
        )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Missing or non-finite CUDA gradient")
    return {
        "loss": float(loss.item()),
        "max_gradient": max(float(gradient.abs().max().item()) for gradient in gradients),
        "official_chunk_autograd_node": expected_node,
    }


@torch.no_grad()
def pretrained_carrier_anchor(
    pretrained: BertForMaskedLM,
    validation: tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 32,
) -> dict[str, float | int]:
    pretrained = pretrained.cuda().eval()
    loss_sum = 0.0
    correct = 0
    masked = 0
    for start in range(0, validation[0].shape[0], batch_size):
        inputs = validation[0][start : start + batch_size].cuda()
        labels = validation[1][start : start + batch_size].cuda()
        attention_mask = torch.ones_like(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = pretrained(
                input_ids=inputs,
                attention_mask=attention_mask,
            ).logits.float()
        mask = labels != -100
        loss_sum += float(F.cross_entropy(logits[mask], labels[mask], reduction="sum").item())
        correct += int(((logits.argmax(dim=-1) == labels) & mask).sum().item())
        masked += int(mask.sum().item())
    metrics = {
        "masked_accuracy": correct / masked,
        "masked_ce": loss_sum / masked,
        "masked_correct": correct,
        "masked_tokens": masked,
    }
    if correct != 1_686 or masked != 4_742:
        raise RuntimeError(f"P018 carrier anchor accuracy drifted: {metrics}")
    if abs(float(metrics["masked_ce"]) - 4.033268768040252) > 1e-5:
        raise RuntimeError(f"P018 carrier anchor CE drifted: {metrics}")
    pretrained.cpu()
    torch.cuda.empty_cache()
    return metrics


def preflight(
    pretrained: BertForMaskedLM,
    validation: tuple[torch.Tensor, torch.Tensor],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    gpu = visible_gpu()
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("CUDA does not expose exactly one device")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(f"Unexpected Transformers version: {transformers.__version__}")
    bert_source = Path(inspect.getfile(BertForMaskedLM)).resolve()
    if sha256(bert_source) != EXPECTED_BERT_SOURCE_SHA256:
        raise RuntimeError(f"Unexpected BertForMaskedLM source: {bert_source}")
    if os.environ.get("FLA_EXPECTED_SOURCE_SHA") != PINNED_FLA_SHA:
        raise RuntimeError("Pinned FLA SHA was not asserted")
    if os.environ.get("FLA_WHEEL_SHA256") != EXPECTED_FLA_WHEEL_SHA256:
        raise RuntimeError("Pinned FLA wheel SHA was not asserted")
    if os.environ.get("FLA_GDN2_SOURCE_SHA256") != EXPECTED_GDN2_SOURCE_SHA256:
        raise RuntimeError("Pinned GDN2 source SHA was not asserted")
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1":
        raise RuntimeError("FLA backend dispatch must be disabled")
    if os.environ.get("FLA_CONV_BACKEND") != "triton":
        raise RuntimeError("FLA short convolution must use Triton")

    fla_provenance = verify_fla_tree(Path(os.environ["FLA_WHEEL"]))
    carrier_anchor = pretrained_carrier_anchor(pretrained, validation)

    direct = build_model(pretrained, future_seed_scale=0.0, direct=True)
    causal = build_model(pretrained, future_seed_scale=0.0)
    future_seed = build_model(pretrained, future_seed_scale=1.0)
    source_lexical = pretrained.bert.embeddings.word_embeddings.weight.detach()
    target_lexical = causal.backbone.embeddings.word_embeddings.weight.detach()
    if not torch.equal(source_lexical.cpu(), target_lexical.cpu()):
        raise RuntimeError("Official lexical table was not copied exactly")
    lexical_table_hash = tensor_digest([("word_embeddings", source_lexical)])
    if lexical_table_hash != tensor_digest([("word_embeddings", target_lexical)]):
        raise RuntimeError("Copied lexical tensor hash differs from official source")
    lexical_contracts = {
        "direct_gdn2": lexical_integrity(direct, lexical_table_hash),
        "causal_gdn2": lexical_integrity(causal, lexical_table_hash),
        "future_seed_gdn2": lexical_integrity(future_seed, lexical_table_hash),
    }
    if state_hash(causal) != state_hash(future_seed):
        raise RuntimeError("Matched scale-0/scale-1 initialization differs")
    if parameter_hash(causal) != parameter_hash(future_seed):
        raise RuntimeError("Matched scale-0/scale-1 parameter tensors differ")
    if sum(p.numel() for p in causal.parameters()) != sum(
        p.numel() for p in future_seed.parameters()
    ):
        raise RuntimeError("Matched arms have different parameter counts")

    direct_state = direct.state_dict()
    causal_shared = shared_state(causal)
    if set(direct_state) != set(causal_shared):
        missing = sorted(set(direct_state) - set(causal_shared))
        extra = sorted(set(causal_shared) - set(direct_state))
        raise RuntimeError(f"Scale-0 state keys differ: missing={missing} extra={extra}")
    shared_init_max_diff = max(
        float((direct_state[name] - causal_shared[name]).abs().max().item())
        for name in direct_state
    )
    if shared_init_max_diff != 0.0:
        raise RuntimeError(f"Scale-0 shared initialization differs: {shared_init_max_diff}")

    initial_state = {
        name: tensor.detach().cpu().clone() for name, tensor in causal.state_dict().items()
    }
    sample_inputs = validation[0][:8].cuda()
    sample_labels = validation[1][:8].cuda()
    direct = direct.cuda().eval()
    causal = causal.cuda().eval()
    future_seed = future_seed.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct_hidden = direct.encode(sample_inputs)
        causal_hidden = causal.encode(sample_inputs)
        direct_logits = direct.mlm_head(direct_hidden[sample_labels != -100]).float()
        causal_logits = causal.mlm_head(causal_hidden[sample_labels != -100]).float()
    scale0_hidden_max_diff = float((direct_hidden - causal_hidden).abs().max().item())
    scale0_output_max_diff = float((direct_logits - causal_logits).abs().max().item())
    if scale0_hidden_max_diff != 0.0 or scale0_output_max_diff != 0.0:
        raise RuntimeError(
            f"Scale-0 identity failed: hidden={scale0_hidden_max_diff} "
            f"logits={scale0_output_max_diff}"
        )

    dependencies = {
        "causal_gdn2": future_dependency(
            causal, validation[0], validation[1], tokenizer.unk_token_id
        ),
        "future_seed_gdn2": future_dependency(
            future_seed, validation[0], validation[1], tokenizer.unk_token_id
        ),
    }
    if dependencies["causal_gdn2"] != 0.0:
        raise RuntimeError(f"Strict causal arm leaks future input: {dependencies}")
    if dependencies["future_seed_gdn2"] <= 0.0:
        raise RuntimeError("FutureSeed arm has no future-input dependency")

    backward = {
        "causal_gdn2": finite_backward(causal, validation[0][:2], validation[1][:2]),
        "future_seed_gdn2": finite_backward(
            future_seed, validation[0][:2], validation[1][:2]
        ),
    }
    gate_gradients = [
        {
            "name": name,
            "max_abs_gradient": float(parameter.grad.detach().abs().max().item()),
        }
        for name, parameter in future_seed.named_parameters()
        if name.endswith("future_seed_logit") and parameter.grad is not None
    ]
    if len(gate_gradients) != LAYERS - 1:
        raise RuntimeError(
            f"Expected {LAYERS - 1} active FutureSeed gates, got {gate_gradients}"
        )
    if any(row["max_abs_gradient"] <= 0.0 for row in gate_gradients):
        raise RuntimeError(f"A FutureSeed gate has zero gradient: {gate_gradients}")
    gate_gradient_max = max(row["max_abs_gradient"] for row in gate_gradients)

    mixers = [layer.sequence_mixer for layer in future_seed.backbone.layers]
    conv_backends = [
        {
            name: getattr(getattr(mixer.layer, name), "backend", None)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        }
        for mixer in mixers
        if isinstance(mixer, ZoologyGDN2FutureSeedMixer)
    ]
    if not conv_backends or any(
        set(layer_backends.values()) != {"triton"}
        for layer_backends in conv_backends
    ):
        raise RuntimeError(f"GDN2 short convolution backend drifted: {conv_backends}")
    source = str(Path(inspect.getfile(GatedDeltaNet2)).resolve())
    result = {
        "gpu": gpu,
        "transformers_version": transformers.__version__,
        "bert_source": str(bert_source),
        "bert_source_sha256": EXPECTED_BERT_SOURCE_SHA256,
        "gdn2_class": f"{GatedDeltaNet2.__module__}.{GatedDeltaNet2.__name__}",
        "gdn2_source": source,
        "fla_sha": PINNED_FLA_SHA,
        "fla_provenance": fla_provenance,
        "gdn2_mode": [mixer.layer.mode for mixer in mixers],
        "gdn2_conv_backend": conv_backends,
        "parameters": sum(p.numel() for p in causal.parameters()),
        "initial_state_hash": state_hash(causal),
        "parameter_hash": parameter_hash(causal),
        "shared_init_max_diff": shared_init_max_diff,
        "scale0_hidden_max_diff": scale0_hidden_max_diff,
        "scale0_output_max_diff": scale0_output_max_diff,
        "future_dependency_mean_abs": dependencies,
        "future_seed_gate_gradient_max": gate_gradient_max,
        "future_seed_gate_gradients": gate_gradients,
        "expected_active_seed_routes": LAYERS - 1,
        "backward": backward,
        "pretrained_carrier_anchor": carrier_anchor,
        "common_pretrained_modules": ["bert.embeddings.word_embeddings.weight"],
        "lexical_table_sha256": lexical_table_hash,
        "lexical_contracts": lexical_contracts,
        "pretrained_encoder_reused": any(
            row["pretrained_bert_module_present"]
            for row in lexical_contracts.values()
        ),
        "trainable_parameters": sum(
            p.numel() for p in causal.parameters() if p.requires_grad
        ),
        "frozen_parameters": sum(
            p.numel() for p in causal.parameters() if not p.requires_grad
        ),
    }
    del direct, causal, future_seed
    torch.cuda.empty_cache()
    return result, initial_state


def token_text(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(token_id)))


@torch.no_grad()
def evaluate(
    model: WordPieceGDN2ForMaskedLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    originals: torch.Tensor,
    tokenizer: Any,
    batch_size: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    masked = 0
    exact = 0
    cases: list[dict[str, Any]] = []
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].cuda()
        batch_labels = labels[start : start + batch_size].cuda()
        mask = batch_labels != -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.encode(batch_inputs)
            logits = model.mlm_head(hidden[mask]).float()
        targets = batch_labels[mask]
        predictions = logits.argmax(dim=-1)
        probabilities = logits.softmax(dim=-1)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").item())
        correct += int((predictions == targets).sum().item())
        masked += int(targets.numel())

        flat_offset = 0
        for local_row in range(batch_inputs.shape[0]):
            positions = torch.nonzero(mask[local_row]).flatten().tolist()
            count = len(positions)
            row_predictions = predictions[flat_offset : flat_offset + count]
            row_probabilities = probabilities[flat_offset : flat_offset + count]
            row_targets = targets[flat_offset : flat_offset + count]
            row_correct = row_predictions == row_targets
            exact += int(bool(row_correct.all().item()))
            masked_rows = []
            for index, position in enumerate(positions):
                top_prob, top_ids = row_probabilities[index].topk(5)
                target_id = int(row_targets[index].item())
                prediction_id = int(row_predictions[index].item())
                masked_rows.append(
                    {
                        "position": position,
                        "target_id": target_id,
                        "target": token_text(tokenizer, target_id),
                        "prediction_id": prediction_id,
                        "prediction": token_text(tokenizer, prediction_id),
                        "correct": prediction_id == target_id,
                        "true_probability": float(row_probabilities[index, target_id].item()),
                        "top5": [
                            {
                                "token_id": int(token_id),
                                "token": token_text(tokenizer, int(token_id)),
                                "probability": float(probability),
                            }
                            for probability, token_id in zip(
                                top_prob.tolist(), top_ids.tolist()
                            )
                        ],
                    }
                )
            global_row = start + local_row
            original_ids = originals[global_row].tolist()
            cases.append(
                {
                    "case_index": global_row,
                    "case_id": (
                        f"{global_row:04d}-"
                        + hashlib.sha256(
                            originals[global_row].contiguous().numpy().tobytes()
                        ).hexdigest()[:16]
                    ),
                    "errors": sum(not row["correct"] for row in masked_rows),
                    "tokens": [token_text(tokenizer, token_id) for token_id in original_ids],
                    "masked_tokens": masked_rows,
                }
            )
            flat_offset += count
    return (
        {
            "masked_ce": loss_sum / masked,
            "masked_accuracy": correct / masked,
            "masked_exact": exact / inputs.shape[0],
            "masked_tokens": masked,
            "examples": int(inputs.shape[0]),
        },
        cases,
    )


def optimizer_groups(
    model: WordPieceGDN2ForMaskedLM,
    *,
    learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    groups: dict[bool, list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        groups.setdefault(no_decay, []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "group_name": "no_decay" if no_decay else "decay",
        }
        for no_decay, parameters in sorted(groups.items())
    ]


def registered_batch(
    train_epochs: list[tuple[torch.Tensor, torch.Tensor]],
    orders: list[torch.Tensor],
    batch_index: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not train_epochs or len(train_epochs) != len(orders):
        raise RuntimeError("Training corruption banks and orders differ")
    windows = len(orders[0])
    if any(len(order) != windows for order in orders):
        raise RuntimeError("Training corruption banks have different sizes")
    start = batch_index * batch_size
    stop = start + batch_size
    if stop > windows * len(orders):
        raise RuntimeError("Registered batch schedule exceeds unique sample pairs")
    input_parts = []
    label_parts = []
    cursor = start
    while cursor < stop:
        bank = cursor // windows
        offset = cursor % windows
        count = min(stop - cursor, windows - offset)
        indices = orders[bank][offset : offset + count]
        input_parts.append(train_epochs[bank][0][indices])
        label_parts.append(train_epochs[bank][1][indices])
        cursor += count
    inputs = torch.cat(input_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    if inputs.shape[0] != batch_size or labels.shape[0] != batch_size:
        raise RuntimeError("Registered batch is not full")
    return inputs, labels


def benchmark_train_step(
    model: WordPieceGDN2ForMaskedLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    warmup_steps: int = 5,
    measured_steps: int = 20,
) -> dict[str, float | int]:
    model.train()
    inputs = inputs.cuda()
    labels = labels.cuda()

    def one_step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, targets = model.masked_logits(inputs, labels)
            loss = F.cross_entropy(logits.float(), targets)
        loss.backward()

    for _ in range(warmup_steps):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measured_steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.zero_grad(set_to_none=True)
    return {
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "elapsed_seconds": elapsed,
        "input_tokens_per_second": inputs.numel() * measured_steps / elapsed,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def paired_case_bootstrap(
    causal_cases: list[dict[str, Any]],
    future_cases: list[dict[str, Any]],
    *,
    samples: int = 10_000,
    seed: int = 91_337,
) -> dict[str, float | int]:
    if len(causal_cases) != 256 or len(future_cases) != 256:
        raise RuntimeError("Paired bootstrap requires exactly 256 windows per arm")
    causal_by_index = {int(row["case_index"]): row for row in causal_cases}
    future_by_index = {int(row["case_index"]): row for row in future_cases}
    if len(causal_by_index) != 256 or len(future_by_index) != 256:
        raise RuntimeError("Paired bootstrap case indices are not unique")
    if set(causal_by_index) != set(future_by_index):
        raise RuntimeError("Matched validation case indices differ")
    rows = []
    for case_index in sorted(causal_by_index):
        causal = causal_by_index[case_index]
        future = future_by_index[case_index]
        if causal["case_id"] != future["case_id"]:
            raise RuntimeError(f"Case IDs differ at index {case_index}")
        if causal["tokens"] != future["tokens"]:
            raise RuntimeError(f"Original tokens differ at index {case_index}")
        if len(causal["masked_tokens"]) != len(future["masked_tokens"]):
            raise RuntimeError(f"Masked target count differs at index {case_index}")
        causal_targets = [
            (int(token["position"]), int(token["target_id"]))
            for token in causal["masked_tokens"]
        ]
        future_targets = [
            (int(token["position"]), int(token["target_id"]))
            for token in future["masked_tokens"]
        ]
        if causal_targets != future_targets:
            raise RuntimeError(f"Masked positions/targets differ at index {case_index}")
        target_count = len(causal["masked_tokens"])
        causal_correct = target_count - int(causal["errors"])
        future_correct = target_count - int(future["errors"])
        causal_nll = -sum(
            math.log(max(float(token["true_probability"]), 1e-30))
            for token in causal["masked_tokens"]
        )
        future_nll = -sum(
            math.log(max(float(token["true_probability"]), 1e-30))
            for token in future["masked_tokens"]
        )
        rows.append(
            [
                float(target_count),
                float(future_correct - causal_correct),
                float(causal_nll - future_nll),
            ]
        )
    tensor = torch.tensor(rows, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, len(rows), (samples, len(rows)), generator=generator
    )
    sampled = tensor[indices].sum(dim=1)
    accuracy_delta = sampled[:, 1] / sampled[:, 0]
    ce_improvement = sampled[:, 2] / sampled[:, 0]
    accuracy_interval = torch.quantile(
        accuracy_delta,
        torch.tensor([0.025, 0.975], dtype=accuracy_delta.dtype),
    ).tolist()
    ce_interval = torch.quantile(
        ce_improvement,
        torch.tensor([0.025, 0.975], dtype=ce_improvement.dtype),
    ).tolist()
    return {
        "windows": len(rows),
        "samples": samples,
        "seed": seed,
        "accuracy_delta_95_lower": float(accuracy_interval[0]),
        "accuracy_delta_95_upper": float(accuracy_interval[1]),
        "ce_improvement_95_lower": float(ce_interval[0]),
        "ce_improvement_95_upper": float(ce_interval[1]),
    }


def run_arm(
    *,
    arm: str,
    pretrained: BertForMaskedLM,
    initial_state: dict[str, torch.Tensor],
    train_epochs: list[tuple[torch.Tensor, torch.Tensor]],
    orders: list[torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    validation_originals: torch.Tensor,
    tokenizer: Any,
    output_dir: Path,
    checkpoint_dir: Path,
    max_steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    lexical_table_hash: str,
) -> dict[str, Any]:
    scale = 0.0 if arm == "causal_gdn2" else 1.0
    set_seed(seed)
    model = build_model(pretrained, future_seed_scale=scale)
    model.load_state_dict(initial_state, strict=True)
    if state_hash(model) != tensor_digest(initial_state.items()):
        raise RuntimeError(f"{arm} did not load the registered shared initialization")
    initial_lexical = lexical_integrity(model, lexical_table_hash)
    model = model.cuda()

    groups = optimizer_groups(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps
    )

    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict[str, Any]] = []
    metrics, _ = evaluate(
        model,
        validation[0],
        validation[1],
        validation_originals,
        tokenizer,
        eval_batch_size,
    )
    curve.append({"step": 0, **metrics})
    print(json.dumps({"arm": arm, "event": "validation", **curve[-1]}), flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    last_loss = math.nan
    registered_pairs = sum(len(order) for order in orders)
    if registered_pairs != max_steps * batch_size:
        raise RuntimeError(
            f"Registered schedule has {registered_pairs} pairs, expected "
            f"{max_steps * batch_size}"
        )
    for batch_index in range(max_steps):
        batch_inputs, batch_labels = registered_batch(
            train_epochs, orders, batch_index, batch_size
        )
        inputs = batch_inputs.cuda(non_blocking=True)
        labels = batch_labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, targets = model.masked_logits(inputs, labels)
            loss = F.cross_entropy(logits.float(), targets)
        global_step = batch_index + 1
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{arm} produced non-finite loss at step {global_step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())
        if global_step % 50 == 0:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "event": "train",
                        "step": global_step,
                        "corruption_bank": (
                            batch_index * batch_size // len(orders[0])
                        ),
                        "loss": last_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                ),
                flush=True,
            )
        if global_step in EVAL_STEPS:
            metrics, _ = evaluate(
                model,
                validation[0],
                validation[1],
                validation_originals,
                tokenizer,
                eval_batch_size,
            )
            curve.append({"step": global_step, **metrics})
            print(
                json.dumps({"arm": arm, "event": "validation", **curve[-1]}),
                flush=True,
            )
            model.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_training = torch.cuda.max_memory_allocated()

    final_metrics, cases = evaluate(
        model,
        validation[0],
        validation[1],
        validation_originals,
        tokenizer,
        eval_batch_size,
    )
    dependency = future_dependency(
        model, validation[0], validation[1], tokenizer.unk_token_id
    )
    suffix = suffix_utility(
        model, validation[0], validation[1], tokenizer.unk_token_id
    )
    benchmark_inputs, benchmark_labels = registered_batch(
        train_epochs, orders, 0, batch_size
    )
    benchmark = benchmark_train_step(
        model,
        benchmark_inputs,
        benchmark_labels,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_lexical = lexical_integrity(model, lexical_table_hash)
    checkpoint_path = checkpoint_dir / f"{arm}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_lexical_hash = tensor_digest(
        [
            (
                "word_embeddings",
                saved_state["backbone.embeddings.word_embeddings.weight"],
            )
        ]
    )
    if saved_lexical_hash != lexical_table_hash:
        raise RuntimeError(f"{arm} checkpoint changed the frozen lexical table")
    opening_step = next(
        (int(row["step"]) for row in curve if row["masked_accuracy"] >= 0.10),
        None,
    )
    score = {
        "arm": arm,
        "future_seed_scale": scale,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_hash": tensor_digest(initial_state.items()),
        "steps": global_step,
        "input_tokens": global_step * batch_size * train_epochs[0][0].shape[1],
        "last_train_loss": last_loss,
        "elapsed_seconds_including_validation": elapsed,
        "peak_training_cuda_memory_bytes": peak_training,
        "metrics": final_metrics,
        "validation_curve": curve,
        "opening_step_at_0.10_accuracy": opening_step,
        "future_dependency_mean_abs": dependency,
        "suffix_utility": suffix,
        "future_seed": futureseed_diagnostics(model),
        "lexical_integrity": {
            "initial": initial_lexical,
            "final": final_lexical,
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_lexical_sha256": saved_lexical_hash,
        },
        "warmed_train_step_benchmark": benchmark,
    }
    (arm_dir / "score.json").write_text(
        json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (arm_dir / "cases.json").write_text(
        json.dumps(cases, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    score["cases"] = cases
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-windows", type=int, default=10_000)
    parser.add_argument("--validation-windows", type=int, default=256)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--train-epochs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1250)
    parser.add_argument("--train-batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registered = {
        "sequence_length": 128,
        "train_windows": 10_000,
        "validation_windows": 256,
        "mask_probability": 0.15,
        "train_epochs": 16,
        "max_steps": 1_250,
        "train_batch": 128,
        "eval_batch": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "seed": 123,
    }
    drift = {
        name: {"registered": expected, "actual": getattr(args, name)}
        for name, expected in registered.items()
        if getattr(args, name) != expected
    }
    if drift:
        raise RuntimeError(f"{PLAN_ID} registered protocol drifted: {drift}")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.checkpoint_file.parent.resolve() != args.model_dir.resolve():
        raise RuntimeError("Registered checkpoint must live directly in model_dir")
    if args.checkpoint_file.suffix not in {".bin", ".safetensors"}:
        raise RuntimeError("Unsupported registered checkpoint format")
    expected_checkpoint_name = (
        "model.safetensors"
        if args.checkpoint_file.suffix == ".safetensors"
        else "pytorch_model.bin"
    )
    if args.checkpoint_file.name != expected_checkpoint_name:
        raise RuntimeError("Registered checkpoint has a noncanonical filename")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if (
        not tokenizer.is_fast
        or tokenizer.mask_token_id != 103
        or tokenizer.vocab_size != VOCAB_SIZE
        or len(tokenizer) != VOCAB_SIZE
    ):
        raise RuntimeError("Unexpected WordPiece vocabulary size")
    pretrained = BertForMaskedLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        attn_implementation="sdpa",
        use_safetensors=args.checkpoint_file.suffix == ".safetensors",
    ).cpu().eval()
    if sum(p.numel() for p in pretrained.parameters()) != 4_416_698:
        raise RuntimeError("Unexpected official BERT-Tiny parameter count")

    train_examples = load_grouped_examples(
        args.train_json, tokenizer, args.sequence_length, args.train_windows
    )
    validation_examples = load_grouped_examples(
        args.validation_json,
        tokenizer,
        args.sequence_length,
        args.validation_windows,
    )
    validation_originals = torch.tensor(
        [example["input_ids"] for example in validation_examples], dtype=torch.long
    )
    validation = corruption(
        validation_examples, tokenizer, args.mask_probability, args.seed
    )
    if int((validation[1] != -100).sum().item()) != 4_742:
        raise RuntimeError("Validation corruption no longer matches P-CAUSAL-018")

    train_epochs = [
        corruption(
            train_examples,
            tokenizer,
            args.mask_probability,
            args.seed + 10_000 + epoch,
        )
        for epoch in range(args.train_epochs)
    ]
    orders = [
        torch.randperm(
            args.train_windows,
            generator=torch.Generator().manual_seed(args.seed + 20_000 + epoch),
        )
        for epoch in range(args.train_epochs)
    ]
    registered_pairs = sum(len(order) for order in orders)
    if registered_pairs != args.max_steps * args.train_batch:
        raise RuntimeError(
            f"Registered schedule has {registered_pairs} unique pairs, expected "
            f"{args.max_steps * args.train_batch}"
        )
    prepared_hash = data_hash(train_epochs, orders, validation)
    prepared = {
        "train_windows": args.train_windows,
        "validation_windows": args.validation_windows,
        "sequence_length": args.sequence_length,
        "mask_probability": args.mask_probability,
        "train_corruptions": args.train_epochs,
        "validation_masked_tokens": int((validation[1] != -100).sum().item()),
        "prepared_tensor_sha256": prepared_hash,
        "registered_sample_pairs": registered_pairs,
        "full_batch_wrap": False,
        "every_registered_pair_consumed_once": True,
    }
    (args.output_dir / "prepared_data.json").write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preflight_result, initial_state = preflight(pretrained, validation, tokenizer)
    preflight_result["prepared_data"] = prepared
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "preflight", **preflight_result}), flush=True)
    if args.preflight_only:
        return

    scores = {}
    for arm in ("causal_gdn2", "future_seed_gdn2"):
        scores[arm] = run_arm(
            arm=arm,
            pretrained=pretrained,
            initial_state=initial_state,
            train_epochs=train_epochs,
            orders=orders,
            validation=validation,
            validation_originals=validation_originals,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_steps=args.max_steps,
            batch_size=args.train_batch,
            eval_batch_size=args.eval_batch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            lexical_table_hash=preflight_result["lexical_table_sha256"],
        )

    causal = scores["causal_gdn2"]
    future_seed = scores["future_seed_gdn2"]
    if causal["parameters"] != future_seed["parameters"]:
        raise RuntimeError("Final matched arm parameter counts differ")
    if causal["initial_state_hash"] != future_seed["initial_state_hash"]:
        raise RuntimeError("Final matched arms did not use the same initialization")
    accuracy_delta = (
        future_seed["metrics"]["masked_accuracy"]
        - causal["metrics"]["masked_accuracy"]
    )
    ce_improvement = (
        causal["metrics"]["masked_ce"] - future_seed["metrics"]["masked_ce"]
    )
    causal_step0 = causal["validation_curve"][0]["masked_ce"]
    causal_ce_drop = causal_step0 - causal["metrics"]["masked_ce"]
    opened = (
        causal["metrics"]["masked_accuracy"] >= 0.10
        and causal_ce_drop >= 0.50
    )
    curve_by_arm = {
        arm: {int(row["step"]): row for row in score["validation_curve"]}
        for arm, score in scores.items()
    }
    advantage_1000 = (
        curve_by_arm["causal_gdn2"][1000]["masked_ce"]
        - curve_by_arm["future_seed_gdn2"][1000]["masked_ce"]
    )
    advantage_1250 = ce_improvement
    accuracy_delta_1000 = (
        curve_by_arm["future_seed_gdn2"][1000]["masked_accuracy"]
        - curve_by_arm["causal_gdn2"][1000]["masked_accuracy"]
    )
    accuracy_delta_1250 = accuracy_delta
    bootstrap = paired_case_bootstrap(causal.pop("cases"), future_seed.pop("cases"))
    accuracy_route_supported = (
        accuracy_delta >= 0.03
        and bootstrap["accuracy_delta_95_lower"] > 0.0
        and accuracy_delta_1000 > 0.0
        and accuracy_delta_1250 > 0.0
    )
    ce_route_supported = (
        ce_improvement >= 0.20
        and bootstrap["ce_improvement_95_lower"] > 0.0
        and advantage_1000 > 0.0
        and advantage_1250 > 0.0
    )
    suffix_supported = (
        future_seed["suffix_utility"]["bootstrap_95_lower"] > 0.0
        and causal["suffix_utility"]["max_abs_change"] == 0.0
    )
    supported = (
        opened
        and (accuracy_route_supported or ce_route_supported)
        and suffix_supported
    )
    weak_signal = (
        opened
        and not supported
        and accuracy_delta < 0.03
        and 0.05 <= ce_improvement < 0.20
        and bootstrap["ce_improvement_95_lower"] > 0.0
        and advantage_1000 > 0.0
        and advantage_1250 > 0.0
    )
    depth_ce_gain = ce_improvement - P019_L2_CE_ADVANTAGE
    depth_accuracy_gain = accuracy_delta - P019_L2_ACCURACY_DELTA
    depth_route_amplified = (
        PLAN_ID == "P-CAUSAL-020"
        and opened
        and suffix_supported
        and ce_improvement >= 0.15
        and depth_ce_gain >= 0.05
        and bootstrap["ce_improvement_95_lower"] > 0.0
        and advantage_1000 > 0.0
    )
    status = (
        "supported"
        if supported
        else (
            "unopened"
            if not opened
            else (
                "depth_amplified_below_strong_gate"
                if depth_route_amplified
                else ("weak_signal" if weak_signal else "no_support")
            )
        )
    )
    comparison = {
        "plan": PLAN_ID,
        "status": status,
        "arms": scores,
        "future_seed_vs_causal": {
            "masked_accuracy_delta": accuracy_delta,
            "masked_ce_improvement": ce_improvement,
            "masked_exact_delta": (
                future_seed["metrics"]["masked_exact"]
                - causal["metrics"]["masked_exact"]
            ),
            "causal_ce_drop_step0_to_1250": causal_ce_drop,
            "accuracy_delta_step1000": accuracy_delta_1000,
            "accuracy_delta_step1250": accuracy_delta_1250,
            "ce_advantage_step1000": advantage_1000,
            "ce_advantage_step1250": advantage_1250,
            "endpoint_advantage_slope": advantage_1250 - advantage_1000,
            "p019_l2_reference_accuracy_delta": P019_L2_ACCURACY_DELTA,
            "p019_l2_reference_ce_advantage": P019_L2_CE_ADVANTAGE,
            "depth_accuracy_advantage_gain": depth_accuracy_gain,
            "depth_ce_advantage_gain": depth_ce_gain,
            "paired_window_bootstrap": bootstrap,
        },
        "registered_gates": {
            "causal_carrier_opened": opened,
            "accuracy_delta_at_least_0.03": accuracy_delta >= 0.03,
            "ce_improvement_at_least_0.20": ce_improvement >= 0.20,
            "accuracy_route_supported": accuracy_route_supported,
            "ce_route_supported": ce_route_supported,
            "positive_suffix_utility_interval": suffix_supported,
            "weak_ce_signal_0.05_to_0.20": weak_signal,
            "depth_ce_advantage_at_least_0.15": ce_improvement >= 0.15,
            "depth_ce_gain_over_l2_at_least_0.05": depth_ce_gain >= 0.05,
            "depth_route_amplified": depth_route_amplified,
            "scientific_support": supported,
        },
        "preflight": preflight_result,
        "protocol": {
            "pretrained_encoder_reused": preflight_result[
                "pretrained_encoder_reused"
            ],
            "common_pretrained_modules": [
                "bert.embeddings.word_embeddings.weight"
            ],
            "lexical_table_frozen_and_tied": all(
                row["frozen"] and row["input_output_tied"]
                for row in preflight_result["lexical_contracts"].values()
            ),
            "model": f"official-FLA GDN2 D128/L{LAYERS}/H4/D32 expand-v1",
            "model_layers": LAYERS,
            "active_future_seed_routes": LAYERS - 1,
            "intervention": "native FutureSeed scale 0 versus 1",
            "train_input_tokens_per_arm": args.max_steps
            * args.train_batch
            * args.sequence_length,
            "prepared_tensor_sha256": prepared_hash,
            "reverse_scan": False,
            "extra_layers_or_steps": False,
        },
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    if not supported:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
