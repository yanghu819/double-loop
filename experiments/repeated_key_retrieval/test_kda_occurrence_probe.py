from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("kda_occurrence_probe.py")
SPEC = importlib.util.spec_from_file_location("kda_occurrence_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_occurrence_rotary_preserves_norm() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 11, 2, 16, generator=generator)
    occurrences = torch.randint(1, 9, (3, 11), generator=generator)
    rotated = MODULE.apply_occurrence_rotary(x, occurrences)
    torch.testing.assert_close(
        rotated.float().norm(dim=-1),
        x.float().norm(dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_same_occurrence_preserves_query_key_dot_product() -> None:
    generator = torch.Generator().manual_seed(11)
    q = torch.randn(2, 5, 3, 32, generator=generator)
    k = torch.randn(2, 5, 3, 32, generator=generator)
    occurrences = torch.randint(1, 17, (2, 5), generator=generator)
    before = (q * k).sum(dim=-1)
    after = (
        MODULE.apply_occurrence_rotary(q, occurrences)
        * MODULE.apply_occurrence_rotary(k, occurrences)
    ).sum(dim=-1)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_generator_labels_requested_target_occurrence() -> None:
    generator = torch.Generator().manual_seed(13)
    batch = MODULE.generate_retrieval_batch(
        batch_size=32,
        sequence_length=128,
        target_repeats=8,
        key_vocab_size=64,
        value_vocab_size=32,
        max_occurrence=64,
        generator=generator,
        device=torch.device("cpu"),
    )
    for index in range(32):
        key = batch.query_key_ids[index]
        matching = batch.key_ids[index].eq(key).nonzero().flatten()
        assert matching.tolist() == batch.target_positions[index].tolist()
        occurrence = int(batch.query_occurrence_ids[index].item())
        expected = batch.value_ids[index, matching[occurrence - 1]]
        assert int(batch.labels[index].item()) == int(expected.item())
        assert batch.occurrence_ids[index, matching].tolist() == list(range(1, 9))


def test_parse_eval_specs() -> None:
    assert MODULE.parse_eval_specs("128:8,512:16") == [(128, 8), (512, 16)]
