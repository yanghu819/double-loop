from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class DirectionalMQARConfig(DataSegmentConfig):
    name: str = "directional_mqar"
    num_kv_pairs: int = 4
    direction: Literal["past", "future"]

    def build(self, seed: int) -> DataSegment:
        return directional_mqar(
            vocab_size=self.vocab_size,
            num_examples=self.num_examples,
            input_seq_len=self.input_seq_len,
            num_kv_pairs=self.num_kv_pairs,
            direction=self.direction,
            seed=seed,
        )


def directional_mqar(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    num_kv_pairs: int,
    direction: str,
    seed: int,
) -> DataSegment:
    """Generate write-before-query or query-before-write associative recall."""
    if vocab_size != 256:
        raise ValueError("The preregistered split requires vocab_size=256")
    if input_seq_len != 64:
        raise ValueError("The preregistered split requires input_seq_len=64")
    if num_kv_pairs != 4:
        raise ValueError("The preregistered split requires num_kv_pairs=4")
    if direction not in {"past", "future"}:
        raise ValueError("direction must be past or future")

    rng = np.random.default_rng(seed)
    filler_vocab = np.arange(0, 64, dtype=np.int64)
    key_vocab = np.arange(64, 160, dtype=np.int64)
    value_vocab = np.arange(160, 256, dtype=np.int64)

    inputs = rng.choice(
        filler_vocab,
        size=(num_examples, input_seq_len),
        replace=True,
    ).astype(np.int64)
    labels = np.full_like(inputs, -100)

    if direction == "past":
        pair_slots = np.arange(0, 16, 2, dtype=np.int64)
        query_slots = np.arange(32, 64, 2, dtype=np.int64)
    else:
        query_slots = np.arange(0, 32, 2, dtype=np.int64)
        pair_slots = np.arange(48, 64, 2, dtype=np.int64)

    for example_idx in range(num_examples):
        keys = rng.choice(key_vocab, size=num_kv_pairs, replace=False)
        values = rng.choice(value_vocab, size=num_kv_pairs, replace=False)
        writes = rng.choice(pair_slots, size=num_kv_pairs, replace=False)
        queries = rng.choice(query_slots, size=num_kv_pairs, replace=False)
        association_order = rng.permutation(num_kv_pairs)

        for query_idx, association_idx in enumerate(association_order):
            write_pos = int(writes[association_idx])
            query_pos = int(queries[query_idx])
            key = int(keys[association_idx])
            value = int(values[association_idx])
            inputs[example_idx, write_pos] = key
            inputs[example_idx, write_pos + 1] = value
            inputs[example_idx, query_pos] = key
            labels[example_idx, query_pos] = value

    return DataSegment(
        inputs=torch.from_numpy(inputs),
        labels=torch.from_numpy(labels),
        slices={
            "direction": direction,
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
        },
    )
