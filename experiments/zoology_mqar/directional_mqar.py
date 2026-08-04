from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class DirectionalMQARConfig(DataSegmentConfig):
    name: str = "directional_mqar"
    num_kv_pairs: int = 4
    direction: Literal["past", "future", "mixed"]

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
    if input_seq_len < 64 or input_seq_len % 8:
        raise ValueError("input_seq_len must be a multiple of 8 and at least 64")
    if num_kv_pairs < 2 or num_kv_pairs % 2:
        raise ValueError("num_kv_pairs must be a positive even number")
    if num_kv_pairs > 64:
        raise ValueError("num_kv_pairs exceeds the disjoint key/value capacity")
    if direction not in {"past", "future", "mixed"}:
        raise ValueError("direction must be past, future, or mixed")

    quarter = input_seq_len // 4
    pairs_per_direction = num_kv_pairs // 2
    slots_per_quarter = quarter // 2
    if pairs_per_direction > slots_per_quarter:
        raise ValueError(
            "Each directional half must fit in one normalized sequence quarter"
        )

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
        pair_slots = np.arange(0, quarter, 2, dtype=np.int64)
        query_slots = np.arange(2 * quarter, input_seq_len, 2, dtype=np.int64)
    elif direction == "future":
        query_slots = np.arange(0, 2 * quarter, 2, dtype=np.int64)
        pair_slots = np.arange(3 * quarter, input_seq_len, 2, dtype=np.int64)
    else:
        future_query_slots = np.arange(0, quarter, 2, dtype=np.int64)
        past_pair_slots = np.arange(quarter, 2 * quarter, 2, dtype=np.int64)
        past_query_slots = np.arange(2 * quarter, 3 * quarter, 2, dtype=np.int64)
        future_pair_slots = np.arange(
            3 * quarter,
            input_seq_len,
            2,
            dtype=np.int64,
        )

    for example_idx in range(num_examples):
        keys = rng.choice(key_vocab, size=num_kv_pairs, replace=False)
        values = rng.choice(value_vocab, size=num_kv_pairs, replace=False)
        association_order = rng.permutation(num_kv_pairs)

        if direction == "mixed":
            future_ids = association_order[: num_kv_pairs // 2]
            past_ids = association_order[num_kv_pairs // 2 :]
            assignments = []
            for association_idx, query_pos, write_pos in zip(
                future_ids,
                rng.choice(
                    future_query_slots,
                    size=pairs_per_direction,
                    replace=False,
                ),
                rng.choice(
                    future_pair_slots,
                    size=pairs_per_direction,
                    replace=False,
                ),
            ):
                assignments.append((association_idx, query_pos, write_pos))
            for association_idx, query_pos, write_pos in zip(
                past_ids,
                rng.choice(
                    past_query_slots,
                    size=pairs_per_direction,
                    replace=False,
                ),
                rng.choice(
                    past_pair_slots,
                    size=pairs_per_direction,
                    replace=False,
                ),
            ):
                assignments.append((association_idx, query_pos, write_pos))
        else:
            writes = rng.choice(pair_slots, size=num_kv_pairs, replace=False)
            queries = rng.choice(query_slots, size=num_kv_pairs, replace=False)
            assignments = [
                (association_idx, queries[query_idx], writes[association_idx])
                for query_idx, association_idx in enumerate(association_order)
            ]

        for association_idx, query_pos, write_pos in assignments:
            write_pos = int(write_pos)
            query_pos = int(query_pos)
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
