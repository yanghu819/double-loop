#!/usr/bin/env python3
"""Stream Sudoku-Extreme CSV rows into the NumPy layout used by EqR."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CELLS = 81


def count_rows(csv_path: Path, min_difficulty: int | None) -> int:
    count = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if min_difficulty is None or int(row["rating"]) >= min_difficulty:
                count += 1
    return count


def encode_chunk(values: list[str], *, blank: bool) -> np.ndarray:
    if blank:
        payload = "".join(value.replace(".", "0") for value in values)
    else:
        payload = "".join(values)
    encoded = np.frombuffer(payload.encode("ascii"), dtype=np.uint8)
    expected = len(values) * CELLS
    if encoded.size != expected:
        raise ValueError(f"Expected {expected} Sudoku cells, got {encoded.size}")
    # EqR reserves token 0 for padding: source 0..9 maps to token 1..10.
    return (encoded.reshape(len(values), CELLS) - (ord("0") - 1)).astype(np.uint8)


def flush_chunk(
    inputs: np.memmap,
    labels: np.memmap,
    start: int,
    questions: list[str],
    answers: list[str],
) -> int:
    end = start + len(questions)
    inputs[start:end] = encode_chunk(questions, blank=True)
    labels[start:end] = encode_chunk(answers, blank=False)
    questions.clear()
    answers.clear()
    return end


def build_dataset(args: argparse.Namespace) -> None:
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    split_dir = output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    row_count = args.expected_rows
    if row_count is None:
        row_count = count_rows(csv_path, args.min_difficulty)
    if row_count <= 0:
        raise ValueError("No Sudoku rows matched the requested split")

    inputs = np.lib.format.open_memmap(
        split_dir / "all__inputs.npy", mode="w+", dtype=np.uint8, shape=(row_count, CELLS)
    )
    labels = np.lib.format.open_memmap(
        split_dir / "all__labels.npy", mode="w+", dtype=np.uint8, shape=(row_count, CELLS)
    )

    questions: list[str] = []
    answers: list[str] = []
    written = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "answer", "rating"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            if args.min_difficulty is not None and int(row["rating"]) < args.min_difficulty:
                continue
            questions.append(row["question"])
            answers.append(row["answer"])
            if len(questions) >= args.chunk_rows:
                written = flush_chunk(inputs, labels, written, questions, answers)
        if questions:
            written = flush_chunk(inputs, labels, written, questions, answers)

    if written != row_count:
        raise ValueError(f"Expected {row_count} rows but wrote {written}")
    inputs.flush()
    labels.flush()

    np.save(split_dir / "all__puzzle_identifiers.npy", np.zeros(row_count, dtype=np.int32))
    indices = np.arange(row_count + 1, dtype=np.int32)
    np.save(split_dir / "all__puzzle_indices.npy", indices)
    np.save(split_dir / "all__group_indices.npy", indices)

    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 11,
        "seq_len": CELLS,
        "num_puzzle_identifiers": 1,
        "total_groups": row_count,
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata) + "\n")
    (output_dir / "identifiers.json").write_text('["<blank>"]\n')
    print(json.dumps({"csv": str(csv_path), "output": str(split_dir), "rows": row_count}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--min-difficulty", type=int)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
