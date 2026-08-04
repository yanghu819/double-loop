from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


MASK_TOKEN_ID = 256
IGNORE_INDEX = -100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_train_rows(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    try:
        rows = payload["data_frame"]["view"]["data"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Unexpected Oxen DataFrame JSON schema") from exc
    texts = [row["text"] for row in rows]
    if not all(isinstance(text, str) for text in texts):
        raise RuntimeError("Train source has non-string text rows")
    return texts


def _load_validation_rows(path: Path) -> list[str]:
    table = pq.read_table(path, columns=["text"])
    texts = table.column("text").to_pylist()
    if not all(isinstance(text, str) for text in texts):
        raise RuntimeError("Validation source has non-string text rows")
    return texts


def _byte_windows(texts: list[str], count: int, seq_len: int) -> np.ndarray:
    stream = "\n".join(texts).encode("utf-8")
    required = count * seq_len
    if len(stream) < required:
        raise RuntimeError(f"Need {required} bytes, source only has {len(stream)}")
    values = np.frombuffer(stream[:required], dtype=np.uint8).copy()
    return values.reshape(count, seq_len)


def _mask_windows(
    windows: np.ndarray,
    *,
    mask_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = windows.astype(np.uint16, copy=True)
    labels = np.full(windows.shape, IGNORE_INDEX, dtype=np.int16)
    for row_index in range(windows.shape[0]):
        positions = rng.choice(windows.shape[1], size=mask_count, replace=False)
        labels[row_index, positions] = windows[row_index, positions]
        inputs[row_index, positions] = MASK_TOKEN_ID
    return inputs, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=20_000)
    parser.add_argument("--valid-examples", type=int, default=2_000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    args = parser.parse_args()

    mask_count = round(args.seq_len * args.mask_rate)
    if mask_count <= 0 or mask_count >= args.seq_len:
        raise ValueError("mask-rate must mask between 1 and seq_len-1 tokens")

    train_windows = _byte_windows(
        _load_train_rows(args.train_json), args.train_examples, args.seq_len
    )
    valid_windows = _byte_windows(
        _load_validation_rows(args.validation_parquet),
        args.valid_examples,
        args.seq_len,
    )
    train_inputs, train_labels = _mask_windows(
        train_windows, mask_count=mask_count, seed=123
    )
    valid_inputs, valid_labels = _mask_windows(
        valid_windows, mask_count=mask_count, seed=124
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_inputs=train_inputs,
        train_labels=train_labels,
        valid_inputs=valid_inputs,
        valid_labels=valid_labels,
    )
    output_sha256 = sha256(args.output)
    manifest = {
        "corpus": "WikiText-103-raw-v1",
        "license": ["CC BY-SA 3.0", "GFDL"],
        "mirror": {
            "provider": "Oxen.ai",
            "repository": "Salesforce/wikitext",
            "commit": "430edf94f37286d0",
            "schema_hash": "d930b7eee9d7f11db9ee0fa4ba45117d",
        },
        "sources": {
            "train": {
                "path": args.train_json.name,
                "sha256": sha256(args.train_json),
                "rows": 20_000,
            },
            "validation": {
                "path": args.validation_parquet.name,
                "sha256": sha256(args.validation_parquet),
            },
        },
        "protocol": {
            "encoding": "UTF-8 bytes",
            "vocab_size": 257,
            "mask_token_id": MASK_TOKEN_ID,
            "ignore_index": IGNORE_INDEX,
            "sequence_length": args.seq_len,
            "mask_rate": args.mask_rate,
            "masked_tokens_per_example": mask_count,
            "train_examples": args.train_examples,
            "validation_examples": args.valid_examples,
            "train_mask_seed": 123,
            "validation_mask_seed": 124,
            "windowing": "first contiguous non-overlapping byte windows",
        },
        "prepared": {
            "path": args.output.name,
            "sha256": output_sha256,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
