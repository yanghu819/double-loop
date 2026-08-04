from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_rows(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        rows = payload["data_frame"]["view"]["data"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Unexpected Oxen DataFrame JSON schema") from exc
    texts = [row["text"] for row in rows]
    if not all(isinstance(text, str) for text in texts):
        raise RuntimeError("Train source contains a non-string text row")
    return texts


def load_validation_rows(path: Path) -> list[str]:
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    if not all(isinstance(text, str) for text in texts):
        raise RuntimeError("Validation source contains a non-string text row")
    return texts


def write_json_lines(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False))
            handle.write("\n")
    temporary.replace(path)


def split_stats(texts: list[str]) -> dict[str, int]:
    return {
        "rows": len(texts),
        "nonempty_rows": sum(bool(text.strip()) for text in texts),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_texts = load_train_rows(args.train_json)
    validation_texts = load_validation_rows(args.validation_parquet)
    if len(train_texts) != 20_000:
        raise RuntimeError(f"Expected 20,000 train rows, got {len(train_texts)}")
    if len(validation_texts) != 3_760:
        raise RuntimeError(
            f"Expected 3,760 validation rows, got {len(validation_texts)}"
        )

    train_output = args.output_dir / "train.json"
    validation_output = args.output_dir / "validation.json"
    write_json_lines(train_output, train_texts)
    write_json_lines(validation_output, validation_texts)

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
            },
            "validation": {
                "path": args.validation_parquet.name,
                "sha256": sha256(args.validation_parquet),
            },
        },
        "prepared": {
            "train": {
                "path": train_output.name,
                "sha256": sha256(train_output),
                **split_stats(train_texts),
            },
            "validation": {
                "path": validation_output.name,
                "sha256": sha256(validation_output),
                **split_stats(validation_texts),
            },
        },
        "protocol": {
            "format": "JSON Lines with one text field per original row",
            "text_transform": "none",
            "tokenization_and_masking": "delegated to pinned Transformers run_mlm.py",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
