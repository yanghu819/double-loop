from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_UPSTREAM_SHA256 = (
    "b5017fb36ffacbe81dc53a0dd77dc3420369a010f5a37fc5263228d7a8811cb1"
)
UPSTREAM_METRIC_LINE = (
    'metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)'
)
OFFLINE_METRIC_LINE = (
    'metric = evaluate.load(os.environ["P009_ACCURACY_METRIC_PATH"], '
    "cache_dir=model_args.cache_dir)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    upstream_sha256 = sha256(args.upstream)
    if upstream_sha256 != EXPECTED_UPSTREAM_SHA256:
        raise RuntimeError(
            f"Unexpected run_mlm.py SHA256: {upstream_sha256}"
        )
    source = args.upstream.read_text(encoding="utf-8")
    if source.count(UPSTREAM_METRIC_LINE) != 1:
        raise RuntimeError("Expected exactly one upstream accuracy metric line")
    patched = source.replace(UPSTREAM_METRIC_LINE, OFFLINE_METRIC_LINE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched, encoding="utf-8")
    args.manifest.write_text(
        json.dumps(
            {
                "upstream": {
                    "repository": "huggingface/transformers",
                    "tag": "v4.46.3",
                    "commit": "052e652d6d53c2b26ffde87e039b723949a53493",
                    "path": "examples/pytorch/language-modeling/run_mlm.py",
                    "sha256": upstream_sha256,
                },
                "patched_sha256": sha256(args.output),
                "only_semantic_diff": (
                    "evaluate.load resolves the pinned local accuracy metric "
                    "path instead of contacting the Hub"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
