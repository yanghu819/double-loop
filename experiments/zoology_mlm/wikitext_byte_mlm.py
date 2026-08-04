from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class WikiTextByteMLMConfig(DataSegmentConfig):
    """Load a fixed, locally prepared WikiText byte-MLM split."""

    name: str = "wikitext_byte_mlm"
    prepared_path: str
    prepared_sha256: str
    split: Literal["train", "valid"]

    def build(self, seed: int) -> DataSegment:
        del seed
        path = Path(self.prepared_path)
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != self.prepared_sha256:
            raise RuntimeError(
                f"Prepared data SHA256 mismatch: {actual_sha256} != "
                f"{self.prepared_sha256}"
            )

        prefix = "train" if self.split == "train" else "valid"
        with np.load(path, allow_pickle=False) as archive:
            inputs = np.asarray(archive[f"{prefix}_inputs"], dtype=np.int64)
            labels = np.asarray(archive[f"{prefix}_labels"], dtype=np.int64)

        expected_shape = (self.num_examples, self.input_seq_len)
        if inputs.shape != expected_shape or labels.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {self.split} shape: {inputs.shape}/{labels.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all((inputs >= 0) & (inputs <= 256)):
            raise RuntimeError("Byte-MLM inputs contain tokens outside [0, 256]")
        label_mask = labels != -100
        if not np.all((labels[label_mask] >= 0) & (labels[label_mask] <= 255)):
            raise RuntimeError("Byte-MLM labels contain non-byte targets")

        return DataSegment(
            inputs=torch.from_numpy(inputs.copy()),
            labels=torch.from_numpy(labels.copy()),
            slices={"split": self.split, "corpus": "wikitext-103-raw-v1"},
        )
