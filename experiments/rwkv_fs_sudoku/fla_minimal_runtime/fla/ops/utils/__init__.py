from __future__ import annotations

import os
import sys
from pathlib import Path


def _extend_real_subpackage(subpath: str) -> None:
    here = Path(__file__).resolve().parent
    configured = os.environ.get("FLA_REAL_PACKAGE_DIR", "").strip()
    bases = [Path(configured)] if configured else []
    bases.extend(Path(entry) for entry in sys.path if entry)
    for base in bases:
        root = base if base.name == "fla" else base / "fla"
        candidate = root / subpath
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved != here and resolved.is_dir():
            __path__.append(str(resolved))
            return


_extend_real_subpackage("ops/utils")
from .cumsum import chunk_local_cumsum
from .index import prepare_chunk_indices, prepare_chunk_offsets

__all__ = ["chunk_local_cumsum", "prepare_chunk_indices", "prepare_chunk_offsets"]

del _extend_real_subpackage
