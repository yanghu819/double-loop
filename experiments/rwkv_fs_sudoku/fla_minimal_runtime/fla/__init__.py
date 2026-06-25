from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.5.2"
__all__: list[str] = []


def _candidate_real_fla_paths() -> list[Path]:
    configured = os.environ.get("FLA_REAL_PACKAGE_DIR", "").strip()
    paths: list[Path] = []
    if configured:
        base = Path(configured)
        paths.append(base if base.name == "fla" else base / "fla")
    for entry in sys.path:
        if not entry:
            continue
        paths.append(Path(entry) / "fla")
    return paths


_here = Path(__file__).resolve().parent
for _candidate in _candidate_real_fla_paths():
    try:
        resolved = _candidate.resolve()
    except OSError:
        continue
    if resolved != _here and (resolved / "ops").is_dir():
        __path__.append(str(resolved))
        break

del _candidate_real_fla_paths, _here
