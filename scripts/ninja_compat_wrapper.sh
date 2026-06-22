#!/usr/bin/env bash
set -euo pipefail

real_ninja="${NINJA_REAL:-/opt/conda/bin/ninja}"

if [[ "$#" -eq 1 && ( "$1" == "--version" || "$1" == "-v" ) ]]; then
  "$real_ninja" "$@" || true
  exit 0
fi

exec "$real_ninja" "$@"
