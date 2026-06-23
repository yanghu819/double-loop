#!/usr/bin/env bash
set -euo pipefail

real_ninja="${NINJA_REAL:-/opt/conda/bin/ninja}"

if [[ "$#" -eq 1 && ( "$1" == "--version" || "$1" == "-v" ) ]]; then
  "$real_ninja" "$@" || true
  exit 0
fi

set +e
"$real_ninja" "$@"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  exit 0
fi

target="$(sed -n 's/^default //p' build.ninja 2>/dev/null | head -n 1 || true)"
if [[ ! -f build.ninja || "${target}" != rwkv7_statepassing_clampw_n*.so ]]; then
  exit "$status"
fi

echo "[ninja_compat_wrapper] real ninja failed with ${status}; falling back to direct RWKV statepassing build for ${target}" >&2

cuda_src="$(sed -n 's#^build rwkv7_statepassing_clampw.cuda.o: cuda_compile ##p' build.ninja | head -n 1)"
cpp_src="$(sed -n 's#^build rwkv7_statepassing_clampw.o: compile ##p' build.ninja | head -n 1)"
cuda_cflags="$(sed -n 's/^cuda_cflags = //p' build.ninja | head -n 1)"
cflags="$(sed -n 's/^cflags = //p' build.ninja | head -n 1)"
ldflags="$(sed -n 's/^ldflags = //p' build.ninja | head -n 1)"
nvcc="$(sed -n 's/^nvcc = //p' build.ninja | head -n 1)"
cxx="$(sed -n 's/^cxx = //p' build.ninja | head -n 1)"

if [[ -z "${cuda_src}" || -z "${cpp_src}" || -z "${cuda_cflags}" || -z "${cflags}" || -z "${ldflags}" || -z "${nvcc}" || -z "${cxx}" ]]; then
  echo "[ninja_compat_wrapper] build.ninja is missing expected RWKV statepassing fields" >&2
  exit "$status"
fi

eval "\"${nvcc}\" --generate-dependencies-with-compile --dependency-output rwkv7_statepassing_clampw.cuda.o.d ${cuda_cflags} -c \"${cuda_src}\" -o rwkv7_statepassing_clampw.cuda.o"
eval "\"${cxx}\" -MMD -MF rwkv7_statepassing_clampw.o.d ${cflags} -c \"${cpp_src}\" -o rwkv7_statepassing_clampw.o"
eval "\"${cxx}\" rwkv7_statepassing_clampw.cuda.o rwkv7_statepassing_clampw.o ${ldflags} -o \"${target}\""
