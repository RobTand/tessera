#!/usr/bin/env bash
# Run a command inside the pinned ROCm vLLM image with Tessera installed as a
# vLLM plugin, on an AMD device.
#
# The HIP sibling of ``tessera_plugin_run.sh``. Three things differ, and each
# is a measured platform fact rather than a preference:
#
# * The GPU arrives through ``--device /dev/dxg`` plus a read-only bind of
#   ``/usr/lib/wsl/lib``. ``--gpus all`` is the NVIDIA container runtime's flag
#   and there is no such runtime here; under WSL2 there is no ``/dev/kfd``
#   either, so neither of the two spellings a native ROCm box would use works.
# * Nothing is linked into ``/usr/local/cuda/include``. The ROCm image ships
#   its own hipcc and headers, which is what ``torch.utils.cpp_extension``
#   reaches for on a ROCm torch.
# * The install is NOT editable. The tree is bind-mounted read-only and
#   ``pip install --no-deps .`` copies it into the container's own layer, so
#   what the plugin resolves is a copy of the bytes this launcher was pointed
#   at and not a directory a later edit can move under a running serve.
#
# A plugin is only a plugin once its ENTRY POINT is in the environment's
# metadata: vLLM discovers it through
# ``importlib.metadata.entry_points(group="vllm.general_plugins")``, so a
# PYTHONPATH is not enough.
#
# Usage: tessera_plugin_run_rocm.sh [docker-args...] -- <bash command>
set -eu
TS=${TS:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-$HOME/agents/t460-cells}
EXT=${EXT:-$RUNS/ext}                 # the window-GEMV JIT build cache
source "$(cd "$(dirname "$0")" && pwd)/runtime_image.sh"
# There is no default for this one. The packaged pin is the sm_121 image, and
# an AMD lane that fell back to it would name a runtime it never ran in.
IMG=${IMG:?set IMG to the ROCm serve image digest (repository@sha256:...)}
runtime_image_require "$IMG" || exit 2
# What the container needs to check its own image against (#132): the reference
# the daemon resolved, injected after the caller's own flags so a caller's -e
# cannot forge it.
imgenv=()
while IFS= read -r _kv; do
  if [ -n "$_kv" ]; then imgenv+=(-e "$_kv"); fi
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
extra=()
detached=0
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in -d|--detach) detached=1 ;; esac
  extra+=("$1"); shift
done
[ "${1:-}" = "--" ] && shift
# --rm only on a foreground run. A detached container that dies during startup
# takes its logs with it under --rm, and the startup failures are exactly the
# ones worth reading (the same reason serve_and_dump_kl.sh does not use it).
rmflag=(--rm); [ "$detached" = 0 ] || rmflag=()
mkdir -p "$EXT"
exec docker run ${rmflag[@]+"${rmflag[@]}"} \
  --device /dev/dxg -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -v "$TS":/work:ro \
  -v "$EXT":/ext -v /mnt/shared:/mnt/shared \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
  -w /work "${extra[@]}" ${imgenv[@]+"${imgenv[@]}"} \
  --entrypoint bash "$IMG" -c '
# The tree is read-only, and a non-editable build writes an egg-info beside the
# sources, so the build reads a COPY in the container layer. The copy is what
# gets installed: the bytes are the mounted tree'"'"'s, and nothing the run does
# can reach back into the checkout.
rm -rf /build && mkdir -p /build
cp -a /work/src /work/pyproject.toml /work/README.md /work/LICENSE /work/MANIFEST.in /build/
pip install --no-deps --no-build-isolation -q /build
echo "[tsrun-rocm] vllm $(python3 -c "import vllm;print(vllm.__version__)" 2>/dev/null); torch $(python3 -c "import torch;print(torch.__version__)" 2>/dev/null); plugin $(python3 -c "import importlib.metadata as m;print(sorted(e.name for e in m.entry_points(group=\"vllm.general_plugins\")))" 2>/dev/null)"
exec bash -c "$*"' bash "$@"
