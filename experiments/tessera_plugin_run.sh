#!/usr/bin/env bash
# TMPDIR and thread bounds are launcher-owned; temporary writes use removable job scratch.
# Run a command inside the vLLM 0.28 image with Tessera installed as a vLLM plugin.
#
# Tessera is pip-installed (not just PYTHONPATH'd) because a plugin is only a
# plugin once its ENTRY POINT is in the environment's metadata: vLLM discovers
# it through ``importlib.metadata.entry_points(group="vllm.general_plugins")``.
# The source tree is bind-mounted read-only at /work and copied to owned scratch
# for an offline install. Metadata and build writes stay in that scratch.
#
# Usage: tessera_plugin_run.sh [container-payload-options...] -- <bash command>
set -eu
export TESSERA_CONTAINER_ENGINE=podman
TS=${TS:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${RUNS:-${TESSERA_GPU_OUT:?set RUNS to the unique job output directory}}
EXT=${EXT:-$RUNS/ext}                 # the NVFP4 decoder's JIT build cache
source "$(cd "$(dirname "$0")" && pwd)/runtime_image.sh"
IMG=${IMG:-$(runtime_image_pin)}
# Refuse a floating image (issue #100); the echo is this wrapper's receipt,
# since it writes no build sidecar of its own.
runtime_image_require "$IMG" || exit 2
# What the container needs to check its own image against (#132): the reference the
# daemon resolved -- never the tag or digest this caller happened to spell --
# and the record it resolved it from.  Injected AFTER "${extra[@]}" below so a
# caller's own -e cannot forge it, which is what makes the census's cross-check
# a check on the launcher rather than on its argument list.
imgenv=()
while IFS= read -r _kv; do
  if [ -n "$_kv" ]; then imgenv+=(-e "$_kv"); fi
done <<<"${RUNTIME_IMAGE_CONTAINER_ENV:-}"
extra=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
mkdir -p "$EXT"
exec python3 "$TS/tools/_flux_gpu_jobspec.py" run --out "$RUNS" --scratch-root "${FLUX_SCRATCH:-/tmp/flux-vanilla}" -- \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$TS/tools":/work/tools:ro -v "$TS/tests":/work/tests:ro \
  -v "$EXT":/ext -v /home/rob/models:/home/rob/models:ro \
  -e TORCH_EXTENSIONS_DIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
  -w /work "${extra[@]}" ${imgenv[@]+"${imgenv[@]}"} \
  --entrypoint bash "$RUNTIME_IMAGE_REFERENCE" -c '
# torch.utils.cpp_extension pulls CUDA headers the stock image does not ship.
# Link ONLY the missing names from the cu13 wheel include dir: putting the whole
# directory on the include path breaks nvcc with __cudaLaunch errors.
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; n=0
for src in "$inc"/*; do name="$(basename "$src")"; [ -e "$dst/$name" ] || { ln -s "$src" "$dst/$name"; n=$((n+1)); }; done
test -e "$dst/cusparse.h" || { echo "cusparse.h unresolved" >&2; exit 1; }
mkdir "$TMPDIR/plugin-project"
cp -a /work/src /work/pyproject.toml "$TMPDIR/plugin-project/"
pip install --no-index --no-deps --no-build-isolation -q "$TMPDIR/plugin-project" >/dev/null 2>&1
echo "[tsrun] linked $n headers; vllm $(python3 -c "import vllm;print(vllm.__version__)" 2>/dev/null); plugin $(python3 -c "import importlib.metadata as m;print([e.name for e in m.entry_points(group=\"vllm.general_plugins\")])" 2>/dev/null)"
exec bash -c "$*"' bash "$@"
