#!/usr/bin/env bash
# Run a command inside the vLLM 0.28 image with Tessera installed as a vLLM plugin.
#
# Tessera is pip-installed (not just PYTHONPATH'd) because a plugin is only a
# plugin once its ENTRY POINT is in the environment's metadata: vLLM discovers
# it through ``importlib.metadata.entry_points(group="vllm.general_plugins")``.
# The source tree is bind-mounted READ-ONLY into /work and the editable install
# writes its egg-info to the container's own layer, so a container running as
# root never writes into the repo.
#
# Usage: tessera_plugin_run.sh [docker-args...] -- <bash command>
set -eu
TS=${TS:-/home/rob/tessera}
RUNS=${RUNS:-/home/rob/tessera-runs/tsplugin}
EXT=${EXT:-$RUNS/ext}                 # the NVFP4 decoder's JIT build cache
source "$(cd "$(dirname "$0")" && pwd)/runtime_image.sh"
IMG=${IMG:-$(runtime_image_pin)}
# Refuse a floating image (issue #100); the echo is this wrapper's receipt,
# since it writes no build sidecar of its own.
runtime_image_require "$IMG" || exit 2
extra=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
mkdir -p "$EXT"
exec docker run --rm --gpus all \
  -v "$TS/src":/work/src:ro -v "$TS/pyproject.toml":/work/pyproject.toml:ro \
  -v "$TS/tools":/work/tools:ro -v "$TS/tests":/work/tests:ro \
  -v "$EXT":/ext -v /home/rob/models:/home/rob/models:ro \
  -e TORCH_EXTENSIONS_DIR=/ext -e TMPDIR=/ext -e TRITON_CACHE_DIR=/ext/triton \
  -w /work "${extra[@]}" -e TESSERA_CENSUS_RUNTIME_IMAGE="$IMG" \
  --entrypoint bash "$IMG" -c '
# torch.utils.cpp_extension pulls CUDA headers the stock image does not ship.
# Link ONLY the missing names from the cu13 wheel include dir: putting the whole
# directory on the include path breaks nvcc with __cudaLaunch errors.
inc="$(python3 -c "import glob; p=sorted(glob.glob(\"/usr/local/lib/python3*/dist-packages/nvidia/cu*/include\")); print(p[0] if p else \"\")")"
dst=/usr/local/cuda/include; n=0
for src in "$inc"/*; do name="$(basename "$src")"; [ -e "$dst/$name" ] || { ln -s "$src" "$dst/$name"; n=$((n+1)); }; done
test -e "$dst/cusparse.h" || { echo "cusparse.h unresolved" >&2; exit 1; }
pip install --no-deps --no-build-isolation -q -e /work >/dev/null 2>&1
echo "[tsrun] linked $n headers; vllm $(python3 -c "import vllm;print(vllm.__version__)" 2>/dev/null); plugin $(python3 -c "import importlib.metadata as m;print([e.name for e in m.entry_points(group=\"vllm.general_plugins\")])" 2>/dev/null)"
exec bash -c "$*"' bash "$@"
