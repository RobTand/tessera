#!/usr/bin/env bash
# Route census of a Tessera checkpoint inside the pinned serving image, with
# THIS checkout's plugin discovered by vLLM (tessera#640).  Submitted through
# PrismaBuild as the admitted action's command; same container discipline as
# routed_fused_tests.sh (checkout read-only at /work, every write under
# <out_dir>, the container runs as the submitting user).
#
# vLLM finds a plugin through ``importlib.metadata.entry_points(group=
# "vllm.general_plugins")``, which reads ``*.dist-info`` directories on
# ``sys.path``.  ``tessera_plugin_run.sh`` satisfied that with a root
# ``pip install -e`` into the container's own layer; a container running as the
# submitting user cannot write there, so this wrapper GENERATES the dist-info
# under <out_dir>/plugin-sp from pyproject.toml -- name, version and the entry
# point, read rather than retyped -- and puts it on PYTHONPATH beside
# /work/src.  The package the entry point names is this checkout's.
#   routed_fused_census.sh <checkout> <out_dir> <model_dir> [census args...]
# Environment:
#   ORACLE_IMAGE          the immutable image reference (required)
#   TESSERA_SERVE_MODE    residency the serve declares (default resident)
#   TESSERA_ROUTED_ENV    one extra KEY=VALUE for the container (optional)
#
# The image may be one whose CUDA toolkit lacks the library headers the JIT
# build of the fused window kernel pulls (the platform's pinned
# ``vllm/vllm-openai`` image): ``experiments/cuda_home_shadow.sh`` runs first
# inside the container and points CUDA_HOME at a user-owned shadow under
# <out_dir> when, and only when, a header is missing.
set -euo pipefail
CHECKOUT=$(realpath "$1"); OUT=$(realpath -m "$2"); MODEL=$(realpath "$3"); shift 3
IMAGE_REF=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared serving image}
source "$CHECKOUT/experiments/runtime_image.sh"
runtime_image_require "$IMAGE_REF"
IMAGE_ENV=()
while IFS= read -r line; do
  [[ -z "$line" ]] || IMAGE_ENV+=(-e "$line")
done <<< "$RUNTIME_IMAGE_CONTAINER_ENV"
mkdir -p "$OUT/home" "$OUT/tmp" "$OUT/triton" "$OUT/torch-ext" "$OUT/plugin-sp"
python3 - "$CHECKOUT/pyproject.toml" "$OUT/plugin-sp" <<'PY'
import pathlib
import sys
import tomllib

project = tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["project"]
name, version = project["name"], project["version"]
groups = project.get("entry-points") or {}
if "vllm.general_plugins" not in groups:
    sys.exit("pyproject.toml publishes no vllm.general_plugins entry point; nothing to discover")
info = pathlib.Path(sys.argv[2]) / f"{name.replace('-', '_')}-{version}.dist-info"
info.mkdir(parents=True, exist_ok=True)
(info / "METADATA").write_text(
    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
lines = []
for group, entries in groups.items():
    lines.append(f"[{group}]")
    lines.extend(f"{key} = {value}" for key, value in entries.items())
    lines.append("")
(info / "entry_points.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"plugin metadata: {info.name} {dict(groups['vllm.general_plugins'])}")
PY
# The receipt commits beside the checkpoint's config.json (the census's own
# sidecar digest is checked against it), so keep a copy in the output.
cp "$MODEL/config.json" "$OUT/config.json"
CPUS=$(python3 -c 'import os; s=sorted(os.sched_getaffinity(0)); print(",".join(map(str,s)))')
HEAD=${TESSERA_HEAD:-$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unknown)}
STATE=${TESSERA_STATE:-$(git -C "$CHECKOUT" status --short 2>/dev/null | tr '\n' ';' || echo unknown)}
echo "host=$(hostname) cpus=$CPUS head=$HEAD state=[$STATE] image=$IMAGE_REF model=$MODEL"
docker run --rm --gpus all --ipc=host --network=host --cpuset-cpus "$CPUS" \
  --user "$(id -u):$(id -g)" \
  -v "$CHECKOUT":/work:ro -v "$OUT":"$OUT" -v "$MODEL":"$MODEL":ro \
  -e HOME="$OUT/home" -e TMPDIR="$OUT/tmp" -e TRITON_CACHE_DIR="$OUT/triton" \
  -e TORCH_EXTENSIONS_DIR="$OUT/torch-ext" -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work/src:"$OUT/plugin-sp" \
  -e HOST_NAME="$(hostname)" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e TESSERA_SERVE_MODE="${TESSERA_SERVE_MODE:-resident}" \
  -e TESSERA_RESEARCH_GLM53_NOPE="${TESSERA_RESEARCH_GLM53_NOPE:-1}" \
  -e TESSERA_ROUTED_FUSED_VERBOSE="${TESSERA_ROUTED_FUSED_VERBOSE:-}" \
  -e PB_ACTION_KEY="${PB_ACTION_KEY:-${PRISMABUILD_ACTION_KEY:-}}" \
  "${IMAGE_ENV[@]}" ${TESSERA_ROUTED_ENV:+-e "$TESSERA_ROUTED_ENV"} \
  --entrypoint bash -w /work "$IMAGE_REF" \
  -c 'source /work/experiments/cuda_home_shadow.sh "$TMPDIR/.." && exec python3 "$@"' bash \
  /work/tools/tessera_route_census.py "$MODEL" "$OUT/census.json" \
  --runtime-image "$IMAGE_REF" --tessera-commit "$HEAD" "$@"
