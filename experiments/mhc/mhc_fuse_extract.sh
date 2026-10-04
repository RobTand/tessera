#!/usr/bin/env bash
# Copy, out of the serving image, the stock mHC sources a fused kernel must
# reproduce bit for bit (#783): vLLM's mHC dispatch, DeepGEMM's sm120 TF32
# pre-norm GEMM, TileLang's templates and compile flags, and the SASS of the
# compiled stock kernels found in a probe cache.  No GPU, no model.
#   mhc_fuse_extract.sh <image> <out_dir> [<probe_cache_home>]
set -euo pipefail
IMAGE=$1; OUT=$(realpath -m "$2"); CACHE=${3:-}
mkdir -p "$OUT"
MOUNTS=(-v "$OUT":/out)
[[ -z "$CACHE" ]] || MOUNTS+=(-v "$(realpath "$CACHE")":/cache:ro)
exec docker run --rm --network=none --user "$(id -u):$(id -g)" "${MOUNTS[@]}" -e HOME=/tmp \
  --entrypoint bash "$IMAGE" -c '
set -u
python3 - <<"PY"
import importlib, os, shutil, pathlib, sys
out = pathlib.Path("/out/src")
def grab(mod):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print("skip", mod, e); return None
    p = pathlib.Path(m.__file__)
    return p
for mod in ["vllm.model_executor.kernels.mhc.tilelang_kernels", "vllm.model_executor.kernels.mhc.tilelang",
            "vllm.model_executor.layers.mhc", "vllm.models.glm5next.common.model", "vllm.utils.deep_gemm"]:
    p = grab(mod)
    if p:
        d = out / "vllm" / mod.replace(".", "_"); d.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / p.name)
        if p.name == "__init__.py" or mod.endswith(".mhc"):
            for q in p.parent.glob("*.py"): shutil.copy(q, d / q.name)
for pkg in ["deep_gemm", "tilelang"]:
    p = grab(pkg)
    if p:
        root = p.parent
        print(pkg, root)
        for q in root.rglob("*"):
            if q.is_file() and q.suffix in (".py", ".cuh", ".h", ".hpp", ".cu") and q.stat().st_size < 2_000_000:
                rel = q.relative_to(root)
                if pkg == "tilelang" and not (str(rel).startswith("src/tl_templates/cuda") or str(rel).startswith("jit") or str(rel).startswith("contrib") or rel.name in ("env.py",) or str(rel).startswith("engine") or str(rel).startswith("transform")):
                    continue
                dst = out / pkg / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(q, dst)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
CUDA=${CUDA_HOME:-/usr/local/cuda}
"$CUDA/bin/nvcc" --version > /out/nvcc_version.txt 2>&1 || true
if [[ -d /cache ]]; then
  mkdir -p /out/sass
  find /cache -name "*.cubin" -o -name "executable.so" | while read -r f; do
    n=$(echo "$f" | sed "s#/cache/##; s#/#_#g")
    "$CUDA/bin/cuobjdump" -sass "$f" > "/out/sass/$n.sass" 2>&1 || true
  done
fi
ls -R /out | head -50
'
