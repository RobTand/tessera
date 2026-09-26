set -euo pipefail
here="$(pwd)"
scratch="$(mktemp -d "${TMPDIR:-$here}/v38gpu.XXXXXX")"
echo "[v38-gpu] host=$(hostname) start=$(date -u +%FT%TZ) head=$(git -C "$here" rev-parse HEAD 2>/dev/null || echo none) scratch=$scratch"
docker run --rm --gpus all --ipc=host --network=host \
  --user "$(id -u):$(id -g)" -e HOME=/scratch -e TMPDIR=/scratch -e TRITON_CACHE_DIR=/scratch/triton \
  -e TORCH_EXTENSIONS_DIR=/scratch/ext -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$here":/work:ro -v "$scratch":/scratch -w /work --entrypoint bash \
  localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5 -c '
python3 -c "import pytest" 2>/dev/null || pip install --user --quiet "pytest==8.*"
python3 -c "import torch,vllm,pytest;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"vllm\",vllm.__version__,\"pytest\",pytest.__version__)"
python3 -m pytest -q -p no:cacheprovider -rs \
  tests/test_serving_native_window.py tests/test_serving_moe_route.py tests/test_serving_moe_selected.py \
  tests/test_serving_moe_bf16_tp1_intake.py tests/test_serving_contract.py tests/test_glm_x_census_cells.py \
  tests/test_serving_attested_wire.py tests/test_route_census_regimes.py tests/test_census_cell_agreement.py \
  tests/test_serving_export_gate.py tests/test_cell_evidence.py tests/test_contract_platform_axis.py \
  tests/test_census_engine_backends.py tests/test_evidence_artifact.py'
rc=$?
rm -rf "$scratch"
exit $rc
