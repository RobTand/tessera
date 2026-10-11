#!/bin/bash
# Run tools/probe_witness_endpoint.py in the vLLM image the endpoint witness was inspected on
# (vLLM 0.30.1rc1.dev336, docs/ARCHITECTURE.md 5.1.2b). Call it from inside a PrismaBuild action
# submitted with --container-image; the docker shim of that action owns scope and placement.
#   WITNESS_IMAGE   image tag to run (default: the inspected runtime)
#   WITNESS_OUT_DIR writable directory for --out (optional)
# Every other argument goes to the probe.
set -eu
checkout=$(pwd -P)
image=${WITNESS_IMAGE:-localhost/prismaquant/spark-vllm-nccl230:nightly-20260929}
mounts=(--mount "type=bind,src=$checkout,dst=$checkout,readonly"
        --mount "type=bind,src=/mnt/shared/tessera-measurements,dst=/mnt/shared/tessera-measurements,readonly")
if [ -n "${WITNESS_OUT_DIR:-}" ]; then
    mounts+=(--mount "type=bind,src=$WITNESS_OUT_DIR,dst=$WITNESS_OUT_DIR")
fi
env_flags=(--env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 --env "PYTHONPATH=$checkout/src" --env HOME=/tmp)
if [ -n "${PRISMABUILD_ACTION_PROGRESS_PATH:-}" ]; then
    progress_dir=$(dirname "$PRISMABUILD_ACTION_PROGRESS_PATH")
    mounts+=(--mount "type=bind,src=$progress_dir,dst=$progress_dir")
    env_flags+=(--env PRISMABUILD_ACTION_PROGRESS_PATH --env PRISMABUILD_ACTION_PROGRESS_TOKEN)
fi
exec docker run --rm --gpus all --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,exec,size=16g \
    --entrypoint /usr/bin/python3 --workdir "$checkout" "${mounts[@]}" "${env_flags[@]}" \
    "$image" tools/probe_witness_endpoint.py "$@"
