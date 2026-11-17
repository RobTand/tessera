"""One worker environment for C1, H1, rows and the specified serve.

The lane source is experiments/tessera_plugin_served.sh:79-87 and
experiments/build_identity.sh:43-47 at the recorded Tessera commit.
The launcher supplies the constants below through its container arguments.
The lane's TESSERA_LANE_DOCKER_EXTRA must carry the same OCI and thread constants.
Do not forward the host environment. Do not print image Config.Env values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

# No optional Tessera lever. The build-identity deterministic switch is unset.
ENV = {"TORCH_EXTENSIONS_DIR": "/ext", "TMPDIR": "/ext",
       "TESSERA_SERVE_MODE": "resident", "TESSERA_GPU_MEM_UTIL": "0.85",
       "HOME": "/root", "HOSTNAME": "tessera-worker", "container": "oci",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MAX_JOBS": "1"}


def specification_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def container_options(scratch):
    """Use fixed worker paths. Host scratch paths never enter the environment."""
    return ["--ipc=host", "--hostname=tessera-worker", "--workdir=/work",
            "-v", f"{scratch}:/ext:rw", "-v", f"{scratch}/home:/root:rw",
            "-v", f"{scratch}/image-inspect.json:/run/numerics/image-inspect.json:ro",
            *[arg for k, v in ENV.items() for arg in ("--env", f"{k}={v}")]]


def lane_extra():
    """Arguments that the serving lane must add to its own four selector inputs."""
    lane = {"TORCH_EXTENSIONS_DIR", "TMPDIR", "TESSERA_SERVE_MODE", "TESSERA_GPU_MEM_UTIL"}
    return shlex.join(["--workdir=/work", "--hostname=tessera-worker",
                       *[arg for k, v in ENV.items() if k not in lane for arg in ("--env", f"{k}={v}")]])


def worker_command(source, module, arguments):
    """Set Python import paths as arguments, without a per-job PYTHONPATH."""
    boot = "import sys,runpy;sys.path[:0]=sys.argv[1:3];sys.argv=sys.argv[3:];runpy.run_module(sys.argv[0],run_name='__main__')"
    return shlex.join(["python3", "-c", boot, str(Path(source) / "src"), "/numerics", module, *map(str, arguments)])
