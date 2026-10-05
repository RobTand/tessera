"""The PR930 A8 control flags, rendered separately from rank ownership."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from managed_window import Refused
from submit import parse_plan, IMAGE

CONTROL = "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"
CONFIG_SHA = "3f5c2c7381aae1c02d486c645ec6015cd1a60eb41faa5686541a15f523d79898"
LONG_CASES = "long_b1_len2100,long_b1_len4000,long_rep_len4000_a,long_rep_len4000_b"
MTP = {"method": "mtp", "num_speculative_tokens": 1, "draft_tensor_parallel_size": 2,
       "moe_backend": "triton"}
GRAPH = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}
PREP = '''set -e
inc="$(python3 -c 'import glob; p=sorted(glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cu*/include")); print(p[0] if p else "")')"
for src in "$inc"/*; do n="$(basename "$src")"; [ -e "/usr/local/cuda/include/$n" ] || ln -s "$src" "/usr/local/cuda/include/$n"; done
rm -rf /ext/tessera && mkdir -p /ext/tessera /ext/tmp
cp -r /tessera-ro/src /tessera-ro/pyproject.toml /ext/tessera/
pip install --no-deps --no-build-isolation -q -e /ext/tessera
python3 -c 'import importlib.metadata as m, tessera.serving as t; print("[ga702] tessera", t.__file__, [e.value for e in m.entry_points(group="vllm.general_plugins")], flush=True)'
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def src_sha(root: Path) -> str:
    # Identical to PR930's relative-path sha256sum | sha256sum spelling.
    lines = "".join(f"{sha(path)}  {path.relative_to(root)}\n"
                    for path in sorted((root / "src").rglob("*.py")))
    return hashlib.sha256(lines.encode()).hexdigest()

def producer_sha() -> str:
    names = ("managed_window.py", "tp2_recipe.py", "rank_window.py", "window_driver.py",
             "submit.py", "watch_window_queue.py", "arm_tp2.sh", "drive_tp2.sh", "plan-artifact.txt")
    return hashlib.sha256("".join(f"{sha(Path(__file__).parent / name)}  {name}\n"
                                 for name in names).encode()).hexdigest()


def inputs(env: dict, *, live: bool, runner=None) -> dict:
    for name in ("TS", "ARTIFACT", "RECEIPTS", "FABRIC"):
        if not env.get(name):
            raise Refused(f"{name} is required; no fabric/source default")
    if env["FABRIC"] not in ("socket", "roce"):
        raise Refused("FABRIC must be socket or roce")
    root, artifact = Path(env["TS"]).resolve(), Path(env["ARTIFACT"])
    if not (artifact / "config.json").is_file():
        raise Refused(f"ARTIFACT has no config.json: {artifact}")
    for name, expected in {"MAX_NUM_SEQS": "4", "MAX_MODEL_LEN": "8448",
                           "KV_BYTES": "2147483648", "FLOOR_GIB": "16",
                           "EXPECT_PEAK_GIB": "98", "MAX_BATCHED": "2048",
                           "GPU_UTIL": "0.5", "MOE_BACKEND": "triton",
                           "SERVE_MODE": "resident", "HEAD_IP": "10.100.96.2",
                           "PEER_IP": "10.100.96.1", "IFACE": "enp1s0f0np0",
                           "IMG": IMAGE, "LONG_CASES": LONG_CASES,
                           "TESSERA_ENV": "TESSERA_FUSED_E4M3_MMA=e4m3",
                           "KERNEL_JSON": '{"enable_flashinfer_autotune":false}'}.items():
        if name in env and env[name] != expected:
            raise Refused(f"the nominated control fixes {name}={expected}; no scope substitution")
    if live:
        if env["FABRIC"] != "socket":
            raise Refused("issues/CEO nominated SOCKET for this A8 control; no fabric substitution")
        if str(artifact) != CONTROL or sha(artifact / "config.json") != CONFIG_SHA:
            raise Refused("nominated A8 control path/config differs")
        if not str(root).startswith("/mnt/shared/") or not env["RECEIPTS"].startswith("/mnt/shared/"):
            raise Refused("live source and receipts must be shared")
        def git(*args):
            argv = ["git", "-C", str(root), *args]
            if runner:
                return runner(argv, limit=10).stdout.strip()
            return subprocess.check_output(argv, text=True, timeout=10).strip()
        head = git("rev-parse", "HEAD")
        if head != env.get("SOURCE_COMMIT") or src_sha(root) != env.get("SOURCE_SHA256"):
            raise Refused("issues-owned frozen SOURCE_COMMIT/SOURCE_SHA256 differs")
        if git("status", "--porcelain"):
            raise Refused("shared source is not clean/frozen")
        if producer_sha() != env.get("PRODUCER_SHA256") or not env.get("PRODUCER_COMMIT"):
            raise Refused("separately sealed producer commit/bytes differ")
    return dict(ts=str(root), artifact=str(artifact), receipts=env["RECEIPTS"],
                fabric=env["FABRIC"], image=IMAGE, src_sha256=src_sha(root),
                config_sha256=sha(artifact / "config.json"),
                source_commit=env.get("SOURCE_COMMIT", "dry-run-unfrozen"),
                producer_commit=env.get("PRODUCER_COMMIT", "dry-run-unfrozen"), producer_sha256=producer_sha(),
                hooks_sha256=sha(root / "experiments/glm53_508_graph_qual/digest/usercustomize.py"),
                equal_script_sha256=sha(root / "experiments/glm53_508_graph_qual/equal-508.py"))


def arm_settings(arm: str, env: dict) -> dict:
    eager = env.get("EAGER", "1")
    if eager not in ("0", "1"):
        raise Refused("EAGER must be 0 or 1")
    compilation = env.get("COMPILATION_JSON", "")
    if eager == "1" and compilation:
        raise Refused("EAGER=1 takes no COMPILATION_JSON")
    if json.loads(env.get("SPEC_JSON", "null")) != MTP:
        raise Refused("the nominated control requires MTP1 and draft TP2")
    if eager == "0" and json.loads(compilation or "null") != GRAPH:
        raise Refused("the nominated graph arm requires the release graph flags")
    return dict(arm=arm, eager=eager, compilation=compilation, spec=env["SPEC_JSON"])


def plan(path: Path) -> list[dict]:
    rows = parse_plan(path)
    if [arm for arm, _ in rows] != ["aE1", "aGR", "aE2"]:
        raise Refused("the finite control is exactly aE1/aGR/aE2; no fourth arm")
    result = [arm_settings(arm, env) for arm, env in rows]
    if [r["eager"] for r in result] != ["1", "0", "1"]:
        raise Refused("control order must be eager/graph/eager")
    if any(set(env) - {"EAGER", "COMPILATION_JSON", "SPEC_JSON", "FABRIC"} for _, env in rows):
        raise Refused("plan cannot override the frozen shared control tuple")
    for arm, (_, env) in zip(result, rows):
        if "FABRIC" in env:
            arm["fabric"] = env["FABRIC"]
    return result


def serve(config: dict, arm: dict, rank: int, *, master_port=29541, api_port=8142) -> list[str]:
    argv = ["vllm", "serve", config["artifact"], "--node-rank", str(rank)]
    argv += ["--headless"] if rank else ["--host", "0.0.0.0", "--port", str(api_port)]
    argv += ["--tensor-parallel-size", "2", "--nnodes", "2", "--master-addr", "10.100.96.2",
             "--master-port", str(master_port), "--distributed-executor-backend", "mp",
             "--kv-cache-dtype", "fp8_ds_mla", "--moe-backend", "triton", "--kernel-config",
             '{"enable_flashinfer_autotune":false}', "--max-model-len", "8448", "--language-model-only",
             "--max-num-seqs", "4", "--max-num-batched-tokens", "2048", "--enable-chunked-prefill",
             "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.5", "--kv-cache-memory-bytes",
             "2147483648", "--trust-remote-code", "--max-logprobs", "20", "--served-model-name", "glm53-artifact"]
    argv += ["--enforce-eager"] if arm["eager"] == "1" else ["--compilation-config", arm["compilation"]]
    return argv + ["--speculative-config", arm["spec"]]


def container(config: dict, arm: dict, identity: dict, out: Path, ext: Path,
              cidfile: Path, image_env: dict, *, master_port=29541, api_port=8142) -> list[str]:
    rank, name = identity["rank"], arm["arm"]
    argv = ["docker", "run", "-d", "--cidfile", str(cidfile), "--name",
            f"ga702-{identity['run_id']}-{identity['nonce'][:8]}-{name}-r{rank}",
            "--network", "host", "--ipc", "host", "--device", "/dev/infiniband", "--gpus", "all",
            "--label", "org.prismaquant.campaign=graph-attest-702", "--label", f"org.prismaquant.run={name}",
            "--label", f"org.prismaquant.graph-window={identity['run_id']}",
            "--label", f"org.prismaquant.attempt={identity['nonce']}",
            "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864", "--cap-add", "IPC_LOCK",
            "--shm-size", "16g"]
    root = Path(config["ts"])
    for source, target in [(root / "src", "/tessera-ro/src:ro"),
                           (root / "pyproject.toml", "/tessera-ro/pyproject.toml:ro"),
                           (ext, "/ext"), (Path("/mnt/shared"), "/mnt/shared:ro"),
                           (out, "/out"), (root / "experiments/glm53_508_graph_qual/digest", "/digest:ro")]:
        argv += ["-v", f"{source}:{target}"]
    env = dict(NCCL_SOCKET_IFNAME="enp1s0f0np0", GLOO_SOCKET_IFNAME="enp1s0f0np0",
               NCCL_IB_HCA="rocep1s0f0,roceP2p1s0f0", NCCL_IB_DISABLE="1" if config["fabric"] == "socket" else "0",
               NCCL_CUMEM_ENABLE="0", NCCL_CUMEM_HOST_ENABLE="0", NCCL_DMABUF_ENABLE="0",
               NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET", TESSERA_RESEARCH_GLM53_NOPE="0",
               TESSERA_SERVE_MODE="resident", VLLM_USE_BREAKABLE_CUDAGRAPH="0",
               VLLM_ALLOW_INSECURE_SERIALIZATION="1", VLLM_SERVER_DEV_MODE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH="/digest", GA_DISPATCH_LOG=f"/out/{name}.rank{rank}.dispatch",
               TORCH_EXTENSIONS_DIR="/ext/torch-ext", TMPDIR="/ext/tmp", TRITON_CACHE_DIR="/ext/triton",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               MAX_JOBS="1", VLLM_HOST_IP=("10.100.96.2", "10.100.96.1")[rank],
               T695_GC_BEFORE_DRAFTER="1", TESSERA_FUSED_E4M3_MMA="e4m3", **image_env)
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    return argv + ["-w", "/ext", "--entrypoint", "bash", config["image"], "-c",
                   PREP + "exec " + shlex.join(serve(config, arm, rank, master_port=master_port, api_port=api_port))]
