"""The PR930 A8 control flags, rendered separately from rank ownership."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from managed_window import MEMORY_POLICY, NOT_COMPUTED, Refused, dev_mode_enabled, seal_check
from submit import parse_plan, IMAGE
from eager_benchmark import GRAPH_SHIP_MODE, LEVER_VALUES, PAIRS as BENCHMARK_PAIRS, pair_refusal

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

PRODUCER_FILES = ("managed_window.py", "tp2_recipe.py", "rank_window.py", "window_driver.py",
                  "submit.py", "watch_window_queue.py", "arm_tp2.sh", "drive_tp2.sh", "plan-artifact.txt",
                  "eager_benchmark.py", "plan-eager-window4.txt", "plan-eager-ship-8192.txt", "plan-graph-ship.txt")


def producer_sha() -> str:
    """Inspection-only disk digest; never an authority to execute reviewed code."""
    if dev_mode_enabled():
        return os.environ.get("PRODUCER_SHA256", NOT_COMPUTED)
    return hashlib.sha256("".join(f"{sha(Path(__file__).parent / name)}  {name}\n"
                                 for name in PRODUCER_FILES).encode()).hexdigest()


def require_producer(root: Path, commit: str, expected: str, *, exact_head: bool, runner=None) -> str:
    """Bind a reviewed Git object to the actual executable bytes; no PB exemption.

    prepare/submit require the exact HEAD. A PB snapshot may have a synthetic
    HEAD, but still needs the reviewed object and identical clean producer files.
    A parentless checkout lacking that object fails closed by name.
    """
    if dev_mode_enabled():
        seal_check("producer identity", {"commit": commit, "digest": expected}, NOT_COMPUTED,
                   where="Window4 producer", refusal=Refused)
        return expected
    def git(*args, binary=False):
        argv = ["git", "-C", str(root), *args]
        try:
            if runner:
                return runner(argv, limit=10, text=not binary).stdout
            return subprocess.check_output(argv, timeout=10, text=not binary, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, OSError, Refused) as exc:
            raise Refused(f"producer reviewed Git object unavailable in this checkout: {commit}: {exc}") from exc
    actual_commit = git("rev-parse", "--verify", commit + "^{commit}").strip() if commit else "missing"
    seal_check("producer commit", commit, actual_commit, where="Window4 producer",
               same=bool(commit) and actual_commit == commit,
               refusal=Refused("producer label is not a full reviewed commit object"))
    if exact_head:
        seal_check("producer checkout", commit, git("rev-parse", "HEAD").strip(),
                   where="Window4 producer", refusal=Refused("producer checkout HEAD differs from PRODUCER_COMMIT"))
    state = git("status", "--porcelain", "--untracked-files=all", "--", "experiments/graph_attest_702").strip()
    seal_check("producer working tree", "clean", state, where="Window4 producer", same=not bool(state),
               refusal=Refused("producer experiments/graph_attest_702 is not clean"))
    lines = []
    for name in PRODUCER_FILES:
        relative = "experiments/graph_attest_702/" + name
        blob = git("show", commit + ":" + relative, binary=True)
        seal_check("producer file", name, "current", where="Window4 producer",
                   same=(root / relative).read_bytes() == blob,
                   refusal=Refused(f"producer on-disk bytes differ from reviewed git show object: {name}"))
        lines.append(f"{hashlib.sha256(blob).hexdigest()}  {name}\n")
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    seal_check("producer digest", expected, digest, where="Window4 producer",
               refusal=Refused("producer git-object digest differs from PRODUCER_SHA256"))
    return digest


def check_control_record(recorded, current, *, where, refusal):
    """Scope/geometry/comparability must still match exactly; only the recorded
    run identity (commits, digests, stamps) is a D32 seal that dev mode stamps."""
    identity = {"source_commit", "producer_commit", "pq_pin_commit", "artifact_authentication", "ts"}
    identity.update(key for key in set(recorded) | set(current) if key.endswith("sha256"))
    comparable = (set(recorded) | set(current)) - identity
    if {key: recorded.get(key) for key in comparable} != {key: current.get(key) for key in comparable}:
        raise Refused(f"{where}: control scope/geometry/comparability differs")
    seal_check("control run identity", {key: recorded.get(key) for key in identity},
               {key: current.get(key) for key in identity}, where=where, refusal=refusal)


def inputs(env: dict, *, live: bool, runner=None) -> dict:
    mode = env.get("WINDOW_MODE", "graph-control")
    development = dev_mode_enabled(env)
    if mode != "graph-control" and mode not in BENCHMARK_PAIRS:
        raise Refused("unknown WINDOW_MODE; no inferred benchmark scope")
    for name in ("TS", "ARTIFACT", "RECEIPTS", "FABRIC"):
        if not env.get(name):
            raise Refused(f"{name} is required; no fabric/source default")
    if env["FABRIC"] not in ("socket", "roce"):
        raise Refused("FABRIC must be socket or roce")
    root, artifact = Path(env["TS"]).resolve(), Path(env["ARTIFACT"])
    if not (artifact / "config.json").is_file():
        raise Refused(f"ARTIFACT has no config.json: {artifact}")
    chunk_baseline = BENCHMARK_PAIRS[mode][0][1] if mode in BENCHMARK_PAIRS else "2048"
    for name, expected in {"MAX_NUM_SEQS": "1" if mode in BENCHMARK_PAIRS else "4", "MAX_MODEL_LEN": "8448",
                           "KV_BYTES": "2147483648", "FLOOR_GIB": f"{MEMORY_POLICY['abort_below_gib']:g}",
                           "EXPECT_PEAK_GIB": str(MEMORY_POLICY["model_kv_estimate_gib"]), "MAX_BATCHED": chunk_baseline,
                           "GPU_UTIL": "0.5", "MOE_BACKEND": "triton",
                           "SERVE_MODE": "resident", "HEAD_IP": "10.100.96.2",
                           "PEER_IP": "10.100.96.1", "IFACE": "enp1s0f0np0",
                           "IMG": IMAGE, "LONG_CASES": LONG_CASES,
                           "TESSERA_ENV": "TESSERA_FUSED_E4M3_MMA=e4m3",
                           "KERNEL_JSON": '{"enable_flashinfer_autotune":false}'}.items():
        if mode in BENCHMARK_PAIRS and name == "MAX_BATCHED" and env.get(name, chunk_baseline) in dict(BENCHMARK_PAIRS[mode]).values():
            continue
        if name in env and env[name] != expected:
            raise Refused(f"the nominated control fixes {name}={expected}; no scope substitution")
    if live:
        if env["FABRIC"] != "socket":
            raise Refused("issues/CEO nominated SOCKET for this A8 control; no fabric substitution")
        seal_check("artifact path", CONTROL, str(artifact), where="Window4 control", environ=env,
                   refusal=Refused("nominated A8 control path/config differs"))
        seal_check("artifact config identity", CONFIG_SHA,
                   NOT_COMPUTED if development else sha(artifact / "config.json"),
                   where="Window4 control", environ=env,
                   refusal=Refused("nominated A8 control path/config differs"))
        if not str(root).startswith("/mnt/shared/") or not env["RECEIPTS"].startswith("/mnt/shared/"):
            raise Refused("live source and receipts must be shared")
        def git(*args):
            argv = ["git", "-C", str(root), *args]
            if runner:
                return runner(argv, limit=10).stdout.strip()
            return subprocess.check_output(argv, text=True, timeout=10).strip()
        seal_check("runtime source", {"commit": env.get("SOURCE_COMMIT"), "digest": env.get("SOURCE_SHA256")},
                   NOT_COMPUTED if development else {"commit": git("rev-parse", "HEAD"), "digest": src_sha(root)},
                   where="Window4 runtime", environ=env,
                   refusal=Refused("issues-owned frozen SOURCE_COMMIT/SOURCE_SHA256 differs"))
        seal_check("runtime working tree", "clean", NOT_COMPUTED if development else git("status", "--porcelain"),
                   where="Window4 runtime", environ=env, same=None if development else not bool(git("status", "--porcelain")),
                   refusal=Refused("shared source is not clean/frozen"))
        require_producer(Path(__file__).resolve().parents[2], env.get("PRODUCER_COMMIT", ""),
                         env.get("PRODUCER_SHA256", ""), exact_head=False, runner=runner)
    result = dict(ts=str(root), artifact=str(artifact), receipts=env["RECEIPTS"],
                fabric=env["FABRIC"], image=IMAGE,
                src_sha256=env.get("SOURCE_SHA256", NOT_COMPUTED) if development else src_sha(root),
                config_sha256=env.get("CONFIG_SHA256", CONFIG_SHA) if development else sha(artifact / "config.json"),
                source_commit=env.get("SOURCE_COMMIT", "dry-run-unfrozen"),
                producer_commit=env.get("PRODUCER_COMMIT", "dry-run-unfrozen"),
                producer_sha256=env.get("PRODUCER_SHA256", NOT_COMPUTED) if development else producer_sha(),
                hooks_sha256=env.get("HOOKS_SHA256", NOT_COMPUTED) if development else sha(root / "experiments/glm53_508_graph_qual/digest/usercustomize.py"),
                equal_script_sha256=env.get("EQUAL_SCRIPT_SHA256", NOT_COMPUTED) if development else sha(root / "experiments/glm53_508_graph_qual/equal-508.py"))
    if mode in BENCHMARK_PAIRS:
        if env["FABRIC"] != "socket":
            raise Refused("Ship graph is A8S/socket/TP2/c1 only" if mode == GRAPH_SHIP_MODE else
                          "Window4 is eager A8S/socket/TP2/c1 only")
        if live:
            from eager_benchmark import bindings
            result.update(bindings(env, artifact))
        else:
            result["window_mode"] = mode
            result["profile_dir"] = str(Path(env["RECEIPTS"]).parent / "profiles")
    return result


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


def pair_arm(name: str, env: dict, mode: str, *, exact_keys=False) -> dict:
    if env.get("MAX_BATCHED") != dict(BENCHMARK_PAIRS[mode]).get(name):
        raise Refused(pair_refusal(mode))
    arm = arm_settings(name, env)
    graph_ship = mode == GRAPH_SHIP_MODE
    fields = {"EAGER", "SPEC_JSON", "FABRIC", "MAX_BATCHED"}
    if graph_ship:
        fields |= {"COMPILATION_JSON", *LEVER_VALUES}
    if (arm["eager"] != ("0" if graph_ship else "1") or env.get("FABRIC") != "socket"
            or (exact_keys and set(env) != fields)):
        raise Refused("Ship graph plan fixes graph/socket/MTP1 and explicit lever env with no other override" if graph_ship else
                      "Window4 plan fixes eager/socket/MTP1 with no other override")
    if graph_ship:
        levers = {key: env.get(key) for key in LEVER_VALUES}
        for key, choices in LEVER_VALUES.items():
            if levers[key] not in choices:
                raise Refused(f"Ship graph plan requires explicit {key}={'/'.join(choices)}")
        enabled = [levers[key] == choices[1] for key, choices in LEVER_VALUES.items()]
        if any(enabled) != (name == BENCHMARK_PAIRS[mode][1][0]):
            raise Refused("Ship graph pair requires all levers off in the first arm and at least one on in the second")
        arm["lever_env"] = levers
    return dict(arm, max_batched=int(env["MAX_BATCHED"]), fabric="socket")


def plan(path: Path, *, mode="graph-control") -> list[dict]:
    rows = parse_plan(path)
    if mode in BENCHMARK_PAIRS:
        expected = BENCHMARK_PAIRS[mode]
        if [(name, env.get("MAX_BATCHED")) for name, env in rows] != expected:
            raise Refused(pair_refusal(mode))
        return [pair_arm(name, env, mode, exact_keys=True) for name, env in rows]
    if mode != "graph-control":
        raise Refused("unknown WINDOW_MODE; no inferred benchmark scope")
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
    benchmark_window = config.get("window_mode") in BENCHMARK_PAIRS
    argv = ["vllm", "serve", config["artifact"], "--node-rank", str(rank)]
    argv += ["--headless"] if rank else ["--host", "0.0.0.0", "--port", str(api_port)]
    argv += ["--tensor-parallel-size", "2", "--nnodes", "2", "--master-addr", "10.100.96.2",
             "--master-port", str(master_port), "--distributed-executor-backend", "mp",
             "--kv-cache-dtype", "fp8_ds_mla", "--moe-backend", "triton", "--kernel-config",
             '{"enable_flashinfer_autotune":false}', "--max-model-len", "8448", "--language-model-only",
             "--max-num-seqs", "1" if benchmark_window else "4", "--max-num-batched-tokens",
             str(arm["max_batched"]) if benchmark_window else "2048", "--enable-chunked-prefill",
             "--no-enable-prefix-caching", "--gpu-memory-utilization", "0.5", "--kv-cache-memory-bytes",
             "2147483648", "--trust-remote-code", "--max-logprobs", "20", "--served-model-name", "glm53-artifact"]
    argv += ["--enforce-eager"] if arm["eager"] == "1" else ["--compilation-config", arm["compilation"]]
    if benchmark_window:
        argv += ["--profiler-config", json.dumps(dict(profiler="torch",
                 torch_profiler_dir=config["profile_dir"], torch_profiler_with_stack=False,
                 torch_profiler_record_shapes=True, ignore_frontend=True), separators=(",", ":"))]
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
    if config.get("window_mode") == GRAPH_SHIP_MODE:
        env.update(arm["lever_env"])
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    local_config = config
    if config.get("window_mode") in BENCHMARK_PAIRS:
        directory = Path(config["profile_dir"]) / arm["arm"]
        argv += ["-v", f"{directory}:{directory}"]
        local_config = dict(config, profile_dir=str(directory))
    return argv + ["-w", "/ext", "--entrypoint", "bash", config["image"], "-c",
                   PREP + "exec " + shlex.join(serve(local_config, arm, rank, master_port=master_port, api_port=api_port))]
