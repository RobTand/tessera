#!/usr/bin/env python3
"""tessera#1185: per-rank driver for the TP2 prefill trace (OFF REPO).

Modes:
  --selftest     CPU dry run: compile-check the hook/client/categorizer,
                 run the categorizer over canned tables and canned steps,
                 assert the four exact bucket sums. No docker, no GPU.
  --gang-rank R  One TP2 rank (R in {0, 1}): full-model eager NoPE serve in
                 the pinned image with the checkout's plugin, mp backend over
                 the RoCE fabric, profiler over four 2048-token chunks plus
                 one decode, outputs to the local and retain dirs.
  --probe        D30 single-box TP1 serve of the small stub export in the
                 pinned image: boots the serve, runs an id-array prompt
                 through one profiled prefill step, asserts the hook wrote
                 chunk0.txt with Self CUDA time. No NCCL, no full weights.
Only --probe and --gang-rank touch a GPU. Rank 0 owns the API, the warmup,
the trigger and the long request; rank 1 follows the shared trigger file.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOOK_DIR = HERE / "hook_dir"
CATEGORIZER = HERE / "categorize.py"
CLIENT = HERE / "client.py"

IMAGE = os.environ.get(
    "TP2TRACE_IMAGE",
    "localhost/prismaquant/spark-vllm-nccl230@sha256:"
    "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
ARTIFACT = os.environ.get(
    "TP2TRACE_ARTIFACT",
    "/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928"
    "/release-t8/exported")
RUN = os.environ.get("TP2TRACE_RUN", "prefill-tp2-trace-20261010")
RETAIN = Path(os.environ.get(
    "TP2TRACE_RETAIN",
    "/mnt/shared/tessera-measurements/" + RUN))
LOCAL = Path(os.environ.get("TP2TRACE_LOCAL", "/home/rob/tmp/" + RUN))
MASTER_PORT = int(os.environ.get("TP2TRACE_MASTER_PORT", "29841"))
API_PORT = int(os.environ.get("TP2TRACE_API_PORT", "8142"))
HOSTS = ("sparklina", "sparky")
IPS = ("10.100.96.2", "10.100.96.1")
KERNEL_JSON = '{"enable_flashinfer_autotune":false}'


def log(msg):
    print(f"[tp2trace] {msg}", flush=True)


def run(argv, **kw):
    log("+ " + " ".join(str(a) for a in argv))
    return subprocess.run(argv, **kw)


def free_gb(path):
    return shutil.disk_usage(path).free / (1 << 30)


def wait_http(url, deadline_s, label, alive=None):
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    log(f"{label} ready")
                    return
        except Exception:
            pass
        if alive is not None and not alive():
            raise SystemExit(f"{label}: container died while waiting")
        time.sleep(5)
    raise SystemExit(f"{label} not ready in {deadline_s}s")


def wait_path(path, deadline_s, label, alive=None):
    end = time.time() + deadline_s
    while time.time() < end:
        if path.exists():
            log(f"{label} present: {path}")
            return
        if alive is not None and not alive():
            raise SystemExit(f"{label}: container died while waiting")
        time.sleep(5)
    raise SystemExit(f"{label} missing after {deadline_s}s: {path}")


def serve_argv(rank, tp, mlm, mbt, mutil, kvbytes):
    argv = ["vllm", "serve", "/model", "--tensor-parallel-size", str(tp)]
    if tp == 2:
        argv += ["--node-rank", str(rank), "--nnodes", "2",
                 "--master-addr", IPS[0], "--master-port", str(MASTER_PORT),
                 "--distributed-executor-backend", "mp"]
        argv += ["--host", "0.0.0.0", "--port", str(API_PORT)] if rank == 0 \
            else ["--headless"]
    else:
        argv += ["--host", "0.0.0.0", "--port", str(API_PORT)]
    argv += ["--enforce-eager", "--attention-backend", "CUSTOM",
             "--kv-cache-dtype", "fp8_ds_mla", "--moe-backend", "triton",
             "--kernel-config", KERNEL_JSON, "--max-model-len", str(mlm),
             "--max-num-batched-tokens", str(mbt), "--enable-chunked-prefill",
             "--no-enable-prefix-caching", "--language-model-only",
             "--kv-cache-memory-bytes", str(kvbytes),
             "--gpu-memory-utilization", str(mutil), "--trust-remote-code",
             "--max-num-seqs", "1", "--served-model-name", "glm53-t8r",
             "--disable-log-stats"]
    return argv


def find_snap():
    """Repo root above this file, flat or nested materialization."""
    here = HERE
    for _ in range(5):
        if (here / "pyproject.toml").is_file() and (here / "src").is_dir():
            return here
        here = here.parent
    raise SystemExit(f"no repo root above {HERE}")


def docker_serve(rank, tp, out, prof_dir, skip, steps, mlm, mbt, mutil,
                 kvbytes):
    """Start the serve container; return the Popen of `docker run -d`."""
    import shlex
    inner = out / "serve-inner.sh"
    prep = "\n".join([
        "set -e",
        "inc=\"$(python3 -c \"import glob; p=sorted(glob.glob("
        "'/usr/local/lib/python3*/dist-packages/nvidia/cu*/include')); "
        "print(p[0] if p else '')\")\"",
        "if [ -n \"$inc\" ]; then for src in \"$inc\"/*; do n=\"$(basename "
        "\"$src\")\"; [ -e \"/usr/local/cuda/include/$n\" ] || "
        "ln -s \"$src\" \"/usr/local/cuda/include/$n\"; done; fi",
        "test -f /tessera-ro/pyproject.toml && test -d /tessera-ro/src",
        "ls /tessera-ro",
        "rm -rf /ext/tessera && cp -r /tessera-ro /ext/tessera",
        "ls /ext/tessera && test -f /ext/tessera/pyproject.toml",
        "python3 --version && python3 -m pip --version",
        "python3 -m pip install --no-deps --no-build-isolation -q -e /ext/tessera",
        "python3 -c \"import importlib.metadata as m; eps=[e.value for e in "
        "m.entry_points(group='vllm.general_plugins')]; "
        "assert any('tessera' in v for v in eps), eps; print('[tp2trace] "
        "plugins:', eps)\"",
        "exec " + shlex.join(serve_argv(rank, tp, mlm, mbt, mutil, kvbytes)),
    ])
    inner.write_text(prep + "\n")
    snap = find_snap()
    plug = os.environ.get("TP2TRACE_PLUGIN", "")
    plug_src = (snap / plug).resolve() if plug else snap
    log(f"snap={snap} plugin_src={plug_src}")
    cpus = ",".join(str(i) for i in sorted(os.sched_getaffinity(0)))
    env = dict(
        NCCL_SOCKET_IFNAME="enp1s0f0np0",
        GLOO_SOCKET_IFNAME="enp1s0f0np0",
        NCCL_IB_HCA="rocep1s0f0,roceP2p1s0f0",
        NCCL_IB_DISABLE="1",
        NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET",
        NCCL_CUMEM_ENABLE="0", NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_DMABUF_ENABLE="0",
        TESSERA_RESEARCH_GLM53_NOPE="1", TESSERA_SERVE_MODE="resident",
        VLLM_ALLOW_INSECURE_SERIALIZATION="1", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MAX_JOBS="1", VLLM_HOST_IP=IPS[rank] if tp == 2 else "127.0.0.1",
        TESSERA_FUSED_E4M3_MMA="e4m3",
        TORCH_EXTENSIONS_DIR="/ext/torch-ext", TMPDIR="/ext/tmp",
        TRITON_CACHE_DIR="/ext/triton", COLUMNS="512",
        PYTHONPATH="/tp2hook", TP2TRACE_PROF_DIR="/prof",
        TP2TRACE_TRIGGER="/retain/trigger",
        TP2TRACE_SKIP=str(skip), TP2TRACE_STEPS=str(steps))
    name = f"tp2trace-{RUN}-r{rank}"
    cmd = ["docker", "run", "-d", "--name", name, "--network", "host",
           "--ipc", "host", "--device", "/dev/infiniband", "--gpus", "all",
           "--cpuset-cpus", cpus, "--ulimit", "memlock=-1:-1",
           "--ulimit", "stack=67108864", "--shm-size", "16g",
           "-v", f"{plug_src}:/tessera-ro:ro",
           "-v", f"{out}/ext:/ext",
           "-v", f"{HOOK_DIR}:/tp2hook:ro",
           "-v", f"{ARTIFACT}:/model:ro",
           "-v", f"{RETAIN}:/retain:ro",
           "-v", f"{prof_dir}:/prof",
           "-w", "/ext"]
    cmd += [x for kv in env.items() for x in ("-e", f"{kv[0]}={kv[1]}")]
    cmd += ["--entrypoint", "bash", IMAGE, "/ext/serve-inner.sh"]
    (out / "ext").mkdir(parents=True, exist_ok=True)
    shutil.copy(inner, out / "ext" / "serve-inner.sh")
    run(["docker", "rm", "-f", name], capture_output=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    cid = (proc.stdout.read() or "").strip()
    if proc.wait() != 0 or not cid:
        raise SystemExit(f"docker run failed: {cid}")
    log(f"container {name} -> {cid[:12]}")
    return name


def client_in_image(args, out, tag):
    """Run client.py inside the pinned image against the rank-0 API."""
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", f"{HERE}:/tp2trace:ro",
           "-v", f"{ARTIFACT}:/model:ro",
           "-e", "PYTHONDONTWRITEBYTECODE=1",
           "--entrypoint", "python3", IMAGE, "/tp2trace/client.py",
           "--url", f"http://127.0.0.1:{API_PORT}",
           "--tokenizer", "/model"] + args
    log(f"client[{tag}]: " + " ".join(args))
    done = run(cmd, capture_output=False)
    if done.returncode:
        raise SystemExit(f"client[{tag}] failed: {done.returncode}")


def retain_copy(out, rank):
    dest = RETAIN / f"rank{rank}"
    dest.mkdir(parents=True, exist_ok=True)
    for src in (HOOK_DIR / "usercustomize.py", CLIENT, CATEGORIZER):
        shutil.copy(src, dest / src.name)
    for name in ("serve-inner.sh", "serve.log", "steps.json"):
        src = (out / name) if name != "steps.json" else (out / "prof" / name)
        if Path(src).exists():
            shutil.copy(src, dest / Path(src).name)
    prof = out / "prof"
    for i in range(9):
        chunk = prof / f"chunk{i}.txt"
        if chunk.exists():
            shutil.copy(chunk, dest / chunk.name)
    (dest / "member.json").write_text(json.dumps(dict(
        host=socket.gethostname(), image=IMAGE, artifact=str(ARTIFACT),
        plugin=os.environ.get("TP2TRACE_PLUGIN", "") or "worktree",
        plugin_commit=os.environ.get("TP2TRACE_PLUGIN_COMMIT", "worktree"),
        ended=time.strftime("%FT%TZ", time.gmtime())), indent=1) + "\n")
    log(f"retained to {dest}")


def container_running(name):
    try:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                              name], capture_output=True, text=True,
                             timeout=15)
    except Exception:
        return False
    return out.stdout.strip() == "true"


def serve_mode(rank, tp, mlm, mbt, mutil, kvbytes, skip, steps, want_tokens,
               warmup_tokens, measure_tokens):
    if tp == 2 and socket.gethostname() != HOSTS[rank]:
        raise SystemExit(
            f"rank {rank} must run on {HOSTS[rank]}, this is "
            f"{socket.gethostname()}")
    if shutil.which("docker") is None:
        raise SystemExit("missing binary: docker")
    if tp == 2 and not os.path.isdir("/sys/class/net/enp1s0f0np0"):
        raise SystemExit("mp fabric NIC enp1s0f0np0 absent on this box")
    if free_gb("/home/rob/tmp") < 8:
        raise SystemExit("less than 8 GiB free on /home/rob/tmp")
    out = LOCAL / f"rank{rank}"
    prof = out / "prof"
    (out / "ext").mkdir(parents=True, exist_ok=True)
    prof.mkdir(parents=True, exist_ok=True)
    for stale in list(prof.glob("chunk*.txt")) + [prof / "steps.json"]:
        try:
            stale.unlink()
            log(f"stale output removed: {stale.name}")
        except OSError:
            pass
    RETAIN.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        name = docker_serve(rank, tp, out, prof, skip, steps, mlm, mbt,
                            mutil, kvbytes)
        if rank == 0:
            try:
                (RETAIN / "trigger").unlink()
                log("stale trigger removed")
            except OSError:
                pass
            wait_http(f"http://127.0.0.1:{API_PORT}/v1/models",
                      900 if tp == 1 else 1800, "vllm api",
                      alive=lambda: container_running(name))
            client_in_image(["--mode", "warmup", "--tokens",
                             str(warmup_tokens), "--max-tokens", "2",
                             "--repeats", "2"], out, "warmup")
            (RETAIN / "trigger").touch()
            log("trigger armed")
            client_in_image(["--mode", "measure", "--tokens",
                             str(measure_tokens), "--max-tokens", "1"], out,
                            "measure")
        else:
            wait_path(RETAIN / "trigger", 2400, "trigger",
                      alive=lambda: container_running(name))
        for i in range(steps):
            wait_path(prof / f"chunk{i}.txt", 1500, f"chunk{i}",
                      alive=lambda: container_running(name))
        wait_path(prof / "steps.json", 300, "steps",
                  alive=lambda: container_running(name))
        got = [s["tokens"] for s in
               json.loads((prof / "steps.json").read_text())["steps"]]
        if got != want_tokens:
            raise SystemExit(f"step tokens {got} != wanted {want_tokens}")
        if "Self CUDA" not in (prof / "chunk0.txt").read_text():
            raise SystemExit("chunk0 table has no Self CUDA column")
        log(f"profiled steps: {got}")
    finally:
        if name is not None:
            with open(out / "serve.log", "w") as fh:
                run(["docker", "logs", name], stdout=fh,
                    stderr=subprocess.STDOUT)
            run(["docker", "rm", "-f", name], capture_output=True)
        retain_copy(out, rank)


def selftest():
    import py_compile
    for src in (HOOK_DIR / "usercustomize.py", CLIENT, CATEGORIZER):
        py_compile.compile(str(src), doraise=True)
        log(f"compiles: {src.name}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gang-rank", type=int, choices=(0, 1))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args(["--gang-rank", "0"])
    log("arg parsing ok")
    table = "\n".join([
        "Name  Self CPU %  Self CPU  CPU total %  CPU total  CPU time avg"
        "  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  # of Calls"
        "  Input Shapes",
        "----  ---------  --------  -----------  ---------  ------------"
        "  ---------  -----------  ----------  -------------  ----------"
        "  --------------",
        "aten::mul"
        "  1.00%  1.000ms  1.00%  1.000ms  1.000ms"
        "  90.000ms  50.00%  90.000ms  2.000ms  45"
        "  [[2048, 4, 4096], [2048, 4, 4096]]",
        "aten::copy_"
        "  0.10%  100.000us  0.10%  100.000us  100.000us"
        "  20.000ms  11.00%  20.000ms  1.000ms  20"
        "  [[1, 2048, 64, 128]]",
        "aten::fill_"
        "  0.05%  50.000us  0.05%  50.000us  50.000us"
        "  6.000ms  3.00%  6.000ms  1.000ms  6"
        "  [[1, 2048, 64, 64]]",
        "aten::cat"
        "  0.04%  40.000us  0.04%  40.000us  40.000us"
        "  7.700ms  4.00%  7.700ms  700.000us  11"
        "  [[2048, 64, 512], [2048, 64, 0]]",
        "aten::masked_fill_"
        "  0.03%  30.000us  0.03%  30.000us  30.000us"
        "  6.400ms  3.00%  6.400ms  581.818us  11"
        "  [[2048, 1, 64, 512], [2048, 1, 64, 512]]",
        "vllm::mhc_fused_post_pre_tilelang"
        "  3.25%  14.446ms  20.08%  89.363ms  1.004ms"
        "  3.325ms  0.70%  3.737ms  41.988us  89"
        "  [[64, 4096], [64, 4, 4096]]",
        "aten::copy_"
        "  0.01%  10.000us  0.01%  10.000us  10.000us"
        "  9.000ms  5.00%  9.000ms  9.000ms  1"
        "  [[32, 4096], [32, 4096]]",
        "vllm::all_reduce"
        "  0.01%  10.000us  0.01%  10.000us  10.000us"
        "  77.800ms  40.00%  77.800ms  855.000us  91"
        "  [[64, 4096], []]",
    ])
    sys.path.insert(0, str(HERE))
    import categorize
    rows = categorize.parse_table(table)
    assert len(rows) == 8, rows
    got = {}
    for rowname, shapes, us, _ in rows:
        bucket = categorize.classify(rowname, shapes)
        got[bucket] = got.get(bucket, 0.0) + us / 1e3
    assert abs(got.get("mhc", 0) - 93.325) < 1e-9, got
    assert abs(got.get("kda_copy", 0) - 26.0) < 1e-9, got
    assert abs(got.get("mla_cat", 0) - 7.7) < 1e-9, got
    assert abs(got.get("mla_fill", 0) - 6.4) < 1e-9, got
    assert None in got, got  # the decoy copy_ and the nccl row stay out
    log(f"canned rows ok: { {k: v for k, v in got.items() if k is not None} }")
    try:
        categorize.parse_table("Name  Self CPU %  # of Calls\n----\n")
        raise SystemExit("CPU-only table must refuse")
    except ValueError:
        log("CPU-only table refuses by name")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i in range(4):
            (tmpdir / f"chunk{i}.txt").write_text(table)
        (tmpdir / "steps.json").write_text(json.dumps(dict(steps=[
            dict(chunk=i, tokens=2048 if i < 4 else 1,
                 execute_model_host_s=0.5) for i in range(5)], skip=0)))
        outp = tmpdir / "rows.json"
        assert categorize.main(["--dir", str(tmpdir), "--out", str(outp)]) \
            == 0
        result = json.loads(outp.read_text())
        assert len(result["chunks"]) == 4, result
        mean = result["mean_per_chunk_ms"]
        assert abs(mean["mhc"] - 93.325) < 1e-9, mean
        assert abs(mean["kda_copy"] - 26.0) < 1e-9, mean
        log(f"canned 4-chunk run ok: {mean}")
    _hook_smoke()
    try:
        import torch  # noqa
        log(f"torch present: {torch.__version__}")
    except ImportError:
        log("torch absent on this box; parser checked against canned text")
    log("SELFTEST PASS")


def _hook_smoke():
    """Drive the hook's trigger/profile/write logic with stub torch/vLLM.

    Catches NameErrors and logic slips the real GPU run would hit late.
    No GPU, no docker: stub modules stand in for torch and vLLM.
    """
    import importlib.util
    import tempfile
    import types

    class _Tables:
        def table(self, sort_by=None, row_limit=None):
            return "Name  Self CUDA  # of Calls\n----  --  --\n"

    class _Prof:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def key_averages(self, group_by_input_shape=False):
            return _Tables()

    torch_stub = types.ModuleType("torch")
    torch_stub.profiler = types.SimpleNamespace(
        ProfilerActivity=types.SimpleNamespace(CPU=0, CUDA=1),
        profile=_Prof)
    torch_stub.cuda = types.SimpleNamespace(synchronize=lambda: None)

    class _RunnerV1:
        def execute_model(self, scheduler_output, *a, **k):
            return "ok"

    class _RunnerV2:
        def execute_model(self, scheduler_output, *a, **k):
            return "ok"

    saved = {}
    for key in ("torch", "vllm", "vllm.v1", "vllm.v1.worker",
                "vllm.v1.worker.gpu",
                "vllm.v1.worker.gpu.model_runner",
                "vllm.v1.worker.gpu_model_runner"):
        saved[key] = sys.modules.pop(key, None)
    try:
        sys.modules["torch"] = torch_stub
        for key in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
            sys.modules[key] = types.ModuleType(key)
        mod = types.ModuleType("vllm.v1.worker.gpu.model_runner")
        mod.GPUModelRunner = _RunnerV2
        sys.modules["vllm.v1.worker.gpu.model_runner"] = mod
        mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")
        mod.GPUModelRunner = _RunnerV1
        sys.modules["vllm.v1.worker.gpu_model_runner"] = mod
        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "prof"
            trigger = Path(tmp) / "trigger"
            os.environ["TP2TRACE_PROF_DIR"] = str(prof)
            os.environ["TP2TRACE_TRIGGER"] = str(trigger)
            os.environ["TP2TRACE_SKIP"] = "1"
            os.environ["TP2TRACE_STEPS"] = "2"
            spec = importlib.util.spec_from_file_location(
                "tp2trace_hook_smoke", str(HOOK_DIR / "usercustomize.py"))
            hook = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hook)
            runner = _RunnerV2()
            sched = types.SimpleNamespace(total_num_scheduled_tokens=64)
            assert runner.execute_model(sched) == "ok"  # idle: no trigger
            assert not prof.exists()
            trigger.touch()
            assert runner.execute_model(sched) == "ok"  # skipped
            assert not prof.exists()
            assert runner.execute_model(sched) == "ok"  # chunk 0
            assert runner.execute_model(sched) == "ok"  # chunk 1
            assert (prof / "chunk0.txt").is_file()
            assert (prof / "chunk1.txt").is_file()
            steps = json.loads((prof / "steps.json").read_text())["steps"]
            assert [s["tokens"] for s in steps] == [64, 64], steps
            assert not trigger.exists()
            log("hook smoke ok: skip/profile/write/trigger-removal")
    finally:
        for key in ("TP2TRACE_PROF_DIR", "TP2TRACE_TRIGGER",
                    "TP2TRACE_SKIP", "TP2TRACE_STEPS"):
            os.environ.pop(key, None)
        for key, mod in saved.items():
            if mod is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = mod
        sys.modules.pop("tp2trace_hook_smoke", None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gang-rank", type=int, choices=(0, 1), default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.probe:
        serve_mode(0, 1, 512, 512, 0.3, 268435456, 0, 1, [64], 32, 64)
        return 0
    if args.gang_rank is not None:
        mlm = int(os.environ.get("TP2TRACE_MLM", "8448"))
        mbt = int(os.environ.get("TP2TRACE_MBT", "2048"))
        mutil = float(os.environ.get("TP2TRACE_UTIL", "0.5"))
        kvbytes = int(os.environ.get("TP2TRACE_KVBYTES", "2147483648"))
        steps = int(os.environ.get("TP2TRACE_STEPS", "4"))
        want = [int(t) for t in
                os.environ.get("TP2TRACE_WANT",
                               "2048,2048,2048,2048").split(",")]
        warm = int(os.environ.get("TP2TRACE_WARM_TOKENS", "32"))
        meas = int(os.environ.get("TP2TRACE_MEASURE_TOKENS", "8192"))
        serve_mode(args.gang_rank, 2, mlm, mbt, mutil, kvbytes, 0, steps,
                   want, warm, meas)
        return 0
    ap.error("one of --selftest, --probe, --gang-rank is required")


if __name__ == "__main__":
    sys.exit(main())
