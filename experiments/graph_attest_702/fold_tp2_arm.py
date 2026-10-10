#!/usr/bin/env python3
"""tessera#1204 fold arm member: one TP2 rank of one arm (base|gate).

Runs inside an admitted PrismaBuild gang member on its Spark. Phases:
P: graph-NONE vLLM serve (release window geometry) with torch profiler,
   timing/generation client on rank 0, Netdata power windows.
T: TR3 scorer (rank 0, in-process gold) against a headless peer (rank 1).
Gate arm sets TESSERA_GLM53_FOLD_SHARED_ADD=1 in every container.
No product code changes; the flag stays default off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SNAP = Path.cwd()
sys.path.insert(0, str(SNAP / "src"))
sys.path.insert(0, str(SNAP / "tools"))
sys.path.insert(0, str(SNAP / "experiments" / "graph_attest_702"))

from managed_window import (CLEANUP_SECONDS, HOSTS, MEMORY_POLICY, Envelope,
                            Refused, atomic_json, check_memory_policy,
                            read_json, require_claim)
import tp2_recipe as recipe
import comparison_arm_identity as owner

IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
         "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
GRAPH_COMPILATION = '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY"}'
MODEL_ID = "glm53-artifact"
API_PORT = 8142
MASTER_PORT = 29541


def sh(argv, **kw):
    return subprocess.run(argv, text=True, capture_output=True, timeout=kw.pop("timeout", 300),
                          **kw)


def barrier(rdv: Path, phase: str, rank: int, env: Envelope, wait_peer: bool = True) -> None:
    path = rdv / "barrier" / f"{phase}.rank{rank}.up"
    atomic_json(path, {"rank": rank, "unix": time.time()})
    if not wait_peer:
        return
    peer = rdv / "barrier" / f"{phase}.rank{1 - rank}.up"
    t0 = time.time()
    while env.remaining() > CLEANUP_SECONDS:
        if peer.exists():
            return
        if (rdv / "barrier" / f"{phase}.failed").exists():
            raise Refused(f"phase {phase} failed on the peer rank")
        if time.time() - t0 > 300:
            t0 = time.time()
            print(f"heartbeat: rank{rank} waits peer at {phase}", flush=True)
        time.sleep(5)
    raise Refused(f"phase {phase}: peer rank did not arrive")


def wait_done(run: Path, phase: str, env: Envelope) -> None:
    t0 = time.time()
    while env.remaining() > CLEANUP_SECONDS:
        if (run / "barrier" / f"{phase}.done").exists():
            return
        if (run / "barrier" / f"{phase}.failed").exists():
            raise Refused(f"phase {phase} failed on the peer rank")
        if time.time() - t0 > 300:
            t0 = time.time()
            print(f"heartbeat: waits {phase}.done", flush=True)
        time.sleep(10)

def mem_gib() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    raise Refused("MemAvailable unreadable")


def serve_lock(action: str, owner_name: str, run: Path) -> None:
    script = SNAP / "experiments" / "serve_lock.sh"
    env = dict(os.environ, SERVE_LOCK_OWNER=owner_name)
    if action == "acquire":
        proc = sh(["bash", "-c", f'source "{script}" && serve_lock_acquire'], env=env)
    else:
        proc = sh(["bash", "-c", f'source "{script}" && serve_lock_release'], env=env)
    (run / f"serve-lock-{action}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 and action == "acquire":
        raise Refused(f"serve lock unavailable: {proc.stdout[-500:]} {proc.stderr[-500:]}")


def docker(argv, out: Path, env: Envelope, tag: str) -> subprocess.CompletedProcess:
    with open(out, "a") as stream:
        proc = subprocess.run(["docker", *argv], text=True, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=max(60, env.remaining()))
    if proc.returncode != 0:
        raise Refused(f"docker {tag} rc={proc.returncode}: see {out}")


def post(base: str, path: str) -> int:
    req = urllib.request.Request(base + path, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status
    except Exception as exc:
        return -1 if "404" in str(exc) else -2


def build_config(setup: dict, ts: str) -> tuple[dict, dict]:
    arm_name = setup["arm"]
    config = {"artifact": setup["artifact"], "window_mode": "window4-eager-2048-4096",
              "profile_dir": str(Path(setup["run"]) / arm_name / "profiles"),
              "image": IMAGE, "fabric": "socket", "ts": ts}
    arm = {"arm": arm_name, "eager": "0", "compilation": GRAPH_COMPILATION,
           "spec": json.dumps(recipe.MTP), "max_batched": 2048, "fabric": "socket"}
    return config, arm


def phase_p(setup: dict, rank: int, env: Envelope, run: Path, owned: dict) -> dict:
    arm_name, fold = setup["arm"], setup["fold"]
    config, arm = build_config(setup, str(SNAP))
    out = run / arm_name
    (out / "profiles").mkdir(parents=True, exist_ok=True)
    (run / f"ext{rank}").mkdir(parents=True, exist_ok=True)
    image_env = {"TESSERA_CENSUS_RUNTIME_IMAGE": IMAGE,
                 "TESSERA_ROUTE_TRACE": f"/out/route-rank{rank}.json"}
    if fold:
        image_env["TESSERA_GLM53_FOLD_SHARED_ADD"] = "1"
    cmd = recipe.container(config, arm, owned, out, run / f"ext{rank}",
                           run / f"p-cid-rank{rank}", image_env,
                           master_port=MASTER_PORT, api_port=API_PORT)
    assert "--compilation-config" in cmd[-1] and "FULL_DECODE_ONLY" in cmd[-1]
    name = f"fold1204-{setup['run_id']}-{arm_name}-r{rank}"
    cmd[cmd.index("--cidfile") + 1] = str(run / f"p-cid-rank{rank}")
    cmd[cmd.index("--name") + 1] = name
    run_cmd = cmd[1:]
    serve_lock("acquire", f"fold1204-{arm_name}-p-r{rank}", run)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    try:
        if rank == 1:
            docker(run_cmd, run / "docker-p1.log", env, "p-run-r1")
            barrier(run, "p", rank, env)
            wait_done(run, "p", env)
        else:
            barrier(run, "p", rank, env)
            docker(run_cmd, run / "docker-p0.log", env, "p-run-r0")
            base = f"http://10.100.96.2:{API_PORT}"
            wait_url(base, env)
            t_start = time.time()
            atomic_json(run / "barrier" / "p.serve-up.json", {"unix": t_start})
            profile_state = post(base, "/start_profile")
            client = [sys.executable, str(SNAP / "tools" / "served_generation_client.py"),
                      "--base-url", base, "--model", MODEL_ID,
                      "--prompts", setup["prompts"], "--out", str(out / "timing.json"),
                      "--lens", "512", "2048", "8192", "--conc", "1", "--trials", "10",
                      "--output", "128", "--label-mode", "graphs-full-decode-only+mode-none",
                      "--label-fabric", "socket", "--label-server", f"fold1204-{arm_name}",
                      "--events", str(out / "events.jsonl")]
            with open(out / "client.log", "w") as stream:
                proc = subprocess.run(client, text=True, stdout=stream,
                                      stderr=subprocess.STDOUT,
                                      timeout=max(60, env.remaining() - CLEANUP_SECONDS))
            t_end = time.time()
            if proc.returncode != 0:
                raise Refused("timing client incomplete")
            if profile_state == 200:
                post(base, "/stop_profile")
            time.sleep(30)
            atomic_json(run / "barrier" / "p.done", {"unix": t_end})
            atomic_json(out / "phase-p.json",
                        {"t_start": t_start, "t_end": t_end, "profile_api": profile_state})
    finally:
        try:
            logs = sh(["docker", "logs", name], timeout=120)
            (out / "logs" / f"p-rank{rank}.log").write_text(logs.stdout + logs.stderr)
        except Exception:
            pass
        sh(["docker", "rm", "-f", name], timeout=120)
        serve_lock("release", f"fold1204-{arm_name}-p-r{rank}", run)
    return {"phase": "p"}


def phase_t(setup: dict, rank: int, env: Envelope, run: Path) -> dict:
    arm_name, fold = setup["arm"], setup["fold"]
    out = run / arm_name / "tr3"
    out.mkdir(parents=True, exist_ok=True)
    engine = read_json(run / "engine.json")
    fold_env = ["-e", "TESSERA_GLM53_FOLD_SHARED_ADD=1"] if fold else []
    mounts = ["-v", f"{setup['pq']}:/pq:ro", "-v", "/mnt/shared:/mnt/shared:ro",
              "-v", f"{run}:{run}", "-v", f"{run}/ext{rank}:/ext",
              "-e", "PQ_GOLD_PEER_DRIVER=experiments.measure_glm_tr3_vllm",
              "-e", "PQ_GOLD_PEER_ROOT=/pq",
              "-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory",
              "-e", "GIT_CONFIG_VALUE_0=*",
              "-e", f"TESSERA_CENSUS_RUNTIME_IMAGE={IMAGE}",
              "-e", "VLLM_HOST_IP=" + ("10.100.96.2", "10.100.96.1")[rank],
              "-e", "NCCL_SOCKET_IFNAME=enp1s0f0np0", "-e", "GLOO_SOCKET_IFNAME=enp1s0f0np0",
              "-e", "NCCL_IB_DISABLE=1", "-e", "NCCL_CUMEM_ENABLE=0",
              "-e", "NCCL_CUMEM_HOST_ENABLE=0", "-e", "NCCL_DMABUF_ENABLE=0",
              "-e", "TORCH_EXTENSIONS_DIR=/ext", "-e", "TMPDIR=/ext", "-e", "TRITON_CACHE_DIR=/ext/triton"]
    prep = ("set -e; rm -rf /ext/tessera && mkdir -p /ext/tessera /ext/tmp; "
            "cp -r /tessera-ro/src /tessera-ro/pyproject.toml /ext/tessera/; "
            "pip install --no-deps --no-build-isolation -q -e /ext/tessera")
    ts_src = str(SNAP / "src")
    serve_lock("acquire", f"fold1204-{arm_name}-t-r{rank}", run)
    try:
        if rank == 1:
            peer = ["run", "-d", "--name", f"fold1204-{setup['run_id']}-{arm_name}-t-r1",
                    "--network", "host", "--ipc", "host", "--device", "/dev/infiniband",
                    "--gpus", "all", "--shm-size", "16g",
                    "-v", f"{ts_src}:/tessera-ro/src:ro",
                    *mounts, *fold_env, "--entrypoint", "bash", IMAGE, "-c",
                    prep + "; cd /pq && exec python3 tools/gold_headless_peer.py "
                    + " ".join(engine["peer_argv"])]
            docker(peer, run / "docker-t1.log", env, "t-peer")
            barrier(run, "t", rank, env)
            wait_done(run, "t", env)
            barrier(run, "t", rank, env)
            sargv = engine["scorer_argv"]
            assert sargv[sargv.index("--output") + 1] == str(out / "full-vocabulary-kl.json")
            cmd = ["run", "-d", "--name", f"fold1204-{setup['run_id']}-{arm_name}-t-r0",
                   "--network", "host", "--ipc", "host", "--device", "/dev/infiniband",
                   "--gpus", "all", "--shm-size", "16g",
                   "-v", f"{ts_src}:/tessera-ro/src:ro",
                   *mounts, *fold_env, "--entrypoint", "bash", IMAGE, "-c",
                   prep + "; cd /pq && exec python3 experiments/measure_glm_tr3_vllm.py "
                   + " ".join(sargv)]
            docker(cmd, run / "docker-t0.log", env, "t-scorer")
            t0 = time.time()
            target = out / "full-vocabulary-kl.json"
            while env.remaining() > CLEANUP_SECONDS:
                if target.exists() and target.stat().st_size > 0:
                    break
                time.sleep(15)
            else:
                raise Refused("TR3 result not written before the envelope closed")
            atomic_json(run / "barrier" / "t.done", {"unix": time.time()})
            atomic_json(run / arm_name / "phase-t.json", {"t_start": t0, "t_end": time.time()})
    finally:
        for r in (0, 1):
            try:
                logs = sh(["docker", "logs", f"fold1204-{setup['run_id']}-{arm_name}-t-r{r}"],
                          timeout=120)
                (out / f"t-rank{r}.log").write_text(logs.stdout + logs.stderr)
            except Exception:
                pass
        sh(["docker", "rm", "-f", f"fold1204-{setup['run_id']}-{arm_name}-t-r{rank}"],
           timeout=120)
        serve_lock("release", f"fold1204-{arm_name}-t-r{rank}", run)
    return {"phase": "t"}


def phase_power(setup: dict, rank: int, env: Envelope, run: Path) -> dict:
    if rank != 0:
        return {"phase": "power", "role": "peer"}
    out = run / setup["arm"]
    if not shutil.which("python3"):
        raise Refused("no python3 for the power instrument")
    for host in ("sparklina", "sparky"):
        for phase in ("p", "t"):
            bounds = read_json(out / f"phase-{phase}.json")
            window = ":".join(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
                              for t in (int(bounds["t_start"]), int(bounds["t_end"]) + 60))
            proc = sh([sys.executable, str(SNAP / "experiments" / "box_power_window.py"),
                       "--host", host, "--label", f"{setup['arm']}-{phase}",
                       "--window", window, "--points", "0",
                       "--out", str(out / f"{host}-{phase}-power.json")], timeout=600)
            if proc.returncode != 0:
                raise Refused(f"power window failed for {host}: {proc.stderr[-300:]}")
    return {"phase": "power"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    key = os.environ.get("PRISMABUILD_ACTION_KEY", "")
    nonce = os.environ.get("PRISMABUILD_ACTION_NONCE", "")
    scope = os.environ.get("PRISMABUILD_ACTION_SCOPE", "")
    if not key or not nonce or not scope:
        raise Refused("fold arm requires an admitted PB action/attempt/scope")  # noqa: EM101
    run = args.run
    setup = read_json(run / "inputs.json")
    if hashlib.sha256((run / "inputs.json").read_bytes()).hexdigest() != os.environ.get(
            "FOLD_INPUT_SHA256", ""):
        raise Refused("inputs differ from the sealed action environment")  # noqa: EM101
    if socket.gethostname() != HOSTS[args.rank]:
        raise Refused(f"rank claimed on the wrong host: {socket.gethostname()}")  # noqa: EM101
    queue = Path(os.environ["PRISMABUILD_QUEUE_ROOT"])
    row = read_json(queue / "claimed" / (key + ".json"))
    if not row.get("gpu_admission"):
        raise Refused("model rank has no admitted GPU evidence")  # noqa: EM101
    owned = dict(rank=args.rank, action_key=key, nonce=nonce, scope_id=scope,
                 host=HOSTS[args.rank], run_id=setup["run_id"],
                 container_owner=os.environ["PRISMABUILD_CONTAINER_OWNER"],
                 claimed_unix=row["claimed_unix"],
                 input_sha256=hashlib.sha256((run / "inputs.json").read_bytes()).hexdigest())
    require_claim(owned, queue)
    check_memory_policy(read_json(run / "memory-policy.json"), where="fold1204 rank")
    end_unix = row["claimed_unix"] + setup["window_seconds"]
    env = Envelope(end_unix)
    if mem_gib() < MEMORY_POLICY["start_gib"]:
        raise Refused(f"host below the start floor {MEMORY_POLICY['start_gib']} GiB")  # noqa: EM101
    roster, _, _ = owner.read_artifact_roster(
        Path(setup["artifact"]), Path(setup["roster"]), setup["roster_sha256"])
    assert roster["manifest_sha256"]
    (run / "barrier").mkdir(parents=True, exist_ok=True)
    atomic_json(run / f"rank{args.rank}.preflight.json",
                {"host": HOSTS[args.rank], "mem_gib": mem_gib(), "owned": owned})
    receipt = {"run_id": setup["run_id"], "arm": setup["arm"], "rank": args.rank,
               "host": HOSTS[args.rank]}
    for key, fn in (("p", phase_p), ("t", phase_t), ("power", phase_power)):
        try:
            receipt[key] = fn(setup, args.rank, env, run, owned) if key != "power" else fn(setup, args.rank, env, run)
        except Exception as exc:
            atomic_json(run / "barrier" / f"{key}.failed", {"rank": args.rank})
            raise Refused(f"phase {key} error: {type(exc).__name__}: {exc}") from exc
    atomic_json(run / f"rank{args.rank}.receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as exc:
        print(f"rank refused: {exc}", file=sys.stderr)
        raise SystemExit(3)
