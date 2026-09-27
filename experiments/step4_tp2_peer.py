#!/usr/bin/env python3
"""One joined rank-1 stock-vLLM headless peer for the step-4 TP2 observer.

The installer execs this only after writing the peer's own per-job runtime
evidence. The head seals a shared plan against those exact bytes before this
process starts vLLM. This is the pinned vLLM 0.28.1rc1 headless MP path, not
a serving-core patch or an independent engine pretending to be rank one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def head_completion_record(path, session_id):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if record.get("session_id") == session_id else None


def wait_for_plan(path, timeout_s, *, finished=None, session_id=None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if finished is not None and head_completion_record(finished, session_id):
            raise RuntimeError("TP2 head finished before a sealed shared plan was available")
        if path.is_file():
            try:
                plan = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
                continue
            if plan.get("world_size") == 2 and plan.get("rank_runtime_evidence"):
                return plan
        time.sleep(1)
    raise TimeoutError(f"shared TP2 observer plan did not appear at {path}")


def peer_engine_args(plan, evidence, *, host_ip):
    bound = plan["rank_runtime_evidence"]["1"]
    if digest(evidence) != bound["sha256"]:
        raise ValueError("peer installer evidence differs from the sealed shared plan")
    engine = dict(plan["selected_configuration"]["engine_args"])
    if (engine.get("nnodes") != 2 or engine.get("node_rank") != 0
            or engine.get("tensor_parallel_size") != 2
            or engine.get("distributed_executor_backend") != "mp"
            or not engine.get("master_addr")):
        raise ValueError("shared plan does not describe an MP TP2 head and peer")
    if host_ip == plan["selected_configuration"]["environment"].get("VLLM_HOST_IP"):
        raise ValueError("peer host IP equals the head host IP")
    engine["node_rank"] = 1
    engine.update(plan["observer_engine_args"])
    engine["model"] = plan["model"]
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serve-mode", choices=("resident", "streamed"), required=True)
    parser.add_argument("--expected-modules", type=json.loads, required=True)
    parser.add_argument("--collector", required=True)
    parser.add_argument("--workspaces", default="-")
    parser.add_argument("--wait-s", type=int, default=600)
    args = parser.parse_args()
    session_path = args.plan.parent.parent / "head-session.json"
    session_id = json.loads(session_path.read_text())["session_id"]
    finished = args.plan.parent.parent / "head-finished.json"
    os.environ["VLLM_HOST_IP"] = args.host_ip
    from experiments.full_engine_worker_identity import actual_host_ip
    actual_host_ip()
    from experiments.step4_cache_preflight import check_worker_caches
    check_worker_caches(args.out / "worker-cache-preflight.json")
    from step4_capture_driver import native_preflight, observer_preflight

    native_preflight(args.out, args.serve_mode, args.expected_modules)
    observer_preflight(args.out, ["--collector", args.collector,
                                        "--workspaces", args.workspaces])
    plan = wait_for_plan(args.plan, args.wait_s, finished=finished, session_id=session_id)
    engine = peer_engine_args(plan, args.evidence, host_ip=args.host_ip)
    # The stock peer uses the same observer plan and bootstrap as the head.
    root = str(Path(__file__).resolve().parents[1])
    if plan["observation_mode"] == "resources":
        os.environ["TESSERA_ENGINE_RESOURCE_PLAN"] = str(args.plan)
        os.environ.pop("TESSERA_ENGINE_TIMING_PLAN", None)
        os.environ["PYTHONPATH"] = root + "/experiments/resource_bootstrap:" + root
    elif plan["observation_mode"] == "timings":
        os.environ["TESSERA_ENGINE_TIMING_PLAN"] = str(args.plan)
        os.environ.pop("TESSERA_ENGINE_RESOURCE_PLAN", None)
        os.environ["PYTHONPATH"] = root
    elif plan["observation_mode"] == "kv":
        os.environ.pop("TESSERA_ENGINE_RESOURCE_PLAN", None)
        os.environ.pop("TESSERA_ENGINE_TIMING_PLAN", None)
        os.environ["PYTHONPATH"] = root
    else:
        raise ValueError("unsupported TP2 observation mode")
    os.environ.update(plan["observer_environment"])
    os.environ["VLLM_HOST_IP"] = args.host_ip
    args.ready.write_text(json.dumps({"schema": "tessera.tp2_peer_ready.v1",
                                      "session_id": session_id,
                                      "plan_sha256": digest(args.plan),
                                      "runtime_evidence_sha256": digest(args.evidence),
                                      "host_ip": args.host_ip}) + "\n")
    # Exact pinned stock headless path in vllm/entrypoints/cli/serve.py.
    from vllm import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    configured = AsyncEngineArgs(**engine).create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER, headless=True)
    if configured.parallel_config.node_rank_within_dp <= 0:
        raise ValueError("configured peer did not resolve to a non-head node")
    # End this exact peer process after the head has finished qualification.
    # The shared sentinel carries the plan digest; no process-name sweep or
    # remote kill is involved.
    def head_finished():
        return head_completion_record(finished, session_id)

    def watch_head():
        while True:
            if head_finished() is not None:
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(1)

    threading.Thread(target=watch_head, name="tp2-head-finished", daemon=True).start()
    executor = MultiprocExecutor(configured, monitor_workers=False)
    try:
        try:
            executor.start_worker_monitor(inline=True)
        except KeyboardInterrupt:
            if head_finished() is None:
                raise
        completion = head_finished()
        if completion is None:
            raise RuntimeError("rank-1 worker ended before the joined head finished")
        if completion.get("plan_sha256") != digest(args.plan):
            raise RuntimeError("head completion differs from the peer's sealed plan")
    finally:
        executor.shutdown()
    (args.out / "peer-completion.json").write_text(json.dumps({
        "schema": "tessera.tp2_peer_completion.v1", "plan_sha256": digest(args.plan),
        "session_id": session_id, "head_finished": True,
        "clean_worker_shutdown": True}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
