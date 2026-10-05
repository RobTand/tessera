"""Prepare/submit the existing graph plan through published PB, not a second scheduler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from managed_window import (CLEANUP_SECONDS, WINDOW_SECONDS, HOSTS, Refused, atomic_json,
                            read_json, terminal_cleanup)
import tp2_recipe as recipe

PB = Path("/mnt/shared/prismabuild-fleet/repo/tools")
CLIENT = "/home/rob/tmp/pb-submit-celestia-20261003/bin/python"
RANK_PYTHON = "/home/rob/venvs/pb-cpu/bin/python"
QUEUE = Path("/mnt/shared/prismabuild-fleet/pb-queue")


def dry_arm(name: str, env: dict):
    config = recipe.inputs(env, live=False)
    arm = recipe.arm_settings(name, env)
    if config.get("window_mode") == recipe.EAGER_MODE:
        expected = {"eager2048": "2048", "eager4096": "4096"}
        if env.get("MAX_BATCHED") != expected.get(name) or arm["eager"] != "1":
            raise Refused("Window4 is exactly eager2048/2048 then eager4096/4096")
        arm["max_batched"] = int(env["MAX_BATCHED"])
    if (Path(config["receipts"]) / name).exists():
        raise Refused("an arm's receipts are never merged into an existing arm")
    print(f"arm {name} (dry run): local PB rank action on each host; start nothing")
    print(f"  5400-second whole-window including rendezvous/all named arms/owned cleanup; {CLEANUP_SECONDS}s cleanup reserve")
    print("  both hosts: MemAvailable >= 114 GiB; sampled physical floor 16 GiB; native threads=1")
    for rank in (0, 1):
        print(f"  serve rank{rank}: {shlex.join(recipe.serve(config, arm, rank))}")
    print("  exact October 5 c1 timing/profile population; no graph receipt" if config.get("window_mode") == recipe.EAGER_MODE else
          "  equality: complete 48-choice pass, complete 48-choice second pass; four long screens")


def rows(root: Path, config: dict, env: dict) -> list[dict]:
    result = []
    for rank in (0, 1):
        result.append(dict(argv=[RANK_PYTHON, "experiments/graph_attest_702/rank_window.py",
                                "--rank", str(rank), "--run", str(root / "inputs.json")],
                           cwd=str(Path(__file__).resolve().parents[2]), tags=[HOSTS[rank]],
                           demand=dict(cpu=8 if rank == 0 else 6, mem_gb=104, gpu=1),
                           gpu_memory_gb=102, exclusive=True, measurement=True, host_class="gb10", max_attempts=1,
                           priority=10, priority_reason=("Goal: exact reviewed A8S eager Window4 MNBT2048/4096"
                                                        if config.get("window_mode") == recipe.EAGER_MODE else
                                                        "Goal: full nominated A8 graph control after exact-head review; one paired window at a time"),
                           container_images=[config["image"]], timeout_s=WINDOW_SECONDS,
                           env={**{key: env[key] for key in ("TS", "ARTIFACT", "RECEIPTS", "FABRIC",
                                                          "SOURCE_COMMIT", "SOURCE_SHA256", "PRODUCER_COMMIT", "PRODUCER_SHA256")},
                                **{key: env[key] for key in ("WINDOW_MODE", "ARTIFACT_MANIFEST", "PQ_PIN_COMMIT") if key in env},
                                "GRAPH_WINDOW_INPUT_SHA256": recipe.sha(root / "inputs.json"),
                                "GRAPH_PEER_WAIT_SECONDS": "3600",
                                **{key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                        "NUMEXPR_NUM_THREADS", "MAX_JOBS")},
                                "PYTHONDONTWRITEBYTECODE": "1"}))
    return result


def read_terminal(identity: dict, queue: Path) -> dict:
    key = identity["action_key"]
    if (queue / "claimed" / (key + ".json")).exists():
        raise Refused(f"scope still claimed: {key}; retain ownership, never drain it")
    paths = [queue / state / (key + ".json") for state in ("done", "failed")]
    found = [p for p in paths if p.exists()]
    if len(found) != 1:
        raise Refused(f"no unambiguous terminal for {key}")
    return read_json(found[0])


def handoffs(path: Path, queue: Path) -> list[dict]:
    supplied = read_json(path)
    if set(supplied) != {"census769"}:
        raise Refused("only the current census769 terminal/physical handoff is required by the superseding CEO order")
    proofs = []
    for name, ranks in supplied.items():
        if len(ranks) != 2 or {rank["host"] for rank in ranks} != set(HOSTS):
            raise Refused(f"{name} requires both exact physical rank handoffs")
        for rank in ranks:
            terminal = read_terminal(rank, queue)
            proof = terminal_cleanup(rank, terminal)
            proofs.append(dict(predecessor=name, identity=rank, cleanup=proof))
    return proofs


def diskcheck(*, for_model=True):
    evidence = []
    for need, hosts, paths in ((8 if for_model else 1, "sparky,sparklina", "/home/rob/tmp"),
                               (1, "sparky,sparklina", "/mnt/shared")):
        done = subprocess.run(["fleet-diskcheck", "--need-gb", str(need), "--hosts", hosts,
                               "--paths", paths], capture_output=True, text=True, timeout=30)
        try:
            proof = json.loads(done.stdout)
        except ValueError as exc:
            raise Refused(f"D1 diskcheck returned no JSON: {done.stdout} {done.stderr}") from exc
        evidence.append(proof)
        if done.returncode or proof.get("pass") is not True:
            raise Refused(f"fresh D1 refused: {proof}")
    return evidence


def prepare(root: Path, path: Path, env: dict, predecessor_path: Path | None, census_keys: list[str], *, role_preflight=False):
    recipe.require_producer(Path(__file__).resolve().parents[2], env.get("PRODUCER_COMMIT", ""),
                            env.get("PRODUCER_SHA256", ""), exact_head=True)
    config = recipe.inputs(env, live=True)
    mode = config.get("window_mode", "graph-control")
    arms = recipe.plan(path, mode=mode)
    if any(arm.get("fabric", config["fabric"]) != config["fabric"] for arm in arms):
        raise Refused("plan and frozen issues-owned fabric differ")
    if not str(root).startswith("/mnt/shared/"):
        raise Refused("rendezvous must be shared and fresh for each submission")
    if env["RECEIPTS"] != str(root / "arms"):
        raise Refused("RECEIPTS must be this fresh invocation ROOT/arms")
    predecessors = handoffs(predecessor_path, QUEUE) if predecessor_path else []
    disks = diskcheck(for_model=not role_preflight)
    root.mkdir(parents=True, exist_ok=False)
    (root / "arms").mkdir()
    if mode == recipe.EAGER_MODE:
        for arm in arms:
            directory = Path(config["profile_dir"]) / arm["arm"]
            directory.mkdir(parents=True, exist_ok=False)
            directory.chmod(0o777)
    setup = dict(schema=("tessera.window4_eager_window.v1" if mode == recipe.EAGER_MODE else "tessera.graph_control_window.v1"), run_id=uuid.uuid4().hex,
                 config=config, arms=arms, window_seconds=WINDOW_SECONDS, cleanup_seconds=CLEANUP_SECONDS,
                 requested_pb_timeout_s=WINDOW_SECONDS, effective_pb_timeout_s=None, peer_wait_seconds=3600,
                 predecessors=predecessors, diskcheck=disks,
                 census_action_keys=census_keys,
                 resources=dict(cpu_total=14, shared_host_memory_gib_total=208, gpu_subset_gib_total=204,
                                exclusive_devices=2, local_output_cap_gib_per_host=8, shared_output_cap_gib=1))
    atomic_json(root / "inputs.json", setup)
    (root / "manifest.json").write_text(json.dumps(rows(root, config, env), indent=2) + "\n")
    print(root / "manifest.json")


def collect(root: Path, queue: Path) -> dict:
    result = dict(ownership_released=False, ranks=[], error=None)
    try:
        for rank in (0, 1):
            owned = read_json(root / f"outcome-rank{rank}.json")
            if owned.get("simulation") is not False:
                raise Refused("a simulated CPU outcome cannot release a live model window")
            terminal = read_terminal(owned, queue)
            proof = terminal_cleanup(owned, terminal)
            if owned.get("cleanup_error") or not owned.get("local_cleanup"):
                raise Refused("rank local cleanup is missing or failed; retain ownership")
            if any(record.get("containers_empty") is not True or record.get("gpu_descendants_empty") is not True
                   for record in owned["local_cleanup"]):
                raise Refused("owned container/GPU descendants not proven empty")
            result["ranks"].append(dict(identity=owned, terminal=terminal, cleanup=proof))
        result["ownership_released"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(root / "handoff.json", result)
    return result


def submit(root: Path, reviews: Path):
    setup = read_json(root / "inputs.json")
    env = dict(os.environ, TS=setup["config"]["ts"], ARTIFACT=setup["config"]["artifact"],
               RECEIPTS=setup["config"]["receipts"], FABRIC=setup["config"]["fabric"],
               SOURCE_COMMIT=setup["config"]["source_commit"], SOURCE_SHA256=setup["config"]["src_sha256"],
               PRODUCER_COMMIT=setup["config"]["producer_commit"], PRODUCER_SHA256=setup["config"]["producer_sha256"])
    for key, field in (("WINDOW_MODE", "window_mode"), ("ARTIFACT_MANIFEST", "artifact_manifest"), ("PQ_PIN_COMMIT", "pq_pin_commit")):
        if field in setup["config"]:
            env[key] = setup["config"][field]
    recipe.require_producer(Path(__file__).resolve().parents[2], env["PRODUCER_COMMIT"],
                            env["PRODUCER_SHA256"], exact_head=True)
    if recipe.inputs(env, live=True) != setup["config"]:
        raise Refused("prepared source/control changed; never restamp or resume an old window")
    if json.loads((root / "manifest.json").read_text()) != rows(root, setup["config"], env):
        raise Refused("prepared PB manifest changed; never restamp admission/resource inputs")
    review = read_json(reviews)
    for who in ("parent", "D5"):
        if review.get(who, {}).get("verdict") != "APPROVE" or review[who].get("head_sha") != env["PRODUCER_COMMIT"]:
            raise Refused(f"missing exact frozen-source {who} review; no real model start")
    if setup["config"].get("window_mode") == recipe.EAGER_MODE:
        for who in ("parent", "D5"):
            candidate = review.get("runtime", {}).get(who, {})
            if candidate.get("verdict") != "APPROVE" or candidate.get("head_sha") != setup["config"]["pq_pin_commit"]:
                raise Refused(f"missing exact corrected runtime candidate {who} review; no Window4 model start")
    for predecessor in setup["predecessors"]:
        terminal_cleanup(predecessor["identity"], read_terminal(predecessor["identity"], QUEUE))
    from datetime import datetime, timezone
    from watch_window_queue import inspect as inspect_queue
    observed = inspect_queue(datetime.now(timezone.utc).isoformat(), setup["census_action_keys"], root / "launch-queue.json")
    if not observed["submit_allowed"]:
        raise Refused("campaign769 is queued/claimed or queue view incomplete; never queue a second paired window")
    atomic_json(root / "launch-diskcheck.json", dict(checks=diskcheck()))
    if setup["config"].get("window_mode") == recipe.EAGER_MODE:
        from eager_benchmark import require_drain
        atomic_json(root / "pact-drain-at-submit.json", require_drain(QUEUE))
    if (root / "submission-started.json").exists():
        raise Refused("this invocation was already submitted; preserve its failed/partial evidence")
    atomic_json(root / "submission-started.json", dict(reviews=review))
    # The published campaign owns fanout and waits. No detach, SSH launcher, retry, or atomic-pair claim.
    with (root / "campaign.log").open("w") as log:
        # Class-scoped GPU measurements must seal live GB10 platform evidence on a Spark (D26).
        import socket
        if socket.gethostname() not in HOSTS:
            raise Refused("measurement submission must use the published PB client on a GB10 origin; celestia has no accelerator evidence")
        done = subprocess.run([sys.executable, str(PB / "pbcampaign.py"), "--transport", "pool", "--wait-s", "6000",
                               "--max-inflight", "2", str(root / "manifest.json")], stdout=log, stderr=subprocess.STDOUT)
    physical = collect(root, QUEUE)
    if not physical["ownership_released"]:
        raise Refused(f"physical handoff unproven; preserve ownership and logs: {physical['error']}")
    if done.returncode or any(r["identity"]["returncode"] for r in physical["ranks"]):
        return 1
    # Bind both rank keys to every arm without pretending the rank0 key owns the other rank.
    keys = [r["identity"]["action_key"] for r in physical["ranks"]]
    manifest = {a["arm"]: dict(action_key=keys[0], rank_action_keys=keys) for a in setup["arms"]}
    atomic_json(root / "receipt-manifest.json", manifest)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--census-key", action="append", default=[], help="current exact campaign769 GPU action key; two keys, no inferred roster")
    ap.add_argument("--prepare-role-preflight", action="store_true", help="prepare CPU role checks, no model output reservation or submission")
    ap.add_argument("--handoffs", type=Path)
    ap.add_argument("--submit", type=Path)
    ap.add_argument("--reviews", type=Path)
    args = ap.parse_args()
    if args.dry_run:
        for arm, env in recipe.parse_plan(args.plan):
            print(f"== arm {arm}")
            dry_arm(arm, {**os.environ, **env})
        recipe.plan(args.plan, mode=os.environ.get("WINDOW_MODE", "graph-control"))
        return 0
    if args.prepare and not args.submit:
        if os.environ.get("WINDOW_MODE", "graph-control") != recipe.EAGER_MODE and len(args.census_key) != 2:
            raise Refused("supply the two exact current campaign769 action keys")
        prepare(args.prepare, args.plan, os.environ, args.handoffs, args.census_key,
                role_preflight=args.prepare_role_preflight)
        return 0
    if args.submit and args.reviews and not args.prepare:
        return submit(args.submit, args.reviews)
    raise Refused("use --dry-run, --prepare ROOT --handoffs JSON, or --submit ROOT --reviews JSON; no direct model launch")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Refused, ValueError, OSError) as exc:
        print(f"graph window refused: {exc}", file=sys.stderr)
        raise SystemExit(3)
