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

from managed_window import (CLEANUP_SECONDS, WINDOW_SECONDS, HOSTS, MEMORY_POLICY, Refused, atomic_json,
                            seal_check, check_memory_policy, read_json, terminal_cleanup)
import tp2_recipe as recipe

PB = Path("/mnt/shared/prismabuild-fleet/repo/tools")
RANK_PYTHON = "/home/rob/venvs/pb-cpu/bin/python"
QUEUE = Path("/mnt/shared/prismabuild-fleet/pb-queue")


def dry_arm(name: str, env: dict):
    config = recipe.inputs(env, live=False)
    arm = recipe.arm_settings(name, env)
    if config.get("window_mode") in recipe.BENCHMARK_PAIRS:
        arm = recipe.pair_arm(name, env, config["window_mode"])
    if "lever_env" in arm:
        print("  explicit per-arm lever env: " + shlex.join(f"{key}={value}" for key, value in arm["lever_env"].items()))
    if (Path(config["receipts"]) / name).exists():
        raise Refused("an arm's receipts are never merged into an existing arm")
    print(f"arm {name} (dry run): local PB rank action on each host; start nothing")
    print(f"  {config.get('window_seconds', WINDOW_SECONDS)}-second whole-window including rendezvous/all named arms/owned cleanup; {CLEANUP_SECONDS}s cleanup reserve")
    print(f"  both hosts: MemAvailable >= {MEMORY_POLICY['start_gib']} GiB; 1 Hz strict <2 GiB dual-rank abort; native threads=1")
    for rank in (0, 1):
        print(f"  serve rank{rank}: {shlex.join(recipe.serve(config, arm, rank))}")
    if config.get("window_mode") in recipe.PHASE_MODES:
        print("  all eleven L2048 seeded outputs; fresh servers per block; matched OFF and separate profiles/power for piece-major only")
    else:
        print("  exact October 5 c1 timing/profile population; no graph receipt" if config.get("window_mode") in recipe.BENCHMARK_PAIRS else
              "  equality: complete 48-choice pass, complete 48-choice second pass; four long screens")


def rows(root: Path, config: dict, env: dict) -> list[dict]:
    result = []
    for rank in (0, 1):
        staged = (dict(data_manifest=config["data_manifest"], residency="stage",
                       residency_ram="auto", residency_share="auto")
                  if config.get("window_mode") in recipe.PHASE_MODES else {})
        result.append(dict(argv=[RANK_PYTHON, "experiments/graph_attest_702/rank_window.py",
                                "--rank", str(rank), "--run", str(root / "inputs.json")],
                           cwd=str(Path(__file__).resolve().parents[2]), tags=[HOSTS[rank]],
                           demand=dict(cpu=8 if rank == 0 else 6, mem_gb=MEMORY_POLICY["host_cap_gib"], gpu=1),
                           gpu_memory_gb=MEMORY_POLICY["gpu_subset_cap_gib"], exclusive=True, measurement=True, host_class="gb10", max_attempts=1,
                           priority=-10 if config.get("window_mode") in recipe.PHASE_MODES else 10, priority_reason=(("Goal: exact reviewed A8S graph lever pair " if config.get("window_mode") == recipe.GRAPH_SHIP_MODE else
                                                        "Goal: exact reviewed A8S eager pair ") + config["window_mode"]
                                                        if config.get("window_mode") in recipe.BENCHMARK_PAIRS else
                                                        "Goal: full nominated A8 graph control after exact-head review; one paired window at a time"),
                           container_images=[config["image"]], timeout_s=config.get("window_seconds", WINDOW_SECONDS),
                           env={**{key: env[key] for key in ("TS", "ARTIFACT", "RECEIPTS", "FABRIC",
                                                          "SOURCE_COMMIT", "SOURCE_SHA256", "PRODUCER_COMMIT", "PRODUCER_SHA256")},
                                **{key: env[key] for key in ("WINDOW_MODE", "ARTIFACT_MANIFEST", "PQ_PIN_COMMIT", "CONTROL_ROOT", "PROFILE_MANIFEST", "DATA_MANIFEST") if key in env},
                                "GRAPH_WINDOW_INPUT_SHA256": recipe.sha(root / "inputs.json"),
                                "GRAPH_PEER_WAIT_SECONDS": str(config.get("peer_wait_seconds", 3600)),
                                **{key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                        "NUMEXPR_NUM_THREADS", "MAX_JOBS")},
                                "PYTHONDONTWRITEBYTECODE": "1"}, **staged))
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
    if mode in recipe.BENCHMARK_PAIRS and mode != recipe.DETERMINISM_MODE:
        for arm in arms:
            directory = Path(config["profile_dir"]) / arm["arm"]
            directory.mkdir(parents=True, exist_ok=False)
            directory.chmod(0o777)
    atomic_json(root / "memory-policy.json", MEMORY_POLICY)
    setup = dict(schema=("tessera.ship_graph_window.v1" if mode == recipe.GRAPH_SHIP_MODE else
                         "tessera.eager_determinism_window.v1" if mode in recipe.PHASE_MODES else
                         "tessera.eager_lever_window.v1" if mode == recipe.EAGER_LEVER_MODE else
                         "tessera.window4_eager_window.v1" if mode in recipe.BENCHMARK_PAIRS else "tessera.graph_control_window.v1"), run_id=uuid.uuid4().hex,
                 config=config, arms=arms, window_seconds=config.get("window_seconds", WINDOW_SECONDS), cleanup_seconds=CLEANUP_SECONDS,
                 requested_pb_timeout_s=config.get("window_seconds", WINDOW_SECONDS), effective_pb_timeout_s=None, peer_wait_seconds=config.get("peer_wait_seconds", 3600),
                 predecessors=predecessors, diskcheck=disks,
                 census_action_keys=census_keys,
                 memory_policy_sha256=recipe.sha(root / "memory-policy.json"),
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
            if owned.get("cleanup_error") or owned.get("staged_input_release_error") or not owned.get("local_cleanup"):
                raise Refused("rank physical or staged-reader cleanup is missing or failed; retain ownership")
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
    for key, field in (("WINDOW_MODE", "window_mode"), ("ARTIFACT_MANIFEST", "artifact_manifest"), ("PQ_PIN_COMMIT", "pq_pin_commit"), ("CONTROL_ROOT", "control_root"), ("PROFILE_MANIFEST", "profile_manifest"), ("DATA_MANIFEST", "data_manifest")):
        if field in setup["config"]:
            env[key] = setup["config"][field]
    recipe.require_producer(Path(__file__).resolve().parents[2], env["PRODUCER_COMMIT"],
                            env["PRODUCER_SHA256"], exact_head=True)
    current = recipe.inputs(env, live=True)
    recipe.check_control_record(setup["config"], current, where="Window4 submission",
                              refusal=Refused("prepared source/control changed; never restamp or resume an old window"))
    stored_rows = json.loads((root / "manifest.json").read_text())
    expected_rows = rows(root, setup["config"], env)
    safety_keys = ("tags", "demand", "gpu_memory_gb", "exclusive", "measurement", "host_class",
                   "max_attempts", "priority", "timeout_s", "data_manifest", "residency", "residency_ram", "residency_share")
    if (
            [{key: row.get(key) for key in safety_keys} for row in stored_rows] !=
            [{key: row.get(key) for key in safety_keys} for row in expected_rows]):
        raise Refused("prepared PB safety/resource manifest changed")
    seal_check("prepared manifest identity", stored_rows, expected_rows, where="Window4 submission",
               refusal=Refused("prepared PB manifest changed; never restamp admission/resource inputs"))
    review = read_json(reviews)
    if recipe.sha(root / "memory-policy.json") != setup["memory_policy_sha256"]:
        raise Refused("prepared memory policy changed; own digest does not match")
    check_memory_policy(read_json(root / "memory-policy.json"), where="Window4 submission")
    # Exact executing-code review is safety, not a recorded run-identity seal.
    code_roots = {str(Path(__file__).resolve().parents[2]), *(row["cwd"] for row in stored_rows)}
    code_heads = {subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"],
                                        text=True, timeout=10).strip() for source in code_roots}
    for who in ("parent", "D5"):
        accepted = review.get(who, {})
        if accepted.get("verdict") != "APPROVE" or code_heads != {accepted.get("head_sha")}:
            raise Refused(f"missing exact executing-code {who} review; no real model start")
    if setup["config"].get("window_mode") in recipe.BENCHMARK_PAIRS:
        # The corrected-runtime candidate review is an exact-head code review of the
        # PQ pin commit, not a recorded run identity: it refuses in dev and certified.
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
    if setup["config"].get("window_mode") in recipe.BENCHMARK_PAIRS:
        from watch_window_queue import other_windows
        live_windows = other_windows(observed)
        atomic_json(root / "pair-isolation.json", dict(complete=True, live_windows=live_windows,
                    reason="One paired window; PB priority election fences lower-priority prices, with no caller drain"))
        if live_windows:
            raise Refused("another paired window is live; never publish a second pair")
    atomic_json(root / "launch-diskcheck.json", dict(checks=diskcheck()))
    if (root / "submission-started.json").exists():
        raise Refused("this invocation was already submitted; preserve its failed/partial evidence")
    atomic_json(root / "submission-started.json", dict(reviews=review))
    # Only the published native driver publishes the group; published pbwait
    # owns completion for BOTH members. No private group writer or dispatcher.
    with (root / "campaign.log").open("w") as log:
        import socket
        if socket.gethostname() not in HOSTS:
            raise Refused("measurement submission must use the published PB client on a GB10 origin; celestia has no accelerator evidence")
        submitted = subprocess.run([sys.executable, str(PB / "pbgang.py"),
                                    "--manifest", str(root / "manifest.json")],
                                   capture_output=True, text=True)
        log.write(submitted.stdout + submitted.stderr)
        log.flush()
        if submitted.returncode:
            raise Refused("native gang publication refused; retain driver/member failure evidence")
        gang = json.loads(submitted.stdout.strip().splitlines()[-1])
        if (gang.get("schema") != "prismabuild.pbgang.v1" or len(gang.get("members", [])) != 2
                or any(not isinstance(key, str) or len(key) != 64 for key in gang["members"])):
            raise Refused("native gang driver returned an incomplete two-member identity")
        atomic_json(root / "native-gang.json", gang)
        wait_argv = [sys.executable, str(PB / "pbwait.py"), "--json", "--wait-s", "6000", *gang["members"]]
        atomic_json(root / "completion-client.json", dict(group=gang, driver_pid=os.getpid(),
                    published_client=str(PB / "pbwait.py"), argv=wait_argv, ownership="Both members and exact physical collector"))
        print(json.dumps(dict(event="native_gang_published", **gang)), flush=True)
        with (root / "member-completion.json").open("w") as completed:
            done = subprocess.run(wait_argv, stdout=completed, stderr=log)
    physical = collect(root, QUEUE)
    if not physical["ownership_released"]:
        raise Refused(f"physical handoff unproven; preserve ownership and logs: {physical['error']}")
    if (done.returncode or any(r["identity"]["returncode"] for r in physical["ranks"])
            or any((root / f"failed-rank{rank}.json").exists() for rank in (0, 1))):
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
    ap.add_argument("--preflight-output", type=Path, help="with --dry-run: bounded actual-input D38 proof, no submission")
    ap.add_argument("--prepare", type=Path)
    ap.add_argument("--census-key", action="append", default=[], help="current exact campaign769 GPU action key; two keys, no inferred roster")
    ap.add_argument("--prepare-role-preflight", action="store_true", help="prepare CPU role checks, no model output reservation or submission")
    ap.add_argument("--handoffs", type=Path)
    ap.add_argument("--submit", type=Path)
    ap.add_argument("--reviews", type=Path)
    args = ap.parse_args()
    if args.dry_run:
        if args.preflight_output:
            if os.environ.get("WINDOW_MODE") not in recipe.PHASE_MODES:
                raise Refused("input preflight output requires the explicit seeded investigation scope")
            from eager_determinism import input_preflight
            config = recipe.inputs(dict(os.environ), live=False)
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.preflight_output, input_preflight(config))
        for arm, env in recipe.parse_plan(args.plan):
            print(f"== arm {arm}")
            dry_arm(arm, {**os.environ, **env})
        recipe.plan(args.plan, mode=os.environ.get("WINDOW_MODE", "graph-control"))
        return 0
    if args.prepare and not args.submit:
        if os.environ.get("WINDOW_MODE", "graph-control") not in recipe.BENCHMARK_PAIRS and len(args.census_key) != 2:
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
