"""Execute the sealed PM timing packet inside one admitted PB attempt.

Only the existing run_direct_arm and owned_cleanup own launch and containment.
The original benchmark argv, native bank and numeric proof are not adapted.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--go-record")
    parser.add_argument("--go-sha256")
    args = parser.parse_args()
    if not os.environ.get("PRISMABUILD_ACTION_KEY"):
        raise ValueError("PM batch controller requires an admitted PrismaBuild attempt")
    if digest(args.packet) != args.packet_sha256:
        raise ValueError("PB packet changed")
    packet = json.loads(Path(args.packet).read_bytes())
    if digest(__file__) != packet["controller_sha256"]:
        raise ValueError("PB controller changed")
    if digest(packet["window"]) != packet["window_sha256"]:
        raise ValueError("original timing window changed")
    window = json.loads(Path(packet["window"]).read_bytes())
    protocol = json.loads(Path(window["protocol"]).read_bytes())
    bindings = {window["protocol"]: window["protocol_sha256"],
                window["numeric_qualification"]["receipt"]: window["numeric_qualification"]["receipt_sha256"],
                **window["execution_owner"]["source_bindings"]}
    bindings.update({str(Path(window["worktree"]) / p): h for p,h in protocol["harness"].items()})
    bindings[protocol["input_manifest"]["path"]] = protocol["input_manifest"]["sha256"]
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("sealed input changed: " + path)
    out = Path(window["output"])
    if out.exists():
        raise ValueError("timing output already exists; no automatic retries")
    root = Path(packet["evidence_root"])
    if args.prepare:
        root.mkdir(exist_ok=False)
        result = {"schema": "tessera.pm_pb_preparation.v1", "action_key": os.environ["PRISMABUILD_ACTION_KEY"],
                  "packet_sha256": args.packet_sha256, "bindings": bindings,
                  "original_window_sha256": packet["window_sha256"],
                  "benchmark_executed": False, "numeric_qualification_reused": True}
        (root / "prepare.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 0
    if not args.go_record or not args.go_sha256 or digest(args.go_record) != args.go_sha256:
        raise ValueError("exact coordinator GPU GO required")
    go = json.loads(Path(args.go_record).read_bytes())
    if go.get("packet_sha256") != args.packet_sha256 or go.get("gpu_go") is not True:
        raise ValueError("GPU GO does not authorize this exact packet")
    if not root.is_dir():
        raise ValueError("CPU preparation evidence missing")
    os.chdir(window["worktree"])
    sys.path[:0] = window["execution_owner"]["namespace_paths"]
    from paired_k32_action import run_direct_arm, owned_cleanup
    def interrupted(signum, frame):
        raise KeyboardInterrupt("PM timing interrupted: " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ, **window["env"])
    env["PB_ACTION_KEY"] = os.environ["PRISMABUILD_ACTION_KEY"]
    started = time.time()
    result = {"schema": "tessera.pm_pb_timing_terminal.v1", "action_key": env["PB_ACTION_KEY"],
              "packet_sha256": args.packet_sha256, "go": go, "start_unix": started,
              "assigned_affinity": sorted(os.sched_getaffinity(0)), "returncode": None,
              "energy_status": "HOLD_pending_independent_coverage_clock_and_instrument_review"}
    try:
        with (root / "controller.log").open("xb") as log:
            result["returncode"] = run_direct_arm(window["argv"], env, log, out,
                                                   root / "pressure-guard.jsonl", deadline_s=600)
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        result["end_unix"] = time.time()
        result["measurement_retention"] = ("benchmark_receipt_present" if (out / "bench_t8r.json").is_file()
            else "benchmark_not_published_partial_inprocess_events_not_retained")
        result["cleanup"] = owned_cleanup(out)
        (root / "action-window.txt").write_text(f"{started} {result['end_unix']}\n")
        (root / "terminal.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if result["returncode"]:
        return result["returncode"]
    (root / "timing").symlink_to(out, target_is_directory=True)
    subprocess.run([sys.executable, "experiments/t8r_speed/routed_gate_netdata.py", str(root)], check=True)
    evidence = {str(p): digest(p) for p in [out / "bench_t8r.json", root / "netdata.json", *sorted(out.glob("torch-*.json"))]}
    (root / "evidence-sha256.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"action_key": env["PB_ACTION_KEY"], "evidence_sha256": evidence}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
