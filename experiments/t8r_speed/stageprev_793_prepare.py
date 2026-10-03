"""Prepare, never execute, the unchanged #793 public-reader PB invocation.

This is an operational relocation record, not a dispatcher or a GPU GO. The
existing ab_arms.sh remains the one-worker, four-process execution owner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile

from ab_stageprev_accept import expected_population, load

FROZEN_COMMIT = "7763d38ff8d5c50ffe928d4e906a986232d4b809"
PACKET_PATH = "experiments/configs/stageprev_793_numeric_packet.json"
EXPECTED_PATH = "experiments/configs/stageprev_793_expected.json"
PB_ROOT = "/mnt/shared/prismabuild-fleet/repo"
OLD_COORDINATOR = "/home/rob/tmp/astra-resume-20261002/t8_performance/history793-native-source"
SL_COORDINATOR = "/home/rob/tmp/astra-resume-20261002/t8_performance/history793-numeric-frozen-7763"
NATIVE_THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "MAX_JOBS")


def proposal(packet, expected, checkout):
    """Render the frozen invocation; no queue, lease or container operations."""
    population = expected_population(expected)
    if expected["phase"] != "numeric" or len(population) != 9:
        raise ValueError("require the frozen nine-case numeric population")
    if packet["cases"] != expected["cases"] or packet["native_files"] != expected["native_files"]:
        raise ValueError("packet population/native bindings differ from acceptance")
    if packet["entrypoint"][0:2] != ["bash", "experiments/t8r_speed/ab_arms.sh"] or packet["entrypoint"][3:] != expected["arms"]:
        raise ValueError("require the existing one-action master/fix owner")
    if packet["entrypoint"][2] != packet["source"]["out"]:
        raise ValueError("output namespace differs from the frozen source preparation")
    resources = packet["resources"]
    if resources != {"cpu": 2, "native_threads": 1, "aggregate_memory_gib": 16,
                     "gpu_memory_gib": 8, "gpu": 1, "exclusive_gb10": True,
                     "timeout_s": 600, "max_attempts": 1}:
        raise ValueError("the frozen admission envelope may not change")
    environment = packet["environment"]
    for key, value in expected["launch"].items():
        observed = environment.get(key, "")
        if (shlex.split(observed) != value if isinstance(value, list) else observed != value):
            raise ValueError("numeric launch differs from its frozen manifest: " + key)
    if environment.get("PB_CLIENT_ROOT") != PB_ROOT or environment.get("AB_EXPECTED_CASES") != EXPECTED_PATH:
        raise ValueError("require the existing published SDK and acceptance owner")
    if environment.get("AB_INPUT_MANIFEST") != packet["readset"]["path"]:
        raise ValueError("require the frozen sealed public-reader manifest")
    if environment.get("BENCH_RO_MOUNTS") != str(Path(packet["readset"]["path"]).parent):
        raise ValueError("require the existing exact readset metadata mount")
    if environment.get("ORACLE_IMAGE") != packet["image"] or "@sha256:" not in packet["image"]:
        raise ValueError("require the exact immutable image declaration")
    if str(checkout) != SL_COORDINATOR:
        raise ValueError("this proposal relocates only to the owned frozen SL checkout")
    argv = ["python3", PB_ROOT + "/tools/pbrun.py", "--cwd", str(checkout),
            "--transport", "pool", "--measurement", "--host-class", "gb10",
            "--exclusive", "--gpu", "--cpus", "2", "--demand", "mem_gb=16",
            "--gpu-memory-gb", "8", "--timeout-s", "600", "--max-attempts", "1",
            "--container-image", packet["image"], "--data-manifest", packet["readset"]["path"],
            "--residency", "stage", "--residency-ram", "off"]
    for key, value in environment.items():
        argv += ["--env", key + "=" + value]
    for key in NATIVE_THREADS:
        argv += ["--env", key + "=1"]
    argv += ["--", *packet["entrypoint"]]
    return {"schema": "tessera.stageprev.coordinator_proposal.v1", "status": "GPU_HOLD",
            "frozen_commit": FROZEN_COMMIT,
            "old_coordinator": {"host": "sparky", "checkout": OLD_COORDINATOR},
            "proposed_coordinator": {"host": "sparklina", "checkout": str(checkout)},
            "unchanged": {"packet_environment": environment, "resources": resources,
                          "readset": packet["readset"], "native_files": packet["native_files"],
                          "source": packet["source"], "cases": packet["cases"],
                          "entrypoint": packet["entrypoint"], "acceptance": EXPECTED_PATH},
            "execution_owner": "experiments/t8r_speed/ab_arms.sh",
            "process_order": ["master", "fix", "fix", "master"],
            "action_count": 1, "worker_count": 1,
            "placement": {"class": "gb10", "hostname_pin": None,
                          "match_required": ["platform", "executable/ABI", "driver",
                                             "compute capability", "device models", "image"]},
            "requires_before_publication": packet["requires_before_publication"],
            "floor_interpretation": "Literal frozen floor/both-Spark requirement retained; relocation does not waive it.",
            "operational_root_allowance_bytes": 1073741824,
            "allowance_scope": "Provisional coordinator/worker Git and container lifecycle allowance, not a measured peak or hard cap.",
            "argv_proposal_only": argv}


def prepare(checkout):
    checkout = Path(checkout).resolve(strict=True)
    if socket.gethostname() != "sparklina":
        raise ValueError("inspect this proposed coordinator from sparklina, never unsafe SP")
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    if git("rev-parse", "HEAD") != FROZEN_COMMIT:
        raise ValueError("coordinator checkout is not the frozen packet commit")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("coordinator checkout has unexpected changes; preserve and stop")
    for relative in (PACKET_PATH, EXPECTED_PATH):
        committed = subprocess.run(["git", "-C", str(checkout), "show", FROZEN_COMMIT + ":" + relative],
                                   check=True, capture_output=True, timeout=15).stdout
        if (checkout / relative).read_bytes() != committed:
            raise ValueError("frozen contract bytes differ: " + relative)
    record = proposal(load(checkout / PACKET_PATH), load(checkout / EXPECTED_PATH), checkout)
    record["packet_sha256"] = hashlib.sha256((checkout / PACKET_PATH).read_bytes()).hexdigest()
    record["coordinator_temp_directory"] = tempfile.gettempdir()
    record["coordinator_floor_observation"] = "Not collected or inferred by this preparation; fresh capacity is a separate gate."
    record["gpu_submitted"] = False
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default=SL_COORDINATOR)
    args = parser.parse_args()
    json.dump(prepare(args.checkout), sys.stdout, sort_keys=True, indent=2)
    print()


if __name__ == "__main__":
    main()
