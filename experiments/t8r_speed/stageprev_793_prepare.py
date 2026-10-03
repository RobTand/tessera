"""Prepare, never execute, the Root-approved #793 operative successor.

The original7763 packet and worker harness stay immutable. This owner only
renders the existing published-PB invocation; ab_arms.sh still owns all four
fresh processes on one worker. Operative readiness is not a numeric GPU GO.
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
ORIGINAL_PACKET_PATH = "experiments/configs/stageprev_793_numeric_packet.json"
ORIGINAL_PACKET_SHA256 = "bd93b379eb8ea0fe93169a28913cd7b9191711f7e50447cd9e679eafb99c467d"
PACKET_PATH = "experiments/configs/stageprev_793_numeric_packet_v2.json"
EXPECTED_PATH = "experiments/configs/stageprev_793_expected.json"
PB_ROOT = "/mnt/shared/prismabuild-fleet/repo"
QUEUE_ROOT = "/mnt/shared/prismabuild-fleet/pb-queue"
OLD_COORDINATOR = "/home/rob/tmp/astra-resume-20261002/t8_performance/history793-native-source"
SL_COORDINATOR = "/home/rob/tmp/astra-resume-20261002/t8_performance/history793-numeric-frozen-7763"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
NATIVE_THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "MAX_JOBS")
ROOT_DECLARATIONS = (
    ("TESSERA_793_SEAL_ROOT", "TESSERA_793_SEAL_MAX_BYTES", "/tmp"),
    ("TESSERA_793_WORK_ROOT", "TESSERA_793_WORK_MAX_BYTES", "/home/rob/tmp"),
    ("TESSERA_793_DOCKER_ROOT", "TESSERA_793_DOCKER_MAX_BYTES", "/var/lib/docker"),
)
APPROVED_CLAUSE = "Fresh >=5% physical free-space floors on every filesystem actually used by the coordinator, admitted worker or storage, plus aggregate declared operational ROOT allowances on used coordinator/worker root filesystems; unchanged capacity and quiet/residency checks; timestamped both-Spark CPU/power/clocks/residency. Sparky is unused for coordination, execution or storage and cannot claim this action under the existing PB-enforced Root/MAX zero-capacity ledger; absent, stale or changed exclusion evidence means HOLD. Preserve the unchanged GB10 class/image/native/public-reader contract, fixed nine cases and one-worker master/fix/fix/master sequence; Root exact #793 GPU GO remains required."
APPROVAL_PROPOSAL_SHA256 = "308d479290a4ccdafa42e5768df4dd15bdbbf143fc5b504f31f73a4f5f4a3fcf"


def original_contract():
    path = SOURCE_ROOT / ORIGINAL_PACKET_PATH
    if hashlib.sha256(path.read_bytes()).hexdigest() != ORIGINAL_PACKET_SHA256:
        raise ValueError("original frozen packet changed; preserve and stop")
    return load(path)


def root_environment():
    result = {"PRISMABUILD_LOCAL_SCRATCH_PAIRS": ",".join(
        root + ":" + maximum for root, maximum, _ in ROOT_DECLARATIONS)}
    for root, maximum, path in ROOT_DECLARATIONS:
        result[root], result[maximum] = path, str(1 << 30)
    return result


def validate_successor(packet, expected):
    original = original_contract()
    allowed = {"schema", "status", "environment", "requires_before_publication"}
    added = {"supersedes", "admission_approval", "operative_admission"}
    if set(packet) != set(original) | added:
        raise ValueError("successor fields differ from the approved cutover")
    if packet["schema"] != "tessera.stageprev.current_numeric_packet.v2":
        raise ValueError("require the versioned operative successor")
    for key in set(original) - allowed:
        if packet[key] != original[key]:
            raise ValueError("scientific/original evidence binding changed: " + key)
    if packet["supersedes"] != {"path": ORIGINAL_PACKET_PATH, "commit": FROZEN_COMMIT,
                                 "sha256": ORIGINAL_PACKET_SHA256}:
        raise ValueError("successor does not bind immutable original7763")
    approval = packet["admission_approval"]
    if (approval != {"source": "local://tessera-k32-floor-scope-approval-20261003.json",
                     "decision": "APPROVE_A_OPERATIONAL_CONTRACT_SUCCESSOR_ONLY",
                     "authority": "Main, existing campaign/executive authority",
                     "decided_utc": "2026-10-03T20:24:37.963260+00:00",
                     "proposal_sha256": APPROVAL_PROPOSAL_SHA256,
                     "launch_authorized": False}):
        raise ValueError("operative wording approval is not numeric GPU authority")
    if packet["requires_before_publication"] != [*original["requires_before_publication"][:-1], APPROVED_CLAUSE]:
        raise ValueError("approved admission successor changed or dropped another gate")
    if packet["environment"] != {**original["environment"], **root_environment()}:
        raise ValueError("only real ROOT/MAX annotations may augment the frozen environment")
    population = expected_population(expected)
    if expected["phase"] != "numeric" or len(population) != 9:
        raise ValueError("require the frozen nine-case numeric population")
    if packet["cases"] != expected["cases"] or packet["native_files"] != expected["native_files"]:
        raise ValueError("packet population/native bindings differ from acceptance")
    for key, value in expected["launch"].items():
        observed = packet["environment"].get(key, "")
        if (shlex.split(observed) != value if isinstance(value, list) else observed != value):
            raise ValueError("numeric launch differs from its frozen manifest: " + key)
    admission = packet["operative_admission"]
    if (admission["coordinator_host"] != "sparklina" or admission["coordinator_checkout"] != SL_COORDINATOR
            or admission["queue_root"] != QUEUE_ROOT or admission["excluded_host"] != "sparky"
            or admission["require_complete_claim_census"] is not True
            or admission["require_current_actual_positive_kind_refusal"] is not True
            or admission["roots_declare_reservation_not_quota"] is not True
            or admission["aggregate_root_growth_bytes"] != 3 * (1 << 30)
            or admission["max_observation_age_s"] != 120):
        raise ValueError("operative exclusion or physical allowance contract changed")
    return original


def public_resource_demand(packet):
    from prismabuild import local_scratch
    roots = local_scratch.scratch_pairs(packet["environment"])
    terms = local_scratch.scratch_terms(packet["environment"])
    if packet["operative_admission"]["exclusion_kind"] != local_scratch.KIND:
        raise ValueError("exclusion must use the actual public local-disk kind")
    if sum(pair["max_bytes"] for pair in roots) != packet["operative_admission"]["aggregate_root_growth_bytes"]:
        raise ValueError("aggregate ROOT allowance does not cover actual sealed pairs")
    return {"cpu": 2, "mem_gb": 16, "gpu": 1, **terms}, roots


def proposal(packet, expected, checkout, prerequisites=None):
    original = validate_successor(packet, expected)
    if str(checkout) != SL_COORDINATOR:
        raise ValueError("this proposal relocates only to the owned frozen SL checkout")
    demand, roots = public_resource_demand(packet)
    argv = ["python3", PB_ROOT + "/tools/pbrun.py", "--cwd", str(checkout),
            "--transport", "pool", "--measurement", "--host-class", "gb10",
            "--exclusive", "--gpu", "--cpus", "2", "--demand", "mem_gb=16",
            "--gpu-memory-gb", "8", "--timeout-s", "600", "--max-attempts", "1",
            "--container-image", packet["image"], "--data-manifest", packet["readset"]["path"],
            "--residency", "stage", "--residency-ram", "off"]
    for key, value in packet["environment"].items():
        argv += ["--env", key + "=" + value]
    for key in NATIVE_THREADS:
        argv += ["--env", key + "=1"]
    argv += ["--", *packet["entrypoint"]]
    from stageprev_793_prerequisites import evaluate
    checks = evaluate(prerequisites, packet) if prerequisites is not None else {
        "status": "HOLD", "unmet": ["Fresh actual prerequisite read-set not supplied"]}
    return {"schema": "tessera.stageprev.coordinator_proposal.v2", "status": "GPU_HOLD",
            "frozen_commit": FROZEN_COMMIT, "original_packet_sha256": ORIGINAL_PACKET_SHA256,
            "old_coordinator": {"host": "sparky", "checkout": OLD_COORDINATOR},
            "proposed_coordinator": {"host": "sparklina", "checkout": str(checkout)},
            "unchanged": {"packet_environment": original["environment"], "resources": packet["resources"],
                          "readset": packet["readset"], "native_files": packet["native_files"],
                          "source": packet["source"], "cases": packet["cases"],
                          "entrypoint": packet["entrypoint"], "acceptance": EXPECTED_PATH},
            "operative_environment": packet["environment"], "public_resource_derived_demand": demand,
            "actual_root_declarations": roots,
            "execution_owner": "experiments/t8r_speed/ab_arms.sh",
            "process_order": ["master", "fix", "fix", "master"], "action_count": 1, "worker_count": 1,
            "placement": {"class": "gb10", "hostname_pin": None,
                          "match_required": ["platform", "executable/ABI", "driver",
                                             "compute capability", "device models", "image"]},
            "requires_before_publication": packet["requires_before_publication"],
            "prerequisite_checks": checks,
            "operational_root_allowance_bytes": packet["operative_admission"]["aggregate_root_growth_bytes"],
            "allowance_scope": "Aggregate real ROOT/MAX reservations, not measured peaks or hard caps; no root redirects.",
            "argv_proposal_only": argv}


def prepare(checkout, prerequisites=None):
    checkout = Path(checkout).resolve(strict=True)
    if socket.gethostname() != "sparklina":
        raise ValueError("inspect this proposed coordinator from sparklina, never unsafe SP")
    def git(*args):
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    if git("rev-parse", "HEAD") != FROZEN_COMMIT:
        raise ValueError("coordinator checkout is not the frozen worker harness")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("coordinator checkout has unexpected changes; preserve and stop")
    for relative in (ORIGINAL_PACKET_PATH, EXPECTED_PATH):
        committed = subprocess.run(["git", "-C", str(checkout), "show", FROZEN_COMMIT + ":" + relative],
                                   check=True, capture_output=True, timeout=15).stdout
        if (checkout / relative).read_bytes() != committed:
            raise ValueError("frozen contract bytes differ: " + relative)
    packet = load(SOURCE_ROOT / PACKET_PATH)
    record = proposal(packet, load(checkout / EXPECTED_PATH), checkout, prerequisites)
    record["packet_sha256"] = hashlib.sha256((SOURCE_ROOT / PACKET_PATH).read_bytes()).hexdigest()
    record["coordinator_temp_directory"] = tempfile.gettempdir()
    record["gpu_submitted"] = False
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default=SL_COORDINATOR)
    parser.add_argument("--prerequisites", type=Path)
    args = parser.parse_args()
    prerequisites = load(args.prerequisites) if args.prerequisites else None
    json.dump(prepare(args.checkout, prerequisites), sys.stdout, sort_keys=True, indent=2)
    print()


if __name__ == "__main__":
    main()
