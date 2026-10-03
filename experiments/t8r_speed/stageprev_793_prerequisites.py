"""Read actual public-PB prerequisites; never claim, mint, retire or launch.

An absent offer is UNKNOWN. A complete current ledger census plus an actual
fresh normal positive-kind denial can establish the approved exclusion. A
source predicate, fabricated denial or stale zero cannot do so.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

from ab_stageprev_accept import load

SCHEMA = "tessera.stageprev.operative_prerequisites.v1"
SDK_SOURCE = "prismabuild.client.PoolQueue.capacity_census+available+latest_denials(include_local=False)"


def _age(stamp, now, maximum):
    return (type(stamp) in (int, float) and math.isfinite(stamp)
            and 0 <= now - stamp <= maximum)


def evaluate(record, packet, *, now=None):
    from stageprev_793_prepare import ROOT_DECLARATIONS, SOURCE_ROOT, PACKET_PATH
    now = time.time() if now is None else now
    admission = packet["operative_admission"]
    kind, maximum = admission["exclusion_kind"], admission["max_observation_age_s"]
    unmet = []
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return {"status": "HOLD", "control_ready": False, "unmet": ["Actual prerequisite read-set missing/wrong schema"]}
    if record.get("source") != SDK_SOURCE or record.get("queue_root") != admission["queue_root"]:
        unmet.append("Wrong public metadata owner/queue")
    if not _age(record.get("observed_unix"), now, maximum):
        unmet.append("Prerequisite read-set stale/future/unknown")
    expected_sha = hashlib.sha256((SOURCE_ROOT / PACKET_PATH).read_bytes()).hexdigest()
    if record.get("packet_sha256") != expected_sha or record.get("gpu_submitted") is not False:
        unmet.append("Wrong operative packet or execution scope")
    ledger = record.get("excluded_ledger", {})
    if not isinstance(ledger, dict):
        ledger = {}
    if (ledger.get("host") != admission["excluded_host"] or ledger.get("complete") is not True
            or ledger.get("unreadable") != [] or ledger.get("changed_during_read") is not False
            or ledger.get("directories_present") is not True
            or not _age(ledger.get("observed_unix"), now, maximum)):
        unmet.append("Excluded-host claim census incomplete/stale/changed/unknown")
    total, available = ledger.get("total"), ledger.get("available")
    if not isinstance(total, dict) or not isinstance(available, dict):
        unmet.append("Claim census is not an actual complete kind map")
    elif (type(total.get(kind, 0)) is not int or type(available.get(kind, 0)) is not int
          or total.get(kind, 0) != 0 or available.get(kind, 0) != 0):
        unmet.append("Excluded-host positive-kind capacity is not zero")
    key = record.get("refusal_action_key")
    if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
        unmet.append("No exact real refusal action identity")
    denials = record.get("actual_denials")
    actual_refusal = False
    for denial in denials if isinstance(denials, list) else []:
        if not isinstance(denial, dict):
            continue
        evidence = denial.get("evidence")
        if not isinstance(evidence, dict):
            continue
        requested = evidence.get("reservation_demand", evidence.get("demand"))
        capacity = evidence.get("capacity_total")
        if not isinstance(requested, dict) or not isinstance(capacity, dict):
            continue
        need = requested.get(kind)
        if (denial.get("host") == admission["excluded_host"]
                and denial.get("action_key") == key
                and denial.get("reason") == "never_fits_capacity"
                and _age(denial.get("denied_unix"), now, maximum)
                and type(need) is int and need > 0
                and type(capacity.get(kind, 0)) is int and capacity.get(kind, 0) == 0):
            actual_refusal = True
    if not actual_refusal:
        unmet.append("Current actual positive-kind refusal absent; source predicate/omitted offer is not proof")
    if (record.get("coordinator_host") != admission["coordinator_host"]
            or record.get("runtime_changed_during_read") is not False
            or not isinstance(record.get("runtime_manifest"), dict)
            or not record["runtime_manifest"].get("generation")):
        unmet.append("Actual coordinator/current runtime identity unknown or changed")
    if record.get("coordinator_root_census_unreadable") != [] or record.get("filesystem_errors") != []:
        unmet.append("Aggregate current ROOT or used-filesystem observation incomplete")
    extra = record.get("conservative_current_root_reservation_bytes")
    if type(extra) is not int or extra < 0:
        unmet.append("Current aggregate ROOT reservations unknown")
        extra = 0
    root_budgets = {packet["environment"][root]: int(packet["environment"][maximum_env])
                    for root, maximum_env, _ in ROOT_DECLARATIONS}
    required_paths = set(root_budgets)
    required_paths.update((packet["environment"]["PB_CLIENT_ROOT"],
                           str(Path(packet["readset"]["path"]).parent), packet["source"]["out"]))
    filesystems, seen = record.get("used_filesystems"), set()
    if not isinstance(filesystems, list) or not filesystems:
        unmet.append("Actual used filesystem observations absent")
        filesystems = []
    for filesystem in filesystems:
        if not isinstance(filesystem, dict):
            unmet.append("Unreadable used filesystem")
            continue
        paths = filesystem.get("paths")
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            unmet.append("Used filesystem path coverage unknown")
            continue
        if seen.intersection(paths) or len(set(paths)) != len(paths):
            unmet.append("Duplicated operation path in filesystem observations")
        seen.update(paths)
        minimum = sum(root_budgets.get(path, 0) for path in paths)
        if any(path in root_budgets for path in paths):
            minimum += extra
        size, free, allowance = (filesystem.get(name) for name in
                                  ("size_bytes", "available_bytes", "aggregate_root_allowance_bytes"))
        if (filesystem.get("host") != admission["coordinator_host"]
                or not _age(filesystem.get("observed_unix"), now, maximum)
                or any(type(value) is not int for value in (size, free, allowance))
                or size <= 0 or free < 0 or allowance < minimum
                or 20 * (free - allowance) < size):
            unmet.append("Used filesystem below fresh5%+aggregate ROOT allowance or unknown")
    if not required_paths <= seen:
        unmet.append("Actual operation paths missing from filesystem read-set")
    if record.get("root_growth_sum_bytes") != admission["aggregate_root_growth_bytes"]:
        unmet.append("Real ROOT/MAX aggregate changed or unknown")
    return {"status": "HOLD" if unmet else "CONTROL_VALID_NOT_GPU_GO",
            "control_ready": not unmet, "unmet": unmet,
            "remaining_gpu_gates": ["Separate Root exact #793 numeric GPU GO",
                                   "Actual admitted worker/platform/driver/image and public reader namespace",
                                   "Actual storage-tier reservation, quiet/residency and both-Spark telemetry",
                                   "Required independent exact operative packet review"]}


def collect(packet, refusal_action_key):
    from prismabuild.client import PoolQueue, SDK_VERSION
    from stageprev_793_prepare import SOURCE_ROOT, PACKET_PATH, public_resource_demand
    if SDK_VERSION != 4:
        raise ValueError("Actual published public SDK4 required")
    admission = packet["operative_admission"]
    if socket.gethostname() != admission["coordinator_host"]:
        raise ValueError("Observe actual healthy coordinator, not unsafe/excluded SP")
    _, roots = public_resource_demand(packet)
    queue = PoolQueue(Path(admission["queue_root"]))
    runtime_path = Path(packet["environment"]["PB_CLIENT_ROOT"]) / "RUNTIME_VERSION.json"
    runtime_before = runtime_path.read_bytes()
    ledger = queue.ledger(admission["excluded_host"])
    total_before, unreadable_before = ledger.capacity_census()
    available = ledger.available()
    total_after, unreadable_after = ledger.capacity_census()
    ledger_stamp = time.time()
    # Public best-effort snapshots only; never record/manufacture a denial.
    denials = queue.latest_denials({refusal_action_key}, include_local=False).get(refusal_action_key, []) if refusal_action_key else []
    offers = queue.offers(max_age_s=admission["max_observation_age_s"])
    excluded_offer = next((offer for offer in offers if offer.get("host") == admission["excluded_host"]), None)
    coordinator_ledger = queue.ledger(admission["coordinator_host"])
    local_total, local_unreadable = coordinator_ledger.capacity_census()
    local_free = coordinator_ledger.available()
    kind = admission["exclusion_kind"]
    held_slots = max(0, local_total.get(kind, 0) - local_free.get(kind, 0))
    # Conservative all-current host Root/spool reservations, including this
    # CPU observer. This is not measured remaining filesystem growth.
    extra_root_allowance = held_slots * (1 << 30)
    paths = {pair["root"]: pair["max_bytes"] for pair in roots}
    for path in (packet["environment"]["PB_CLIENT_ROOT"],
                 str(Path(packet["readset"]["path"]).parent), packet["source"]["out"]):
        paths.setdefault(path, 0)
    filesystems, errors = {}, []
    for path, allowance in paths.items():
        try:
            device = os.stat(path).st_dev
            space = os.statvfs(path)
            slot = filesystems.setdefault(device, {"host": socket.gethostname(), "device": device,
                "paths": [], "size_bytes": space.f_blocks * space.f_frsize,
                "available_bytes": space.f_bavail * space.f_frsize,
                "aggregate_root_allowance_bytes": 0, "observed_unix": time.time()})
            slot["paths"].append(path)
            slot["available_bytes"] = min(slot["available_bytes"], space.f_bavail * space.f_frsize)
            slot["aggregate_root_allowance_bytes"] += allowance
        except OSError as exc:
            errors.append({"path": path, "error": str(exc)})
    for slot in filesystems.values():
        if any(path in {pair["root"] for pair in roots} for path in slot["paths"]):
            slot["aggregate_root_allowance_bytes"] += extra_root_allowance
    runtime_after = runtime_path.read_bytes()
    return {"schema": SCHEMA, "source": SDK_SOURCE, "observed_unix": time.time(),
        "queue_root": admission["queue_root"], "coordinator_host": socket.gethostname(),
        "runtime_manifest": json.loads(runtime_after),
        "runtime_changed_during_read": runtime_before != runtime_after,
        "excluded_ledger": {"host": admission["excluded_host"], "total": total_after,
            "available": available, "complete": not (unreadable_before or unreadable_after),
            "unreadable": [*unreadable_before, *unreadable_after],
            "changed_during_read": total_before != total_after,
            "directories_present": ledger.free_dir.is_dir() and ledger.held_dir.is_dir(),
            "observed_unix": ledger_stamp, "free_path": str(ledger.free_dir), "held_path": str(ledger.held_dir)},
        "refusal_action_key": refusal_action_key, "actual_denials": denials,
        "denial_read_path": str(queue.root / "reservations" / admission["excluded_host"] / "adaptive" / "claim-denials.json"),
        "excluded_offer": excluded_offer,
        "offer_observation": "FRESH_RECORDED_OFFER" if excluded_offer else "UNKNOWN_OMITTED_NOT_A_ZERO_MEASUREMENT",
        "used_filesystems": list(filesystems.values()), "filesystem_errors": errors,
        "coordinator_root_census_unreadable": local_unreadable,
        "conservative_current_root_reservation_bytes": extra_root_allowance,
        "root_growth_sum_bytes": sum(pair["max_bytes"] for pair in roots),
        "packet_sha256": hashlib.sha256((SOURCE_ROOT / PACKET_PATH).read_bytes()).hexdigest(),
        "gpu_submitted": False}


def main():
    from stageprev_793_prepare import PACKET_PATH, SOURCE_ROOT, SL_COORDINATOR, prepare
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refusal-action-key", default=os.environ.get("PRISMABUILD_ACTION_KEY", ""))
    args = parser.parse_args()
    packet = load(SOURCE_ROOT / PACKET_PATH)
    metadata = collect(packet, args.refusal_action_key)
    json.dump({"prerequisite_readset": metadata, "proposal": prepare(SL_COORDINATOR, metadata)},
              sys.stdout, sort_keys=True, indent=2)
    print()


if __name__ == "__main__":
    main()
