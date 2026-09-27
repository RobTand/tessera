"""Join two actual TP ranks of the #399 timing observer, without admission.

This is a v2 observation. It keeps every rank's raw capture/profile paths and
digests, per-sample direct gaps and owner segments. Qualification can remain
false with usable diagnostic data; no missing quantity is replaced by zero.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from experiments.full_engine_timings import analyze_profile_partition

SCHEMA = "tessera.full_engine_timing_observation.v2"
POLICY_SCHEMA = "tessera.full_engine_observer_impact_policy.v1"
RUN_SCHEMA = "tessera.stock_engine_raw_timing_run.v1"
COMMON_DIGESTS = ("configuration_sha256", "model_sha256", "runtime_manifest_sha256",
                  "workload_sha256", "assignment_sha256", "canonical_units_sha256")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check(passed, reason=None, **detail):
    return {"passed": passed is True, "reason": None if passed else reason, **detail}


def _policy(plan, capture_dir):
    bound = plan.get("observer_impact_policy")
    if not isinstance(bound, dict):
        return None, "no versioned observer-impact policy was supplied", None
    path = Path(bound.get("path", ""))
    if not path.is_file() and path.is_absolute() and path.parts[:2] == ("/", "out"):
        path = Path(capture_dir).parent.joinpath(*path.parts[2:])
    if not path.is_file() or _sha(path) != bound.get("sha256"):
        return None, "bound observer-impact policy bytes are unavailable or changed", None
    value = json.loads(path.read_text())
    maximum = value.get("max_abs_relative_difference")
    if (value.get("schema") != POLICY_SCHEMA or type(maximum) not in (float, int)
            or not math.isfinite(maximum) or maximum < 0):
        return None, "observer-impact policy has an unsupported schema or bound", None
    return value, None, {"path": str(path), "sha256": bound["sha256"]}


def _host_log(launch_dir, rank):
    if launch_dir is None:
        return None
    return Path(launch_dir) / f"rank-{rank}" / "host-vitals.log"


def _host_log_identity(path, *, host_ip, device_uuid):
    if path is None or not path.is_file():
        return _check(False, "rank host-vitals log is absent")
    sampled = []
    for line in path.read_text().splitlines():
        fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        if "compute_apps" in fields:
            sampled.append((fields.get("host_ip"), fields.get("gpu_uuid")))
    return _check(bool(sampled) and all(pair == (host_ip, device_uuid) for pair in sampled),
                  "host-vitals samples do not bind this worker's host IP and physical GPU UUID",
                  samples=len(sampled))


def _read_arm(capture_dir, arm, sample, armed, worker):
    """Read one actual worker arm and independently replay its raw profile."""
    rank = worker["rank"]
    directory = Path(worker["directory"])
    if not directory.is_absolute() or not directory.exists():
        directory = Path(capture_dir) / directory.name
    files = {name: directory / f"{name}.json" for name in ("capture", "profile", "partition")}
    if any(not path.is_file() for path in files.values()):
        return {"rank": rank, "arm": arm, "sample": sample, "error": "raw capture/profile/partition file is missing",
                "files": {name: {"path": str(path), "sha256": None} for name, path in files.items()}}
    payloads = {name: path.read_bytes() for name, path in files.items()}
    raw = {name: json.loads(payload) for name, payload in payloads.items()}
    refs = {name: {"path": str(files[name]), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()}
    del payloads
    capture, profile, published = (raw[name] for name in ("capture", "profile", "partition"))
    identity = capture.get("identity") or {}
    error = None
    if (capture.get("schema") != "tessera.full_engine_timing_capture.v2"
            or capture.get("arm") != arm or identity.get("rank") != rank
            or identity.get("world_size") != 2 or identity.get("device_id") != worker.get("device_id")
            or identity.get("device_uuid") != worker.get("device_uuid")
            or identity.get("host") != worker.get("host")
            or (armed.get("pid"), armed.get("rank"), armed.get("world_size"),
                armed.get("device_id"), armed.get("device_uuid"), armed.get("host")) !=
               (worker.get("pid"), rank, 2, worker.get("device_id"), worker.get("device_uuid"),
                worker.get("host"))):
        error = "actual worker/capture identity disagrees with the arm's rank and device"
    elif capture.get("profile_sha256") != refs["profile"]["sha256"]:
        error = "capture's raw profiler digest disagrees with the profiler bytes"
    replay = analyze_profile_partition(capture, profile)
    if (published.get("schema") != "tessera.full_engine_timing_partition.v2"
            or published.get("status") != replay["status"]
            or published.get("issues") != replay["issues"]
            or published.get("steps") != replay["steps"]):
        error = "saved partition differs from replay of the raw profiler and capture"
    counts = (_count_operations(profile, capture) if arm == "control" or replay["status"] != "observed_same_run_partition"
              else {"by_step": {step["step_id"]: len(step["gpu_operations"]) for step in replay["steps"]},
                    "outside_steps": 0,
                    "total": sum(len(step["gpu_operations"]) for step in replay["steps"])})
    return {"rank": rank, "arm": arm, "sample": sample, "files": refs,
            "identity": identity, "capture": capture,
            "partition_status": replay["status"], "partition_issues": replay["issues"],
            "gpu_operations": counts, "error": error}


def _count_operations(profile, capture):
    from experiments.full_engine_timing_observation import count_gpu_operations
    return count_gpu_operations(profile, capture)


def _rank_terms(arms, samples):
    result = {}
    for phase in ("prefill", "decode"):
        rows = []
        for sample in samples:
            arm = next((row for row in arms if row["arm"] == "partition" and row["sample"] == sample), None)
            if arm is None or "capture" not in arm:
                return None
            step = next((step for step in arm["capture"]["steps"] if step["phase"] == phase), None)
            if step is None:
                return None
            rows.append(step)
        order = [unit["unit_id"] for unit in rows[0]["units"]]
        if any([unit["unit_id"] for unit in row["units"]] != order for row in rows):
            return None
        result[phase] = {"samples": samples,
                         "whole_step_ms": [row["whole_step_ms"] for row in rows],
                         "fixed_gaps_ms": [row["fixed_gaps_ms"] for row in rows],
                         "fixed_gap_sum_ms": [math.fsum(row["fixed_gaps_ms"]) for row in rows],
                         "candidate_ms": {unit: [next(item["elapsed_ms"] for item in row["units"]
                                                       if item["unit_id"] == unit) for row in rows]
                                          for unit in order},
                         "segments": [row["segments"] for row in rows], "unit_order": order}
    return result


def build_tp2_timing_observation(capture_dir, *, launch_dir=None):
    from experiments.full_engine_timing_observation import host_exclusivity, _parse_vitals, worker_exclusivity

    capture_dir = Path(capture_dir)
    plan = json.loads((capture_dir / "observer-plan.json").read_text())
    run = json.loads((capture_dir / "run.json").read_text())
    if run.get("schema") != RUN_SCHEMA or plan.get("world_size") != 2:
        raise ValueError("TP2 timing observation needs a TP2 plan and raw timing run")
    plan_path, run_path = capture_dir / "observer-plan.json", capture_dir / "run.json"
    plan_sha, run_sha = _sha(plan_path), _sha(run_path)
    samples = list(range(plan.get("timing_samples", 0)))
    policy, policy_error, policy_ref = _policy(plan, capture_dir)
    rows, errors = [], []
    seen = set()
    for entry in run.get("arms", []):
        arm, sample = entry.get("arm"), entry.get("sample")
        if arm not in {"control", "partition"} or sample not in samples:
            errors.append("arm/sample lies outside the declared timing plan")
            continue
        armed, workers = entry.get("armed", []), entry.get("workers", [])
        if (len(armed) != 2 or len(workers) != 2
                or {row.get("rank") for row in armed} != {0, 1}
                or {row.get("rank") for row in workers} != {0, 1}):
            errors.append(f"{arm}/{sample} lacks two distinct actual ranks")
            continue
        by_armed = {row["rank"]: row for row in armed}
        for worker in workers:
            key = (arm, sample, worker["rank"])
            if key in seen:
                errors.append(f"duplicate rank arm/sample: {key}")
                continue
            seen.add(key)
            rows.append(_read_arm(capture_dir, arm, sample, by_armed[worker["rank"]], worker)
                  | {"tokens": entry.get("tokens")})
    expected = {(arm, sample, rank) for arm in ("control", "partition")
                for sample in samples for rank in (0, 1)}
    qualifications = {
        "plan_binding": _check(run.get("plan_sha256") == plan_sha,
                                "raw timing run does not bind the observer plan bytes"),
        "complete_world": _check(len(samples) >= 3 and seen == expected and not errors,
                                 "missing/duplicate rank arms or fewer than three samples", errors=errors),
        "positive_warmup": _check(run.get("warmup_requests", 0) >= 1,
                                  "positive identical-token warmup was not recorded"),
        "identical_tokens": _check(isinstance(run.get("warmup_tokens"), list)
                                   and len(run["warmup_tokens"]) == 2
                                   and bool(rows) and all(row["tokens"] == run["warmup_tokens"]
                                                          for row in rows), "generated token IDs differ"),
    }
    digest_identity = {name: plan["identity"].get(name) for name in COMMON_DIGESTS}
    rank_records = []
    for rank in (0, 1):
        arms = sorted((row for row in rows if row["rank"] == rank),
                      key=lambda row: (row["sample"], row["arm"]))
        identities = [row.get("identity") or {} for row in arms]
        device_uuids = {identity.get("device_uuid") for identity in identities}
        hosts = {json.dumps(identity.get("host"), sort_keys=True) for identity in identities}
        process_ids = {row["capture"].get("exclusivity", {}).get("own_pid") for row in arms
                       if "capture" in row}
        checks = {
            "raw_evidence": _check(len(arms) == 2 * len(samples) and all(row.get("error") is None for row in arms),
                                   "a rank arm lacks authentic raw capture/profile/partition evidence",
                                   errors=[row.get("error") for row in arms if row.get("error")]),
            "rank_identity": _check(bool(identities) and all(identity.get("rank") == rank
                                     and identity.get("world_size") == 2
                                     and all(identity.get(name) == value for name, value in digest_identity.items())
                                     for identity in identities)
                                     and len(device_uuids) == 1 and None not in device_uuids
                                     and len(hosts) == 1 and "null" not in hosts
                                     and len(process_ids) == 1 and None not in process_ids,
                                     "rank, device, process or shared identity changes across arms"),
            "stream_coverage": _check(bool(arms) and all(row.get("partition_status") == "observed_same_run_partition"
                                      and not row.get("partition_issues")
                                      for row in arms if row["arm"] == "partition"),
                                      "raw trace does not cover each step and owner on joined streams"),
        }
        controls = {row["sample"]: row for row in arms if row["arm"] == "control"}
        parts = {row["sample"]: row for row in arms if row["arm"] == "partition"}
        count_ok = True
        overhead, collection = {}, {}
        for sample in samples:
            left, right = controls.get(sample), parts.get(sample)
            if left is None or right is None or "gpu_operations" not in left or "gpu_operations" not in right:
                count_ok = False
                continue
            collection[str(sample)] = {"control": left["gpu_operations"], "partition": right["gpu_operations"]}
            if (left["gpu_operations"]["by_step"] != right["gpu_operations"]["by_step"]
                    or left["gpu_operations"]["outside_steps"] or right["gpu_operations"]["outside_steps"]):
                count_ok = False
            if "capture" not in left or "capture" not in right:
                continue
            for control_step, partition_step in zip(left["capture"]["steps"], right["capture"]["steps"]):
                phase = partition_step["phase"]
                overhead.setdefault(phase, []).append({"sample": sample,
                    "control_whole_step_ms": control_step["whole_step_ms"],
                    "partition_whole_step_ms": partition_step["whole_step_ms"],
                    "difference_ms": partition_step["whole_step_ms"] - control_step["whole_step_ms"]})
        checks["observed_operation_agreement"] = _check(count_ok, "control/partition GPU operation counts differ",
                                                          by_sample=collection)
        health = [row.get("capture", {}).get("profiler_collection_health") or {} for row in arms]
        checks["profiler_collection_health"] = _check(
            False, "no owned cumulative CUPTI loss witness covers Kineto reads, global and NCCL queues",
            witnesses=health)
        spans = [(row["capture"]["observer_span"]["started_unix_ns"] / 1e9,
                  row["capture"]["observer_span"]["finished_unix_ns"] / 1e9)
                 for row in arms if "capture" in row]
        host = host_exclusivity(_parse_vitals(_host_log(launch_dir, rank)), spans)
        host_path = _host_log(launch_dir, rank)
        host_ref = {"path": str(host_path) if host_path is not None else None,
                    "sha256": _sha(host_path) if host_path is not None and host_path.is_file() else None}
        worker = worker_exclusivity(arms) if all("capture" in row for row in arms) and arms else _check(False, "missing worker arm")
        checks["device_exclusivity"] = _check(host["passed"] and worker["passed"],
                                               "host or worker device census is incomplete",
                                               host=host, worker=worker)
        checks["host_log_identity"] = _host_log_identity(
            host_path, host_ip=(identities[0].get("host") or {}).get("ip") if identities else None,
            device_uuid=next(iter(device_uuids)) if len(device_uuids) == 1 else None)
        if policy is None:
            checks["observer_impact"] = _check(False, policy_error)
        else:
            maximum = policy["max_abs_relative_difference"]
            valid = bool(overhead) and all(row["control_whole_step_ms"] > 0
                       and abs(row["difference_ms"]) / row["control_whole_step_ms"] <= maximum
                       for values in overhead.values() for row in values)
            checks["observer_impact"] = _check(valid, "measured observer impact exceeds the explicit policy",
                                                 policy=policy_ref, overhead=overhead)
        terms = _rank_terms(arms, samples) if checks["raw_evidence"]["passed"] else None
        checks["sample_terms"] = _check(terms is not None, "rank samples or phase/owner order are incomplete")
        rank_records.append({"rank": rank, "device_id": identities[0].get("device_id") if identities else None,
                             "device_uuid": next(iter(device_uuids)) if len(device_uuids) == 1 else None,
                             "host": identities[0].get("host") if len(hosts) == 1 and identities else None,
                             "process_id": next(iter(process_ids)) if len(process_ids) == 1 else None,
                             "arms": [{key: row[key] for key in ("arm", "sample", "files")}
                                      | {"gpu_operations": row.get("gpu_operations"), "error": row.get("error")}
                                      for row in arms], "raw_terms": terms,
                             "qualification": checks, "observer_overhead": overhead,
                             "host_vitals": host_ref})
    distinct = {record["device_uuid"] for record in rank_records}
    qualifications["distinct_devices"] = _check(len(distinct) == 2 and None not in distinct,
                                                  "TP2 rank captures do not name distinct physical devices")
    host_ips = [(record.get("host") or {}).get("ip") for record in rank_records]
    qualifications["distinct_hosts"] = _check(len(set(host_ips)) == 2 and None not in host_ips,
                                               "TP2 rank captures do not name distinct actual host IPs")
    established = all(row["passed"] for row in qualifications.values()) and all(
        check["passed"] for record in rank_records for check in record["qualification"].values())
    terms = None
    if established:
        ranks = {str(record["rank"]): record["raw_terms"] for record in rank_records}
        medians = {str(record["rank"]): {phase: median(record["raw_terms"][phase]["fixed_gap_sum_ms"])
                 for phase in ("prefill", "decode")} for record in rank_records}
        terms = {"ranks": ranks, "rank_median_fixed_ms": medians,
                 "slowest_rank_median_fixed_ms": {phase: max(medians[str(rank)][phase] for rank in (0, 1))
                                                  for phase in ("prefill", "decode")},
                 "aggregation": "sum direct gaps within sample, median samples per rank, then maximum rank median"}
    failed = [name for name, row in qualifications.items() if not row["passed"]]
    failed += [f"rank{record['rank']}.{name}" for record in rank_records
               for name, row in record["qualification"].items() if not row["passed"]]
    return {"schema": SCHEMA, "run_identity": digest_identity, "world_size": 2,
            "plan_sha256": run["plan_sha256"], "timing_samples": len(samples),
            "fixture_provenance": run.get("fixture_provenance"),
            "raw_inputs": {"run": {"path": str(run_path), "sha256": run_sha},
                           "plan": {"path": str(plan_path), "sha256": plan_sha},
                           "observer_impact_policy": policy_ref},
            "ranks": rank_records, "qualification": qualifications,
            "partition": {"established": established,
                          "reason": None if established else "failed qualification: " + ", ".join(failed),
                          "terms": terms},
            "scope": "TP2 eager/resident/batch-one observations; no fixed timing admission from producer labels"}
