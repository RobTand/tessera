"""The same-run noncandidate timing observation of one timing pass (tessera#399).

A timing pass (``capture_full_engine_resources --observation-mode timings``)
runs the canonical two-step workload -- a cold 512-token prefill and a
one-token decode -- once per sample under two arms: ``control`` (the stock
engine, profiled) and ``partition`` (the same engine with every canonical
native apply bracketed by CUDA events on the main stream). The recorder
(``full_engine_timings``) already derives, per partition-arm step, the whole
step's elapsed time between a start event and a join of the main stream's end
event with the stock output copy event, every unit's elapsed time, and the
fixed gaps between consecutive units; its profile analysis proves from the
Kineto trace that every GPU launch of the step lies in that step, on the main
or the copy stream, and that the units' launches are on the main stream.

This module reads every arm of one pass and derives the observation the
resource report carries as ``timing_captures``: the partition's terms per
phase (fixed gap sum, per-unit elapsed, whole step), sample by sample and
never aggregated, and the qualification each term rests on -- stream
coverage, identical tokens across the arms, one worker, the canonical step
shape, equal GPU-operation counts between the arms of a sample (collection
health), a device-exclusivity witness from the worker's NVML census and the
launcher's host samples, and the disclosed observer overhead (partition
minus control whole-step time, never subtracted). ``partition.established``
is true only when every check passed; a witness that could not be taken is
a failed check, not a pass.
"""
import argparse
import bisect
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OBSERVATION_SCHEMA = "tessera.full_engine_timing_observation.v1"
RUN_SCHEMA = "tessera.stock_engine_raw_timing_run.v1"
ARM_CAPTURE_SCHEMA = "tessera.full_engine_timing_capture.v1"
PARTITION_STATUS = "observed_same_run_partition"
PHASES = ("prefill", "decode")

#: The run-identity digests the resource report binds this observation to.
RUN_IDENTITY_DIGESTS = ("configuration_sha256", "model_sha256", "runtime_manifest_sha256",
                        "workload_sha256", "assignment_sha256", "canonical_units_sha256")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check(passed, **detail):
    return {"passed": passed is True, **detail}


def count_gpu_operations(profile, capture):
    """Per step, the GPU operations the trace attributes to that step's range.

    The same launch join the recorder's analysis uses, restricted to counting:
    a GPU operation is in a step when its unique runtime/driver launch lies
    inside that step's CPU range on the same thread. It is computed for both
    arms so their counts can be compared; the control arm has no unit ranges
    and no partition analysis of its own.
    """
    ranges = {name: [] for name, row in capture["ranges"].items() if row["kind"] == "step"}
    launches, gpu = {}, []
    for event in profile["traceEvents"]:
        if (event.get("ph") == "X" and event.get("cat") == "user_annotation"
                and event.get("name") in ranges):
            ranges[event["name"]].append(event)
        category = event.get("cat", "")
        if category in {"cuda_runtime", "cuda_driver"} and "correlation" in event.get("args", {}):
            launches.setdefault(event["args"]["correlation"], []).append(event)
        if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            gpu.append(event)
    counts = {capture["ranges"][name]["step_id"]: 0 for name in ranges}
    unattributed = 0
    for operation in gpu:
        paired = launches.get(operation.get("args", {}).get("correlation"), [])
        if len(paired) != 1:
            unattributed += 1
            continue
        launch = paired[0]
        matched = [capture["ranges"][name]["step_id"] for name, rows in ranges.items()
                   if len(rows) == 1 and launch.get("pid") == rows[0].get("pid")
                   and launch.get("tid") == rows[0].get("tid")
                   and rows[0]["ts"] <= launch["ts"] <= launch["ts"] + launch["dur"] <= rows[0]["ts"] + rows[0]["dur"]]
        if len(matched) == 1:
            counts[matched[0]] += 1
        else:
            unattributed += 1
    return {"by_step": counts, "outside_steps": unattributed, "total": len(gpu)}


def _parse_vitals(path):
    """``[(unix_seconds, compute_apps or None)]`` from the launcher's host-vitals log."""
    samples = []
    if path is None or not Path(path).exists():
        return samples
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        apps = None
        for item in parts[1:]:
            if item.startswith("compute_apps="):
                value = item[len("compute_apps="):]
                apps = [] if value == "" else value.split(";")
        samples.append((stamp, apps))
    return samples


def host_exclusivity(samples, spans, cadence_s=5.0):
    """The host-sampled device-exclusivity witness over the arms' spans.

    A sample is inside a span when its second-resolution stamp falls within
    the span widened by one cadence on each side, so the samples around a
    short arm are read rather than none. The witness passes only when every
    such sample lists exactly one compute process and it is the same process
    throughout. A log without the ``compute_apps`` field cannot witness.
    """
    stamps = [stamp for stamp, _apps in samples]
    inside, pids = [], set()
    for begin, end in spans:
        lo = bisect.bisect_left(stamps, begin - cadence_s)
        hi = bisect.bisect_right(stamps, end + cadence_s)
        inside.extend(samples[lo:hi])
    if not samples or any(apps is None for _stamp, apps in samples):
        return _check(False, reason="host-vitals log carries no compute_apps field", samples=len(inside))
    if not inside:
        return _check(False, reason="no host sample falls within one cadence of any arm span", samples=0)
    counts = sorted({len(apps) for _stamp, apps in inside})
    for _stamp, apps in inside:
        pids.update(apps)
    return _check(counts == [1] and len(pids) == 1, samples=len(inside), compute_app_counts=counts,
                  compute_apps=sorted(pids), cadence_s=cadence_s,
                  scope=("host nvidia-smi compute-process census at the launcher's cadence over the arms' "
                         "spans widened by one cadence; a foreign process alive only between samples "
                         "is not seen"))


def worker_exclusivity(arms):
    """The worker's NVML census at every arm and finish: exactly one process, constant."""
    counts, pids, missing = set(), set(), []
    for arm in arms:
        exclusivity = arm["capture"].get("exclusivity")
        for moment in ("at_arm", "at_finish"):
            census = (exclusivity or {}).get(moment)
            if not isinstance(census, dict) or census.get("available") is not True:
                missing.append(f"{arm['arm']}/{arm['sample']}/{moment}")
                continue
            counts.add(census["count"])
            pids.update(row["pid"] for row in census["processes"])
    if missing:
        return _check(False, reason="NVML census unavailable at: " + ", ".join(missing))
    return _check(counts == {1} and len(pids) == 1, process_counts=sorted(counts), pids=sorted(pids),
                  scope=("NVML compute-process census in the worker at arm and at finish of every arm; "
                         "the pid namespace is the driver's, so count and constancy are the witness"))


def build_timing_observation(capture_dir, *, launch_dir=None):
    """Derive the observation from one timing pass's capture directory."""
    capture_dir = Path(capture_dir)
    plan = json.loads((capture_dir / "observer-plan.json").read_text())
    run = json.loads((capture_dir / "run.json").read_text())
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError("unsupported timing run schema: " + str(run.get("schema")))
    arms = []
    for entry in run["arms"]:
        # ``armed`` and ``workers`` are collective_rpc results: one entry per
        # worker, and this observation covers exactly one worker.
        if len(entry["workers"]) != 1 or len(entry["armed"]) != 1:
            raise ValueError("a timing arm reports exactly one worker")
        entry = dict(entry, armed=entry["armed"][0])
        directory = Path(entry["workers"][0]["directory"])
        if not directory.is_absolute() or not directory.exists():
            directory = capture_dir / directory.relative_to(directory.parents[1]) if len(directory.parents) > 1 else capture_dir / directory.name
        capture = json.loads((directory / "capture.json").read_text())
        if capture.get("schema") != ARM_CAPTURE_SCHEMA or capture["arm"] != entry["arm"]:
            raise ValueError(f"arm capture at {directory} is not the {entry['arm']} arm it is listed as")
        partition = json.loads((directory / "partition.json").read_text())
        profile = json.loads((directory / "profile.json").read_text())
        arms.append({"arm": entry["arm"], "sample": entry["sample"], "directory": str(directory),
                     "tokens": entry["tokens"], "armed": entry["armed"],
                     "capture": capture, "partition": partition,
                     "capture_sha256": _digest(directory / "capture.json"),
                     "gpu_operations": count_gpu_operations(profile, capture)})
    samples = sorted({arm["sample"] for arm in arms})
    by_key = {(arm["arm"], arm["sample"]): arm for arm in arms}
    checks = {}

    # One worker process across every arm.
    pids = {arm["armed"]["pid"] for arm in arms}
    checks["single_worker"] = _check(len(pids) == 1, pids=sorted(pids))

    # Identical generated tokens: warmup and every arm.
    tokens = {tuple(arm["tokens"]) for arm in arms} | {tuple(run["warmup_tokens"])}
    checks["identical_tokens"] = _check(len(tokens) == 1, tokens=run["warmup_tokens"])

    # Both arms present for every sample, in the planned count.
    complete = all((name, sample) in by_key for sample in samples for name in ("control", "partition"))
    checks["arms_complete"] = _check(complete and len(samples) == plan["timing_samples"],
                                     samples=samples, planned=plan["timing_samples"])

    # The canonical step shape on every arm: prefill(512) then decode(1).
    shapes_ok = all([(step["phase"], step["scheduled_tokens"]) for step in arm["capture"]["steps"]]
                    == [("prefill", 512), ("decode", 1)] for arm in arms)
    checks["step_shape"] = _check(shapes_ok)

    # Stream coverage and recomposition: the recorder's own analysis of every
    # partition arm, and every GPU operation of every step on a joined stream.
    partition_arms = [arm for arm in arms if arm["arm"] == "partition"]
    statuses = {arm["partition"]["status"] for arm in partition_arms}
    issues = [issue for arm in partition_arms for issue in arm["partition"]["issues"]]
    checks["stream_coverage"] = _check(bool(partition_arms) and statuses == {PARTITION_STATUS} and not issues,
                                       statuses=sorted(statuses), issues=issues)

    # Collection health: the two arms of a sample saw the same number of GPU
    # operations in each step, and none outside the steps.
    counts_ok, count_detail = True, {}
    for sample in samples:
        control, partition = by_key.get(("control", sample)), by_key.get(("partition", sample))
        if control is None or partition is None:
            counts_ok = False
            continue
        row = {"control": control["gpu_operations"], "partition": partition["gpu_operations"]}
        count_detail[str(sample)] = row
        if (row["control"]["by_step"] != row["partition"]["by_step"]
                or row["control"]["outside_steps"] or row["partition"]["outside_steps"]):
            counts_ok = False
    checks["gpu_operation_counts_agree"] = _check(counts_ok, by_sample=count_detail)

    # Device exclusivity: the worker's NVML census and the host's samples.
    spans = [(arm["capture"]["observer_span"]["started_unix_ns"] / 1e9,
              arm["capture"]["observer_span"]["finished_unix_ns"] / 1e9) for arm in arms]
    host = host_exclusivity(_parse_vitals(None if launch_dir is None else Path(launch_dir) / "host-vitals.log"), spans)
    worker = worker_exclusivity(arms)
    checks["device_exclusivity"] = _check(host["passed"] and worker["passed"], host=host, worker=worker)

    # Observer overhead, disclosed per sample and phase, never subtracted.
    overhead = {}
    for sample in samples:
        control, partition = by_key.get(("control", sample)), by_key.get(("partition", sample))
        if control is None or partition is None:
            continue
        for c_step, p_step in zip(control["capture"]["steps"], partition["capture"]["steps"]):
            overhead.setdefault(p_step["phase"], []).append(
                {"sample": sample, "control_whole_step_ms": c_step["whole_step_ms"],
                 "partition_whole_step_ms": p_step["whole_step_ms"],
                 "difference_ms": p_step["whole_step_ms"] - c_step["whole_step_ms"]})

    established = all(check["passed"] for check in checks.values())
    terms = None
    if established:
        terms = {}
        for phase in PHASES:
            rows = [(arm["sample"], step) for arm in partition_arms for step in arm["capture"]["steps"]
                    if step["phase"] == phase]
            rows.sort()
            units = {}
            for _sample, step in rows:
                for unit in step["units"]:
                    units.setdefault(unit["unit_id"], []).append(unit["elapsed_ms"])
            terms[phase] = {
                "samples": [sample for sample, _step in rows],
                "whole_step_ms": [step["whole_step_ms"] for _s, step in rows],
                "fixed_gap_sum_ms": [sum(step["fixed_gaps_ms"]) for _s, step in rows],
                "fixed_gaps_ms": [step["fixed_gaps_ms"] for _s, step in rows],
                "candidate_ms": units,
                "control_whole_step_ms": [row["control_whole_step_ms"] for row in overhead.get(phase, [])],
                "unit_order": [unit["unit_id"] for unit in rows[0][1]["units"]] if rows else [],
            }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {
        "schema": OBSERVATION_SCHEMA,
        "run_identity": {name: plan["identity"].get(name) for name in RUN_IDENTITY_DIGESTS},
        "rank": None, "world_size": None,
        "plan_sha256": run["plan_sha256"],
        "process_id": next(iter(pids)) if len(pids) == 1 else None,
        "timing_samples": plan["timing_samples"],
        "canonical_unit_ids": partition_arms[0]["capture"]["canonical_unit_ids"] if partition_arms else None,
        "arms": [{key: arm[key] for key in ("arm", "sample", "directory", "capture_sha256", "gpu_operations")}
                 | {"partition_status": arm["partition"]["status"],
                    "steps": [{key: step[key] for key in ("step_id", "phase", "scheduled_tokens", "whole_step_ms",
                                                          "main_stream_id", "copy_stream_id")}
                              | {"fixed_gap_sum_ms": sum(step["fixed_gaps_ms"]),
                                 "units": {unit["unit_id"]: unit["elapsed_ms"] for unit in step["units"]}}
                              for step in arm["capture"]["steps"]]}
                 for arm in arms],
        "qualification": checks,
        "observer_overhead": overhead,
        "partition": {
            "established": established,
            "reason": None if established else "failed qualification: " + ", ".join(failed),
            "terms": terms,
            "scope": ("same-run noncandidate partition of the canonical two-step workload: per phase, "
                      "fixed_gap_sum_ms is the CUDA-event time on the main stream outside every native "
                      "apply (before the first, between consecutive ones, after the last up to the join "
                      "of the main end event with the stock output copy event) and candidate_ms the "
                      "per-unit apply elapsed; whole_step_ms recomposes as their sum within binary32 "
                      "rounding; samples are listed, never aggregated; observer overhead is disclosed, "
                      "not subtracted"),
        },
        "scope": ("derived from every arm of one timing pass under the qualification above; "
                  "established only when every check passed, and a witness that could not be taken "
                  "is a failed check"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="the timing pass's capture directory")
    parser.add_argument("--launch-dir", type=Path, help="the launcher output directory holding host-vitals.log")
    parser.add_argument("--output", type=Path, required=True, help="where to write timing-observation.json")
    args = parser.parse_args()
    observation = build_timing_observation(args.capture_dir, launch_dir=args.launch_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observation, sort_keys=True, indent=1) + "\n")
    print(json.dumps({"artifact": str(args.output), "sha256": _digest(args.output),
                      "established": observation["partition"]["established"],
                      "reason": observation["partition"]["reason"],
                      "qualification": {name: check["passed"] for name, check in observation["qualification"].items()}},
                     indent=1), flush=True)
    return 0 if observation["partition"]["established"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
