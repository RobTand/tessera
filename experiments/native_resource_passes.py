"""Persistent pass-R and pass-T runners for the transfer contract (#654).

Pass R warms up once, then brackets each rate of one persistent process with
rate:<id>:begin|end markers around load->prepare->prime->evict->settle, stops
the collector once, then derives every per-rate window from that single
continuous trace; the weights are gone before the end marker, so a retained
allocation is a window refusal rather than an invisible leak. Pass T never
constructs a collector; it re-derives operator identities and times each rate
against the pass-R record. Both runners are engine-injected: the native engine
binds prepare_native_operator/apply_complete/time_apply, and CPU fakes bind
protocol doubles. These runners make no reuse decision and assert no noise
authority: the published run report is the measurement artifact, and what a
client may conclude from it is the client's alone.

Engine protocol: family(fmt) context; warmup(rate) one load/apply/evict cycle
BEFORE the first marker so lazy one-time allocations land in the trace's
initialization baseline, never in the first rate window; load(rate) -> payload;
prepare(rate, payload) -> prepared; prime(prepared, payload) full apply
including any TP2 all-reduce; identity(prepared, payload) -> binding;
evict(prepared, payload); settle() synchronize + empty-cache inside the
bracket; process(); time(prepared, payload) -> per-phase single-apply samples.
"""
from __future__ import annotations

import re

import copy
import json
from experiments.native_resource_trace import ContinuousTrace

PASS_R_SCHEMA, PASS_T_SCHEMA = ("tessera.native_resource_pass_r.v1",
                               "tessera.native_resource_pass_t.v1")
RUN_SCHEMA = "tessera.native_persistent_run.v1"
RUNTIME_IDENTITY_SCHEMA = "tessera.native_resource_identity.v1"
def _runtime_identity(identity):
    keys = {"schema", "image_digest", "tessera_package_sha256", "vllm_package_sha256",
            "nccl_version", "family", "world_size", "native_runtime_sha256"}
    if not isinstance(identity, dict) or set(identity) != keys or identity["schema"] != RUNTIME_IDENTITY_SCHEMA:
        raise ValueError("persistent run runtime identity is incomplete")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity["image_digest"]):
        raise ValueError("persistent run image digest is not immutable")
    for key in ("tessera_package_sha256", "vllm_package_sha256", "native_runtime_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError("persistent run package identity is invalid: " + key)
    version = identity["nccl_version"]
    if (not isinstance(version, list) or len(version) != 3 or
            any(type(v) is not int or v < 0 for v in version)):
        raise ValueError("persistent run NCCL version is missing")
    if type(identity["world_size"]) is not int or identity["world_size"] not in (1, 2):
        raise ValueError("persistent run world must be TP1 or TP2")
    if not re.fullmatch(r"TESSERA_(?:BF16|E4M3|E2M1)_K[12]", identity["family"]):
        raise ValueError("persistent run family identity is invalid")


RATE_PATTERN = r"TESSERA_[A-Z0-9]+_K[1-9][0-9]*_R[0-9]+"


def _validate_rates(rates, runtime_identity):
    rates = list(rates)
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("pass rates must be a nonempty roster without duplicates")
    if any(not re.fullmatch(RATE_PATTERN, rate) for rate in rates):
        raise ValueError("pass requires exactly one valid format family")
    families = {rate.rsplit("_R", 1)[0] for rate in rates}
    if len(families) != 1 or families.pop() != runtime_identity["family"]:
        raise ValueError("pass rates must belong to the runtime identity family")
    return rates


def _records_by_rate(records, rates):
    if not isinstance(records, dict) or set(records) != set(rates):
        raise ValueError("pass records must key every roster rate exactly")
    return records


def run_pass_resource(rates, engine, collector, *, runtime_identity, trace_path,
                      device_id, context_id):
    """One persistent collector process: mark each rate, stop once, derive windows.

    The finish() call sits in a finally: the roster and runtime validation
    AND any rate that raises mid-loop all flush the one continuous trace the
    process contributed to, instead of leaving a half-written record behind.
    Windows are derived only after the whole roster survived, from that
    single trace.
    """
    markers = []
    trace = None
    try:
        _runtime_identity(runtime_identity)
        rates = _validate_rates(rates, runtime_identity)
        identity = dict(runtime_identity)
        with engine.family(identity["family"]):
            engine.warmup(rates[0])
            for rate in rates:
                begin = collector.mark(f"rate:{rate}:begin")
                payload = engine.load(rate)
                prepared = engine.prepare(rate, payload)
                engine.prime(prepared, payload)
                bound = engine.identity(prepared, payload)
                engine.evict(prepared, payload)
                del prepared, payload
                engine.settle()
                end = collector.mark(f"rate:{rate}:end")
                if end <= begin:
                    raise ValueError("rate end marker does not follow its begin")
                markers.append((rate, begin, end, bound))
    finally:
        trace = collector.finish(trace_path)
    windows = ContinuousTrace(trace, device_id=device_id, context_id=context_id)
    records = {}
    for rate, begin, end, bound in markers:
        window = windows.window(f"rate:{rate}")
        if window["begin_ns"] != begin or window["end_ns"] != end:
            raise ValueError(f"window markers for {rate} are not this process's")
        records[rate] = {"schema": PASS_R_SCHEMA, "status": "observed", "rate": rate,
                         "q256": int(rate.rsplit("_R", 1)[1]),
                         "runtime_identity": identity,
                         "collector": {"started": True,
                                       "library_sha256": collector.library_sha256},
                         "process": engine.process(), "binding": bound,
                         "device_id": device_id, "context_id": context_id,
                         "window": window}
    return records


def run_pass_timing(rates, engine, resource_records, *, runtime_identity):
    """Separate collector-free process: re-derive identities, then time each rate.

    There is no collector parameter by construction; a timing leg that started
    collection can never be produced by this runner. Every pass-R input must be
    an observed record of this runtime keyed by its own wire rate, and the
    identity digest must equal the record's binding before timing.
    """
    rates = _validate_rates(rates, runtime_identity)
    _runtime_identity(runtime_identity)
    identity = dict(runtime_identity)
    records = _records_by_rate(resource_records, rates)
    for rate, record in records.items():
        if (not isinstance(record, dict) or record.get("schema") != PASS_R_SCHEMA
                or record.get("status") != "observed" or record.get("rate") != rate
                or not isinstance(record.get("binding"), dict)
                or record.get("runtime_identity") != identity):
            raise ValueError(f"pass-T input for {rate} is not an observed "
                             f"pass-R record of this runtime")
        window = record.get("window")
        if (not isinstance(window, dict) or window.get("status") != "observed"
                or window.get("interval") != f"rate:{rate}"):
            raise ValueError(f"pass-T input for {rate} carries no observed window "
                             f"of its own rate; a record without its collector "
                             "evidence cannot license timing")
    timings = {}
    order = {}
    time_in_process = 0
    with engine.family(identity["family"]):
        for rate in rates:
            payload = engine.load(rate)
            prepared = engine.prepare(rate, payload)
            bound = engine.identity(prepared, payload)
            if bound != records[rate]["binding"]:
                raise ValueError(f"prepared identity for {rate} differs from its pass-R record")
            time_in_process += 1
            ordinal = time_in_process
            timings[rate] = engine.time(prepared, payload)
            order[rate] = ordinal
            engine.evict(prepared, payload)
            del prepared, payload
            engine.settle()
    return {rate: {"schema": PASS_T_SCHEMA, "status": "observed", "rate": rate,
                   "q256": int(rate.rsplit("_R", 1)[1]),
                   "runtime_identity": identity, "collector_started": False,
                   "time_in_process": order[rate],
                   "process": engine.process(), "binding": records[rate]["binding"],
                   "samples_ms": timings[rate]}
            for rate in rates}

def assemble_run_report(resource_records, timing_records, *, trace_path):
    """Bind both passes' records and the single continuous trace into one report.

    The report is the published measurement artifact: every pass-R record
    carries its derived window (whose ``trace_sha256`` is the canonical digest
    of the one trace), every pass-T record carries its samples, and the report
    itself binds that digest and the collector library once. A record set that
    disagrees about the trace, the collector, the device or the runtime is
    refused rather than published.
    """
    if not isinstance(resource_records, dict) or not resource_records:
        raise ValueError("run report requires pass-R records")
    if not isinstance(timing_records, dict) or not timing_records:
        raise ValueError("run report requires pass-T records")
    if set(resource_records) != set(timing_records):
        raise ValueError("pass-R and pass-T records must key the same roster")
    trace_digests = {record["window"]["trace_sha256"] for record in resource_records.values()}
    if len(trace_digests) != 1:
        raise ValueError("pass-R records derive from more than one trace")
    trace_sha256 = trace_digests.pop()
    collectors = {record["collector"]["library_sha256"] for record in resource_records.values()}
    if len(collectors) != 1:
        raise ValueError("pass-R records name more than one collector library")
    identities = {json.dumps(record["runtime_identity"], sort_keys=True)
                  for record in list(resource_records.values()) + list(timing_records.values())}
    if len(identities) != 1:
        raise ValueError("run records do not share one runtime identity")
    devices = {(record["device_id"], record["context_id"]) for record in resource_records.values()}
    if len(devices) != 1:
        raise ValueError("pass-R records do not share one device/context")
    device_id, context_id = devices.pop()
    for rate, record in timing_records.items():
        if record.get("collector_started") is not False:
            raise ValueError(f"pass-T record for {rate} claims a started collector")
    import hashlib
    try:
        with open(trace_path, "rb") as handle:
            file_sha256 = hashlib.sha256(handle.read()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read trace file {str(trace_path)!r}: {error}") from error
    return {"schema": RUN_SCHEMA, "schema_version": 1,
            "runtime_identity": copy.deepcopy(next(iter(resource_records.values()))["runtime_identity"]),
            "trace": {"file": str(trace_path), "sha256": trace_sha256,
                      "file_sha256": file_sha256,
                      "collector_library_sha256": collectors.pop()},
            "device_id": device_id, "context_id": context_id,
            "pass_r": copy.deepcopy(resource_records),
            "pass_t": copy.deepcopy(timing_records)}


def _load_entry(specifier):
    """Resolve ``module:attr`` to a factory object; import-time injection only."""
    import importlib
    module_name, separator, attribute = specifier.partition(":")
    if not separator or not attribute:
        raise ValueError(f"engine/collector specifier {specifier!r} must be module:attr")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ValueError(f"module {module_name!r} has no {attribute!r}") from error


def _read_json(path, label):
    import json
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"cannot read {label} file {str(path)!r}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} file {str(path)!r} must hold a JSON object")
    return value


def _write_json(path, value):
    import json
    import os
    try:
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"cannot write output file {str(path)!r}: {error}") from error


def _main(argv=None):
    """Producer entry point: pass-r, pass-t, assemble.

    Downstream consumers reach the runners only through this CLI (paths in,
    paths out, non-zero exit on refusal); the one-rate form of ``pass-r`` +
    ``assemble`` is the fresh single-rate report producer. Engines and
    collectors are injected by ``module:factory`` specifiers -- the native
    engine factory lives with the engine module; CPU protocol doubles live in
    the test tree.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="experiments.native_resource_passes",
        description="Produce a tessera.native_persistent_run.v1 measurement.")
    commands = parser.add_subparsers(dest="command", required=True)

    pass_r = commands.add_parser("pass-r", help="one persistent collector process")
    pass_r.add_argument("--rates", required=True,
                        help="comma-separated wire rates, one family, no repeats")
    pass_r.add_argument("--fixtures", required=True,
                        help="JSON file mapping wire rates to fixture directories "
                             "(production wiring: NativeTransferEngine)")
    pass_r.add_argument("--collector-library", required=True,
                        help="path to the CUPTI collector shared library")
    pass_r.add_argument("--engine", default=None,
                        help="TEST-ONLY override: module:factory, zero-arg")
    pass_r.add_argument("--collector", default=None,
                        help="TEST-ONLY override: module:factory, called with the engine")
    pass_r.add_argument("--runtime-identity", required=True,
                        help="JSON file with the runtime identity object")
    pass_r.add_argument("--trace", required=True, help="trace JSON output path")
    pass_r.add_argument("--out", required=True, help="pass-R records JSON output path")
    pass_r.add_argument("--device-id", type=int, required=True)
    pass_r.add_argument("--context-id", type=int, required=True)

    pass_t = commands.add_parser("pass-t", help="separate collector-free process")
    pass_t.add_argument("--rates", required=True)
    pass_t.add_argument("--fixtures", required=True,
                        help="JSON file mapping wire rates to fixture directories")
    pass_t.add_argument("--engine", default=None, help="TEST-ONLY override: module:factory")
    pass_t.add_argument("--records", required=True, help="pass-R records JSON path")
    pass_t.add_argument("--runtime-identity", required=True)
    pass_t.add_argument("--out", required=True, help="pass-T records JSON output path")

    assemble = commands.add_parser("assemble", help="bind both passes and the trace")
    assemble.add_argument("--resource", required=True, help="pass-R records JSON path")
    assemble.add_argument("--timing", required=True, help="pass-T records JSON path")
    assemble.add_argument("--trace", required=True, help="trace JSON path")
    assemble.add_argument("--out", required=True, help="run report JSON output path")

    args = parser.parse_args(argv)
    if args.command == "pass-r":
        from experiments.native_operator_resources import NativeMemoryCollector
        from experiments.native_transfer_engine import NativeTransferEngine
        rates = [rate for rate in args.rates.split(",") if rate]
        identity = _read_json(args.runtime_identity, "runtime identity")
        # The collector starts FIRST, before anything that could load Torch or
        # CUDA libraries; the engine (which loads them) comes second. The
        # test-only module:factory override skips package attestation: CPU
        # doubles run outside the runtime the identity names.
        if args.collector is not None and args.engine is not None:
            engine = _load_entry(args.engine)()
            collector = _load_entry(args.collector)(engine)
        else:
            fixtures = _read_json(args.fixtures, "fixtures")
            missing = [rate for rate in rates if rate not in fixtures]
            if missing:
                raise ValueError(f"fixtures file is missing sampled rates: {missing}")
            collector = NativeMemoryCollector(args.collector_library)
            engine = NativeTransferEngine({rate: fixtures[rate] for rate in rates})
        records = run_pass_resource(rates, engine, collector,
                                     runtime_identity=identity,
                                     trace_path=args.trace,
                                     device_id=args.device_id,
                                     context_id=args.context_id)
        _write_json(args.out, records)
    elif args.command == "pass-t":
        from experiments.native_transfer_engine import NativeTransferEngine
        rates = [rate for rate in args.rates.split(",") if rate]
        identity = _read_json(args.runtime_identity, "runtime identity")
        if args.engine is not None:
            engine = _load_entry(args.engine)()
        else:
            fixtures = _read_json(args.fixtures, "fixtures")
            missing = [rate for rate in rates if rate not in fixtures]
            if missing:
                raise ValueError(f"fixtures file is missing sampled rates: {missing}")
            engine = NativeTransferEngine({rate: fixtures[rate] for rate in rates})
        resource = _read_json(args.records, "pass-R records")
        records = run_pass_timing(rates, engine, resource,
                                  runtime_identity=identity)
        _write_json(args.out, records)
    else:
        resource = _read_json(args.resource, "pass-R records")
        timing = _read_json(args.timing, "pass-T records")
        report = assemble_run_report(resource, timing, trace_path=args.trace)
        _write_json(args.out, report)


if __name__ == "__main__":
    import sys

    try:
        _main()
    except ValueError as error:
        print(f"native-resource-passes: {error}", file=sys.stderr)
        raise SystemExit(2) from error
