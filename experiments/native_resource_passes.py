"""Persistent pass-R and pass-T runners for the transfer contract (#654).

Pass R warms up once, then brackets each rate of one persistent process with
rate:<id>:begin|end markers around load->prepare->prime->evict->settle, stops
the collector once, then derives every per-rate window from that single
continuous trace; the weights are gone before the end marker, so a retained
allocation is a window refusal rather than an invisible leak. Pass T never
constructs a collector; it re-derives operator identities and times each rate
against the pass-R record. Both runners are engine-injected: the native engine
binds prepare_native_operator/apply_complete/time_apply, and CPU fakes bind
protocol doubles. These runners add no noise-band authority and perform no
qualification; qualify_transfer remains the only transfer authority, and a
pass-R record alone never licenses reuse.

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

from experiments.native_resource_transfer import ContinuousTrace, _runtime

PASS_R_SCHEMA, PASS_T_SCHEMA = ("tessera.native_resource_pass_r.v1",
                               "tessera.native_resource_pass_t.v1")
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
        _runtime(runtime_identity)
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
    _runtime(runtime_identity)
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
