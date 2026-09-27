"""Closed-world evidence for qualified native resource transfer (#654).

This module does not authorize a noise band or turn a timing screen into a price.
A continuous, terminal CUPTI trace is the resource authority; an explicit passing
three-way qualification is the transfer authority. Unsupported pools remain a
refusal, not an empty live set. CPU fixture qualifications are not GPU evidence.
"""
from __future__ import annotations

import copy
from typing import Any, cast
import hashlib
import json
import math
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter

from experiments.native_operator_resources import (
    MEMORY_API_OPERATIONS,
    _integer,
    api_base_name,
    memory_row_domain,
    ownership_operation,
    validate_cupti_capture,
)

PHASES = ("prefill", "decode")
RUNTIME_SCHEMA = "tessera.native_resource_transfer_runtime.v1"
QUALIFICATION_SCHEMA = "tessera.native_resource_transfer_qualification.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _requests(rows):
    counts = Counter((r["bytes"], r["source"], r["memory_kind"]) for r in rows
                     if r["operation"] == "allocate")
    return [{"bytes": size, "source": source, "memory_kind": kind, "count": count}
            for (size, source, kind), count in sorted(counts.items())]


class ContinuousTrace:
    """Index a finished trace once; retain original rows, never invented events.

    All memory history is validated through stop, including events outside a
    requested window. Starts snapshot the exact live allocation records, and
    window requests retain their sources. ``view`` only compacts fully matched
    historical lifetimes; it is for the unchanged warmed-apply scalar analyzer,
    not an alternate completeness rule. Original trace bytes remain the receipt
    identity. Build/use this index on a terminal immutable trace.
    """

    def __init__(self, trace, *, device_id, context_id):
        self.pid = validate_cupti_capture(trace)
        self.device = _integer(device_id)
        self.context = _integer(context_id)
        self.trace = trace
        self.trace_sha256 = digest(trace)
        start, stop = _integer(trace["start_ns"], 1), _integer(trace["end_ns"], 1)
        if stop <= start:
            raise ValueError("continuous trace stop precedes start")
        paired = {}
        for marker in trace["markers"]:
            name, separator, side = marker["name"].rpartition(":")
            if not separator or side not in ("begin", "end"):
                raise ValueError("continuous trace marker must name begin or end")
            point = _integer(marker["timestamp_ns"], 1)
            if not start < point < stop or side in paired.setdefault(name, {}):
                raise ValueError("duplicate/out-of-range continuous trace marker")
            paired[name][side] = point
        if not paired or any(set(p) != {"begin", "end"} or p["begin"] >= p["end"]
                             for p in paired.values()):
            raise ValueError("exactly one paired interval required per marker")
        self.intervals = {name: (p["begin"], p["end"]) for name, p in paired.items()}
        ordered = sorted(self.intervals.values())
        for index in range(len(ordered) - 1):
            if ordered[index][1] >= ordered[index + 1][0]:
                raise ValueError("rate windows overlap in the continuous trace")
        self.first_begin = min(begin for begin, _ in self.intervals.values())
        self.apis = trace["api_events"]
        if not self.apis:
            raise ValueError("no observed CUDA API records")
        self.correlations = {}
        api_order = []
        for i, api in enumerate(self.apis):
            a, b = _integer(api["start_ns"], 1), _integer(api["end_ns"], 1)
            if b < a or not isinstance(api["name"], str):
                raise ValueError("CUDA API timestamps are reversed or name is invalid")
            key = (_integer(api["process_id"], 1), _integer(api["correlation_id"]))
            self.correlations.setdefault(key, []).append(i)
            api_order.append((a, b, i))
        # Sweep API intervals against all marker points once. Nested outer-rate
        # and inner-apply windows are supported without rescanning the prefix.
        api_order.sort()
        self.window_apis = {name: set() for name in self.intervals}
        active, opened, cursor = {}, set(), 0
        points = sorted((t, side, name) for name, (begin, end) in self.intervals.items()
                        for t, side in ((begin, 0), (end, 1)))
        for point, side, name in points:
            while cursor < len(api_order) and api_order[cursor][0] <= point:
                a, b, i = api_order[cursor]
                for interval in opened:
                    self.window_apis[interval].add(i)
                if b >= point:
                    active[i] = b
                cursor += 1
            active = {i: b for i, b in active.items() if b >= point}
            if side == 0:
                self.window_apis[name].update(active)
                opened.add(name)
            else:
                opened.remove(name)
        self.events = sorted(trace["memory_events"], key=lambda r: r["timestamp_ns"])
        if not self.events:
            raise ValueError("no observed allocation records; collection not demonstrated")
        self.times = []
        for row in self.events:
            t = _integer(row["timestamp_ns"], 1)
            if not start <= t <= stop:
                raise ValueError("allocation lies outside continuous collection")
            self.times.append(t)
        self.starts, self.ends, self.peaks = {}, {}, {}
        # No memory operation may hide between rate windows: the pricing prefix
        # is the whole persistent process, not only the windows a rate asked for.
        begins = [begin for begin, _ in ordered]
        last_end = ordered[-1][1]
        for point in self.times:
            if self.first_begin <= point <= last_end:
                index = bisect_right(begins, point) - 1
                if index < 0 or point > ordered[index][1]:
                    raise ValueError(
                        f"memory operation at {point} lies outside rate windows (unpriced gap)")
        live, opened, cursor = {}, set(), 0
        for point, side, name in points:
            # A begin snapshot precedes records exactly on its boundary; an end
            # snapshot follows them, matching analyze_trace's inclusive window.
            while cursor < len(self.events) and (self.times[cursor] < point or
                                                (side == 1 and self.times[cursor] == point)):
                row = self.events[cursor]
                self._apply(live, row)
                for interval in opened:
                    self.peaks[interval] = max(self.peaks[interval], sum(r["bytes"] for r in live.values()))
                cursor += 1
            if side == 0:
                self.starts[name] = tuple(live.values())
                self.peaks[name] = sum(r["bytes"] for r in live.values())
                opened.add(name)
            else:
                self.ends[name] = tuple(live.values())
                opened.remove(name)
        # A malformed tail must not disappear because no rate asked for it.
        for row in self.events[cursor:]:
            self._apply(live, row)
        first = min(self.intervals, key=lambda name: self.intervals[name][0])
        self.initial_live = self.starts[first]

    def _apply(self, live, row):
        if row["process_id"] != self.pid:
            raise ValueError("allocation belongs to another process")
        # One home classifies the row's domain: host rows are outside the
        # device-byte scope, and anything unsupported is refused right here.
        kind = memory_row_domain(row)
        if kind == "host":
            return  # host allocations are outside the established device-byte scope
        address, size = _integer(row["address"], 1), _integer(row["bytes"], 1)
        context = _integer(row["context_id"])
        static_zero = (kind == "static" and self.trace["cupti_version"] == 130001 and
                       context == row["correlation_id"] == row["stream_id"] == 0)
        if row["device_id"] != self.device or not (context == self.context or static_zero):
            raise ValueError("allocation device/context differs from operator")
        key = (kind, context, address)
        if row["operation"] == "allocate":
            if not isinstance(row["source"], str) or not row["source"]:
                raise ValueError("allocation request source is unknown")
            if key in live:
                raise ValueError("duplicate live allocation address")
            live[key] = row
        elif row["operation"] == "free":
            previous = live.pop(key, None)
            if previous is None or previous["bytes"] != size:
                raise ValueError("free lacks matching allocation bytes/context")
        else:
            raise ValueError("unknown memory operation")

    def _rows(self, interval):
        begin, end = self.intervals[interval]
        return self.events[bisect_left(self.times, begin):bisect_right(self.times, end)]

    def _apis(self, interval):
        indices = set(self.window_apis[interval])
        for row in self._rows(interval):
            indices.update(self.correlations.get((row["process_id"], row["correlation_id"]), ()))
        return [self.apis[i] for i in sorted(indices)]

    def _coverage(self, interval):
        begin, end = self.intervals[interval]
        required, observed = {}, set()
        for api in self._apis(interval):
            name = api_base_name(api["name"])
            a, b = api["start_ns"], api["end_ns"]
            if a > end or b < begin:
                continue
            if name.startswith(("cudaGraph", "cuGraph")):
                raise ValueError("CUDA graph execution is outside eager resource scope")
            operation = ownership_operation(name, api["return_value"])
            if operation == "unsupported":
                raise ValueError("unsupported or failed allocation API: " + name)
            if operation is not None:
                if api["process_id"] != self.pid or not begin <= a <= b <= end:
                    raise ValueError("allocation API crosses the rate process/interval")
                key = (self.pid, api["correlation_id"])
                if len(self.correlations[key]) != 1 or key in required:
                    raise ValueError("ambiguous allocation API correlation")
                required[key] = api
        for row in self._rows(interval):
            if memory_row_domain(row) == "host":
                continue
            if row["memory_kind"] == 6:
                # analyze_trace refuses module/static activity inside apply;
                # the persistent window must refuse it with the same words.
                raise ValueError("module/static allocation or free occurred during apply")
            key = (self.pid, row["correlation_id"])
            api = required.get(key)
            if api is None or key in observed:
                raise ValueError("memory operation has no unambiguous in-window API correlation")
            name = api_base_name(api["name"])
            if not api["start_ns"] <= row["timestamp_ns"] <= api["end_ns"]:
                raise ValueError("memory operation timestamp is outside its API")
            if row["operation"] != MEMORY_API_OPERATIONS[name]:
                raise ValueError("allocation operation disagrees with its API")
            observed.add(key)
        if observed != set(required):
            raise ValueError("allocation API lacks its matching memory operation")

    def window(self, interval):
        self._coverage(interval)
        baseline, final = self.starts[interval], self.ends[interval]
        if baseline != self.initial_live or final != baseline:
            raise ValueError("rate changed the one-time baseline or retained allocations after eviction")
        begin, end = self.intervals[interval]
        base = sum(row["bytes"] for row in baseline)
        rows = [row for row in self._rows(interval) if row["memory_kind"] in (3, 6)]
        return {"schema": "tessera.native_rate_resource_window.v1", "status": "observed",
                "trace_sha256": self.trace_sha256, "process_id": self.pid,
                "collection_start_ns": self.trace["start_ns"], "interval": interval,
                "begin_ns": begin, "end_ns": end, "baseline_live": list(baseline),
                "end_live": list(final), "baseline_bytes": base,
                "window_peak_bytes": self.peaks[interval],
                "transient_peak_bytes": self.peaks[interval] - base,
                "allocation_requests": _requests(rows),
                "initialization_requests": _requests(self.initial_live)}


def stratified_rates(rates, count):
    """Stratified integer rates: first, true median and last always present.

    The interior points are the evenly spaced grid positions resolved onto
    unused indices, so the last rate can never be crowded out by the median.
    """
    if (not isinstance(rates, list) or len(rates) < 5 or any(type(r) is not int for r in rates)
            or rates != list(range(rates[0], rates[-1] + 1))):
        raise ValueError("qualification requires the full consecutive integer rate domain")
    if type(count) is not int or not 5 <= count <= len(rates):
        raise ValueError("stratified sample needs between five and every rate")
    last = len(rates) - 1
    middle = last // 2
    picks = {0, last, middle}
    for anchor in (round(i * last / (count - 1)) for i in range(1, count - 1)):
        if len(picks) >= count:
            break
        picks.add(anchor)
    index = 1
    while len(picks) < count:
        picks.add(index)
        index += 1
    if not {0, middle, last} <= picks or len(picks) != count:
        raise ValueError("stratified sample must keep the first, median and last rate")
    return [rates[i] for i in sorted(picks)]


def _runtime(identity):
    keys = {"schema", "image_digest", "tessera_package_sha256", "vllm_package_sha256",
            "nccl_version", "family", "world_size", "native_runtime_sha256"}
    if not isinstance(identity, dict) or set(identity) != keys or identity["schema"] != RUNTIME_SCHEMA:
        raise ValueError("resource transfer runtime identity is incomplete")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity["image_digest"]):
        raise ValueError("resource transfer image digest is not immutable")
    for key in ("tessera_package_sha256", "vllm_package_sha256", "native_runtime_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", identity[key]):
            raise ValueError("resource transfer runtime package identity is invalid: " + key)
    version = identity["nccl_version"]
    if (not isinstance(version, list) or len(version) != 3 or
            any(type(v) is not int or v < 0 for v in version)):
        raise ValueError("resource transfer NCCL version is missing")
    if type(identity["world_size"]) is not int or identity["world_size"] not in (1, 2):
        raise ValueError("resource transfer world must be TP1 or TP2")
    if not re.fullmatch(r"TESSERA_(?:BF16|E4M3|E2M1)_K[12]", identity["family"]):
        raise ValueError("resource transfer family identity is invalid")


def _samples(values):
    """Median of single-apply samples; three is time_apply's own iterations
    floor (bench_native_operator refuses iterations < 3), not a statistic
    chosen here — the timing authority is the band, never this median."""
    if (not isinstance(values, list) or len(values) < 3 or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError("qualification timing requires positive finite single-apply samples")
    return statistics.median(values)


BAND_SOURCE = "fresh_process_repeat_r5_pooled_log"
GATE_KIND = "not_detected"
FRESH_REPEATS = 5
ALPHA = 0.01


def _band_raw(noise_band):
    """Validate the raw band shape; no caller-asserted number survives this."""
    if (not isinstance(noise_band, dict)
            or set(noise_band) != {"source", "gate_kind", "raw"}
            or noise_band["source"] != BAND_SOURCE
            or noise_band["gate_kind"] != GATE_KIND):
        raise ValueError("timing evidence must be raw fresh-process repeats under a "
                         "not-detected gate; caller-asserted band numbers are refused")
    raw = noise_band["raw"]
    if (not isinstance(raw, dict) or set(raw) != {"eps", "phases"}
            or not isinstance(raw["eps"], dict)
            or set(raw["eps"]) != {"samples_ms", "eps_source"}
            or not isinstance(raw["eps"]["samples_ms"], list) or not raw["eps"]["samples_ms"]
            or not isinstance(raw["eps"]["eps_source"], str) or not raw["eps"]["eps_source"]
            or not isinstance(raw["phases"], dict) or set(raw["phases"]) != set(PHASES)):
        raise ValueError("timing raw evidence needs both phases and the timer's own samples")
    return raw


def _phase_gate(phase, raw_phase, selected, eps_ms, K):
    """FOLLOWUP-7 per-phase gate over raw evidence.

    Everything is re-derived here from raw samples: the r fresh-process
    medians per rate, the pooled log-scale s, the timer floor, the per-rate
    variance screen, the residuals against the common factor and the drift
    test. The gate is a not-detected gate; its MDE is stamped, never hidden.
    """
    from scipy import stats as _st

    if (not isinstance(raw_phase, dict)
            or set(raw_phase) != {"fresh", "persistent"}):
        raise ValueError(f"{phase} raw timing evidence needs fresh repeats and "
                         "the persistent pass")
    fresh, persistent = raw_phase["fresh"], raw_phase["persistent"]
    if not isinstance(fresh, dict) or not fresh:
        return None, [f"{phase}: phase has no raw fresh timing evidence"]
    if len(fresh) < 2:
        return None, [f"{phase}: timing gate requires at least two sampled rates"]
    if set(fresh) != {str(rate) for rate in selected}:
        raise ValueError(f"{phase} raw fresh rates must key the sampled roster exactly")
    r, k = FRESH_REPEATS, len(selected)
    x, medians = {}, {}
    phase_fresh_ids = []
    for key, legs in fresh.items():
        if not isinstance(legs, list) or len(legs) != r:
            raise ValueError(f"{phase} rate {key} needs exactly {r} fresh processes")
        values, procs = [], []
        for leg in legs:
            if (not isinstance(leg, dict) or set(leg) != {"process", "samples_ms"}
                    or not isinstance(leg["process"], dict)):
                raise ValueError(f"{phase} rate {key} fresh legs carry process and samples only")
            values.append(_samples(leg["samples_ms"]))
            procs.append(digest(leg["process"]))
        if len(set(procs)) != r:
            return None, [f"{phase} rate {key}: fresh repeats must be separate processes"]
        # Cross-rate distinctness: the same r processes must not time every
        # rate. All k*r identities in a phase are pairwise distinct.
        phase_fresh_ids.extend(procs)
        medians[key] = values
        x[key] = [math.log(v) for v in values]
    if len(set(phase_fresh_ids)) != k * r:
        return None, [f"{phase}: fresh repeats must be separate processes "
                      "across rates"]
    if (not isinstance(persistent, dict)
            or set(persistent) != {"process", "rates"}
            or not isinstance(persistent["process"], dict)
            or not isinstance(persistent["rates"], list) or not persistent["rates"]):
        raise ValueError(f"{phase} persistent timing evidence names one process and its rates")
    entries = {}
    for entry in persistent["rates"]:
        if (not isinstance(entry, dict)
                or set(entry) != {"q256", "samples_ms", "time_in_process"}
                or entry["q256"] not in selected):
            raise ValueError(f"{phase} persistent rates must carry q256, samples and time in process")
        entries[entry["q256"]] = entry
    if len(entries) != len(persistent["rates"]) or set(entries) != set(selected):
        raise ValueError(f"{phase} persistent rates must key the sampled roster exactly")
    ordered = sorted(persistent["rates"], key=lambda e: e["time_in_process"])
    times = [e["time_in_process"] for e in ordered]
    if (any(type(t) is not int for t in times)
            or any(times[i + 1] <= times[i] for i in range(len(times) - 1))):
        return None, [f"{phase}: persistent rates are not ordered in time"]
    persistent_id = digest(persistent["process"])
    # Mixed-source table: every persistent row must come from the one pass-T
    # process, and no fresh-process row may sit inside the persistent table.
    if persistent_id in set(phase_fresh_ids):
        return None, [f"{phase}: persistent table mixes fresh-process rows "
                      "into the pass-T evidence"]
    y = {str(entry["q256"]): math.log(_samples(entry["samples_ms"])) for entry in ordered}

    reasons = []
    if any(len(set(vals)) < 2 for vals in x.values()):
        reasons.append("timer_cannot_resolve_process_noise: a rate has fewer than "
                       "two distinct fresh medians")
        return None, reasons
    m = {key: statistics.fmean(vals) for key, vals in x.items()}
    s_i = {key: statistics.stdev(vals) for key, vals in x.items()}
    df = k * (r - 1)
    s = math.sqrt(sum((value - m[key]) ** 2 for key, vals in x.items()
                      for value in vals) / df)
    eps_log = {key: math.log(1.0 + eps_ms / math.exp(m[key])) for key in m}
    if s <= 0.0 or max(eps_log.values()) >= s:
        reasons.append("timer_cannot_resolve_process_noise: the timer cannot "
                       "resolve process noise")
        return None, reasons

    factor = math.sqrt((1.0 + 1.0 / r) * (1.0 - 1.0 / k))
    t_crit = float(_st.t.ppf(1.0 - ALPHA / (2.0 * K), df))
    fallback, ratios = [], {}
    for key in sorted(x, key=int):
        others = math.sqrt(sum((value - m[other]) ** 2
                               for other in x if other != key
                               for value in x[other]) / ((k - 1) * (r - 1)))
        ratio = float("inf") if others == 0.0 else (s_i[key] ** 2) / (others ** 2)
        p_screen = float(_st.f.sf(ratio, r - 1, (k - 1) * (r - 1)))
        ratios[key] = {"variance_ratio": ratio if math.isfinite(ratio) else None,
                       "p": p_screen}
        if p_screen < ALPHA / k:
            fallback.append(key)

    d = {key: y[key] - m[key] for key in x}
    d_bar = statistics.fmean(d.values())
    residuals = {}
    for key in sorted(x, key=int):
        if key in fallback:
            v = (1.0 + 1.0 / r) * ((1.0 - 1.0 / k) ** 2 * s_i[key] ** 2
                                   + sum(s_i[other] ** 2 for other in s_i
                                         if other != key) / k ** 2)
            c_ii = (1.0 + 1.0 / r) * (1.0 - 1.0 / k) ** 2
            c_ij = (1.0 + 1.0 / r) / k ** 2
            den = ((c_ii * s_i[key] ** 2) ** 2
                   + sum((c_ij * s_i[other] ** 2) ** 2
                         for other in s_i if other != key))
            df_i = (r - 1) * v ** 2 / den
            threshold = float(_st.t.ppf(1.0 - ALPHA / (2.0 * K), df_i)) * math.sqrt(v)
            threshold += eps_log[key]
            variance_log = v
        else:
            threshold = t_crit * factor * s + eps_log[key]
            variance_log = (s * factor) ** 2
        delta = d[key] - d_bar
        residuals[key] = {"d_minus_d_bar": delta, "threshold_log": threshold,
                          "variance_log": variance_log, "fallback": key in fallback,
                          "passed": abs(delta) <= threshold}
        if not residuals[key]["passed"]:
            reasons.append(f"{phase} rate {key}: residual exceeds the not-detected gate")

    if len(set(d.values())) > 1:
        spearman = cast("tuple[Any, Any]", _st.spearmanr([d[str(e["q256"])] for e in ordered],
                                                         times))
        drift = {"rho": float(spearman[0]), "p": float(spearman[1])}
    else:
        drift = {"rho": None, "p": None}
    if drift["p"] is not None and drift["p"] < ALPHA / K:
        reasons.append(f"persistent_mode_drifts: {phase} (Spearman p={drift['p']:.4g})")

    f_ratio = (sum((value - d_bar) ** 2 for value in d.values()) / (k - 1)
               / (s ** 2 * (1.0 + 1.0 / r)))
    f_p = float(_st.f.sf(f_ratio, k - 1, df))
    failing = [key for key, row in residuals.items() if not row["passed"]]
    if failing and f_p < ALPHA and not (set(failing) & set(fallback)):
        reasons.append(f"persistent_noise_exceeds_fresh: {phase}")
    try:
        wilcoxon_p = float(cast(Any, _st.wilcoxon(list(d.values()))).pvalue)
    except ValueError:
        wilcoxon_p = None
    spread = statistics.stdev(list(d.values())) / math.sqrt(k)
    hw = float(_st.t.ppf(0.995, k - 1)) * spread
    interval = [d_bar - hw, d_bar + hw]
    mde80 = (t_crit + float(_st.norm.ppf(0.8))) * factor * s
    report = {"s_log": s, "df": df, "mde80_log": mde80,
              "fallback_rates": fallback,
              "fallback_note": ("df is per-rate and near r-1; a fallback rate "
                                "certifies almost nothing") if fallback else None,
              "variance_screen": ratios, "residuals": residuals,
              "d_log": d, "d_bar_log": d_bar, "d_bar_interval_log": interval,
              "drift_spearman": drift, "wilcoxon_p": wilcoxon_p,
              "f_diagnostic": {"ratio": f_ratio, "p": f_p},
              "fresh_medians_ms": {key: medians[key] for key in medians},
              "persistent_medians_ms": {key: math.exp(y[key]) for key in y},
              "persistent_process": persistent["process"],
              "persistent_order": [entry["q256"] for entry in ordered],
              "persistent_ordinals": {str(entry["q256"]): entry["time_in_process"]
                                      for entry in ordered},
              "fresh_processes": phase_fresh_ids}
    return report, reasons


def _timing_gate(noise_band, selected):
    """Both phases plus the shared eps; one persistent process across phases."""
    raw = _band_raw(noise_band)
    eps_samples = [value for value in raw["eps"]["samples_ms"]
                   if type(value) in (int, float) and math.isfinite(value) and value > 0]
    if not eps_samples:
        return ({"gate_kind": GATE_KIND, "alpha": ALPHA, "r": FRESH_REPEATS,
                 "phases": {}},
                ["timer_cannot_resolve_process_noise: the timer produced no positive sample"])
    k = len(selected)
    K = 2 * k + 2
    eps_ms = min(eps_samples)
    reasons, phases, processes = [], {}, []
    fresh_ids = set()
    for phase in PHASES:
        report, phase_reasons = _phase_gate(phase, raw["phases"][phase],
                                            selected, eps_ms, K)
        reasons.extend(phase_reasons)
        if report is not None:
            phases[phase] = report
            processes.append(digest(report["persistent_process"]))
            fresh_ids.update(report["fresh_processes"])
    if len(processes) == len(PHASES) and len(set(processes)) != 1:
        reasons.append("persistent samples must come from one process")
    return ({"gate_kind": GATE_KIND, "alpha": ALPHA, "r": FRESH_REPEATS, "K": K,
             "eps_ms": eps_ms, "eps_source": raw["eps"]["eps_source"],
             "fresh_processes": sorted(fresh_ids), "phases": phases}, reasons)


def _continuous(trace, device_id, context_id):
    if (not isinstance(trace, dict) or type(device_id) is not int
            or type(context_id) is not int):
        raise ValueError("a continuous trace leg must carry the trace and "
                         "integer device and context ids")
    return ContinuousTrace(trace, device_id=device_id, context_id=context_id)


def _persistent_rows(noise_band):
    """One persistent process and its per-rate raw samples, re-read from raw."""
    raw = _band_raw(noise_band)
    rows = {"process": None, "samples": {}, "ordinals": {}}
    for phase in PHASES:
        persistent = raw["phases"][phase]["persistent"]
        process = digest(persistent["process"])
        if rows["process"] not in (None, process):
            raise ValueError("persistent samples must come from one process")
        rows["process"] = process
        for entry in persistent["rates"]:
            rows["samples"].setdefault(entry["q256"], {})[phase] = entry["samples_ms"]
            known = rows["ordinals"].setdefault(entry["q256"], entry["time_in_process"])
            if known != entry["time_in_process"]:
                raise ValueError("persistent time in process disagrees between phases")
    return rows


def qualify_transfer(runtime_identity, *, rates, cases, noise_band, resource_trace):
    """Compare fresh/R/T evidence; any failed check retains fresh-process mode.

    The fresh leg is produced exactly like pass R minus persistence: one fresh
    process per sampled rate (the retained ground-truth repeat of the r=5 set),
    collector started before the CUDA libraries, one bracketed
    rate:<family>_R<rate>:begin|end window per rate, one trace. The fresh leg
    carries its own trace; pass R is one continuous trace handed in whole.
    Windows are re-derived here with ContinuousTrace from those traces — a
    caller-asserted window is never read — and compared field by field.
    observe_apply+analyze_trace is NOT a fresh producer: it brackets only the
    apply and yields no allocation_requests, transient peak or initialization
    multiset.

    The timing authority is FOLLOWUP-7's not-detected gate over raw evidence:
    r=5 fresh-process medians per sampled rate and phase, pooled log-scale
    process noise, a per-rate variance screen with a heteroscedastic fallback,
    a drift test over time in process, and a stamped common factor with its
  interval and MDE. Every number is re-derived here from raw samples; a
    caller-asserted band cannot survive the shape check. TP2 is refused until
    its cases are bound to a cut axis and rank. A qualification-only T screen
    is not itself decision-timing admission.
    """
    _runtime(runtime_identity)
    if runtime_identity["world_size"] != 1:
        raise ValueError("world 2 requires cases bound to their cut axis and rank")
    raw = _band_raw(noise_band)
    k = len(raw["phases"]["prefill"]["fresh"])
    selected = stratified_rates(rates, k)
    gate, gate_reasons = _timing_gate(noise_band, selected)
    persistent_rows = _persistent_rows(noise_band)
    resource_index = None
    resource_digest = None
    first_resource_ids = None
    owed = {(rate, None) for rate in selected}
    seen, checks, reasons = [], [], list(gate_reasons)
    fresh_processes, resource_processes, timing_processes = [], [], []
    for case in cases:
        if not isinstance(case, dict):
            reasons.append("qualification case must be an object")
            continue
        key = (case.get("rate"), case.get("cut_axis"))
        if key not in owed or key in seen:
            reasons.append(f"unexpected or duplicate qualification case {key}")
            continue
        seen.append(key)
        fresh, resource, timing = (case.get("fresh"), case.get("resource"),
                                   case.get("timing"))
        failed = []
        if not (isinstance(fresh, dict) and isinstance(resource, dict)
                and isinstance(timing, dict)):
            reasons.append(f"{key}: a qualification leg is missing")
            continue
        bindings = [leg.get("binding") for leg in (fresh, resource, timing)]
        world = (bindings[0].get("runtime", {}).get("execution", {}).get("tensor_parallel")
                 if isinstance(bindings[0], dict) else None)
        if (any(not isinstance(binding, dict) for binding in bindings) or
                bindings[1] != bindings[0] or bindings[2] != bindings[0]):
            failed.append("prepared identity binding differs between legs")
        elif world != runtime_identity["world_size"]:
            failed.append("prepared identity binding world differs from the qualified runtime")
        processes = [leg.get("process") for leg in (fresh, resource, timing)]
        if any(not isinstance(process, dict) for process in processes):
            failed.append("process identity is missing for a qualification leg")
        elif len({digest(process) for process in processes}) != 3:
            failed.append("fresh, resource and timing passes must be distinct processes")
        for bucket, process in zip((fresh_processes, resource_processes, timing_processes),
                                   processes, strict=True):
            if isinstance(process, dict):
                bucket.append(digest(process))
        # Pass R is one continuous trace: the first resource leg fixes its
        # device and context ids and every later leg must agree with them.
        ids = (resource.get("device_id"), resource.get("context_id"))
        if first_resource_ids is None:
            first_resource_ids = ids
            resource_index = _continuous(resource_trace, ids[0], ids[1])
            resource_digest = resource_index.trace_sha256
        elif ids != first_resource_ids:
            failed.append("resource legs disagree on the pass-R device or context")
        interval = f"rate:{runtime_identity['family']}_R{key[0]}"
        derived = []
        for leg, index, trace_digest in ((fresh, None, None), (resource, resource_index, resource_digest)):
            if index is None:
                try:
                    ct = _continuous(leg.get("trace"), leg.get("device_id"),
                                     leg.get("context_id"))
                except ValueError:
                    failed.append("fresh leg trace is malformed")
                    derived.append(None)
                    continue
                index, trace_digest = ct, ct.trace_sha256
            if leg.get("interval") != interval or interval not in index.intervals:
                failed.append("window is missing, unobserved or not this rate")
                derived.append(None)
                continue
            window = index.window(interval)
            if not isinstance(leg.get("process"), dict) or window["process_id"] != leg["process"]["pid"]:
                failed.append("window process differs from its leg")
            derived.append((window, trace_digest))
        if all(item is not None for item in derived):
            if derived[0][1] == derived[1][1]:
                failed.append("fresh and resource windows must come from distinct traces")
            for field in ("allocation_requests", "transient_peak_bytes",
                          "initialization_requests"):
                if derived[0][0][field] != derived[1][0][field]:
                    failed.append(field)
        if type(timing.get("collector_started")) is not bool or timing["collector_started"]:
            failed.append("timing collector was started")
        if (isinstance(timing.get("process"), dict)
                and persistent_rows["process"] is not None
                and digest(timing["process"]) != persistent_rows["process"]):
            failed.append("timing process differs from the raw band's persistent pass")
        # The drift axis is never caller-asserted: the timing leg carries the
        # pass-T ordinal and it must equal the band's own entry for this rate.
        if (type(timing.get("time_in_process")) is not int
                or timing["time_in_process"] != persistent_rows["ordinals"].get(key[0])):
            failed.append("timing time in process differs from the raw band")
        timing_samples = timing.get("samples_ms")
        if (not isinstance(timing_samples, dict) or set(timing_samples) != set(PHASES)
                or any(not isinstance(v, list) for v in timing_samples.values())):
            failed.append("timing samples must carry both phases from the raw band")
        else:
            for phase in PHASES:
                if timing_samples[phase] != persistent_rows["samples"].get(key[0], {}).get(phase):
                    failed.append(f"timing samples differ from the raw band: {phase}")
        d_log = {}
        for phase in PHASES:
            report = gate["phases"].get(phase)
            if report is not None and str(key[0]) in report["d_log"]:
                d_log[phase] = report["d_log"][str(key[0])]
        checks.append({"rate": key[0], "cut_axis": key[1], "d_log": d_log,
                       "failures": failed})
        reasons.extend(f"{key}: {reason}" for reason in failed)
    reasons.extend(f"missing qualification case {key}" for key in sorted(owed - set(seen), key=str))
    # Each fresh ground truth is its own process; a reused process is a reused
    # reference, not an independent one. Passes R and T are each one persistent
    # process across the sampled rates.
    if len(set(fresh_processes)) != len(fresh_processes):
        reasons.append("fresh reference process was reused across rates")
    if len(set(resource_processes)) > 1:
        reasons.append("resource pass must be one persistent process across sampled rates")
    if len(set(timing_processes)) > 1:
        reasons.append("timing pass must be one persistent process across sampled rates")
    # Fresh evidence is disjoint from the persistent passes: no fresh leg —
    # band repeat or case ground truth — may share a process with pass R or T.
    band_fresh = set(gate.get("fresh_processes") or [])
    persistent_ids = set(resource_processes) | set(timing_processes)
    if band_fresh & persistent_ids:
        reasons.append("fresh repeats share a process with a persistent pass")
    if set(fresh_processes) & persistent_ids:
        reasons.append("fresh reference process was reused across rates")
    if band_fresh & set(fresh_processes):
        reasons.append("fresh reference process was reused across rates")
    # The drift axis is derived, never labelled: pass R ran the sampled rates
    # in the order its single trace records, and the band's pass-T ordinals
    # must agree with that order. A relabelled band cannot survive this.
    if resource_index is not None:
        prefix = f"rate:{runtime_identity['family']}_R"
        trace_order = [int(name[len(prefix):]) for name in
                       sorted((n for n in resource_index.intervals
                               if n.startswith(prefix)),
                              key=lambda n: resource_index.intervals[n][0])]
        for phase in PHASES:
            report = gate["phases"].get(phase)
            if report is not None and report["persistent_order"] != trace_order:
                reasons.append(f"persistent timing order disagrees with the "
                               f"pass-R trace: {phase}")
    result = {"schema": QUALIFICATION_SCHEMA, "runtime_identity": copy.deepcopy(runtime_identity),
              "status": "failed" if reasons else "passed", "fallback": "fresh_process" if reasons else None,
              "rates": rates, "noise_band": copy.deepcopy(noise_band), "cases": copy.deepcopy(cases),
              "checks": checks, "timing_gate": gate, "reasons": reasons}
    result["qualification_id"] = digest(result)
    return result


def require_transfer(transfer, *, runtime_identity, qualifications, resource_records,
                     resource_trace, device_id, context_id):
    """Consumer gate: recompute the qualification and bind its pass-R evidence.

    A qualification ID alone proves nothing: the qualification is recomputed
    from its own stored evidence, and every pass-R record the consumer reads
    must re-derive from the very continuous trace handed in — window SHA,
    interval and every derived number. A record whose window does not come
    from that trace, at any rate, is refused.
    """
    _runtime(runtime_identity)
    if (not isinstance(transfer, dict) or set(transfer) != {"qualification_id", "runtime_identity"} or
            transfer["runtime_identity"] != runtime_identity):
        raise ValueError("resource transfer runtime identity differs or qualification is missing")
    if not isinstance(qualifications, list) or any(not isinstance(q, dict) for q in qualifications):
        raise ValueError("qualification registry entries must be objects")
    matches = [q for q in qualifications if q.get("qualification_id") == transfer["qualification_id"]]
    if len(matches) != 1:
        raise ValueError("resource transfer requires one known qualification")
    qualification = matches[0]
    if qualification["runtime_identity"] != runtime_identity or qualification["status"] != "passed":
        raise ValueError("resource transfer has no passing runtime qualification")
    recomputed = qualify_transfer(runtime_identity, rates=qualification["rates"], cases=qualification["cases"],
                                  noise_band=qualification["noise_band"], resource_trace=resource_trace)
    if recomputed != qualification or recomputed["status"] != "passed":
        raise ValueError("resource transfer qualification evidence was changed or failed")
    if not isinstance(resource_records, list):
        raise ValueError("resource transfer requires the pass-R records it binds")
    index = {}
    for record in resource_records:
        if not isinstance(record, dict) or not isinstance(record.get("rate"), str):
            raise ValueError("pass-R records must be objects keyed by their wire rate")
        if record["rate"] in index:
            raise ValueError("pass-R record rate appears twice")
        index[record["rate"]] = record
    trace = _continuous(resource_trace, device_id, context_id)
    for name, record in index.items():
        window = record.get("window", {})
        if (window.get("trace_sha256") != trace.trace_sha256
                or window.get("interval") != f"rate:{name}"
                or window.get("interval") not in trace.intervals):
            raise ValueError(f"pass-R record {name} is not a window of the handed trace")
        derived = trace.window(window["interval"])
        for field in ("allocation_requests", "transient_peak_bytes", "initialization_requests",
                      "baseline_bytes", "window_peak_bytes", "baseline_live", "end_live",
                      "process_id", "begin_ns", "end_ns"):
            if derived[field] != window.get(field):
                raise ValueError(f"pass-R record {name} window does not re-derive "
                                 "from the handed trace")
    for case in qualification["cases"]:
        rate_name = f"{runtime_identity['family']}_R{case['rate']}"
        record = index.get(rate_name)
        if record is None or record.get("binding") != case["resource"].get("binding"):
            raise ValueError(f"pass-R record {rate_name} is not the qualified binding")
    return qualification
