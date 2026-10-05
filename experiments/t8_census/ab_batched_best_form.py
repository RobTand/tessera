"""Batched units x best-form window Viterbi on the census's own unit path: identity, then time.

THE COMMON OWNER, ON THE ORIGINAL SOURCE.  The stack is planned exactly as the
exporter plans it -- ``plan_expert_stack`` over the REAL checkpoint config and
the REAL source's routed shapes -- so membership is the whole declared expert
population with original names, shapes and Hessian keys: nothing truncated,
renumbered, rewritten or synthetic, and no loadable checkpoint is produced.
The timed work is the exporter's ONE fresh batch encode/finish owner,
``tessera.export_serving.fresh_joined_encode``, over the batches
``plan_joined_encodes`` schedules in the loop's own order; the per-unit anchor
is the exporter's own per-unit finish (``encode_linear_planes`` +
``parse_unit_artifact`` + ``pack_fused``).  Identity is claimed bitwise, per
finished blob, per unit, against that anchor, per join-key width.

The join key is ``(stack, rows, cols)``: down/up orientations flush as
different keys, and the owner's ``batch_observed`` list -- one append per
actual ``encode_linears_planes`` call -- is this record's width evidence.  A
width-N claim is never made from the requested knob when the histogram says
tails ran narrower.

AUTH.  The probe runs inside the scoped producer environment, under the
interpreter ``TESSERA_PRODUCER_PYTHON`` selects, and authenticates it with the
core-owned ``tessera.export_serving.authenticate_producer_python()`` (env-only;
``None`` -- no interpreter selected -- refuses the run).  The Hessians are the
real capture, bound to the real producer authority's canonical capture by
``ActivationSource.from_capture``.

BUDGET.  The full stack is 864 units -- more than one bounded action -- so the
probe times a PREFIX of the owner's own schedule: batches in schedule order
until ``--budget-s`` (default 2700, the WHOLE action's ceiling: binding,
planning, JIT warmup, anchors and profiling ride in it) says the next batch no
longer fits, at which point the run stops and the unmeasured remainder is
named exactly (batches and units of the schedule that did not run).  Nothing
is silently truncated: the schedule, and the prefix line, are in the record.
The timed workload is encode+frame+verify (the owner includes the manifest
parse and the container pack); full export IO is not measured.  Work/J derives
each batch's OWN observed NVML joules (timestamped samples written beside the
record); the Netdata ``nvidia_smi.gpu_power_draw`` harvest joins by
``power.device_uuid`` on each batch's exact UTC window.

CLAIMS.  Estimates are labeled estimates: the aggregate rate over the timed
prefix yields a projection-EQUIVALENT 288-unit stack estimate and a
complete-864-unit estimate -- never an individually measured gate/up/down
stack time, never a measured full export, and no whole-gamut speed claim from
the staged rungs.

usage (inside the scoped producer environment; the launcher sets
TESSERA_PRODUCER_PYTHON, TESSERA_PRODUCER_SOURCE and TESSERA_GIT):
  python experiments/t8_census/ab_batched_best_form.py OUT --hessian H.json \\
      --producer-authority A.py --rungs E4M3:768,E2M1:640 [--batch 8]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
# Tessera comes from the selected interpreter's installed package; the source
# tree is a binding target (TESSERA_PRODUCER_SOURCE), never an import root.
# Only the experiments sibling is an import root, derived from __file__.
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera.export import served_recipe                                   # noqa: E402
from tessera.export_serving import (expert_stacks, fresh_joined_encode,     # noqa: E402
                                    grid_for, load_producer_authority,
                                    plan_expert_stack, plan_joined_encodes,
                                    quantizable)
from tessera.structure import STRUCTURE_ROUTED_MOE                         # noqa: E402

from profile_unit_encode import (DEFAULT_DIGEST_CACHE, SRC,                    # noqa: E402
                                 Power, bind_source, read_tensor)

BEST_ENV = "TESSERA_WINDOW_BEST_FORM"

#: Planning prior only, never a result: the coarse batch-1 profile at the
#: historical 1280 rung.  A batch with no warm measurement of its own is
#: staged against this and the plan says so.
COARSE_PRIOR_S_PER_UNIT = 15.2

#: The restored plan rungs the staged launch names.  Pinned in the help text
#: because guessing a rung is how 1280/1408 pretended to qualify the gamut.
RESTORED_RUNGS = ("GA layer3 E4M3@768, layer4 BF16@960, layer5 BF16@1088, "
                  "layer6 BF16@1152; GB layer3 E2M1@640, layer4 E2M1@768")


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def set_best(on: bool):
    if on:
        os.environ[BEST_ENV] = "1"
    else:
        os.environ.pop(BEST_ENV, None)


def leg_evidence(*, s_per_unit: float, units: int, wall_s: float, power: dict) -> dict:
    """The measured facts; work/J derives the batch's OWN observed joules.

    ``power["joules"]`` is the NVML trapezoid integral over the timestamped
    samples (``Power.stop``); with no samples there is no work/J -- a
    mean-times-time fabrication is never substituted.
    """
    joules = power.get("joules") if power else None
    return {"s_per_unit": round(s_per_unit, 4), "units": units,
            "wall_s": round(wall_s, 3), "joules_observed": joules,
            "units_per_kJ": round(units * 1000.0 / joules, 3) if joules else None}


def extrapolations(s_per_unit: float, *, projections_measured, basis: dict) -> dict:
    """Population-weighted estimates; a truncated prefix is not a population."""
    ext = {"cohort_seconds_per_unit": round(s_per_unit, 4),
           "complete_stack": {}, "basis": basis}
    costs = basis.get("shape_costs", {})
    population = basis.get("shape_population", {})
    missing = sorted(key for key in population if key not in costs)
    if len(set(projections_measured)) != 3 or not population or missing:
        ext["complete_stack"]["unmeasured"] = (
            f"completed projections={sorted(projections_measured)}; "
            f"missing population shape costs={missing or 'population unavailable'}")
        return ext
    population_units = sum(population.values())
    population_seconds = sum(population[key] * costs[key] for key in population)
    population_rate = population_seconds / population_units
    ext["seconds_per_unit"] = round(population_rate, 4)
    ext["projection_equivalent_288_unit_stack_estimate_min"] = round(population_rate * 288 / 60, 3)
    ext["complete_stack"]["complete_864_unit_stack_estimate_min"] = round(population_rate * 864 / 60, 3)
    ext["complete_stack"]["kind"] = (
        "extrapolation weighted by the declared shape population; "
        "not a measured full export (source I/O and wire writes excluded)")
    return ext


def plan_stages(stages, *, budget_s: float, elapsed_s: float) -> dict:
    """Decide each stage against the WHOLE action's remaining budget.

    Every stage -- a warm batch, an anchor set, a profile capture -- is paid
    from the same ceiling.  A stage that does not fit the remainder is skipped
    and named in ``unmeasured``; nothing is silently dropped and the ceiling
    is never extended.
    """
    remaining = budget_s - elapsed_s
    decisions, unmeasured = [], []
    for stage in stages:
        projected = round(stage["units"] * stage["s_per_unit_prior"], 3)
        if projected <= remaining:
            remaining -= projected
            decision = "run"
        else:
            decision = "skip"
            unmeasured.append(stage["stage"])
        decisions.append({"stage": stage["stage"], "decision": decision,
                          "projected_s": projected, "remaining_s": round(remaining, 3)})
    return {"stages": decisions, "unmeasured": unmeasured,
            "projected_total_s": round(elapsed_s + sum(d["projected_s"] for d in decisions
                                                       if d["decision"] == "run"), 3)}


def parse_rungs(spec: str, grid: str | None) -> list[tuple[str, int]]:
    """``GRID:Q256`` pairs; a bare rate takes ``--grid``; nothing is defaulted."""
    grids = ("E4M3", "E2M1x2", "E2M1", "BF16")
    out = []
    for item in spec.split(","):
        item = item.strip()
        name, sep, rate = item.partition(":")
        if sep:
            if name not in grids or not rate.isdigit() or int(rate) <= 0:
                raise SystemExit(f"--rungs entry {item!r} is not GRID:Q256 with grid one of {grids}")
            out.append((name, int(rate)))
        elif item.isdigit() and int(item) > 0:
            if grid is None:
                raise SystemExit(f"--rungs entry {item!r} is a bare rate and needs --grid")
            out.append((grid, int(item)))
        else:
            raise SystemExit(f"--rungs entry {item!r} is neither GRID:Q256 nor a positive rate")
    if not out:
        raise SystemExit("--rungs named no rung")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--grid", default=None,
                    help="grid for bare rung rates (E4M3, E2M1, E2M1x2, BF16)")
    ap.add_argument("--rungs", required=True,
                    help="comma list of GRID:Q256 rungs, actual restored plan rungs only -- "
                         + RESTORED_RUNGS
                         + " -- e.g. E4M3:768,E2M1:640; the historical 1280/1408 defaults "
                           "qualified no actual rung and are never a default here")
    ap.add_argument("--hessian", type=Path, required=True)
    ap.add_argument("--producer-authority", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8,
                    help="units per joined call requested of the owner; the widths that "
                         "actually run are batch_observed, never this knob")
    ap.add_argument("--budget-s", type=float, default=2700.0,
                    help="the WHOLE action's ceiling; the schedule prefix stops when the "
                         "next batch no longer fits, and the remainder is named")
    ap.add_argument("--bind-max-s", type=float, default=420.0,
                    help="stop before any timed batch when stamping the shards cost more "
                         "than this; the partial receipt with the binding is the record")
    ap.add_argument("--source-digest-cache", type=Path, default=DEFAULT_DIGEST_CACHE,
                    help="the source's own stable shard-digest cache (default: the "
                         "source-digests beside --src); never a second cache")
    ap.add_argument("--no-profile", action="store_true")
    return ap


def anchor_units(positions, units, src, grid, q256, recipe, activation, out: Path, label: str):
    """The exporter's own per-unit finish for these positions: the seq anchor.

    Returns the per-unit blob digests and the timing/power facts.  This is the
    exporter's per-unit path (``encode_linear_planes`` + ``parse_unit_artifact``
    + ``pack_fused``), not a probe re-implementation of it.
    """
    from tessera.export import encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.export_serving import packed_expert_weight
    # Match the owner's timing boundary: source reads precede both timers.
    members = [(units[i], read_tensor(src, units[i]["source_tensor"])) for i in positions]
    set_best(False)
    torch.cuda.synchronize()
    power = Power()
    start = utc()
    power.start()
    t0 = time.perf_counter()
    digests = {}
    for unit, source in members:
        weight = packed_expert_weight(source, unit).to("cuda", torch.float32).contiguous()
        extra = activation.for_unit(unit["tensor"], weight.shape[1], "cuda",
                                    scale_plane=recipe.scale_plane)
        exported, _artifact, _forests = encode_linear_planes(
            weight, grid=grid, q256=q256, body=recipe.body,
            name=unit["tensor"], verify=True, **extra)
        extra.clear()
        parse_unit_artifact(exported.blob, device="cuda")
        blob = pack_fused([(unit["projection"], exported.rows, exported.blob)])
        torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
        digests[unit["tensor"]] = sha256_bytes(blob)
        del weight, exported
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    pw = power.stop(series_path=out / f"{label}-anchor-power.jsonl")
    return {"digests": digests, "units": len(positions),
            "start_utc": start, "end_utc": utc(), "wall_s": round(wall, 3),
            "s_per_unit": round(wall / len(positions), 4), "power": pw}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


def run_owner_batch(positions, units, stack_plan, recipe, activation, src, out: Path, label: str):
    """One scheduled batch through THE owner, timed, with its own power window.

    The owner returns one ``(exported, blob, payload, own_global, manifest)``
    per member, in member order; the batch record keeps every blob digest (the
    identity comparison set), the owner's own ``batch_observed`` widths, and
    the workload label: encode+frame+verify, not full export IO.
    """
    members = [(units[i], read_tensor(src, units[i]["source_tensor"])) for i in positions]
    observed = []
    set_best(True)
    torch.cuda.synchronize()
    power = Power()
    start = utc()
    power.start()
    t0 = time.perf_counter()
    results = fresh_joined_encode(
        members, stack_plan=stack_plan, activation=activation,
        device="cuda", no_verify=False, batch_observed=observed)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    pw = power.stop(series_path=out / f"{label}-owner-power.jsonl")
    set_best(False)
    digests = {unit["tensor"]: sha256_bytes(result[1])
               for (unit, _source), result in zip(members, results)}
    del members, results
    return {"positions": list(positions),
            "units": [units[i]["tensor"] for i in positions],
            "key": f"{units[positions[0]]['rows']}x{units[positions[0]]['cols']}",
            "widths_observed": observed, "digests": digests,
            "start_utc": start, "end_utc": utc(),
            "wall_s": round(wall, 3), "s_per_unit": round(wall / len(positions), 4),
            "power": pw,
            "workload": "encode+frame+verify (fresh_joined_encode with no_verify=False, "
                        "verify symmetric with the seq anchor; incl. manifest parse and "
                        "pack_fused); full export IO not measured"}


def cuda_profile(fn, out: Path, label: str, trace_bound: int = 200_000):
    """One CUDA-only kineto capture of ``fn()``, aggregated from the raw events."""
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    events = prof.profiler.kineto_results.events()

    def ns(e, what):
        f = getattr(e, f"{what}_ns", None)
        if f is not None:
            return int(f())
        return int(getattr(e, f"{what}_us")() * 1000)

    dev, host = {}, {}
    intervals = []
    for e in events:
        name = e.name()
        dur = ns(e, "duration")
        if "CUDA" in str(e.device_type()):
            s = ns(e, "start")
            intervals.append((s, s + dur))
            d = dev.setdefault(name, [0, 0])
            d[0] += 1
            d[1] += dur
        else:
            h = host.setdefault(name, [0, 0])
            h[0] += 1
            h[1] += dur
    intervals.sort()
    busy = 0
    gaps = []
    cur_s, cur_e = None, None
    for s, e in intervals:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (intervals[-1][1] - intervals[0][0]) if intervals else 0
    gaps.sort()
    dev_total = sum(v[1] for v in dev.values())
    rec = {
        "label": label, "wall_s": round(wall, 3), "device_ops": len(intervals),
        "device_span_s": round(span / 1e9, 4), "device_busy_s": round(busy / 1e9, 4),
        "device_busy_frac_of_wall": round(busy / 1e9 / wall, 4) if wall else None,
        "device_busy_frac_of_span": round(busy / span, 4) if span else None,
        "gaps": {"count": len(gaps), "sum_s": round(sum(gaps) / 1e9, 4),
                 "p50_us": round(gaps[len(gaps) // 2] / 1e3, 3) if gaps else None,
                 "p90_us": round(gaps[int(len(gaps) * 0.9)] / 1e3, 3) if gaps else None,
                 "p99_us": round(gaps[int(len(gaps) * 0.99)] / 1e3, 3) if gaps else None},
        "kernels_by_device_time": [
            {"name": k[:160], "calls": v[0], "device_ms": round(v[1] / 1e6, 3),
             "mean_us": round(v[1] / v[0] / 1e3, 3),
             "frac_of_device": round(v[1] / dev_total, 4) if dev_total else None}
            for k, v in sorted(dev.items(), key=lambda kv: -kv[1][1])[:25]],
        "host_runtime_calls": [
            {"name": k[:120], "calls": v[0], "ms": round(v[1] / 1e6, 3)}
            for k, v in sorted(host.items(), key=lambda kv: -kv[1][1])[:25]],
    }
    if len(events) <= trace_bound:
        trace = out / f"trace-{label}.json"
        prof.export_chrome_trace(str(trace))
        rec["trace"] = str(trace)
    else:
        rec["trace"] = f"not written: {len(events)} events > bound {trace_bound}"
    return rec


class Count:
    """Viterbi calls by (columns, rate) and the synced seconds inside the fused call.

    Measurement instrumentation only: it wraps the owner's own calls while
    they run and restores them exactly; it never decides grouping or order.
    """

    def __init__(self):
        from tessera import encode as enc
        from tessera import window_viterbi as wv
        self.enc, self.wv = enc, wv
        self.calls: dict[str, int] = {}
        self.fused_s = 0.0

    def __enter__(self):
        enc, wv = self.enc, self.wv
        self._real = (enc.viterbi_window, wv.viterbi_window_fused)
        real_window, real_fused = self._real

        def window(targets, vectors, window_bits, rate, *a, **kw):
            key = f"cols={targets.shape[1]} rows={targets.shape[0]} L={window_bits} R={rate}"
            self.calls[key] = self.calls.get(key, 0) + 1
            return real_window(targets, vectors, window_bits, rate, *a, **kw)

        def fused(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = real_fused(*a, **kw)
            torch.cuda.synchronize()
            self.fused_s += time.perf_counter() - t0
            return r

        enc.viterbi_window, wv.viterbi_window_fused = window, fused
        return self

    def __exit__(self, *exc):
        self.enc.viterbi_window, self.wv.viterbi_window_fused = self._real
        return False

    def record(self):
        return {"calls": self.calls, "fused_synced_s": round(self.fused_s, 3)}


def main() -> int:
    started = time.perf_counter()
    # The core-owned auth: TESSERA_PRODUCER_PYTHON selects the scoped producer
    # interpreter, TESSERA_PRODUCER_SOURCE binds the checkout's src/tessera; a
    # None return (no selection) refuses the run -- it is not a pass.
    from tessera.export_serving import authenticate_producer_python
    auth = authenticate_producer_python()
    if auth is None:
        raise SystemExit("TESSERA_PRODUCER_PYTHON must select the scoped producer "
                         "interpreter; an unauthenticated run encodes nothing")
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("a GPU measurement with no GPU")
    args.out.mkdir(parents=True, exist_ok=True)

    rungs = parse_rungs(args.rungs, args.grid)

    # The exporter's own construction: real authority (the launch supplies the
    # campaign's actual authority file), real Hessians bound to its canonical
    # capture, exporter LDLQ defaults.
    from tessera.export import DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA
    from tessera.export import ActivationSource
    _authority, canonical_capture = load_producer_authority(args.producer_authority)
    activation = ActivationSource.from_capture(
        args.hessian, canonical_capture=canonical_capture, ldlq_sigma=DEFAULT_LDLQ_SIGMA,
        ldlq_block=DEFAULT_LDLQ_BLOCK, refit_reach_floor=False)

    # The REAL stack plan on the ORIGINAL source: whole declared membership,
    # the exporter's own refusal rules -- nothing truncated or synthetic.
    config = json.loads((args.src / "config.json").read_text())
    _shards, _shapes, _expert_shapes, routed_shapes = quantizable(args.src)
    stacks = expert_stacks(routed_shapes)
    stack = f"model.language_model.layers.{args.layer}.mlp.experts"
    if stack not in stacks:
        raise SystemExit(f"source {args.src} holds no routed stack {stack}")

    rec = {
        "schema": "tessera.t8_census.ab_batched_best_form.v4",
        "auth": auth,
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                               cwd=ROOT).stdout.strip() or None,
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "host": os.uname().nodename, "src": str(args.src), "layer": args.layer,
        "stack": stack, "batch": args.batch,
        "hessian": str(args.hessian), "producer_authority": str(args.producer_authority),
        "budget_s": args.budget_s, "bind_max_s": args.bind_max_s,
        "coarse_prior_s_per_unit": COARSE_PRIOR_S_PER_UNIT,
        "restored_rungs": RESTORED_RUNGS,
        "netdata": {"metric": "nvidia_smi.gpu_power_draw",
                    "join": "each batch's power.window_utc on power.device_uuid; the "
                            "in-process NVML joules here are device-scoped and stand "
                            "separate from any host-side series",
                    "harvest": "lead, by these UTC windows, after the actual run"},
        "rungs": {}, "unmeasured": [],
    }
    outp = args.out / "ab_batched_best_form.json"

    def save():
        outp.write_text(json.dumps(rec, indent=1, default=str))

    # Bind the ORIGINAL shards these units live on through the stable cache --
    # before anything is timed; a cold binding that eats its budget stops here.
    index = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    layer_names = sorted(n for n in index if f".layers.{args.layer}.mlp.experts." in n)
    bound = bind_source(args.src, layer_names, args.source_digest_cache)
    rec["source_binding"] = {k: bound[k] for k in
                             ("shards", "digest_s", "cache", "cached_shards", "hashed_shards")}
    rec["source_binding"]["receipt"] = bound["receipt"]
    rec["source_binding"]["per_unit_weight_binding"] = {
        name: {"shard": index[name], "shard_sha256": bound["identity"]["files"][index[name]]}
        for name in layer_names}
    rec["source_identity"] = bound["identity"]
    if bound["digest_s"] > args.bind_max_s:
        rec["unmeasured"].append({"stage": "all timed batches",
                                  "reason": "source binding consumed its budget",
                                  "bind_s": bound["digest_s"], "bind_max_s": args.bind_max_s})
        save()
        print(json.dumps({"stopped": "bind budget", "unmeasured": rec["unmeasured"]}, indent=1))
        return 3
    save()

    for rung_i, (grid_name, q) in enumerate(rungs):
        grid = grid_for(grid_name)
        plan_record = plan_expert_stack(stack, stacks[stack], grid, q, config=config)
        units = plan_record["units"]
        keys = [(u["stack"], u["rows"], u["cols"]) for u in units]
        schedule = plan_joined_encodes(keys, args.batch)
        recipe = served_recipe(grid, q, STRUCTURE_ROUTED_MOE)
        stack_plan = {stack: plan_record}
        projections_in_stack = sorted({u["projection"] for u in units})
        r = rec["rungs"][f"{grid_name}:{q}"] = {
            "grid": grid_name, "q256": q,
            "membership": {"experts": len(stacks[stack]), "units": len(units),
                           "projections": projections_in_stack,
                           "projection_unit_counts": {
                               p: sum(1 for u in units if u["projection"] == p)
                               for p in projections_in_stack},
                           "unit_shapes": sorted({f"{u['projection']}:{u['rows']}x{u['cols']}"
                                                  for u in units})},
            "schedule": {"requested_batch": args.batch, "batches": len(schedule),
                         "units": len(units)},
            "widths": {}, "batches": [], "unmeasured": []}

        # Warm + anchor the FIRST batch of each join key: the width compiles and
        # captures its plan through the owner, and the exporter's per-unit
        # finish anchors those exact units bitwise.
        seen_keys = set()
        warm_and_anchor = []
        for positions in schedule:
            k = keys[positions[0]]
            if k in seen_keys:
                continue
            seen_keys.add(k)
            warm_and_anchor.append(positions)
        stages = []
        for positions in warm_and_anchor:
            k = keys[positions[0]]
            stages.append({"stage": f"warm+anchor {k[1]}x{k[2]}", "units": len(positions) * 2,
                           "s_per_unit_prior": COARSE_PRIOR_S_PER_UNIT})
        plan = plan_stages(stages, budget_s=args.budget_s, elapsed_s=time.perf_counter() - started)
        r["stage_plan"] = plan
        for entry in plan["unmeasured"]:
            rec["unmeasured"].append({"stage": f"{grid_name}:{q}/{entry}", "reason": "budget"})
        decision = {d["stage"]: d["decision"] for d in plan["stages"]}

        for positions in warm_and_anchor:
            k = keys[positions[0]]
            width_key = f"{k[1]}x{k[2]}"
            if decision.get(f"warm+anchor {width_key}", "run") == "skip":
                continue
            batch = run_owner_batch(positions, units, stack_plan, recipe, activation,
                                    args.src, args.out, f"{grid_name}-{q}-w{width_key}")
            anchor = anchor_units(positions, units, args.src, grid, q, recipe, activation,
                                  args.out, f"{grid_name}-{q}-w{width_key}")
            differing = sorted(n for n in batch["digests"]
                               if batch["digests"][n] != anchor["digests"].get(n))
            r["widths"][width_key] = {
                "owner_batch": {k2: batch[k2] for k2 in
                                ("positions", "units", "widths_observed", "wall_s",
                                 "s_per_unit", "start_utc", "end_utc", "power", "workload")},
                "anchor_seq": {k2: anchor[k2] for k2 in
                               ("units", "wall_s", "s_per_unit", "start_utc", "end_utc",
                                "power")},
                "identity": {"units": len(positions),
                             "same_bytes": len(positions) - len(differing),
                             "differing_units": differing,
                             "batch_observed": observed_summary(batch["widths_observed"])},
                "blobs": batch["digests"]}
            print(f"[{grid_name}:{q}] w{width_key}: owner {batch['s_per_unit']:.3f} s/unit, "
                  f"anchor {anchor['s_per_unit']:.3f} s/unit, same "
                  f"{len(positions) - len(differing)}/{len(positions)}, "
                  f"widths {batch['widths_observed']}", flush=True)
            save()
            if differing:
                rec["failure"] = {"reason": "joined encode differs from per-unit anchor",
                                  "rung": f"{grid_name}:{q}", "units": differing}
                save()
                return 4

        # Capture the required before/after packet before the timed prefix can
        # spend the rest of the action. Use observed warm costs, not a guess.
        if not args.no_profile:
            r["captures"] = {}
            for positions in warm_and_anchor:
                k = keys[positions[0]]
                width_key = f"{k[1]}x{k[2]}"
                measured_width = r["widths"].get(width_key)
                if measured_width is None:
                    continue
                projected = (2 * measured_width["owner_batch"]["wall_s"]
                             + measured_width["anchor_seq"]["wall_s"])
                if time.perf_counter() - started + projected > args.budget_s:
                    rec["unmeasured"].append({"stage": f"{grid_name}:{q}/profiles/{width_key}",
                                              "reason": "budget", "projected_s": projected})
                    save()
                    return 3
                with Count() as c:
                    run_owner_batch(positions, units, stack_plan, recipe, activation,
                                    args.src, args.out, f"{grid_name}-{q}-{width_key}-counted")
                capture = {"counted_batch": c.record()}
                capture["profiled_anchor"] = cuda_profile(
                    lambda: anchor_units(positions, units, args.src, grid, q, recipe,
                                         activation, args.out, f"{grid_name}-{q}-{width_key}-profseq"),
                    args.out, f"anchor-{grid_name}-q{q}-{width_key}")
                r["captures"][width_key] = capture
                save()
                capture["profiled_batch"] = cuda_profile(
                    lambda: run_owner_batch(positions, units, stack_plan, recipe, activation,
                                            args.src, args.out, f"{grid_name}-{q}-{width_key}-prof"),
                    args.out, f"owner-{grid_name}-q{q}-{width_key}")
                save()

        # Timed: the owner's schedule, IN ORDER, until the budget says stop.
        timed_units, timed_wall, timed_joules = 0, 0.0, 0.0
        stopped = None
        for b_i, positions in enumerate(schedule):
            k = keys[positions[0]]
            width_key = f"{k[1]}x{k[2]}"
            prior = (r["widths"].get(width_key, {}).get("owner_batch", {})
                     .get("s_per_unit", COARSE_PRIOR_S_PER_UNIT))
            projected = len(positions) * prior
            elapsed = time.perf_counter() - started
            if elapsed + projected > args.budget_s:
                stopped = b_i
                remaining_batches = schedule[b_i:]
                r["unmeasured_batches"] = {
                    "reason": "budget", "at_batch_index": b_i,
                    "batches_remaining": len(remaining_batches),
                    "units_remaining": sum(len(p) for p in remaining_batches),
                    "elapsed_s": round(elapsed, 3), "projected_next_batch_s": round(projected, 3)}
                rec["unmeasured"].append({
                    "stage": f"{grid_name}:{q}/schedule_batches_{b_i}..{len(schedule) - 1}",
                    "reason": "budget",
                    "batches": len(remaining_batches),
                    "units": sum(len(p) for p in remaining_batches)})
                save()
                break
            batch = run_owner_batch(positions, units, stack_plan, recipe, activation,
                                    args.src, args.out, f"{grid_name}-{q}-b{b_i:03d}")
            batch.pop("digests", None)
            batch.update(leg_evidence(s_per_unit=batch["s_per_unit"],
                                      units=len(positions), wall_s=batch["wall_s"],
                                      power=batch["power"]))
            r["batches"].append(batch)
            timed_units += len(positions)
            timed_wall += batch["wall_s"]
            timed_joules += batch["power"].get("joules") or 0.0
            if b_i % 4 == 0:
                save()
        if stopped is None:
            r["unmeasured_batches"] = {"reason": None, "note": "schedule complete"}

        # Summary per width from the COMPLETED timed batches only.
        by_width = {}
        for batch in r["batches"]:
            by_width.setdefault(batch["key"], []).append(batch)
        r["summary"] = {}
        for width_key, batches in sorted(by_width.items()):
            spu = sum(b["wall_s"] for b in batches) / sum(b["units"] for b in batches)
            joules = sum(b["power"].get("joules") or 0.0 for b in batches) or None
            units_n = sum(b["units"] for b in batches)
            w = [b["power"].get("mean_w") for b in batches if b["power"].get("mean_w")]
            r["summary"][width_key] = {
                "batches_timed": len(batches), "units_timed": units_n,
                "s_per_unit": round(spu, 4),
                "joules_observed": round(joules, 3) if joules else None,
                "units_per_kJ": round(units_n * 1000.0 / joules, 3) if joules else None,
                "mean_w": round(sum(w) / len(w), 2) if w else None,
                "widths_observed": sorted({tuple(b["widths_observed"]) for b in batches})}
        completed_units = [u for b in r["batches"] for u in
                           ([units[i] for i in b["positions"]])]
        measured = tuple(sorted({u["projection"] for u in completed_units}))
        if r["batches"]:
            spu = (sum(b["wall_s"] for b in r["batches"]) / timed_units if timed_units else 0.0)
            r["extrapolated"] = extrapolations(
                spu, projections_measured=measured,
                basis={"grid": grid_name, "q256": q, "layer": args.layer,
                       "batch": args.batch, "units_completed": timed_units,
                       "projections": projections_in_stack,
                       "projection_unit_counts": {p: sum(u["projection"] == p for u in completed_units)
                                                  for p in measured},
                       "unit_shapes": r["membership"]["unit_shapes"],
                       "shape_costs": {key: row["s_per_unit"] for key, row in r["summary"].items()},
                       "shape_population": {f"{u['rows']}x{u['cols']}": sum(
                           (v["rows"], v["cols"]) == (u["rows"], u["cols"]) for v in units)
                           for u in units},
                       "effective_group_widths_observed": sorted(
                           {tuple(b["widths_observed"]) for b in r["batches"]}),
                       "schedule_batches_total": len(schedule),
                       "workload": "encode+frame+verify; full export IO not measured"})

    rec["scope"] = {
        "rungs_measured": sorted(key for key, row in rec["rungs"].items() if row["batches"]),
        "whole_gamut_claim": "none: only the staged rungs were measured; every omitted "
                             "rung is named in unmeasured or was never staged",
        "unmeasured": rec["unmeasured"],
    }
    save()
    print(f"wrote {outp}")
    return 0


def observed_summary(widths):
    """Histogram of the owner's per-call widths, plus the max actually run."""
    hist = {}
    for w in widths:
        hist[w] = hist.get(w, 0) + 1
    return {"histogram": hist, "max_width": max(widths) if widths else 0,
            "calls": len(widths)}


if __name__ == "__main__":
    raise SystemExit(main())
