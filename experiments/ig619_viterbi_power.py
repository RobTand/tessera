#!/usr/bin/env python3
"""tessera#619: a matched A/B of the window Viterbi's two schedules at GLM
routed-expert shapes, with power, joules and bytes, on one GB10.

The window Viterbi that encodes every ``TESSERA_E4M3_K1`` and
``TESSERA_BF16_K1`` unit has two spellings of one recurrence:

  front  ``TESSERA_WINDOW_BEST_FORM`` unset or ``0``, the default: each step
         writes the ``2^L`` front, and the L2 budget holds 32 columns of it.
  best   ``TESSERA_WINDOW_BEST_FORM=1``: each step carries the class minimum,
         and the same budget holds 16x the columns at ``L=14``, ``R=4``.

The flag is read per call, so both run in ONE process on ONE tensor on ONE
board, seconds apart.  That is what makes a ratio between them a result: a box
difference cannot sit inside it.  Every arm is a fresh encode of the same
synthetic units, ``encode_linear_planes(..., verify=False)`` at the GLM expert
shape (gate/up 2048x4096) at the two rungs the issue measured, BF16 q256 1024
and E4M3 q256 896, weights-only, the window body over the CHANNEL plane.

Per rung the arms are:

  front#1 best#1 best#2 front#2   ABBA, one thread.  Drift in clocks or
                                  temperature falls on both forms alike, and
                                  the two repeats of a form say how big it was.
  front/rs0 best/rs0              E4M3 only (a fractional rate is two rates):
                                  ``TESSERA_WINDOW_RATE_STREAMS=0``, so the two
                                  rate chains queue on one stream.  The gap to
                                  the ABBA arms is what two chains in flight
                                  already buy.
  front/Nt best/Nt                N threads, each on its own CUDA stream, each
                                  with its own plans, the units split evenly:
                                  "two units in flight".

A ratio between arms is only a result if the arms wrote the same bytes, and
the probe decides that, not the reader: the sha256 of every unit's blob is
compared across ALL arms of a rung, a difference is recorded and the process
exits 3.  An earlier version of this probe recorded ``byte_identical: false``
and still exited 0.

Power is read two ways, against the 140 W envelope.  The in-process sampler is
one streaming ``nvidia-smi`` at 500 ms (watts and the SM clock, so a power cap
shows as a clock and not only as a number), bracketed by each arm's own wall
clock.  The box's Netdata is the second instrument, read through
``experiments/box_power_window.py`` over the same windows; its 10 s groups are
kept only when they lie wholly inside a window, so a 20 s arm has one or two.
J/unit is the arm's mean watts over its units/s, and a pooled figure is total
joules over total units.

One unit per rung is also profiled (``torch.profiler``) under each form, for
the kernel that holds the time, its launches and microseconds each, and the
fraction of the encode's wall clock the device spent in kernels.  Profiling
never shares a run with timing.

NOT measured here: real GLM weights (the units are N(0, 0.02^2) in bf16, which
fixes the shapes, rates and kernel sequence and not the values), served KL
(the bytes are identical, so there is nothing to price), and any box but the one
the action landed on.  Run it on each box you want a reading for.

Usage, through PrismaBuild (``--gpu``, ``PYTHONPATH=src``)::

    python experiments/ig619_viterbi_power.py \\
        --publish-dir /mnt/shared/tessera-measurements/issue-619-ab-20261011
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# The tree this file ships with, never whichever ``tessera`` the interpreter's
# site-packages happens to carry: the first three arms of tessera#619 failed
# because a stale venv copy shadowed the snapshot (tests/test_experiment_import_roots.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import box_power_window  # noqa: E402

ENVELOPE_W = box_power_window.ENVELOPE_W

#: The schedule -> its ``TESSERA_WINDOW_BEST_FORM`` spelling.  ``front`` is
#: written ``0`` and not left unset so an inherited value cannot move an arm.
FORMS = {"front": "0", "best": "1"}
_BEST_FORM_ENV = "TESSERA_WINDOW_BEST_FORM"
_RATE_STREAMS_ENV = "TESSERA_WINDOW_RATE_STREAMS"
#: Knobs a caller may have set that move the plans' widths or kernels; recorded
#: in the receipt so a reading is never mistaken for the defaults.
_RECORDED_ENV = (_BEST_FORM_ENV, _RATE_STREAMS_ENV, "TESSERA_WINDOW_GRAPH",
                 "TESSERA_WINDOW_L2_BYTES", "TESSERA_WINDOW_BEST_TILE",
                 "TESSERA_WINDOW_SCAN_UNROLL", "PYTHONPATH")
#: A Netdata group is 10 s; the last arm's group is not complete until it ends.
_NETDATA_SETTLE_S = 12.0
_GRIDS = ("bf16", "e4m3")


class IdentityError(RuntimeError):
    """Two arms of one comparison wrote different bytes."""


# -- the bookkeeping that carries the claim ------------------------------------

def arm_schedule(forms):
    """ABBA: every form twice, the second pass in reverse, so the run ends on
    the form it began with and drift cannot favour one of them."""
    forms = list(forms)
    return forms + forms[::-1]


def check_identity(arms):
    """Refuse unless every arm encoded the same units to the same bytes.

    ``arms`` are dicts with ``name`` and ``sha256`` (one hex digest per unit,
    in unit order).  The first arm is the reference; the error names the arm
    that differs and the first unit that does.  An empty comparison proves
    nothing and is refused too.
    """
    if not arms or not arms[0]["sha256"]:
        raise IdentityError("no arm encoded a unit: there is nothing to compare")
    ref = arms[0]
    for arm in arms[1:]:
        if len(arm["sha256"]) != len(ref["sha256"]):
            raise IdentityError(
                f"arm {arm['name']!r} encoded {len(arm['sha256'])} units, "
                f"{ref['name']!r} encoded {len(ref['sha256'])}")
        for i, (want, got) in enumerate(zip(ref["sha256"], arm["sha256"])):
            if want != got:
                raise IdentityError(
                    f"arm {arm['name']!r} unit {i} differs from {ref['name']!r}: "
                    f"sha256 {got[:12]} against {want[:12]}")


def split_units(units, threads):
    """Unit indices per thread, round-robin.  A remainder is refused: the
    arms would otherwise encode different work."""
    if units < 1 or threads < 1 or units % threads:
        raise ValueError(f"{units} units do not split evenly over {threads} threads")
    return [list(range(t, units, threads)) for t in range(threads)]


def pool(arms):
    """Total units over total wall, and total joules over total units.

    A mean of per-arm ratios would weight a short arm like a long one; the
    energy is each arm's mean watts times its own seconds.
    """
    units = sum(a["units"] for a in arms)
    wall = sum(a["wall_s"] for a in arms)
    joules = None
    if all(a["power"].get("mean_w") is not None for a in arms):
        joules = sum(a["power"]["mean_w"] * a["wall_s"] for a in arms)
    return {"units": units, "wall_s": round(wall, 3),
            "units_per_s": units / wall,
            "j_per_unit": None if joules is None else joules / units,
            "mean_w": None if joules is None else joules / wall}


def compare(base, cand):
    """``cand`` against ``base``: faster is above 1, cheaper is below 1."""
    jb, jc = base.get("j_per_unit"), cand.get("j_per_unit")
    return {"speedup": cand["units_per_s"] / base["units_per_s"],
            "joule_ratio": (jc / jb) if jb and jc else None}


def netdata_power_summary(series):
    """The box's GPU power over one window, from ``box_power_window.collect``.

    An empty window is reported as no reading and never as 0 W: "the GPU drew
    nothing" and "Netdata has no whole group here" are different facts.
    """
    entry = series.get("nvidia_smi.gpu_power_draw") or {}
    if "error" in entry:
        return {"error": entry["error"]}
    stats = (entry.get("stats") or {}).get("power_draw") or {}
    cover = entry.get("coverage") or {}
    out = {"accepted_groups": cover.get("accepted_groups", 0),
           "accepted_coverage_s": cover.get("accepted_coverage_s", 0),
           "unobserved_requested_s": cover.get("unobserved_requested_s"),
           "mean_w": stats.get("mean")}
    if out["mean_w"] is not None:
        out.update(min_w=stats.get("min"), max_w=stats.get("max"),
                   mean_fraction_of_envelope=round(out["mean_w"] / ENVELOPE_W, 4))
    return out


def netdata_power(host, after, before):
    if int(before) <= int(after):
        # Netdata groups are whole seconds at the finest; this window holds none.
        return {"error": f"window of {before - after:.2f} s holds no whole Netdata group"}
    points = max(4, min(600, (int(before) - int(after)) // 10))
    try:
        series = box_power_window.collect(host, int(after), int(before), points)
    except Exception as exc:  # noqa: BLE001 - a missing reading is recorded, not fatal
        return {"error": f"{type(exc).__name__}: {exc}"}
    return netdata_power_summary(series)


def parse_sample(line):
    """``(watts, sm_mhz)`` from one ``nvidia-smi`` CSV line, ``None`` where a
    field is unreadable.  The clock is context: it is never a reason to drop a
    power sample, and an unreadable power field drops the whole line."""
    fields = [f.strip() for f in line.split(",")]
    try:
        watts = float(fields[0])
    except (ValueError, IndexError):
        return None, None
    try:
        return watts, float(fields[1])
    except (ValueError, IndexError):
        return watts, None


class Power:
    """Sample ``nvidia-smi`` power.draw as one streaming subprocess for the run.

    One ``nvidia-smi -lms`` process, read line by line with arrival
    timestamps: no spawn per sample, so a loaded box cannot starve the
    sampler behind fork storms.  ``errors`` counts unparsed lines, so a
    silent window is distinguishable from an idle one.
    """

    def __init__(self, period_ms: int = 500):
        self._period_ms = period_ms
        self._samples: list[tuple[float, float]] = []
        self._clocks: list[tuple[float, float]] = []
        self._errors = 0
        self._stop = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _fields(self) -> str:
        """``power.draw,clocks.sm`` if the driver answers it, else power alone:
        a query the driver refuses would leave the whole run with no power."""
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.sm",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=30)
            if done.returncode == 0:
                return "power.draw,clocks.sm"
        except Exception:
            pass
        return "power.draw"

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self._fields()}",
                 "--format=csv,noheader,nounits", f"-lms={self._period_ms}"],
                stdout=subprocess.PIPE, text=True)
        except Exception:
            self._proc = None
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        try:
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        self._thread.join(timeout=10)

    def _run(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            stamp = time.time()
            watts, mhz = parse_sample(line)
            if watts is None:
                self._errors += 1
                continue
            self._samples.append((stamp, watts))
            if mhz is not None:
                self._clocks.append((stamp, mhz))

    def window(self, t0: float, t1: float) -> dict:
        sel = [s for s in self._samples if t0 <= s[0] <= t1]
        out: dict = {"points": len(sel), "sampler_errors": self._errors}
        if not sel:
            return out
        watts = [s[1] for s in sel]
        out.update({
            "mean_w": round(sum(watts) / len(watts), 2),
            "max_w": round(max(watts), 2),
            "min_w": round(min(watts), 2),
            "mean_fraction_of_envelope": round(sum(watts) / len(watts) / ENVELOPE_W, 4),
            "max_fraction_of_envelope": round(max(watts) / ENVELOPE_W, 4),
        })
        mhz = [c[1] for c in self._clocks if t0 <= c[0] <= t1]
        if mhz:
            out.update({"mean_sm_mhz": round(sum(mhz) / len(mhz)), "min_sm_mhz": round(min(mhz))})
        return out


# -- the arguments --------------------------------------------------------------

def _rungs(text):
    out = []
    for item in text.split(","):
        grid, _, q256 = item.partition(":")
        if grid not in _GRIDS or not q256.isdigit():
            raise argparse.ArgumentTypeError(
                f"{item!r} is not GRID:Q256 with GRID one of {', '.join(_GRIDS)}")
        out.append((grid, int(q256)))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--units", type=int, default=12,
                    help="units each arm encodes; split evenly over --threads")
    ap.add_argument("--threads", type=int, default=2,
                    help="units in flight in the concurrent arms")
    ap.add_argument("--rungs", type=_rungs, default=_rungs("bf16:1024,e4m3:896"),
                    help="GRID:Q256 list; q256 not a multiple of 256 is a mixed-rate stack")
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--table", metavar="RECEIPT", default=None,
                    help="print markdown tables of a receipt and exit; reads no device")
    ap.add_argument("--out", default="/tmp/ig619_receipt.json")
    ap.add_argument("--publish-dir", default=None,
                    help="also write the receipt here, opened exclusively so a "
                         "published receipt is never overwritten")
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        split_units(args.units, args.threads)
    except ValueError as exc:
        ap.error(str(exc))
    return args


# -- the encodes ------------------------------------------------------------------

def _blob_sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@contextlib.contextmanager
def _knobs(form, rate_streams=True):
    """The two per-call knobs for one arm, restored on the way out."""
    saved = {k: os.environ.get(k) for k in (_BEST_FORM_ENV, _RATE_STREAMS_ENV)}
    os.environ[_BEST_FORM_ENV] = FORMS[form]
    if rate_streams:
        os.environ.pop(_RATE_STREAMS_ENV, None)
    else:
        os.environ[_RATE_STREAMS_ENV] = "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _weights(rows, cols, units, q256):
    import torch

    gen = torch.Generator(device="cpu").manual_seed(61900 + q256)
    out = [(torch.randn(rows, cols, generator=gen) * 0.02)
           .to(device="cuda", dtype=torch.bfloat16).contiguous()
           for _ in range(units)]
    torch.cuda.synchronize()
    return out


def _encode(weight, name, grid, q256):
    from tessera.export import encode_linear_planes

    return encode_linear_planes(weight, grid=grid, q256=q256, name=name,
                                verify=False)[0].blob


def _run_single(form, weights, names, grid, q256, rate_streams=True):
    with _knobs(form, rate_streams):
        return [_encode(w, n, grid, q256) for w, n in zip(weights, names)]


def _timed(fn):
    """``fn()`` bracketed by device syncs; its result, seconds and UTC window."""
    import torch

    torch.cuda.synchronize()
    t0, w0 = time.perf_counter(), time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (w0, time.time())


def _run_threads(form, weights, names, grid, q256, threads, rate_streams=True,
                 around=None):
    """The units on ``threads`` threads, each on its own CUDA stream.

    Plans, rate streams and captures are per thread, so each thread warms its
    own on one unit (serially: a capture is one at a time per process and any
    CUDA call from another thread during it faults it), then all of them wait.
    The clock starts, and ``around`` (a profiler, say) opens, only after every
    thread has warmed, and stops when the last has finished.  Returns the
    blobs in unit order, the seconds, and the UTC window.
    """
    import torch

    parts = split_units(len(weights), threads)
    blobs = [None] * len(weights)
    failed: list[BaseException] = []
    warm = threading.Lock()
    ready = threading.Barrier(threads + 1)
    go = threading.Event()

    def work(part):
        try:
            with torch.cuda.stream(torch.cuda.Stream()):
                with warm:
                    _encode(weights[part[0]], names[part[0]], grid, q256)
                    torch.cuda.current_stream().synchronize()
                ready.wait()
                go.wait()
                for i in part:
                    blobs[i] = _encode(weights[i], names[i], grid, q256)
                torch.cuda.current_stream().synchronize()
        except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
            failed.append(exc)
            ready.abort()

    with _knobs(form, rate_streams):
        pool_ = [threading.Thread(target=work, args=(p,), daemon=True) for p in parts]
        for thread in pool_:
            thread.start()
        try:
            ready.wait()
        except threading.BrokenBarrierError:
            go.set()
            for thread in pool_:
                thread.join()
            raise failed[0] if failed else RuntimeError("a worker thread did not warm")
        torch.cuda.synchronize()
        with (around() if around else contextlib.nullcontext()):
            t0, w0 = time.perf_counter(), time.time()
            go.set()
            for thread in pool_:
                thread.join()
            torch.cuda.synchronize()
            seconds, window = time.perf_counter() - t0, (w0, time.time())
    if failed:
        raise failed[0]
    return blobs, seconds, window


def _arm(name, form, threads, rate_streams, units, blobs, seconds, window, power):
    ups = units / seconds
    reading = power.window(*window)
    return {"name": name, "form": form, "threads": threads,
            "rate_streams": rate_streams, "units": units,
            "wall_s": round(seconds, 3), "units_per_s": round(ups, 4),
            "power": reading,
            "j_per_unit": (round(reading["mean_w"] / ups, 2)
                           if reading.get("points") else None),
            "utc_unix": list(window), "sha256": [_blob_sha(b) for b in blobs]}


def _plan_diag() -> list:
    """Each cached plan's width, batch count and step count, per shape.

    Names whether an arm runs launch-bound: batches times steps is the
    step-kernel launch count, width times low is the grid behind each launch.
    """
    from tessera import window_viterbi

    plans, _seen = window_viterbi._window_maps()
    out = []
    for key, plan in plans.items():
        (device, rows, cols, arity, size, rate, chunk, has_w, budget, scan,
         best, tile) = key
        out.append({
            "rows": rows, "cols": cols, "arity": arity, "size": size,
            "rate": rate, "chunk": chunk, "has_weights": has_w,
            "budget_bytes": budget, "best_form": best,
            "steps": plan.steps, "width": plan.width,
            "batches": plan.batches, "grid": list(plan.grid)})
    return out


def _kernel_table(prof, seconds, top=6):
    """What the device did during one profiled encode.

    ``device_over_wall`` is the summed device time of every kernel over the
    encode's wall clock: about 1 means a single serial chain with the device
    never idle, and above 1 means kernels overlapped.
    """
    rows = [(e.key, e.self_device_time_total or 0.0, e.count)
            for e in prof.key_averages()]
    rows = sorted((r for r in rows if r[1] > 0), key=lambda r: r[1], reverse=True)
    total_us = sum(us for _, us, _ in rows)
    return {"wall_s": round(seconds, 3), "device_s": round(total_us / 1e6, 3),
            "device_over_wall": round(total_us / 1e6 / seconds, 3),
            "top": [{"name": name[:96], "self_device_ms": round(us / 1e3, 3),
                     "count": count,
                     "us_per_launch": round(us / count, 2) if count else None}
                    for name, us, count in rows[:top]]}


def _profile(form, weights, names, grid, q256, threads=1):
    import torch
    from torch.profiler import ProfilerActivity, profile

    if threads == 1:
        with _knobs(form):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.perf_counter()
                _encode(weights[0], names[0], grid, q256)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
        return _kernel_table(prof, seconds)
    holder = {}

    @contextlib.contextmanager
    def around():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            holder["prof"] = prof
            yield

    _, seconds, _ = _run_threads(form, weights[:threads], names[:threads], grid,
                                 q256, threads, around=around)
    return _kernel_table(holder["prof"], seconds)


# -- one rung -----------------------------------------------------------------------

def _guard(rung, phase, fn):
    """Run one phase of a rung.  A failure is recorded under ``rung["errors"]``
    and the other phases still run, so one broken arm or profile does not throw
    away the readings already taken."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        rung.setdefault("errors", {})[phase] = f"{type(exc).__name__}: {exc}"
        print(f"  {rung['grid']}@{rung['q256']} {phase} FAILED: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


def _rung(args, rung, power):
    from tessera import window_viterbi
    from tessera.alphabet import BF16_GRID, E4M3_GRID

    grid_name, q256 = rung["grid"], rung["q256"]
    grid = BF16_GRID if grid_name == "bf16" else E4M3_GRID
    weights = _weights(args.rows, args.cols, args.units, q256)
    # The unit name is part of the wire's manifest, so every arm uses the same.
    names = [f"u{i}" for i in range(args.units)]
    window_viterbi.window_plan_cache_clear()

    def record(name, form, threads, rate_streams, blobs, seconds, window):
        arm = _arm(name, form, threads, rate_streams, args.units, blobs, seconds,
                   window, power)
        rung["arms"].append(arm)
        print(f"  {grid_name}@{q256} {name}: {arm['units_per_s']} u/s "
              f"{arm['power'].get('mean_w')} W", flush=True)

    def single(name, form, rate_streams=True):
        blobs, seconds, window = _timed(
            lambda: _run_single(form, weights, names, grid, q256, rate_streams))
        record(name, form, 1, rate_streams, blobs, seconds, window)

    def threaded(form):
        blobs, seconds, window = _run_threads(form, weights, names, grid, q256,
                                              args.threads)
        record(f"{form}/{args.threads}t", form, args.threads, True, blobs, seconds,
               window)

    # One unit per form on this thread pays the Triton compile and each shape's
    # persistent plan, so no timed arm measures a capture.
    for form in FORMS:
        _run_single(form, weights[:1], names[:1], grid, q256)
    seen = {}
    for form in arm_schedule(FORMS):
        seen[form] = seen.get(form, 0) + 1
        _guard(rung, f"{form}#{seen[form]}", lambda: single(f"{form}#{seen[form]}", form))
    if q256 % 256:
        for form in FORMS:
            _guard(rung, f"{form}/rs0", lambda: single(f"{form}/rs0", form, False))
    rung["plans"] = _guard(rung, "plans", _plan_diag)
    for form in FORMS:
        _guard(rung, f"{form}/{args.threads}t", lambda: threaded(form))

    if rung["arms"]:                 # no arm ran: ``errors`` says why, identity has no claim
        try:
            check_identity(rung["arms"])
            rung["identity"] = {"ok": True, "arms": len(rung["arms"]),
                                "units": args.units}
            rung["comparisons"] = _guard(rung, "comparisons",
                                         lambda: _comparisons(rung["arms"], args.threads))
        except IdentityError as exc:
            rung["identity"] = {"ok": False, "detail": str(exc)}
    if not args.no_profile:
        rung["profile"] = {}
        for form in FORMS:
            rung["profile"][form] = _guard(
                rung, f"profile/{form}",
                lambda: _profile(form, weights, names, grid, q256))
        if grid_name == "bf16":
            for form in FORMS:
                rung["profile"][f"{form}/{args.threads}t"] = _guard(
                    rung, f"profile/{form}/{args.threads}t",
                    lambda: _profile(form, weights, names, grid, q256,
                                     threads=args.threads))


def _comparisons(arms, threads):
    """Each axis against its own control, only for the arms that exist."""
    by = {a["name"]: a for a in arms}
    out = {}
    one = {}
    for form in FORMS:
        if f"{form}#1" in by and f"{form}#2" in by:
            one[form] = pool([by[f"{form}#1"], by[f"{form}#2"]])
    if len(one) == len(FORMS):
        out["pooled_single_thread"] = one
        out["best_over_front"] = compare(one["front"], one["best"])
        out["repeat_spread"] = {form: compare(by[f"{form}#1"], by[f"{form}#2"])
                                for form in FORMS}
    for key, name, base in (
            ("threads_over_one", lambda f: f"{f}/{threads}t", lambda f: one.get(f)),
            ("rate_streams_on_over_off", lambda f: f"{f}/rs0", None)):
        got = {}
        for form in FORMS:
            if form not in one or name(form) not in by:
                continue
            if base is not None:        # the control is the single-thread pool
                got[form] = compare(one[form], by[name(form)])
            else:                        # rs0 is the control; the pooled ON arms are the candidate
                got[form] = compare(by[name(form)], one[form])
        if got:
            out[key] = got
    return out


# -- the receipt a person reads ---------------------------------------------------------

def _num(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def render_tables(rec) -> str:
    """Markdown tables of one receipt, written by the code that wrote it, so a
    doc quotes the receipt and not a retyping of it."""
    lines = [f"{rec['host']}: {rec['device']}, torch {rec['torch']}, head "
             f"{str(rec.get('head'))[:10]}, idle "
             f"{_num((rec.get('idle_power') or {}).get('mean_w'))} W"]
    for rung in rec["rungs"]:
        lines += ["", f"{rung['grid']} q256 {rung['q256']}, {rung['units']} units of "
                      f"{rung['rows']}x{rung['cols']}", "",
                  "| arm | units/s | mean W | of 140 W | SM MHz | J/unit | Netdata W | groups |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in rung["arms"]:
            power, box = arm["power"], arm.get("netdata") or {}
            lines.append(
                f"| {arm['name']} | {arm['units_per_s']:.4f} | {_num(power.get('mean_w'))} "
                f"| {_num(power.get('mean_fraction_of_envelope'), 3)} "
                f"| {power.get('mean_sm_mhz', '-')} | {_num(arm['j_per_unit'], 1)} "
                f"| {_num(box.get('mean_w'))} | {box.get('accepted_groups', '-')} |")
        cmp = rung.get("comparisons") or {}
        pooled = cmp.get("pooled_single_thread") or {}
        if pooled:
            lines += ["", "| pooled ABBA | units/s | mean W | J/unit |", "|---|---:|---:|---:|"]
            for form, row in pooled.items():
                lines.append(f"| {form} | {row['units_per_s']:.4f} | {_num(row['mean_w'])} "
                             f"| {_num(row['j_per_unit'], 1)} |")
        for key, label in (("best_over_front", "best over front"),
                           ("repeat_spread", "second ABBA arm over first"),
                           ("threads_over_one", "threads over one"),
                           ("rate_streams_on_over_off", "rate streams on over off")):
            value = cmp.get(key)
            if not value:
                continue
            rows = value if "speedup" not in value else {"": value}
            for form, row in rows.items():
                lines.append(f"- {label}{' (' + form + ')' if form else ''}: speedup "
                             f"{row['speedup']:.3f}x, joule ratio {_num(row['joule_ratio'], 3)}")
        block = rung.get("abba_block") or {}
        if block:
            box, own = block.get("netdata") or {}, block.get("sampler") or {}
            lines.append(f"- ABBA block, Netdata {_num(box.get('mean_w'))} W over "
                         f"{box.get('accepted_groups', 0)} groups; in-process "
                         f"{_num(own.get('mean_w'))} W over {own.get('points', 0)} samples")
        if rung.get("profile"):
            lines += ["", "| profile (one unit) | wall s | device s | device/wall | top kernel "
                          "| launches | us each |", "|---|---:|---:|---:|---|---:|---:|"]
            for name, table in rung["profile"].items():
                if not table:
                    continue
                top = (table.get("top") or [{}])[0]
                lines.append(f"| {name} | {table['wall_s']} | {table['device_s']} "
                             f"| {table['device_over_wall']} | {top.get('name', '-')} "
                             f"| {top.get('count', '-')} | {top.get('us_per_launch', '-')} |")
        if rung.get("plans"):
            lines += ["", "| plan | rate | steps | width | batches | grid |", "|---|---:|---:|---:|---:|---|"]
            for plan in rung["plans"]:
                lines.append(f"| {'best' if plan['best_form'] else 'front'} | {plan['rate']} "
                             f"| {plan['steps']} | {plan['width']} | {plan['batches']} "
                             f"| {plan['grid']} |")
        if rung.get("errors"):
            lines += ["", f"errors: {rung['errors']}"]
        identity = rung.get("identity") or {}
        lines.append(f"- identity: {identity}")
    return "\n".join(lines)


# -- the run ---------------------------------------------------------------------------

def _git(*argv):
    here = os.path.dirname(os.path.abspath(__file__))
    done = subprocess.run(["git", *argv], capture_output=True, text=True, cwd=here)
    return done.stdout.strip() or None


def _publish(directory, receipt, host):
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = os.path.join(directory, f"receipt-{host}-{stamp}.json")
    with open(path, "x") as handle:          # never overwrite a published receipt
        handle.write(json.dumps(receipt, indent=1))
    return path


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.table:
        with open(args.table) as handle:
            print(render_tables(json.load(handle)))
        return 0

    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"error": "no CUDA device"}))
        return 2
    from tessera import window_viterbi

    props = torch.cuda.get_device_properties(0)
    host = socket.gethostname()
    rec: dict = {
        "schema": "tessera.ig619_viterbi_ab/1",
        "host": host, "device": torch.cuda.get_device_name(),
        "uuid": str(props.uuid), "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "head": _git("rev-parse", "HEAD"), "parent": _git("rev-parse", "HEAD^"),
        "fused_available": window_viterbi.fused_available(),
        "env": {k: os.environ.get(k, "") for k in _RECORDED_ENV},
        "envelope_w": ENVELOPE_W,
        "l2_cache_size": int(props.L2_cache_size),
        "l2_budget_bytes": window_viterbi._l2_budget(torch.device("cuda", 0)),
        "weights": {"kind": "synthetic", "dist": "N(0, 0.02^2), bf16",
                    "seed": "61900 + q256"},
        "args": vars(args),
        "rungs": [],
    }
    with Power() as power:
        time.sleep(args.idle_s)
        t_idle = time.time()
        time.sleep(args.idle_s)
        rec["idle_power"] = power.window(t_idle - args.idle_s, t_idle)
        for grid_name, q256 in args.rungs:
            print(f"rung {grid_name}@{q256}", flush=True)
            rung = {"grid": grid_name, "q256": q256, "rows": args.rows,
                    "cols": args.cols, "units": args.units, "arms": []}
            rec["rungs"].append(rung)
            _guard(rung, "rung", lambda: _rung(args, rung, power))
        time.sleep(_NETDATA_SETTLE_S)
        for rung in rec["rungs"]:
            arms = rung.get("arms", [])
            for arm in arms:
                arm["netdata"] = netdata_power("127.0.0.1", *arm["utc_unix"])
            abba = [a for a in arms if a["name"][-2:] in ("#1", "#2")]
            if abba:
                span = [min(a["utc_unix"][0] for a in abba),
                        max(a["utc_unix"][1] for a in abba)]
                rung["abba_block"] = {
                    "utc_unix": span, "netdata": netdata_power("127.0.0.1", *span),
                    "sampler": power.window(*span)}
    text = json.dumps(rec, indent=1)
    with open(args.out, "w") as handle:
        handle.write(text)
    if args.publish_dir:
        rec["published"] = _publish(args.publish_dir, rec, host)
        print(f"published {rec['published']}", flush=True)
    print(json.dumps(rec, separators=(",", ":")))
    if any(r.get("identity", {}).get("ok") is False for r in rec["rungs"]):
        return 3
    return 1 if any(r.get("errors") for r in rec["rungs"]) else 0


if __name__ == "__main__":
    sys.exit(main())
