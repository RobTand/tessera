"""One bounded paired A/B: the class-minimum step against the front step.

The candidate is ``TESSERA_WINDOW_BEST_FORM``.  Substituting the branch cost
into the class minimum removes the ``2^L`` front from the recurrence, so the
step stores ``2^(L-R)`` floats where it stored ``2^L``, and ``_layout`` sizes
the two resident arrays from the class width instead of the front width -- so
the same L2 budget admits ``2^R`` times as many columns, which is more blocks
over an unchanged serial chain.

Bytes are not the claim.  Front bytes at fixed ``L`` are provably rate
invariant, and this tree's own bench still moves 34/78/75% across the rate
axis and draws 20 W more doing it, so a byte ratio does not predict a second.
The claim is seconds and joules on the production shape.

Three arms, because two would leave the mechanism unattributed:

  front     the current fused step.
  best      the candidate, at the width its smaller resident set earns.
  best@w32  the candidate held to the front form's INTERNAL width, by
            narrowing the L2 budget for that arm alone.  The chunk stays the
            production chunk, so the epilogue's min, its sse accumulation and
            its traceback call count are identical to the other arms and the
            only thing that moved is the width.  A first attempt held the
            width by passing chunk=32 instead; that moved the outer loop too,
            and its sse said so.  The difference between this arm and
            ``front`` is the store and the front alone; the difference
            between it and ``best`` is the width.

Phases in ONE process, on ONE tensor, in this order:

  identity  every arm's states and ``sse`` compared to the reference as
            bytes, before any clock is read.  A faster wrong answer is not a
            result, so the timing never runs if this does not hold.
  timing    ABC blocks, unprofiled, each block long enough that a 1 Hz power
            sampler sees it.  Energy is integrated over each block's OWN
            interval, so the joules are that arm's joules and not a mean of
            everybody's.  The report is total work over total joules.

``--mode pbprofile`` runs one call per arm and nothing else, so PrismaBuild's
own ``--profile torch`` can wrap it; the arms are separable in that trace by
kernel name.  Profiling never shares a run with timing.

Usage::

    python experiments/window_viterbi_best_form_ab.py --mode time \\
        --out /mnt/shared/tessera-measurements/.../ab.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera import window_viterbi as wv  # noqa: E402
from tessera.alphabet import E4M3_GRID  # noqa: E402
from tessera.encode import (grid_vector_table, viterbi_window,  # noqa: E402
                            window_table)

# (arm, width_cap) -- width_cap None runs the width the arm's resident set
# earns; a number holds the arm to that internal width by narrowing the L2
# budget alone.  The chunk NEVER changes: chunk is the outer loop, and moving
# it moves the epilogue's min, its sse accumulation and its traceback call
# count as well as the plan's width, which is three changes in a control that
# is supposed to isolate one.
ARMS = (("front", None), ("best", None), ("best@w32", 32))

# (name, window_bits, rate, arity, rows, cols) -- the production groups.
CONFIGS = [
    ("R3", 14, 3, 1, 4096, 192),
    ("R4", 14, 4, 1, 4096, 64),
]
ENVELOPE_W = 140.0


class Power:
    """nvidia-smi at 1 Hz, every sample stamped, so energy is an integral.

    A mean power over one arm's repeats multiplied by another arm's seconds
    is not that arm's energy.  Keeping the stamps lets each block's joules be
    integrated over that block's own interval.
    """

    def __init__(self, hz: float = 1.0):
        self.samples: list[tuple[float, float]] = []
        self._hz = hz
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.samples.append((time.time(), float(out.splitlines()[0])))
            except Exception:
                pass
            self._stop.wait(1.0 / self._hz)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=3)

    def energy(self, t0: float, t1: float):
        """Trapezoidal joules over ``[t0, t1]``, refused unless the interval is bracketed.

        Integrating between the samples that happen to fall inside an
        interval covers less than the interval whenever the first sample is
        after ``t0`` or the last is before ``t1`` -- and the work denominator
        does not shrink with it, so a partially covered block reports a work
        per joule that is too GOOD.  A block with no sample at or before
        ``t0`` and none at or after ``t1`` therefore returns no energy at all
        rather than a flattering one.  ``covered`` reports the fraction
        anyway, so a marginal block is visible instead of silently dropped.
        """
        pts = sorted((t, w) for t, w in self.samples if t0 - 5.0 <= t <= t1 + 5.0)
        before = [p for p in pts if p[0] <= t0]
        after = [p for p in pts if p[0] >= t1]
        inside = [p for p in pts if t0 <= p[0] <= t1]
        use = ([before[-1]] if before else []) + inside + ([after[0]] if after else [])
        bracketed = bool(before and after)
        if len(use) < 2:
            return None, dict(bracketed=False, covered=0.0, samples=[])
        j = 0.0
        covered = 0.0
        for (ta, wa), (tb, wb) in zip(use, use[1:]):
            a, b = max(ta, t0), min(tb, t1)
            if b <= a:
                continue
            # linear in power between samples
            wl = wa + (wb - wa) * ((a - ta) / (tb - ta)) if tb > ta else wa
            wr = wa + (wb - wa) * ((b - ta) / (tb - ta)) if tb > ta else wb
            j += 0.5 * (wl + wr) * (b - a)
            covered += b - a
        span = max(t1 - t0, 1e-9)
        meta = dict(bracketed=bracketed, covered=round(covered / span, 4),
                    samples=[[round(t, 3), w] for t, w in use])
        if not bracketed:
            return None, meta
        return j, meta


def _inputs(window_bits, rate, arity, rows, cols, dev):
    """The encoder's own table, not a random one.

    ``window_table`` is a seeded permutation of equal-mass quantiles snapped
    to the grid, so its value distribution -- and therefore the branch costs
    the scan compares -- is the distribution production compares.
    """
    grid = E4M3_GRID
    vectors = grid_vector_table(grid, dev)
    codes = window_table(grid, window_bits, sigma=1.0, seed=0, device=dev)
    table = vectors[codes.long()].contiguous()
    if arity != table.shape[1]:
        table = table[:, :arity].contiguous()
    torch.manual_seed(0)
    targets = torch.randn(rows, cols, device=dev)
    weights = torch.rand(rows, cols, device=dev) + 0.5
    return targets, table, weights


def _select(arm: str):
    """Pick the spelling.  One environment write; the plan cache is keyed on it."""
    os.environ[wv._BEST_FORM_ENV] = "0" if arm == "front" else "1"
    os.environ[wv._GRAPH_ENV] = "1"


def _budget_for(arm, width_cap, window_bits, rate):
    """The byte budget that makes ``_layout`` choose ``width_cap``.

    ``_layout`` takes ``budget // (2 * resident * 4)``, so a budget of exactly
    ``2 * resident * 4 * width_cap`` lands on that width.  ``_L2_BUDGET`` is
    in the plan-cache key, so the two budgets coexist in one process, and the
    module states outright that it is a measurement knob and never a
    correctness one: every value returns the same bytes, sse included.  This
    control therefore has to match the other arms on sse as well as states,
    and the harness asserts it.
    """
    if width_cap is None:
        return None
    size = 1 << window_bits
    resident = size if arm == "front" else (size >> rate)
    return 2 * resident * 4 * width_cap


def _call(arm, width_cap, targets, vectors, window_bits, rate, weights):
    _select(arm)
    saved = wv._L2_BUDGET
    wv._L2_BUDGET = _budget_for(arm, width_cap, window_bits, rate) or saved
    try:
        return viterbi_window(targets, vectors, window_bits, rate,
                              weights=weights, impl="fused")
    finally:
        wv._L2_BUDGET = saved


class _Spy:
    """Keep the ``CompiledKernel`` a launch returns, so regs are the real regs."""

    def __init__(self, jit):
        self.jit = jit
        self.compiled = []

    def __getitem__(self, grid):
        inner = self.jit[grid]

        def call(*args, **kwargs):
            out = inner(*args, **kwargs)
            self.compiled.append(out)
            return out

        return call


def _registers(arm, width_cap, targets, vectors, window_bits, rate, weights):
    ks = list(wv._kernels())
    slot = 0 if arm == "front" else 5              # _step / _step_best
    spy = _Spy(ks[slot])
    held = list(ks)
    held[slot] = spy
    saved = wv._CACHE.get("k")
    wv._CACHE["k"] = tuple(held)
    wv.window_plan_cache_clear()
    try:
        _call(arm, width_cap, targets, vectors, window_bits, rate, weights)
    finally:
        wv._CACHE["k"] = saved
        wv.window_plan_cache_clear()
    if not spy.compiled:
        return None
    ck = spy.compiled[-1]
    return dict(n_regs=getattr(ck, "n_regs", None),
                n_spills=getattr(ck, "n_spills", None),
                shared=getattr(getattr(ck, "metadata", None), "shared", None))


def _plan_shape(arm, width_cap, window_bits, rate, cols, dev):
    size = 1 << window_bits
    resident = size if arm == "front" else (size >> rate)
    saved = wv._L2_BUDGET
    wv._L2_BUDGET = _budget_for(arm, width_cap, window_bits, rate) or saved
    try:
        _, width, descs = wv._layout(dev, size, cols, 512, resident)
    finally:
        wv._L2_BUDGET = saved
    return dict(resident=resident, width=width, batches=len(descs),
                l2_budget_bytes=_budget_for(arm, width_cap, window_bits, rate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("time", "pbprofile"), default="time")
    ap.add_argument("--blocks", type=int, default=3, help="ABC blocks")
    ap.add_argument("--min-block-s", type=float, default=5.0,
                    help="inner repeats are sized so a block runs at least this "
                         "long, since a 1 Hz sampler cannot see 40 ms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args()

    dev = "cuda"
    records = []
    want = [c for c in CONFIGS if not a.configs or c[0] in a.configs]
    for name, L, R, arity, rows, cols in want:
        targets, vectors, weights = _inputs(L, R, arity, rows, cols, dev)
        rec = dict(config=name, window_bits=L, rate=R, arity=arity, rows=rows,
                   cols=cols, weighted=True, table="E4M3_GRID window_table "
                   "sigma=1.0 seed=0",
                   plan={arm: _plan_shape(arm, width_cap, L, R, cols, dev)
                         for arm, width_cap in ARMS})

        # -- identity, before any clock --------------------------------------
        wv.window_plan_cache_clear()
        ref_states, ref_sse = viterbi_window(targets, vectors, L, R,
                                             weights=weights, impl="reference")
        rec["identity"] = {}
        for arm, width_cap in ARMS:
            s, e = _call(arm, width_cap, targets, vectors, L, R, weights)
            rec["identity"][arm] = dict(
                states_equal=bool(torch.equal(s, ref_states)),
                sse=e.hex(),
                # Every arm runs the production chunk, so every arm's sse is
                # summed in the reference's order and compared as bytes.  A
                # control that could not be compared here was a control that
                # had changed more than the one thing it names.
                sse_equal=bool(e == ref_sse))
            del s
        del ref_states
        torch.cuda.empty_cache()
        if not all(v["states_equal"] and v["sse_equal"]
                   for v in rec["identity"].values()):
            rec["verdict"] = "an arm does not return the reference's answer"
            records.append(rec)
            continue

        if a.mode == "pbprofile":
            # PrismaBuild's torch mode is a contract, not a wrapper: the
            # profiler is in-process by construction, so the action exports
            # its own Chrome trace to the path the fleet names.  An action
            # that asks for the mode and writes nothing fails, and rightly:
            # a receipt filed for a profile-less run poisons the key.
            from torch.profiler import ProfilerActivity, profile, record_function
            out = os.environ.get("PRISMABUILD_PROFILE_TORCH_OUT")
            if not out:
                raise SystemExit(
                    "pbprofile mode needs PRISMABUILD_PROFILE_TORCH_OUT; run "
                    "this under pbrun --profile torch")
            for arm, width_cap in ARMS:                      # capture, untraced
                _call(arm, width_cap, targets, vectors, L, R, weights)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                for arm, width_cap in ARMS:
                    # Both candidate arms run _step_best, so the kernel name
                    # alone would not separate them; the marker does.
                    with record_function(f"arm:{arm}"):
                        _call(arm, width_cap, targets, vectors, L, R, weights)
                        torch.cuda.synchronize()
            # One action, one config, one trace.  The first version ran both
            # configs and exported both to the SAME fleet path, so R4
            # overwrote R3 and the receipt named a trace that was no longer
            # the run it was filed for.  A profiled run is keyed on being
            # profiled, so a trace that is silently the other config's is
            # worse than none.
            if len(want) != 1:
                raise SystemExit(
                    "pbprofile takes exactly one --configs entry: the fleet "
                    f"names one path and {len(want)} configs would overwrite "
                    "each other in it")
            keep = Path(a.out).with_name(f"{name}.chrome-trace.json.gz")
            keep.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(keep))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(keep, out)
            rec["pb_profile"] = dict(
                trace=out, retained=str(keep), exists=Path(out).exists(),
                bytes=keep.stat().st_size,
                sha256=hashlib.sha256(keep.read_bytes()).hexdigest())
            rec["kernels"] = {
                ev.key[:56]: dict(us=round(ev.self_device_time_total, 1),
                                  calls=ev.count)
                for ev in sorted(prof.key_averages(),
                                 key=lambda e: -e.self_device_time_total)[:10]
                if ev.self_device_time_total > 0}
            print(json.dumps(rec), flush=True)
            records.append(rec)
            continue

        rec["registers"] = {arm: _registers(arm, width_cap, targets, vectors, L, R,
                                            weights)
                            for arm, width_cap in ARMS}

        # -- timing ----------------------------------------------------------
        # One clear, then every arm's plan is built and NOTHING clears again:
        # clearing per repeat would time a graph capture and call it a step.
        wv.window_plan_cache_clear()
        single = {}
        for arm, width_cap in ARMS:
            _call(arm, width_cap, targets, vectors, L, R, weights)   # capture
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _call(arm, width_cap, targets, vectors, L, R, weights)
            torch.cuda.synchronize()
            single[arm] = time.perf_counter() - t0
        inner = {arm: max(1, int(a.min_block_s / single[arm]) + 1)
                 for arm, _ in ARMS}
        rec["inner_repeats"] = inner
        rec["single_call_s"] = {k: round(v, 5) for k, v in single.items()}

        power = Power().start()
        blocks = {arm: [] for arm, _ in ARMS}
        try:
            for _ in range(a.blocks):
                for arm, width_cap in ARMS:
                    torch.cuda.synchronize()
                    w0, t0 = time.time(), time.perf_counter()
                    for _ in range(inner[arm]):
                        _call(arm, width_cap, targets, vectors, L, R, weights)
                    torch.cuda.synchronize()
                    dt, w1 = time.perf_counter() - t0, time.time()
                    blocks[arm].append(dict(seconds=dt, wall_start=w0,
                                            wall_end=w1, calls=inner[arm]))
        finally:
            power.stop()

        steps = rows // arity
        work_per_call = steps * cols * (1 << L)      # branch evaluations
        rec["work_per_call"] = work_per_call
        rec["timing"] = {}
        for arm, _ in ARMS:
            bs = blocks[arm]
            for b in bs:
                j, meta = power.energy(b["wall_start"], b["wall_end"])
                b["joules"] = round(j, 2) if j else None
                b["power"] = meta
                b["power_w_mean"] = (round(j / b["seconds"], 2)
                                     if j and b["seconds"] else None)
                b["seconds_per_call"] = round(b["seconds"] / b["calls"], 5)
            secs = [b["seconds"] for b in bs]
            calls = sum(b["calls"] for b in bs)
            # Only blocks whose energy was measured over their whole interval
            # go into work per joule, and their work is the only work counted.
            paid = [b for b in bs if b["joules"]]
            total_j = sum(b["joules"] for b in paid) if paid else None
            total_work = work_per_call * sum(b["calls"] for b in paid)
            energy_seconds = sum(b["seconds"] for b in paid)
            rec["timing"][arm] = dict(
                blocks=[{k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in b.items()} for b in bs],
                seconds_per_call_min=round(min(secs) / bs[0]["calls"], 5),
                seconds_per_call_mean=round(statistics.fmean(secs) / bs[0]["calls"], 5),
                seconds_per_call_median=round(statistics.median(secs) / bs[0]["calls"], 5),
                total_calls=calls, total_seconds=round(sum(secs), 4),
                energy_blocks=len(paid), energy_seconds=round(energy_seconds, 4),
                total_joules=round(total_j, 1) if total_j else None,
                power_w_mean=(round(total_j / energy_seconds, 2)
                              if total_j else None),
                power_envelope_frac=(round(total_j / energy_seconds / ENVELOPE_W, 3)
                                     if total_j else None),
                work_per_joule=(round(total_work / total_j, 1) if total_j else None))
        f = rec["timing"]["front"]
        for arm in ("best", "best@w32"):
            t = rec["timing"][arm]
            rec[f"speedup_{arm}"] = round(
                f["seconds_per_call_median"] / t["seconds_per_call_median"], 4)
            if t["work_per_joule"] and f["work_per_joule"]:
                rec[f"work_per_joule_ratio_{arm}"] = round(
                    t["work_per_joule"] / f["work_per_joule"], 4)
        records.append(rec)
        del targets, vectors, weights
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)

    with open(a.out, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
