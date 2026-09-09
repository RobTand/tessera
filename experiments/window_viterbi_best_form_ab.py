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
The claim is seconds and joules on the production shape, and that is what this
measures.

Three phases in ONE process, on ONE tensor, in this order:

  identity  every arm's states and ``sse`` compared to the reference as
            bytes, before any clock is read.  A faster wrong answer is not a
            result, so the timing never runs if this does not hold.
  timing    ABAB, unprofiled, both arms alike, with the board's power beside
            the seconds.  Interleaved because the box warms.
  profile   one rep per arm, each inside its OWN ``torch.profiler`` context,
            traces written out and hashed.  Last, so the profiler cannot
            reach the seconds above; both arms profiled, neither compared
            against an unprofiled one.

Register counts come off the launched ``CompiledKernel``, not from prose:
"lower register pressure" is a claim about a number, so the number is here.

Usage::

    python experiments/window_viterbi_best_form_ab.py \
        --out /home/rob/tmp/triage/best_form_ab.json \
        --trace-dir /home/rob/tmp/triage/best_form_traces
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tessera import window_viterbi as wv  # noqa: E402
from tessera.encode import viterbi_window  # noqa: E402
from window_viterbi_bench import _Power, neighbours, table_for  # noqa: E402

ARMS = ("front", "best")

# (name, window_bits, rate, arity, rows, cols) -- the production groups.
CONFIGS = [
    ("R3", 14, 3, 1, 4096, 192),
    ("R4", 14, 4, 1, 4096, 64),
]


def _inputs(window_bits, rate, arity, rows, cols, dev):
    torch.manual_seed(0)
    vectors = table_for(window_bits, arity, dev)
    scale = vectors.abs().max() / 4
    targets = torch.randn(rows, cols, device=dev) * scale
    weights = torch.rand(rows, cols, device=dev) + 0.5
    return targets, vectors, weights


def _select(arm: str, clear: bool = False):
    """Pick the spelling.

    The plan cache is keyed on the knob, so both plans coexist and switching
    arms is one environment write.  ``clear`` is for the phases that want a
    cold plan; the timing loop must NOT use it.  The first run of this A/B
    cleared per repetition, so every timed rep re-planned and re-captured its
    graphs -- and the front form, at six batches to the best form's one, paid
    six captures to its one.  That is a real cost of the front form, but it is
    a per-call cost only on a cold cache, and the encoder's cache is not cold.
    Clearing per rep timed the capture and called it the step.
    """
    os.environ[wv._BEST_FORM_ENV] = "1" if arm == "best" else "0"
    os.environ[wv._GRAPH_ENV] = "1"
    if clear:
        wv.window_plan_cache_clear()


def _run(arm, targets, vectors, window_bits, rate, weights):
    _select(arm, clear=True)
    return viterbi_window(targets, vectors, window_bits, rate,
                          weights=weights, impl="fused")


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


def _registers(arm, targets, vectors, window_bits, rate, weights):
    ks = list(wv._kernels())
    slot = 5 if arm == "best" else 0                # _step_best / _step
    spy = _Spy(ks[slot])
    held = list(ks)
    held[slot] = spy
    saved = wv._CACHE.get("k")
    wv._CACHE["k"] = tuple(held)
    try:
        _run(arm, targets, vectors, window_bits, rate, weights)
    finally:
        wv._CACHE["k"] = saved
        wv.window_plan_cache_clear()
    if not spy.compiled:
        return None
    ck = spy.compiled[-1]
    return dict(n_regs=getattr(ck, "n_regs", None),
                n_spills=getattr(ck, "n_spills", None),
                shared=getattr(getattr(ck, "metadata", None), "shared", None))


def _plan_shape(arm, window_bits, rate, cols, dev):
    size = 1 << window_bits
    resident = (size >> rate) if arm == "best" else size
    _, width, _ = wv._layout(dev, size, cols, 512, resident)
    return dict(resident=resident, width=width,
                batches=(cols + width - 1) // width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="ABAB pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args()

    dev = "cuda"
    records = []
    for name, L, R, arity, rows, cols in CONFIGS:
        if a.configs and name not in a.configs:
            continue
        targets, vectors, weights = _inputs(L, R, arity, rows, cols, dev)
        rec = dict(config=name, window_bits=L, rate=R, arity=arity, rows=rows,
                   cols=cols, chunk=512, weighted=True,
                   neighbours=neighbours(),
                   plan={arm: _plan_shape(arm, L, R, cols, dev) for arm in ARMS})

        # -- identity, before any clock -----------------------------------
        ref_states, ref_sse = viterbi_window(targets, vectors, L, R,
                                             weights=weights, impl="reference")
        answers = {}
        for arm in ARMS:
            s, e = _run(arm, targets, vectors, L, R, weights)
            answers[arm] = (s, e)
        rec["identity"] = {
            arm: dict(states_equal=bool(torch.equal(answers[arm][0], ref_states)),
                      sse=answers[arm][1].hex(), sse_equal=answers[arm][1] == ref_sse)
            for arm in ARMS}
        rec["identity"]["arms_agree"] = bool(
            torch.equal(answers["front"][0], answers["best"][0])
            and answers["front"][1] == answers["best"][1])
        del answers, ref_states
        torch.cuda.empty_cache()
        if not (rec["identity"]["arms_agree"]
                and all(rec["identity"][arm]["states_equal"]
                        and rec["identity"][arm]["sse_equal"] for arm in ARMS)):
            rec["verdict"] = "the arms do not agree; no timing was taken"
            records.append(rec)
            continue

        rec["registers"] = {arm: _registers(arm, targets, vectors, L, R, weights)
                            for arm in ARMS}

        # -- timing, unprofiled, interleaved ------------------------------
        # Warm both plans, then never clear again: the timed reps replay the
        # captured graphs the encoder replays, not a fresh capture.
        for arm in ARMS:
            _run(arm, targets, vectors, L, R, weights)
            _select(arm)
            viterbi_window(targets, vectors, L, R, weights=weights, impl="fused")
        torch.cuda.synchronize()
        timing = {arm: [] for arm in ARMS}
        power = {arm: [] for arm in ARMS}
        for _ in range(a.reps):
            for arm in ARMS:
                _select(arm)
                torch.cuda.synchronize()
                with _Power() as p:
                    t0 = time.perf_counter()
                    viterbi_window(targets, vectors, L, R, weights=weights,
                                   impl="fused")
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                timing[arm].append(dt)
                if p.samples:
                    power[arm].append(statistics.fmean(p.samples))
        steps = rows // arity
        work = steps * cols * (1 << L)                     # branch evaluations
        rec["timing"] = {}
        for arm in ARMS:
            s = min(timing[arm])
            w = statistics.fmean(power[arm]) if power[arm] else None
            rec["timing"][arm] = dict(
                seconds=[round(x, 4) for x in timing[arm]],
                seconds_min=round(s, 4),
                seconds_median=round(statistics.median(timing[arm]), 4),
                power_w_mean=round(w, 2) if w else None,
                power_envelope_frac=round(w / 140.0, 3) if w else None,
                work_per_joule=round(work / (s * w), 1) if w else None)
        f, b = rec["timing"]["front"], rec["timing"]["best"]
        rec["speedup_min"] = round(f["seconds_min"] / b["seconds_min"], 4)
        if f["work_per_joule"] and b["work_per_joule"]:
            rec["work_per_joule_ratio"] = round(
                b["work_per_joule"] / f["work_per_joule"], 4)

        # -- profile, last, one fresh context per arm ----------------------
        if a.trace_dir:
            os.makedirs(a.trace_dir, exist_ok=True)
            from torch.profiler import ProfilerActivity, profile
            rec["profile"] = {}
            for arm in ARMS:
                _select(arm)
                viterbi_window(targets, vectors, L, R, weights=weights,
                               impl="fused")
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.CUDA]) as prof:
                    t0 = time.perf_counter()
                    viterbi_window(targets, vectors, L, R, weights=weights,
                                   impl="fused")
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                path = os.path.join(a.trace_dir, f"{name}-{arm}.json")
                prof.export_chrome_trace(path)
                table = os.path.join(a.trace_dir, f"{name}-{arm}.txt")
                with open(table, "w") as fh:
                    fh.write(prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=15,
                        max_name_column_width=64))
                kernels = {}
                for ev in prof.key_averages():
                    if ev.self_device_time_total > 0:
                        kernels[ev.key[:48]] = dict(
                            us=round(ev.self_device_time_total, 1),
                            calls=ev.count)
                rec["profile"][arm] = dict(
                    seconds_profiled=round(dt, 4),
                    trace=path,
                    trace_sha256=hashlib.sha256(
                        open(path, "rb").read()).hexdigest(),
                    table=table,
                    kernels=dict(sorted(kernels.items(),
                                        key=lambda kv: -kv[1]["us"])[:8]))
        records.append(rec)
        del targets, vectors, weights
        torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)

    with open(a.out, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
