"""The dense-GEMM algorithm sweep's decision rules, torch-free (#806, #850).

One owner for every rule a sweep decision reads: which sample statistic an
A/B arm is decided on, how the non-bitwise differing fractions are
summarized, when a reference qualifies a pick for the serving table, and
which input paths a run may stat or open.  The sweep driver imports this
module; tests import it directly by path, so the rules are exercised
without torch or cuBLASLt.
"""
from __future__ import annotations

import statistics


def abba_summary(reference_observations, pick_observations):
    """Summarize an interleaved A/B group: (ref_med, pick_med, saving_ms or None).

    The conventional sample median, like the bench owner's ``summarize()``: the
    mean of the two middle order statistics for an even group.  The recovered
    ``sorted(x)[len(x) // 2]`` is an upper order statistic, which reports the
    slower arm of an even-sized ABBA group as if it were the middle.
    """
    ref_med = statistics.median(reference_observations)
    pick_med = statistics.median(pick_observations)
    return ref_med, pick_med, (ref_med - pick_med if pick_med < ref_med else None)


def non_bitwise_fraction_summary(diffs):
    """Summarize the non-bitwise candidates' differing fractions.

    The conventional sample median, with the raw ordered population retained
    so the summary can be re-derived without re-running.
    """
    ordered = sorted(diffs)
    return dict(count=len(ordered), min=ordered[0] if ordered else None,
                median=statistics.median(ordered) if ordered else None, raw=ordered)


def reference_qualified(matches_served_kernel, deterministic):
    """A reference qualifies only when it is the kernel the served shape
    dispatched AND it repeats bit-deterministically.
    """
    return bool(matches_served_kernel and deterministic)


def admit_pick(matches_served_kernel, deterministic, saving_ms):
    """Admit a pick to the serving table: (admitted, refusal).

    Fail closed: only a faster pick under a qualified reference is admitted.
    The refusal names the unmet reference predicate whenever the reference is
    unqualified, whether or not a pick exists to refuse -- a mismatch stays
    diagnostic and the table stays closed.
    """
    if not reference_qualified(matches_served_kernel, deterministic):
        return False, ("reference kernel does not match the served kernel"
                       if not matches_served_kernel
                       else "reference does not repeat bit-deterministically")
    if saving_ms is None:
        return False, None
    return True, None


def resolve_real_input(allow_capture, capture, rows, k, exists, load):
    """Resolve the ``real`` distribution: recorded capture, named gap, or not_measured.

    ``capture`` is ``(name, cols, path)`` or None; ``exists``/``load`` are
    injected so a run (and these tests) observe exactly which paths are
    stat'ed or opened.  A synthetic-only run (``allow_capture`` False) is a
    screen: neither ``exists`` nor ``load`` is called for any capture path,
    and the real distribution is reported as ``not_measured``.
    """
    if not allow_capture:
        return {"real": None, "real_source": "not_measured"}
    if capture is None:
        return {"real": None, "real_source": None}
    name, cols, path = capture
    if not exists(path):
        return {"real": None, "real_source": f"missing {path}"}
    x = load(path)
    x = x[:, :cols] if cols else x
    if x.shape[1] != k or x.shape[0] < rows:
        return {"real": None,
                "real_source": f"{name}: shape {tuple(x.shape)} does not cover [{rows}, {k}]"}
    return {"real": x[:rows], "real_source": name + (f" (columns 0:{cols})" if cols else "")}
