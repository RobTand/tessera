"""The dense-GEMM algorithm sweep's decision rules, torch-free (#806, #850).

One owner for every rule a sweep decision reads: which sample statistic an
A/B arm is decided on, how the non-bitwise differing fractions are
summarized, when a reference qualifies a pick for the serving table, and
which input paths a run may stat or open.  The sweep driver imports this
module; tests import it directly by path, so the rules are exercised
without torch or cuBLASLt.

This extraction preserves the recovered sweep's behavior exactly (the OLD
rules); the #850 fixes change the bodies, one commit per rule.
"""
from __future__ import annotations


def abba_summary(reference_observations, pick_observations):
    """Summarize an interleaved A/B group: (ref_med, pick_med, saving_ms or None).

    OLD RULE (pre-#850): the ``sorted(x)[len(x) // 2]`` upper order statistic.
    """
    ref_med = sorted(reference_observations)[len(reference_observations) // 2]
    pick_med = sorted(pick_observations)[len(pick_observations) // 2]
    return ref_med, pick_med, (ref_med - pick_med if pick_med < ref_med else None)


def non_bitwise_fraction_summary(diffs):
    """Summarize the non-bitwise candidates' differing fractions.

    OLD RULE (pre-#850): the ``sorted(x)[len(x) // 2]`` upper order statistic,
    and no raw population retained.
    """
    ordered = sorted(diffs)
    return dict(count=len(ordered), min=ordered[0] if ordered else None,
                median=ordered[len(ordered) // 2] if ordered else None)


def reference_qualified(matches_served_kernel, deterministic):
    """When a reference qualifies picks for the serving table.

    OLD RULE (pre-#850): recorded only -- every reference qualifies.
    """
    return True


def admit_pick(matches_served_kernel, deterministic, saving_ms):
    """Admit a pick to the serving table: (admitted, refusal).

    The refusal names the unmet reference predicate whenever the reference is
    unqualified, whether or not a pick exists to refuse.

    OLD RULE (pre-#850): no gate -- a faster pick is admitted regardless of
    its reference.
    """
    return saving_ms is not None, None


def resolve_real_input(allow_capture, capture, rows, k, exists, load):
    """Resolve the ``real`` distribution from a recorded capture.

    ``capture`` is ``(name, cols, path)`` or None; ``exists``/``load`` are
    injected so a run (and these tests) observe exactly which paths are
    stat'ed or opened.

    OLD RULE (pre-#850): ``allow_capture`` is accepted but not consulted --
    a present capture is always opened.
    """
    del allow_capture
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
