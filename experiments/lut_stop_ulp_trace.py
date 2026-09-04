"""Issue #106: does ``_fit_lut``'s eps stop test decide anything a real encode reaches?

``2f6a15a`` replaced ``_fit_lut``'s swap accept test ``cost < base * (1 - 1e-9)``
with ``cost < base * (1 - torch.finfo(cost.dtype).eps)``.  On the float32 costs
production accumulates that is ~119x looser, so it can only *reject* swaps the
literal accepted -- and the set it rejects is exactly one thing:

    ``base`` is a float32 value widened to a Python float, so
    ``base * (1 - step)`` is exact in float64 and the test accepts iff
    ``base - cost > base * step``.  ``base * eps`` is one ulp of ``base``'s
    binade, and ``base * 1e-9`` is far below the smallest step float32 can
    express at all (``eps / 2 = 5.96e-8`` relative).

    So the whole behavioural difference is: **a trial that improves the running
    cost by at most one ulp of its binade was accepted before and is rejected
    now** -- one representable step inside a binade, two at its foot, where the
    steps below are half-width.  Everything larger is taken by both tests.
    ``tests/test_lut_stop_ulp_band.py`` pins that enumeration.

That is a derivation, not a measurement, so this script measures it.  It wraps
``encode._lut_cost`` and mirrors the swap loop's state -- ``_fit_lut`` is the
only caller, and the pass-start call is the one whose table equals the table the
loop currently holds, while every trial differs from it in exactly one entry --
and for every trial records both verdicts:

    old: cost < base * (1 - 1e-9)          new: cost < base * (1 - eps)

The mirror follows the **new** verdict, which is what the running code does, so
it stays in lockstep with the real encode whatever it finds.  Zero disagreements
over a corpus therefore proves the two encoders take the identical trajectory on
that corpus -- identical state, identical decision, by induction -- rather than
merely hashing equal once.  Each disagreement is additionally checked against
the derivation: its improvement must fall inside the one-ulp band.

The corpus is ``audit_byte_baseline.py``'s own encode matrix and release rows
(the LUT-plane ones are all that reach ``_fit_lut`` at all) plus, with
``--full-unit``, whole real Qwen3-0.6B Linears with their real Hessian at an
E2M1x2 sub-cap rung.

    python experiments/lut_stop_ulp_trace.py trace.json [--full-unit]

CPU only, by construction -- it runs while a GPU measurement is in flight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tessera.encode as encode

OLD_LITERAL = 1e-9


class Tracer:
    """Mirror of ``_fit_lut``'s swap loop, reading every accept decision twice."""

    def __init__(self) -> None:
        self.depth = 0
        self.base: "float | None" = None
        self.current: "torch.Tensor | None" = None
        self.rows: list = []
        self.invocations = 0
        self.trials = 0
        self.pass_starts = 0
        self.dtypes: dict = {}
        self.accepts_new = 0
        self.accepts_old = 0
        self.disagreements: list = []
        self.min_positive_rel = math.inf
        self.label = ""

    # -- lifecycle -------------------------------------------------------
    def begin(self) -> None:
        # Nested ``_fit_lut`` never happens, but a reset that assumed so
        # would be a silent lie if it ever did.
        assert self.depth == 0, "nested _fit_lut: the mirror's state is not a stack"
        self.depth += 1
        self.base, self.current = None, None
        self.invocations += 1

    def end(self) -> None:
        self.depth -= 1
        self.base, self.current = None, None

    # -- the decision ----------------------------------------------------
    def observe(self, table: torch.Tensor, cost_t: torch.Tensor) -> None:
        if self.depth == 0:                 # not inside _fit_lut: nothing to mirror
            return
        dtype = str(cost_t.dtype)
        self.dtypes[dtype] = self.dtypes.get(dtype, 0) + 1
        cost = float(cost_t)
        if self.current is None or torch.equal(table, self.current):
            # ``base_cost = _lut_cost(s, w, table)`` at the top of a pass.
            self.base, self.current = cost, table.clone()
            self.pass_starts += 1
            return
        self.trials += 1
        base = self.base
        eps = torch.finfo(cost_t.dtype).eps
        old = cost < base * (1.0 - OLD_LITERAL)
        new = cost < base * (1.0 - eps)
        if old:
            self.accepts_old += 1
        if new:
            self.accepts_new += 1
        if cost < base:
            rel = (base - cost) / base if base > 0 else math.inf
            self.min_positive_rel = min(self.min_positive_rel, rel)
        if old != new:
            self.disagreements.append({
                "label": self.label,
                "invocation": self.invocations,
                "base": repr(base),
                "cost": repr(cost),
                "rel_improvement": repr((base - cost) / base if base else None),
                "old_accepts": bool(old),
                "new_accepts": bool(new),
                "dtype": dtype,
                # The band the derivation predicts: at most one ulp of base's
                # binade.  A difference outside it would say the derivation is
                # wrong, which is the point of checking rather than asserting.
                "within_one_ulp_band": bool(0.0 < base - cost <= base * eps),
            })
        if new:                              # the running code's own verdict
            self.base, self.current = cost, table.clone()

    def summary(self) -> dict:
        return {
            "fit_lut_invocations": self.invocations,
            "swap_pass_starts": self.pass_starts,
            "swap_trials": self.trials,
            "cost_dtypes": self.dtypes,
            "accepts_under_new_eps_test": self.accepts_new,
            "accepts_under_old_1e-9_test": self.accepts_old,
            "decisions_that_differ": len(self.disagreements),
            "all_differences_within_one_ulp_band": all(
                d["within_one_ulp_band"] for d in self.disagreements
            ),
            "smallest_positive_relative_improvement_seen": (
                repr(self.min_positive_rel) if self.min_positive_rel < math.inf else None
            ),
            "differences": self.disagreements[:64],
        }


def install(tracer: Tracer):
    orig_fit, orig_cost = encode._fit_lut, encode._lut_cost

    def traced_fit(*args, **kwargs):
        tracer.begin()
        try:
            return orig_fit(*args, **kwargs)
        finally:
            tracer.end()

    def traced_cost(targets, weights, table):
        out = orig_cost(targets, weights, table)
        tracer.observe(table, out)
        return out

    encode._fit_lut, encode._lut_cost = traced_fit, traced_cost
    return orig_fit, orig_cost


def run_audit_matrix(tracer: Tracer) -> dict:
    """Every encode row of ``audit_byte_baseline.py``, traced."""
    import audit_byte_baseline as abb

    out = {}
    for label, grid, q256, rows, cols in abb._cases():
        for weighting in ("none", "scale"):
            key = f"{label}/{weighting}"
            tracer.label = key
            before = tracer.trials
            try:
                abb.encode_shape_case(label, grid, q256, rows, cols, weighting)
                out[key] = tracer.trials - before
            except Exception as exc:
                out[key] = f"REFUSED {type(exc).__name__}: {exc}"
    payload = abb.load_value_slice()
    for case in abb._value_cases():
        tracer.label = case.label
        before = tracer.trials
        try:
            abb.encode_value_case(case, payload)
            out[case.label] = tracer.trials - before
        except Exception as exc:
            out[case.label] = f"REFUSED {type(exc).__name__}: {exc}"
    # The release rows too: they ride the E2M1 cap recipe, so they are on the
    # LUT plane and they run the scale refit, which is a second ``_fit_lut``
    # caller the encode matrix reaches only through the value cases.
    tracer.label = "release"
    before = tracer.trials
    abb.release_hashes()
    out["release_rows"] = tracer.trials - before
    return out


#: Whole real Linears with their real Hessian, cut by
#: ``experiments/audit_byte_baseline.py``'s recipe on the box that holds the
#: model and the capture.  The audit harness's value cases cut the committed
#: slice to 16x128; the swap loop's near-ties are a property of how many halves
#: there are, so the receipt needs at least one unit at its shipped width.
REAL_UNITS = "/mnt/shared/ts106-arms/real_units_qwen06b.pt"


def run_full_unit(tracer: Tracer, units: "list[str] | None" = None) -> dict:
    """Whole real Linears, real H, on the LUT-plane wire -- traced."""
    import hashlib

    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.export import HESSIAN_IDENTITY, ActivationSource, encode_linear, wire_recipe

    blob = torch.load(REAL_UNITS, map_location="cpu", weights_only=False)
    provenance = {f: blob["provenance"][f] for f in HESSIAN_IDENTITY}
    grid, q256 = tuple_grid(E2M1_GRID, 2), 512      # E2M1x2 below the cap: LUT plane
    plane = wire_recipe(grid, q256).scale_plane
    out = {}
    for unit in units or list(blob["units"]):
        weight = blob["units"][unit]["weight"].float()
        H = blob["units"][unit]["H"].float()
        source = ActivationSource({unit: H}, provenance, ldlq_sigma=None)
        kwargs = source.for_unit(f"{unit}.weight", weight.shape[1], scale_plane=plane)
        tracer.label = f"{unit}/e2m1x2-sub-512"
        before = tracer.trials
        enc = encode_linear(weight, grid=grid, q256=q256,
                            trellis_weighting="scale", **kwargs)
        out[unit] = {
            "shape": list(weight.shape),
            "grid": grid.name,
            "q256": q256,
            "scale_plane": str(plane),
            "sha256": hashlib.sha256(enc.blob).hexdigest(),
            "swap_trials": tracer.trials - before,
        }
        print(unit, out[unit], flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--full-unit", action="store_true")
    ap.add_argument("--units", default="")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tracer = Tracer()
    install(tracer)

    result = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "old_literal": OLD_LITERAL,
        "float32_eps": float(torch.finfo(torch.float32).eps),
        "audit_matrix_trials_per_case": run_audit_matrix(tracer),
    }
    if args.full_unit:
        result["full_unit"] = run_full_unit(
            tracer, [u for u in args.units.split(",") if u] or None
        )
    result["summary"] = tracer.summary()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    s = result["summary"]
    print(json.dumps({k: v for k, v in s.items() if k != "differences"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
