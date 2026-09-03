"""Byte and decode baselines for the 2026-09-02 math-audit fix passes.

Fifteen checkpoints exist and the wire is a compatibility surface, so every
audit fix has to say -- and prove -- whether it changes the bytes an encoder
emits or the tensor a reader decodes.  This is the proof harness both halves
of that claim are made with.

    python experiments/audit_byte_baseline.py before.json     # at HEAD
    ...apply the fix...
    python experiments/audit_byte_baseline.py after.json
    python experiments/audit_byte_baseline.py --diff before.json after.json

``encode`` hashes the serialised unit for two matrices.  The **shape** matrix
(``_cases``) is a fixed grid of (grid, rung, shape, seed, weighting) on
``randn`` weights -- the recipes the exporter selects, plus the shapes that
exercise a partial trailing superblock.  The **value** matrix (``_value_cases``)
encodes a real weight slice against a real Hessian through ``ActivationSource``,
which is the only way the encoder's activation-aware arithmetic is reachable at
all.  ``decode`` hashes the tensor every ``.tessera`` file on this box decodes
to, which is the half that matters for the artifacts already written: a fix may
legitimately change future bytes, but it may never change what today's bytes
mean.

**Why the second matrix exists** (issue #39).  The shape matrix alone reported
``0 changed of 36`` for the CHANNEL-refit collapse fix (merge ``2b8ffe9``), and
that zero was correct and meant nothing: every case ran with no refit metric, no
LDLQ factor and no completion, so the conditions the fix touched were
unreachable.  Re-measured here on the 18-encode shape matrix, reverting the
``B > 0`` hold at ``scale_channel.py``'s refit still prints ``0 changed of 18``;
the same revert moves every CHANNEL value case.  A byte proof is only a proof of
the arithmetic its corpus reaches, and shape arithmetic is not value arithmetic.

The value matrix is deliberately small -- five encodes of sixteen rows -- and
costs about 13% of the encode half's wall-clock (measured 15.7s against the
shape matrix's 117s, same process, box at load 108).  A proof harness that is
too slow to run before *and* after every fix stops being run, which is a worse
failure than a narrow one.

Two conditions the audit fixes also touched stay out of reach here, and this
harness does not claim them: ``land_at_least``'s ``inf`` branch needs a reach
floor above fp16's range over the unit's global scale, which no real slice
produces, and ``shared_lut_global``'s subnormal range check lives in the fused
lane, which ``encode_linear`` never calls.  Both are pinned by unit tests
instead.

CPU only, by construction -- it runs while a GPU measurement is in flight.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from tessera.alphabet import (
    BF16_GRID,
    E2M1_GRID,
    E4M3_GRID,
    PayloadGrid,
    tuple_grid,
)
from tessera.export import (
    HESSIAN_IDENTITY,
    ActivationSource,
    encode_linear,
    wire_recipe,
)

ARTIFACT_GLOBS = (
    "/home/rob/tessera-runs/*/*/cache/wire/*.tessera",
    "/home/rob/tessera-runs/*/cache/wire/*.tessera",
)

#: The committed real (weight, H) pair the value matrix encodes.  Written by
#: ``experiments/make_audit_value_slice.py``, which records why it has to be
#: real: the ``B <= 0`` condition is a property of H's *off-diagonal*
#: structure, and no synthetic H tried reaches it.
VALUE_SLICE = Path(__file__).resolve().parents[1] / "tests" / "data" / "audit_value_slice.pt"

# (label, grid factory, q256, rows, cols).  Column counts 640 and 384 carry a
# partial trailing superblock on purpose -- that is the shape the granule
# arithmetic floors.
def _cases():
    g2 = tuple_grid(E2M1_GRID, 2)
    return [
        ("e2m1x2-cap-512c",   g2,         896, 64, 512),
        ("e2m1x2-cap-640c",   g2,         896, 64, 640),
        ("e2m1x2-cap-384c",   g2,         896, 32, 384),
        ("e2m1x2-sub-512c",   g2,         512, 64, 512),
        ("e2m1-256-512c",     E2M1_GRID,  256, 64, 512),
        ("e4m3-1024-256c",    E4M3_GRID, 1024, 32, 256),
        ("e4m3-1024-320c",    E4M3_GRID, 1024, 32, 320),
        # The fourth serialisable grid.  It was added after this matrix was
        # written, so until now a change to `BF16_WINDOW_BITS` or
        # `BF16_CHANNEL_SIGMA` -- exactly the two constants issue #18 asks
        # someone to search -- moved real bytes and this harness reported
        # "0 changed".  Same shapes as the E4M3 rows on purpose: the two
        # recipes differ only in the alphabet the window table snaps to, so a
        # digest that moves on one and not the other localises the change.
        ("bf16-1024-256c",    BF16_GRID, 1024, 32, 256),
        ("bf16-1024-320c",    BF16_GRID, 1024, 32, 320),
    ]


class ValueCase(NamedTuple):
    """One encode of the real slice under a named activation-aware recipe.

    ``source`` is the ``ActivationSource`` field set -- the exporter's own
    object, so what gets hashed is the recipe an export selects rather than a
    hand-assembled approximation of it; ``encode`` is the extra
    ``encode_linear`` keywords.  Sixteen rows of 128 columns, because each of
    these runs the trellis several times and the matrix has to stay runnable
    before *and* after every fix.
    """

    label: str
    grid: PayloadGrid
    q256: int
    source: dict
    # Read, never mutated -- and a NamedTuple default is shared by every case,
    # so it is spelled unwritable rather than trusted to stay that way.
    encode: "MappingProxyType" = MappingProxyType({})
    rows: int = 16
    cols: int = 128


# The value matrix.  Each case names a condition the shape matrix cannot
# reach; the text after the slash is the arm, the way ``/none`` and ``/scale``
# are the weighting arms above.
#
# An empty ``source`` is every ``ActivationSource`` default, named by naming
# nothing: a case that spelled ``ldlq_sigma=1.0`` would stop tracking
# ``DEFAULT_LDLQ_SIGMA`` the moment that default moved, and a moved default is
# exactly the byte change this harness exists to catch.  ``ldlq_sigma=None`` is
# LDLQ off, and it is off everywhere but that one case: the block loop costs
# about 3x the encode, and one case is enough that a change to ``block_ldl`` or
# to the block schedule moves a digest.  Every other case keeps the metric,
# which is the condition this matrix was added for.
def _value_cases():
    g2 = tuple_grid(E2M1_GRID, 2)
    return [
        # CHANNEL plane, exact full-H quadratic (DEFAULT_REFIT_OBJECTIVE's
        # "channel": "hessian") plus LDLQ: the exporter's whole default recipe
        # for the E4M3 wire, and the case the CHANNEL ``B > 0`` hold moves.
        ValueCase("e4m3-1024-128c/hessian+ldlq", E4M3_GRID, 1024, dict()),
        # The same plane on the other grid that ships it, at the same shape --
        # exactly as the shape matrix pairs E4M3 with BF16: a digest that moves
        # on one and not the other localises the change to the alphabet rather
        # than to the plane.
        ValueCase("bf16-1024-128c/hessian", BF16_GRID, 1024,
                  dict(ldlq_sigma=None)),
        # The reach floor, which is the only caller of ``land_at_least``.
        ValueCase("e4m3-1024-128c/hessian+reach", E4M3_GRID, 1024,
                  dict(ldlq_sigma=None, refit_reach_floor=True)),
        # The LUT plane's own metric refit: a 1-D diagonal weight
        # ("lut16": "h^1.0"), through a different function than CHANNEL's.
        ValueCase("e2m1x2-sub-512-128c/h1", g2, 512,
                  dict(ldlq_sigma=None)),
        # The second rate axis.  A window body has no completion axis, so this
        # rides the TCQ body -- which is also the only body whose completion
        # argmin has more than one descendant to choose between.
        ValueCase("e2m1-256-128c/completion", E2M1_GRID, 256,
                  dict(ldlq_sigma=None), dict(completion=2)),
    ]


def load_value_slice() -> dict:
    """The committed real weight and Hessian, or a refusal naming the recipe.

    Skipping the value matrix when the slice is absent would be the exact
    failure the matrix exists to close: a corpus that quietly shrinks back to
    the arm reaching nothing, while the harness still prints a total and reads
    like a proof.  A missing slice stops the run instead.
    """
    if not VALUE_SLICE.exists():
        raise FileNotFoundError(
            f"{VALUE_SLICE} is missing, so every value case would be skipped and "
            f"the encode total would silently shrink to the shape matrix -- which "
            f"reaches none of the encoder's activation-aware arithmetic. Cut it "
            f"again with `python experiments/make_audit_value_slice.py` on the box "
            f"that holds the source model and the H capture."
        )
    payload = torch.load(VALUE_SLICE, map_location="cpu", weights_only=False)
    absent = [f for f in HESSIAN_IDENTITY if payload.get(f) is None]
    if absent:
        raise ValueError(
            f"{VALUE_SLICE} carries no {absent}: a fixture that cannot say which "
            f"capture shaped its digests is not evidence about any capture"
        )
    return payload


def encode_value_case(case: ValueCase, slice_payload: dict) -> str:
    """The digest for one value case.  The CLI and the coverage test share it."""
    weight = slice_payload["weight"][: case.rows, : case.cols].float()
    # A principal submatrix of a second-moment matrix is the second-moment
    # matrix of those columns, so the sliced H is a real H for the sliced
    # weight and not a truncation of one.
    H = slice_payload["H"][: case.cols, : case.cols].float()
    unit = slice_payload["unit"]
    provenance = {f: slice_payload[f] for f in HESSIAN_IDENTITY}
    source = ActivationSource({unit: H}, provenance, **case.source)
    # The plane comes from the resolved recipe and is never re-derived: an
    # objective chosen for a plane the encode is not on prices a different
    # artifact than it ships.
    plane = wire_recipe(case.grid, case.q256).scale_plane
    kwargs = source.for_unit(f"{unit}.weight", case.cols, scale_plane=plane)
    exported = encode_linear(
        weight, grid=case.grid, q256=case.q256,
        # The exporter's weighting, not ``encode_unit``'s signature default:
        # these cases exist to hash the recipe that ships.
        trellis_weighting="scale", **kwargs, **case.encode,
    )
    return hashlib.sha256(exported.blob).hexdigest()


def encode_shape_case(label, grid, q256, rows, cols, weighting) -> str:
    """The digest for one shape case.  The CLI and the coverage test share it."""
    # zlib, not hash(): PYTHONHASHSEED is randomised per process, so a
    # built-in hash would reseed the weights on every run and every digest
    # below would be noise.
    torch.manual_seed(zlib.crc32(f"{label}/{weighting}".encode()) & 0xFFFF)
    w = torch.randn(rows, cols)
    unit = encode_linear(w, grid=grid, q256=q256, trellis_weighting=weighting)
    return hashlib.sha256(unit.blob).hexdigest()


def encode_hashes() -> dict:
    out = {}
    for label, grid, q256, rows, cols in _cases():
        for weighting in ("none", "scale"):
            key = f"{label}/{weighting}"
            try:
                out[key] = encode_shape_case(label, grid, q256, rows, cols, weighting)
            except Exception as exc:            # a refusal is part of the baseline
                out[key] = f"REFUSED {type(exc).__name__}: {exc}"
    payload = load_value_slice()
    for case in _value_cases():
        try:
            out[case.label] = encode_value_case(case, payload)
        except Exception as exc:            # a refusal is part of the baseline
            out[case.label] = f"REFUSED {type(exc).__name__}: {exc}"
    return out


def decode_hashes() -> dict:
    from tessera.unit_artifact import read_unit_artifact  # late: keeps import cheap

    out = {}
    paths = sorted({p for g in ARTIFACT_GLOBS for p in glob.glob(g)})
    for path in paths:
        key = path.replace("/home/rob/tessera-runs/", "")
        try:
            blob = open(path, "rb").read()
            tensor = read_unit_artifact(blob)
            digest = hashlib.sha256(
                tensor.to(torch.float32).contiguous().numpy().tobytes()
            ).hexdigest()
        except Exception as exc:
            digest = f"FAILED {type(exc).__name__}: {exc}"
        out[key] = digest
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()

    if a.diff:
        before = json.load(open(a.diff[0]))
        after = json.load(open(a.diff[1]))
        changed = 0
        for section in ("encode", "decode"):
            b, c = before.get(section, {}), after.get(section, {})
            for key in sorted(set(b) | set(c)):
                if b.get(key) != c.get(key):
                    changed += 1
                    print(f"{section} CHANGED {key}\n    before {b.get(key)}\n    after  {c.get(key)}")
        print(f"{changed} changed of {sum(len(before.get(s, {})) for s in ('encode','decode'))}")
        return 1 if changed else 0

    report = {"encode": encode_hashes()}
    if not a.encode_only:
        report["decode"] = decode_hashes()
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.path:
        open(a.path, "w").write(text + "\n")
        value = len(_value_cases())
        # The split is printed, not just the total: a reader of a "0 changed"
        # needs to know how much of it was shape arithmetic, since that is the
        # half that answered zero to the CHANNEL fixes (issue #39).
        print(f"wrote {a.path}: {len(report['encode'])} encodes "
              f"({len(report['encode']) - value} shape, {value} value), "
              f"{len(report.get('decode', {}))} decodes")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
