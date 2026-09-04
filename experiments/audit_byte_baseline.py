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
harness does not claim them: the shared upward-landing helper's ``inf`` branch
needs a reach floor above fp16's range over the unit's global scale, which no
real slice produces, and ``shared_lut_global``'s subnormal range check lives in
the fused lane, which ``encode_linear`` never calls.  Both are pinned by unit
tests instead.

``release`` is the third matrix, and it exists because the first two are blind
to the RELEASE plane: ``export.encode_linear`` has no ``released_positions``
keyword at all, so no ``encode`` row can carry a release and no artifact on this
box carries one either (issue #27, 2026-09-03 -- one of 642 ``.tessera`` files
has a RELEASE plane, at 512 columns).  A change to the release quota therefore
reported "0 changed" through a harness that could not see it.  These rows go
through ``encode_unit``/``build_unit_artifact`` directly, at both a complete and
a partial trailing superblock, and hash the placement as well as the bytes.

``layout`` is the fourth (issue #143), and it is the same failure one more
time.  The first three matrices encode only what ``wire_recipe`` selects and
only whole units, so three of the ten planes a reader will read were written by
no row at all: **SCALE_BASE** (the S6b plane, which ``export.py``'s
``scale_plane=`` override still writes and ``unit_artifact._read_scale_planes``
still accepts, though no recipe selects it), **DIAG_SU**/**DIAG_SV as segment
2a** (``with_diagonals=``, refused under a CHANNEL plane and therefore reachable
only on a block plane), and **INITIAL_STATE** (the shard plane schema minor 4
added, together with ``planes.SHARD_PLANE_ORDER``, a tenth ``plane_elements``
entry and a ``PER_SUPERBLOCK`` RELEASE descriptor -- all layout, and all of it
proved by nothing here).  A shard row hashes the parent bytes, the shard bytes
and the start state separately, so a move localises to the encoder, the cutter
or the state replay.  ``tests/test_audit_byte_baseline.py`` derives the
coverage claim from ``planes.SHARD_PLANE_ORDER`` and ``ScalePlaneKind`` rather
than restating it, so the next plane is noisy rather than silent.

CPU only, by construction -- it runs while a GPU measurement is in flight.
That is not incidental for the shard rows: ``tests/test_slice_unit.py`` gates
the whole slicing surface on ``torch.cuda.is_available()`` ("the encoder is a
CUDA path", ``:161-163``), which these rows disprove by running.
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
from tessera.manifest import ScalePlaneKind

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
        # The refit's reach floor, which covers ``land_at_least`` separately
        # from the initial reach landing exercised by the LDLQ case above.
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


class LayoutCase(NamedTuple):
    """One encode whose *layout* is the condition, not its arithmetic.

    The shape and value matrices both encode what ``wire_recipe`` selects, on
    whole units.  That leaves three planes a reader accepts and this harness
    never wrote, and each is reached by one keyword or one cut rather than by a
    different rung:

    ``encode``
        extra ``encode_linear`` keywords.  ``scale_plane=S6B`` is the plane no
        recipe selects and every reader still decodes; ``with_diagonals=True``
        is segment 2a, which a CHANNEL plane refuses (its row scale *is* the
        DIAG_SV field), so it is spelled on a block-plane grid.
    ``cut``
        ``((r0, r1), (c0, c1))``: parse the encoded unit and cut a shard out of
        it with ``slicing.slice_unit``, the way a rank does at load.  A row cut
        is what puts the INITIAL_STATE plane on the wire, and the column cut
        alongside it makes the block scale planes restrict too.

    Columns are chosen to share ``_plan_for``'s memo with an existing row (512
    for the E2M1 rows, 256 for the E4M3 one), so the matrix grows by encodes
    and not by forest and window-table builds.
    """

    label: str
    grid: PayloadGrid
    q256: int
    rows: int
    cols: int
    # Read, never mutated -- see ValueCase.encode.
    encode: "MappingProxyType" = MappingProxyType({})
    cut: "tuple[tuple[int, int], tuple[int, int]] | None" = None


def _layout_cases():
    return [
        # SCALE_BASE.  ``export.encode_linear_planes``'s ``scale_plane``
        # override is caller-facing and ``_read_scale_planes`` accepts what it
        # writes, so an S6b artifact is a thing that exists; before this row a
        # change to ``encode._refit_scales`` or to the E8M0 base packing moved
        # its bytes and the harness reported "0 changed".
        LayoutCase("s6b-e2m1-256-512c", E2M1_GRID, 256, 32, 512,
                   MappingProxyType({"scale_plane": ScalePlaneKind.S6B})),
        # DIAG_SU and DIAG_SV as segment 2a: the rank-1 channel diagonals, fitted
        # and packed.  The CHANNEL rows above fill DIAG_SV with a row scale,
        # which is a different producer of the same plane and does not cover
        # ``diagonals.fit_diagonals`` at all.
        LayoutCase("diag-e2m1-256-512c", E2M1_GRID, 256, 32, 512,
                   MappingProxyType({"with_diagonals": True})),
        # INITIAL_STATE under the TCQ body, where its width is the convolutional
        # code's memory; the column cut restricts the LUT block plane with it.
        LayoutCase("shard-e2m1-256-512c/tcq", E2M1_GRID, 256, 32, 512,
                   cut=((16, 32), (256, 512))),
        # INITIAL_STATE under the WINDOW body, where its width is ``window_bits``
        # instead -- a different element width in the same plane, and the wire
        # the shipping E4M3 recipe writes.
        LayoutCase("shard-e4m3-1024-256c/window", E4M3_GRID, 1024, 32, 256,
                   cut=((16, 32), (0, 256))),
    ]


def written_planes(blob: bytes) -> "frozenset":
    """The plane kinds this artifact actually carries, off its own manifest.

    Read from ``manifest.plane_order`` zipped against the terminal's
    ``plane_elements`` -- the pair every reader indexes -- so "the matrix writes
    this plane" is measured on the bytes rather than declared beside them.
    """
    from tessera.container import parse

    art = parse(blob)
    return frozenset(
        kind
        for kind, count in zip(art.manifest.plane_order, art.terminal.plane_elements)
        if count
    )


def encode_layout_case(case: LayoutCase) -> "dict[str, bytes]":
    """The byte strings one layout case pins.  The CLI and the tests share it.

    A whole-unit case pins its bytes.  A cut case pins three things, because a
    shard has three ways to move: ``parent`` (the encoder), ``bytes`` (the
    cutter and the shard's own layout) and ``state`` (the replay that produces
    the start state).  A digest that moves on ``bytes`` alone is not a digest
    that moves on all three.
    """
    from tessera.slicing import slice_unit
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    torch.manual_seed(zlib.crc32(case.label.encode()) & 0xFFFF)
    weight = torch.randn(case.rows, case.cols)
    parent = encode_linear(
        weight, grid=case.grid, q256=case.q256,
        trellis_weighting="scale", **case.encode,
    )
    if case.cut is None:
        return {"bytes": parent.blob}
    # From bytes alone, exactly as a rank loading the artifact does: the cut is
    # a property of the wire, not of the encoder object that happens to be in
    # this process.
    parsed = parse_unit_artifact(parent.blob)
    shard = slice_unit(parsed, rows=case.cut[0], cols=case.cut[1])
    _m, _r, blob = build_unit_artifact(
        shard, case.label, parsed.forests,
        case.q256 * case.grid.arity, parsed.code,
    )
    return {
        "parent": parent.blob,
        "bytes": blob,
        "state": shard.initial_state.cpu().numpy().tobytes(),
    }


def layout_hashes() -> dict:
    out = {}
    for case in _layout_cases():
        keys = ("parent", "bytes", "state") if case.cut else ("bytes",)
        try:
            for key, payload in encode_layout_case(case).items():
                out[f"{case.label}/{key}"] = hashlib.sha256(payload).hexdigest()
        except Exception as exc:        # a refusal is part of the baseline
            for key in keys:
                out[f"{case.label}/{key}"] = f"REFUSED {type(exc).__name__}: {exc}"
    return out


#: ``(label, grid factory, q256, rows, cols, released, cut)`` for the RELEASE
#: plane.  512 columns is a whole number of superblocks and 640 and 320 are not,
#: which is the only distinction the release quota draws.  ``cut`` is the shard
#: extent for the one row that is also cut: a released unit's shard is the only
#: artifact that carries a ``PER_SUPERBLOCK`` RELEASE descriptor, because its
#: counts are its parent's restricted and no quota reproduces them
#: (``unit_artifact._release_placement``).
def _release_cases():
    return [
        ("e2m1-cap-512c-rel3000",  E2M1_GRID, None, 32, 512, 3000, None),
        ("e2m1-cap-640c-rel3000",  E2M1_GRID, None, 32, 640, 3000, None),
        ("e2m1-cap-320c-rel96",    E2M1_GRID, None, 32, 320,   96, None),
        ("e2m1-cap-640c-rel4000",  E2M1_GRID, None,  8, 640, 4000, None),
        # A released unit's shard: 256 columns is the superblock, and a release
        # forces the column granularity to it.
        ("e2m1-cap-512c-rel3000-shard", E2M1_GRID, None, 32, 512, 3000,
         ((16, 32), (256, 512))),
    ]


def release_hashes() -> dict:
    from tessera.encode import encode_unit
    from tessera.export import _plan_for, tcq_cap_q256, wire_recipe
    from tessera.manifest import ScalePlaneKind
    from tessera.trellis import ConvCode
    from tessera.unit_artifact import build_unit_artifact

    code = ConvCode(memory=6)
    out = {}
    for label, grid, q256, rows, cols, released, cut in _release_cases():
        keys = ["bytes", "placement"]
        if cut is not None:
            keys += ["shard-bytes", "shard-placement"]
        if q256 is None:
            q256 = tcq_cap_q256(grid)
        recipe = wire_recipe(grid, q256)
        torch.manual_seed(zlib.crc32(label.encode()) & 0xFFFF)
        weight = torch.randn(rows, cols) * 0.02
        sigma = (
            recipe.channel_sigma
            if recipe.scale_plane is ScalePlaneKind.CHANNEL
            else None
        )
        try:
            rates, forests = _plan_for(grid, q256, cols, recipe.body, sigma)
            unit = encode_unit(
                weight, forests, rates, code, completion=0,
                released_positions=released, span=recipe.span,
                scale_plane=recipe.scale_plane, body=recipe.body,
                window_bits=recipe.window_bits, window_seed=recipe.window_seed,
                window_sigma=recipe.window_sigma,
                channel_sigma=recipe.channel_sigma, scale_refit=2,
            )
            _m, _r, blob = build_unit_artifact(
                unit, label, forests, q256 * grid.arity, code
            )
            shard = None
            if cut is not None:
                shard = _release_shard(blob, label, cut, q256 * grid.arity)
        except Exception as exc:        # a refusal is part of the baseline
            for key in keys:
                out[f"{label}/{key}"] = f"REFUSED {type(exc).__name__}: {exc}"
            continue
        out[f"{label}/bytes"] = hashlib.sha256(blob).hexdigest()
        # The placement, hashed separately: the bytes would also move if the
        # codes moved, and this half localises the change to the quota.
        out[f"{label}/placement"] = hashlib.sha256(
            unit.release_index.cpu().numpy().tobytes()
        ).hexdigest()
        if shard is not None:
            for key, payload in shard.items():
                out[f"{label}/shard-{key}"] = hashlib.sha256(payload).hexdigest()
    return out


def _release_shard(blob: bytes, label: str, cut, q256: int) -> "dict[str, bytes]":
    """The shard of a released unit: the one artifact with per-superblock counts.

    A whole unit's released set is ``grammar.release_quota`` of its total and a
    reader regenerates it; a shard's is the restriction of its parent's, so the
    counts travel on the wire as the RELEASE descriptor's ``PER_SUPERBLOCK``
    counts (``unit_artifact._release_placement``).  That descriptor is written
    by no other row in any matrix here.  The placement is hashed beside the
    bytes for the same reason the whole unit's is.
    """
    from tessera.slicing import slice_unit
    from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

    parsed = parse_unit_artifact(blob)
    shard = slice_unit(parsed, rows=cut[0], cols=cut[1])
    _m, _r, shard_blob = build_unit_artifact(
        shard, f"{label}.shard", parsed.forests, q256, parsed.code
    )
    return {
        "bytes": shard_blob,
        "placement": shard.release_index.cpu().numpy().tobytes(),
    }


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
        # Over the sections the two files actually hold, never a roster of the
        # sections this file knew about when it was written: a matrix added
        # later and left out of such a roster is a matrix ``--diff`` silently
        # reports "0 changed" for, which is the failure the whole harness is
        # against.  ``layout`` (issue #143) would have been exactly that.
        for section in sorted(set(before) | set(after)):
            b, c = before.get(section, {}), after.get(section, {})
            for key in sorted(set(b) | set(c)):
                if b.get(key) != c.get(key):
                    changed += 1
                    print(f"{section} CHANGED {key}\n    before {b.get(key)}\n    after  {c.get(key)}")
        print(f"{changed} changed of "
              f"{sum(len(rows) for rows in before.values())}")
        return 1 if changed else 0

    report = {
        "encode": encode_hashes(),
        "layout": layout_hashes(),
        "release": release_hashes(),
    }
    if not a.encode_only:
        report["decode"] = decode_hashes()
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.path:
        open(a.path, "w").write(text + "\n")
        value = len(_value_cases())
        # The split is printed, not just the total: a reader of a "0 changed"
        # needs to know how much of it was shape arithmetic, since that is the
        # half that answered zero to the CHANNEL fixes (issue #39), and how
        # much was the release rows, which the encode matrix structurally
        # cannot carry (issue #27), or the layout rows, which carry the three
        # planes neither of the first two writes (issue #143).
        print(f"wrote {a.path}: {len(report['encode'])} encodes "
              f"({len(report['encode']) - value} shape, {value} value), "
              f"{len(report['layout'])} layout rows, "
              f"{len(report['release'])} release rows, "
              f"{len(report.get('decode', {}))} decodes")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
