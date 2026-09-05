"""How ``tests/data/legacy/*.tessera`` were made -- a record, not a tool.

Run ONCE at master ``da2b371`` (2026-09-04, the last commit before schema
minor 7) with that tree's own encoder and writer::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python tests/data/legacy/generate_legacy_blobs.py tests/data/legacy

Every blob is therefore a minor-6 artifact in the minor 0-6 plane layout,
written by code that had never heard of ``PlaneLayout``.  ``manifest.json``
records, per blob, the header minor, the plane order the writer used, the
terminal's ``plane_elements``, the COMPLETION descriptor's counts, and the
SHA-256 of the tensor the *pre-change* reader decoded it to; plus the
encoder identity and per-fixture digests of that tree.

``tests/test_ladder_wire.py`` holds the current reader to those blobs (same
decoded tensor, same order, same minor) and holds the ``PlaneLayout.LEGACY``
writer to the same bytes.  Re-running this script at any later tree writes
minor-7 artifacts and would overwrite the evidence; do not.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.container import parse
from tessera.encode import encode_unit
from tessera.export import (
    DEFAULT_CODE, DEFAULT_GROUP, DEFAULT_HALF, DEFAULT_SCALE_REFIT,
    _plan_for, encode_linear_planes, wire_recipe,
)
from tessera.manifest import ScalePlaneKind
from tessera.planes import PlaneKind
from tessera.slicing import slice_unit
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact, read_unit_artifact

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
GRIDS = {g.name: g for g in SERIALISABLE_GRIDS.values()}
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def weight(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g) * 0.02


cases = {}


def put(label, blob):
    art = parse(blob)
    dec = read_unit_artifact(blob)
    (out / f"{label}.tessera").write_bytes(blob)
    cases[label] = {
        "header_minor": blob[10],
        "schema_minor": art.manifest.schema_minor,
        "plane_order": [k.name for k in art.manifest.plane_order],
        "plane_elements": list(art.terminal.plane_elements),
        "completion_counts": list(art.manifest.plane(PlaneKind.COMPLETION).counts),
        "decoded_sha256": hashlib.sha256(dec.contiguous().float().numpy().tobytes()).hexdigest(),
        "decoded_shape": list(dec.shape),
        "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "blob_bytes": len(blob),
    }
    print(label, cases[label]["header_minor"], cases[label]["plane_elements"], cases[label]["completion_counts"])


# 1. The one whose COMPLETION packing changes at minor 7: E2M1 TCQ below the
#    cap, LUT plane, two superblocks, at full depth and at depth 1.
e, u, f = encode_linear_planes(weight(16, 512, 1), grid=GRIDS["E2M1"], q256=256, name="u", completion=None)
put("e2m1-256-cfull-lut-512c", e.blob)
e, u, f = encode_linear_planes(weight(16, 512, 2), grid=GRIDS["E2M1"], q256=256, name="u", completion=1)
put("e2m1-256-c1-lut-512c", e.blob)
# 2. The S6b plane with completion.
e, u, f = encode_linear_planes(weight(16, 512, 3), grid=GRIDS["E2M1"], q256=256, name="u", completion=None, scale_plane=ScalePlaneKind.S6B)
put("e2m1-256-cfull-s6b-512c", e.blob)
# 3. Segment 2a diagonals on the LUT plane.
e, u, f = encode_linear_planes(weight(16, 256, 4), grid=GRIDS["E2M1"], q256=256, name="u", completion=None, with_diagonals=True)
put("e2m1-256-cfull-lut-diag-256c", e.blob)
# 4. The four shipping wires at default completion: E2M1x2 cap (TCQ/LUT),
#    E2M1x2 sub-cap (window/LUT), E4M3 (window/CHANNEL), BF16 (window/CHANNEL).
e, u, f = encode_linear_planes(weight(16, 256, 5), grid=GRIDS["E2M1x2"], q256=896, name="u")
put("e2m1x2-896-tcq-lut-256c", e.blob)
e, u, f = encode_linear_planes(weight(16, 256, 6), grid=GRIDS["E2M1x2"], q256=640, name="u")
put("e2m1x2-640-window-lut-256c", e.blob)
e, u, f = encode_linear_planes(weight(16, 256, 7), grid=GRIDS["E4M3"], q256=1024, name="u")
put("e4m3-1024-window-channel-256c", e.blob)
e, u, f = encode_linear_planes(weight(16, 256, 8), grid=GRIDS["BF16"], q256=1024, name="u")
put("bf16-1024-window-channel-256c", e.blob)
# 5. A row shard of a completion unit (INITIAL_STATE plane, the shard order).
parent_e, _, _ = encode_linear_planes(weight(16, 512, 9), grid=GRIDS["E2M1"], q256=256, name="u", completion=None)
put("e2m1-256-cfull-lut-512c-parent", parent_e.blob)
parsed = parse_unit_artifact(parent_e.blob)
shard = slice_unit(parsed, rows=(8, 16))
_m, _r, blob = build_unit_artifact(shard, "u", parsed.forests, 256, parsed.code)
put("e2m1-256-cfull-lut-512c-shard-r8-16", blob)
# 6. A released unit, assembled the way encoder_identity._release_blob does.
grid = GRIDS["E2M1"]
q256 = 768
recipe = wire_recipe(grid, q256)
rates, forests = _plan_for(grid, q256, 256, recipe.body, None)
unit = encode_unit(
    weight(16, 256, 10), forests, rates, DEFAULT_CODE, completion=0, released_positions=256,
    group=DEFAULT_GROUP, half=DEFAULT_HALF, scale_refit=DEFAULT_SCALE_REFIT, span=recipe.span,
    scale_plane=recipe.scale_plane, trellis_weighting="scale", body=recipe.body,
    window_bits=recipe.window_bits, window_seed=recipe.window_seed,
    window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
)
_m, _r, blob = build_unit_artifact(unit, "u", forests, q256 * grid.arity, DEFAULT_CODE)
put("e2m1-768-release256-256c", blob)

from tessera import encoder_identity as ei  # noqa: E402

meta = {
    "generated_at_commit": commit,
    "encoder_fixture_id": ei.encoder_fixture_id().hex(),
    "fixture_digests": ei.fixture_digests(),
    "cases": cases,
}
(out / "manifest.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
print("encoder_fixture_id", meta["encoder_fixture_id"])
