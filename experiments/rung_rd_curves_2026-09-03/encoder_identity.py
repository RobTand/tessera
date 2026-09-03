"""Does this commit's encoder still write the bytes PrismaQuant priced and served?

153 commits sit between the served receipt (Tessera ``11b007c``, blobs written
at ``3d419e7``) and the HEAD this experiment encodes at, and several of them
touch the E4M3/CHANNEL path (``45a8f19`` held the CHANNEL refit to B > 0,
``bace528`` made the refit objective a property of the plane).  If HEAD's
encode differs from the priced blobs then the rate-distortion table below
measures a different object than the served anchors it is validated against,
and every candidate cost scored on it is scored on the wrong curve.

So this runs first and it is a gate, not a note.  Every ``.tessera`` blob in
PrismaQuant's ``verify_cache/wire`` -- the re-encode of exactly the rungs the
allocator chose, which is what the receipt's section 7 table was scored on --
is re-encoded here from the same source weight at the same weights-only
settings and compared three ways: payload bytes, the packed planes, and the
decoded FP8 tile the W8A8 route multiplies.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import E4M3_GRID
from tessera.decode import materialize_fp8
from tessera.export import encode_linear_planes
from tessera.unit_artifact import parse_unit_artifact

PQ_WIRE = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b/verify_cache/wire")
MODEL = Path("/home/rob/models/Qwen3-0.6B")
NAME_RE = re.compile(r"^model__layers__0__(.+)__TESSERA_E4M3_K1_R(\d+)\.tessera$")


def sha(x) -> str:
    if isinstance(x, torch.Tensor):
        return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return repr(x)[:24]


def fingerprint(parsed) -> dict:
    unit = parsed.unit
    out = {}
    for field in ("body_bits", "scale_lut", "scale_global", "scale_row", "scale_rows",
                  "scale_channel", "window_table", "table"):
        value = getattr(unit, field, None)
        if value is not None:
            out[field] = sha(value)
    return out


def main() -> int:
    picks = []
    for path in sorted(PQ_WIRE.iterdir()):
        m = NAME_RE.match(path.name)
        if m:
            picks.append((m.group(1).replace("__", "."), int(m.group(2)), path))
    if not picks:
        raise SystemExit(f"no E4M3 blobs under {PQ_WIRE}")

    rows = []
    with safe_open(str(MODEL / "model.safetensors"), framework="pt") as handle:
        for role, rung, path in picks:
            name = f"model.layers.0.{role}.weight"
            weight = handle.get_tensor(name).to("cuda", torch.float32).contiguous()
            exported, _u, _f = encode_linear_planes(
                weight, grid=E4M3_GRID, q256=rung, name=name, verify=True)
            theirs = path.read_bytes()
            mine_p = parse_unit_artifact(bytes(exported.blob), device="cuda")
            their_p = parse_unit_artifact(theirs, device="cuda")
            fa, fb = fingerprint(mine_p), fingerprint(their_p)
            fields = sorted(set(fa) | set(fb))
            planes_same = all(fa.get(f) == fb.get(f) for f in fields)
            ma = materialize_fp8(mine_p.unit, mine_p.forests, mine_p.code)
            mb = materialize_fp8(their_p.unit, their_p.forests, their_p.code)
            tile_same = all(sha(a) == sha(b) for a, b in zip(ma, mb))
            rows.append(dict(role=role, rung=rung, payload_bytes=exported.exact_bytes,
                             blob_mine=len(exported.blob), blob_pq=len(theirs),
                             blob_bytes_equal=bytes(exported.blob) == theirs,
                             planes_equal=planes_same, decoded_tile_equal=tile_same,
                             fields=fields))
            print(f"{role:22s} R{rung:5d}  payload {exported.exact_bytes:9d}B  "
                  f"blob {len(exported.blob):9d}/{len(theirs):9d}  "
                  f"bytes {'==' if rows[-1]['blob_bytes_equal'] else 'DIFFER'}  "
                  f"planes {'==' if planes_same else 'DIFFER'}  "
                  f"tile {'==' if tile_same else 'DIFFER'}", flush=True)

    verdict = all(r["decoded_tile_equal"] for r in rows)
    print()
    print("blobs byte-identical :", all(r["blob_bytes_equal"] for r in rows))
    print("planes identical     :", all(r["planes_equal"] for r in rows))
    print("decoded tiles identical:", verdict)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is not None:
        out.write_text(json.dumps({"rows": rows, "identical": verdict}, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
