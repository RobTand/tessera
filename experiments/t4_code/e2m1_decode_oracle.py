"""Oracle for the E2M1x2 window decode into the FP4 MMA's operands (sm_121a).

    python3 experiments/t4_code/e2m1_decode_oracle.py --out DIR [--quick]

Encodes real GLM-5.3-Flash routed-expert weights with the window body over
the LUT16 plane (``export.E2M1X2_SUBCAP_RECIPE``'s wire) at every run table
the grammar emits -- one-run [r] and two-run [r, r + 1] for r = 1..8 -- at
L = 12 and L = 14, packs each unit with the lane's packer, the compact loader
(``compact_prep.prepare_window_lut_compact``, 512 TUPLES per tile; checked
against the documented ``tests/window_pack_reference.pack_bitstream``), maps
columns with the fused lane's run pair and block descriptors
(``routed_fused.run_pair`` / ``block_desc``), and checks three links:

1. **decode**: the kernel's B operand (packed E2M1, K-contiguous) equals the
   reader's own tuple codes (``decode._decode_window``) nibble for nibble,
   and its UE4M3 scales equal the unit's LUT bytes at every (row, k16 group);
2. **reader**: those codes and scale bytes reproduce ``read_unit_artifact``
   bit for bit, as ``value(code) * (e4m3(byte) * global)`` in the reader's
   own order -- so the operands mean what the artifact means;
3. **mma**: the B operand, staged in the consumer's shared-memory layout and
   read with ldmatrix, through ``mma.sync ... kind::mxf4nvf4 ... scale_vec::4X``
   with a one-hot A, gives ``value(code) * e4m3(byte)`` exactly at every
   (row, k): the MMA takes each k16 group's scale from the byte the decode
   wrote for it.

The first tensor also runs TP2 rank 1's row cut: the loader carries the
window state before its first tuple, and the cut decodes to the WHOLE unit's
codes on those rows.

Writes ``DIR/e2m1_decode_oracle.json``; exits nonzero on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tessera import routed_fused as rf                     # noqa: E402
from tessera.decode import _decode_window, unit_scale_field  # noqa: E402
from tessera.export import encode_linear                     # noqa: E402
from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact  # noqa: E402
from tessera.lane_planes import lut_scale_bytes            # noqa: E402
from tessera.manifest import BodyKind, ScalePlaneKind       # noqa: E402
from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact  # noqa: E402
from window_pack_reference import pack_bitstream            # noqa: E402

from t4_code_compare import GRID, Source, P                  # noqa: E402

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

# q256 -> run table: bits per tuple = q256 / 128 (``rate_set(q256 * 2 / 256)``).
Q256 = (128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024)


def build_ext(out: Path):
    from torch.utils.cpp_extension import load

    build = out / "build"
    build.mkdir(parents=True, exist_ok=True)
    return load(name="e2m1_decode_oracle", sources=[str(HERE / "e2m1_decode_oracle.cu")],
                extra_include_paths=[str(HERE)], build_directory=str(build),
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a"],
                verbose=False)


def check(ext, blob, unit, L, q256, dev, cut=None):
    """One unit (``cut``: a weight-row range, the TP row cut) through the lane's packer."""
    t0 = time.time()
    steps, cols = unit.body_bits.shape
    if int(unit.half) != 16 or int(unit.window_bits) != L:
        return {"error": f"unit half {unit.half} / window_bits {unit.window_bits}"}
    rows_all = steps * GRID.arity
    r0, r1 = cut if cut is not None else (0, rows_all)
    rows = r1 - r0
    rates = tuple(int(r) for r in unit.rates)
    # The lane's packer: the compact loader, straight from the wire bytes.
    wire = parse_compact_wire(blob, device=dev, name="w")
    lu = prepare_window_lut_compact(wire, device=dev, **({"rows": cut} if cut is not None else {}))
    rep = lu.rep
    packer_bad = 0
    if cut is None:   # the documented packer, over the reader's body (the unit test covers cuts)
        ref = pack_bitstream(unit.body_bits.cpu(), rates)
        packer_bad = int(not (torch.equal(rep.words.cpu(), ref.words) and torch.equal(rep.perm.cpu(), ref.perm)
                              and torch.equal(rep.runs.cpu(), ref.runs)))
    pair, why = rf.run_pair(rep.runs, cols)
    if pair is None:
        return {"error": f"run_pair refused: {why}"}
    n_lo = int(pair[2])
    bdesc = rf.block_desc(rep.perm, n_lo, cols).reshape(-1).contiguous()
    init_perm = lu.permuted_start_state()
    has_init = init_perm is not None
    if init_perm is None:
        init_perm = torch.zeros(1, dtype=torch.int32, device=dev)
    B, SF = ext.decode(rep.words.contiguous(), int(rep.tile_words), int(rep.n_tiles), pair.to(dev),
                       bdesc.to(dev), init_perm.contiguous(), has_init, lu.codes, lu.scale_plane.contiguous(),
                       lu.scale_lut.contiguous(), rows, cols, L)
    torch.cuda.synchronize()

    # 1. decode: nibble for nibble against the reader's tuple codes of the WHOLE unit
    codes = _decode_window(unit, GRID, torch.int64)                   # [steps, cols] tuple codes
    nib = torch.stack([codes >> 4, codes & 15], dim=1).reshape(rows_all, cols)[r0:r1]   # row 2s high
    want_b = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    got = B.to(torch.int64)
    got_nib = torch.stack([got & 15, got >> 4], dim=2).reshape(rows, cols)
    nib_bad = int((got_nib != nib).sum())
    byte_bad = int((B != want_b).sum())
    lut16 = lut_scale_bytes(unit.scale_lut, dev)
    idx = unit.scale_refine.to(dev).reshape(rows_all, cols // unit.half).long()[r0:r1]
    want_sf = lut16[idx]
    sf_bad = int((SF != want_sf).sum())

    # 2. reader: codes and scale bytes reproduce the artifact's weights bit for bit
    w_reader = read_unit_artifact(blob, device=dev)[r0:r1]
    scale = (want_sf.view(torch.float8_e4m3fn).float() * float(lu.global_scale))
    scale = scale.repeat_interleave(unit.half, dim=1)
    w_ops = E2M1_VALUES.to(dev)[got_nib] * scale
    field = unit_scale_field(unit, rows_all, cols)[r0:r1]
    reader_bad = int((w_ops.to(w_reader.dtype) != w_reader).sum())
    scale_bad = int((scale != field.to(scale.dtype)).sum())

    # 3. mma: one-hot A through the block-scaled FP4 MMA
    D = ext.mma(B, SF, rows, cols)
    torch.cuda.synchronize()
    want_d = E2M1_VALUES.to(dev)[nib] * SF.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    mma_bad = int((D != want_d).sum())                                 # values: the sign of a zero is immaterial
    mma_nan = int(torch.isnan(D).sum())
    runs = [tuple(int(v) for v in r) for r in rep.runs.tolist()]
    res = {"q256": q256, "L": L, "rows": rows, "cols": cols, "cut": list(cut) if cut else None,
           "runs": runs, "n_lo": n_lo, "tile_words": int(rep.tile_words), "n_tiles": int(rep.n_tiles),
           "has_init": has_init, "reader_dtype": str(w_reader.dtype), "packer_mismatch": packer_bad,
           "nibble_mismatch": nib_bad, "byte_mismatch": byte_bad, "scale_byte_mismatch": sf_bad,
           "reader_mismatch": reader_bad, "scale_field_mismatch": scale_bad, "mma_mismatch": mma_bad,
           "mma_nan": mma_nan, "secs": round(time.time() - t0, 2)}
    res["ok"] = all(res[k] == 0 for k in ("packer_mismatch", "nibble_mismatch", "byte_mismatch",
                                          "scale_byte_mismatch", "reader_mismatch", "scale_field_mismatch",
                                          "mma_mismatch", "mma_nan"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="one tensor, q256 in {192, 512, 576}, L=14")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    ext = build_ext(out)
    src = Source()
    # Real routed-expert units: gate [2048, 6144] and down [6144, 2048] at L20,
    # and a gate cut to 1664 rows (832 tuples: a partial second tile, zero-code
    # padding) -- the fused lane's shapes and its padding case.
    tensors = [("L20.e0.gate_proj", f"{P}20.mlp.experts.0.gate_proj.weight", None),
               ("L20.e0.down_proj", f"{P}20.mlp.experts.0.down_proj.weight", None),
               ("L42.e0.up_proj.rows1664", f"{P}42.mlp.experts.0.up_proj.weight", 1664)]
    grid = [(L, q) for L in (12, 14) for q in Q256]
    if a.quick:
        tensors = tensors[:1]
        grid = [(14, q) for q in (192, 512, 576)]
    srcs = {}
    for f in sorted(HERE.glob("e2m1_*")):
        srcs[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    report = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "tessera_head": os.environ.get("TESSERA_HEAD"),
              "image": os.environ.get("ORACLE_IMAGE"), "host": os.environ.get("HOST_NAME"),
              "sources": srcs, "cases": []}
    path = out / "e2m1_decode_oracle.json"
    failures = 0
    for tag, key, rows in tensors:
        w = src.get(key)
        if rows is not None:
            w = w[:rows]
        w = w.contiguous().to(dev).float()
        for L, q in grid:
            cases = []
            try:
                exported = encode_linear(w, grid=GRID, q256=q, body=BodyKind.WINDOW,
                                         scale_plane=ScalePlaneKind.LUT, window_bits=L)
                unit = parse_unit_artifact(exported.blob, device=dev).unit
                cuts = [None]
                if tag == tensors[0][0]:
                    # TP2 rank 1's row cut: its first tuples decode from the carried state
                    n = int(unit.body_bits.shape[0]) * GRID.arity
                    cuts.append((n // 2, n))
                for cut in cuts:
                    try:
                        cases.append(check(ext, exported.blob, unit, L, q, dev, cut))
                    except Exception as exc:  # noqa: BLE001
                        cases.append({"cut": list(cut) if cut else None, "error": repr(exc)[:800]})
            except Exception as exc:  # noqa: BLE001
                cases.append({"error": "encode: " + repr(exc)[:800]})
            for r in cases:
                r.update(tensor=tag, q256=q, L=L)
                r.setdefault("ok", False)
                report["cases"].append(r)
                failures += 0 if r["ok"] else 1
                print(json.dumps(r), flush=True)
            path.write_text(json.dumps(report, indent=1))
            torch.cuda.empty_cache()
    report["failures"] = failures
    report["cases_total"] = len(report["cases"])
    path.write_text(json.dumps(report, indent=1))
    print(f"DONE cases={len(report['cases'])} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
