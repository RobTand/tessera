#!/usr/bin/env python
"""Is a fused module's decode driven by each member's OWN manifest, or by one
module-level ``(grid, q256)``?  (#37, question 1.)

Encodes a REAL fused group -- Qwen3-0.6B layer 0 ``q_proj``/``k_proj``/
``v_proj`` -- at THREE DIFFERENT rungs on the E4M3 grid, and compares element
for element:

* every role prepared and decoded ALONE (a one-member container, the shape the
  exporter writes for an unfused Linear), against
* the same three roles prepared and decoded as ONE fused module,

and both against ``tessera.decode.materialize_fp8``, the reference decoder.

If the decode read one module-level rate, the mixed-rung module could not
reproduce the singles: three roles at 1024/900/1200 would all be decoded on
whichever rate the module declared.  Nothing here touches the sidecar scheme
(``serving.scheme.parse_tessera_blob_for_scheme``) -- that gate is the *other*
half of #37 and is measured by ``tests/test_fused_member_rungs.py``.  This
script is about the DECODER.

The rungs are an argument, not a constant, because the default three are all
inside ONE window band (E4M3 resolves L=14 across its whole range, and BF16
does up to 3584).  A module key of ``(family, grid, body, plane)`` also admits
members whose window TABLES differ in size, and "the decode is per member"
measured only at equal L would not have covered that.  ``--rungs
1024,3600,3900`` on the BF16 grid is the cross-band arm: window_bits 14, 15
and 16 in one fused module.

    python experiments/fused_member_rung_identity.py [--src DIR] [--grid E4M3|BF16]
                                                     [--rungs A,B,C]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors import safe_open

from tessera.alphabet import BF16_GRID, E4M3_GRID
from tessera.fused import pack_fused, parse_fused
from tessera.unit_artifact import parse_unit_artifact

#: role -> (checkpoint tensor, q256).  Three rungs, none of them equal, all
#: inside the E4M3 family's published ``reader_rate_range_q256`` [256, 2048].
GROUP = (("q_proj", "model.layers.0.self_attn.q_proj.weight", 1024),
         ("k_proj", "model.layers.0.self_attn.k_proj.weight", 900),
         ("v_proj", "model.layers.0.self_attn.v_proj.weight", 1200))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/home/rob/models/Qwen3-0.6B"))
    ap.add_argument("--grid", choices=("E4M3", "BF16"), default="E4M3")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated q256 per role, in GROUP order; "
                         "1024,3600,3900 on --grid BF16 spans L=14/15/16")
    args = ap.parse_args()

    group = GROUP
    if args.rungs:
        wanted = [int(r) for r in args.rungs.split(",")]
        if len(wanted) != len(GROUP):
            ap.error(f"--rungs needs {len(GROUP)} values, one per role")
        group = tuple((role, tensor, q) for (role, tensor, _), q
                      in zip(GROUP, wanted))

    from tessera.export import encode_linear_planes

    if args.grid == "E4M3":
        from tessera.decode import materialize_fp8 as materialize
        from tessera.serving.fp8_route import prepare_tessera_fp8_module as prepare
        grid = E4M3_GRID
    else:
        from tessera.decode import materialize_bf16 as materialize
        from tessera.serving.bf16_route import prepare_tessera_bf16_module as prepare
        grid = BF16_GRID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, grid {grid.name}")

    shard = sorted(args.src.glob("*.safetensors"))[0]
    blobs = []
    with safe_open(str(shard), framework="pt") as handle:
        for role, tensor, q256 in group:
            weight = handle.get_tensor(tensor).to(device, torch.float32).contiguous()
            exported, unit, _forests = encode_linear_planes(
                weight, grid=grid, q256=q256, name=role, verify=False)
            print(f"  {role:7s} {tuple(weight.shape)}  q256={q256:5d}  "
                  f"rates={sorted({int(r) for r in unit.rates})}  L={unit.window_bits}  "
                  f"{len(exported.blob)} B  {float(exported.bpp):.4f} bpp")
            blobs.append((role, exported.rows, exported.blob))

    parsed = [(m.name, parse_unit_artifact(m.blob, device=device))
              for m in parse_fused(pack_fused(blobs))]
    # Each member's rung, read back from that member's OWN manifest.  The
    # manifest's root rate is per CODE; arity is 1 on both scalar grids.
    rungs = [int(p.manifest.branch.root_q256) // p.grid.arity for _, p in parsed]
    print(f"  per-member q256 from each member's own manifest: {rungs}")
    assert rungs == [q for _, _, q in group], rungs
    assert len(set(rungs)) == len(rungs), "the point of the run is that they differ"

    singles = [prepare([(name, p)], device=device) for name, p in parsed]
    fused = prepare(parsed, device=device)

    tile, scale = fused.decode(), fused.row_scale()
    cat_tile = torch.cat([m.decode() for m in singles], 0)
    cat_scale = torch.cat([m.row_scale() for m in singles], 0)
    assert torch.equal(tile, cat_tile), "the fused decode is not the per-role decode"
    assert torch.equal(scale, cat_scale), "the fused row scale is not the per-role row scale"

    offset = 0
    for name, p in parsed:
        reference, reference_scale = materialize(p.unit, p.forests, p.code)
        rows = reference.shape[0]
        assert torch.equal(tile[offset:offset + rows], reference.to(device)), name
        assert torch.equal(scale[offset:offset + rows],
                           reference_scale.to(device, torch.float32).reshape(-1)), name
        offset += rows
    assert offset == tile.shape[0]

    widths = sorted({int(p.unit.window_bits) for _, p in parsed})
    print(f"PASS  {tile.numel()} tile elements and {scale.numel()} row scales identical "
          f"across the fused and the per-role decode, at rungs {rungs}, "
          f"window_bits {widths}"
          f"{' (CROSS-BAND)' if len(widths) > 1 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
