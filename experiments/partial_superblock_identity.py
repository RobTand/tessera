"""Is a trailing partial superblock CORRECT?  (#44's question (a).)

Question (a) is separable from the cost half and answerable without a quiet
box: does the trailing partial superblock -- the last superblock of a unit
whose column count is not a whole number of 256 -- decode to the same values
those columns would decode to inside a *complete* superblock?

Three parts, in the order they gate each other:

1. **What widths are even admissible.**  #44 asks for `k*256 + 1` and
   `k*256 - 1`.  Both are odd, and NVFP4 packs two nibbles to a byte, so
   neither can exist as an NVFP4 tile at all.  The serving plugin says the
   same thing earlier and by name (`serving.scheme` gates on
   `ROUTES[family]["columns_multiple"]`, which is 16).  This part runs every
   gate on every candidate width and prints what refuses and what admits, so
   the reachable set is measured rather than argued.

2. **Cross-width code identity.**  At `released_positions=0` and an integral
   root (q256=768 -> root 3, so the Bresenham rate schedule is all-3s at every
   width), narrowing a tensor to a partial-superblock width must not move a
   single body bit of the columns it keeps.  Everything the encoder does is
   then position-local: the S6b scale groups are cut from the FLATTENED tensor
   (`encode._pack_scales`), so they align to row starts exactly when
   `cols % group == 0` -- which is why a width that is a multiple of 16 but
   not of 32 is a DIFFERENT treatment (#57) and is reported separately rather
   than mixed into the pair.

3. **Release over the partial block.**  #22's fix put the trailing block in
   the release quota and #27's fix made that quota width-proportional.  This
   checks the density the trailing block actually gets, that the encoder's
   placement and the reader's regeneration agree there, and that the artifact
   round-trips byte-for-byte at every admissible width (sha256 over the blob
   and over each plane).

Weight space, CPU-runnable, no timing claim.  The cost half is
`experiments/partial_superblock_loadcost.py`.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tessera.alphabet import build_forest
from tessera.decode import (
    decode_codes_mixed,
    materialize_nvfp4,
    release_order,
    reconstruct_unit,
)
from tessera.encode import _canonical_release_order, encode_unit
from tessera.errors import GrammarError
from tessera.grammar import (
    bresenham_rate_schedule,
    release_quota,
    root_from_q256,
    superblock_count,
    superblock_widths,
)
from tessera.manifest import RotationState, ScalePlaneKind

from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

Q256 = 768                      # root 3: integral, so the schedule is width-invariant
SUPERBLOCK = 256
GROUP, HALF = 32, 16
ROWS = int(os.environ.get("PSI_ROWS", "512"))
SOURCE = "/mnt/shared/tessera-ts44/gate_proj.pt"
DEV = os.environ.get("PSI_DEVICE", "cpu")
CC = ConvCode(memory=6)

# 5120 = 20 x 256 exactly (the only width any published load figure covers).
# 4864 = 19 x 256 exactly -- the second conforming control, adjacent to 4896.
# 4896 = 19 x 256 + 32   -- a nearly-EMPTY trailing partial block.
# 5088 = 19 x 256 + 224  -- a nearly-FULL trailing partial block.
# 4880 = 19 x 256 + 16   -- admissible to the plugin, but %32 == 16, so it also
#                           straddles an S6b group (#57): a second treatment.
# 4865 = 19 x 256 + 1    -- #44's "one column past a superblock multiple".
# 4863 = 18 x 256 + 255  -- #44's "one column short of one".  Both odd, and
#                           k*256 +- 1 is odd for every k, which is the whole
#                           point of part 1.  (5121 is not reachable from this
#                           tensor -- it is 5120 wide -- so the +1 case is taken
#                           one superblock down, where the arithmetic is the same.)
WIDTHS = [4865, 4863, 5119, 4880, 4864, 4896, 5088, 5120]
PAIR = [4864, 4896, 5088, 5120]


def sha(t) -> str:
    if isinstance(t, (bytes, bytearray)):
        return hashlib.sha256(bytes(t)).hexdigest()[:16]
    if t is None:
        return "-"
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def head(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


def load() -> torch.Tensor:
    W = torch.load(SOURCE, map_location="cpu")
    return W[:ROWS].to(DEV).float().contiguous()


def encode(W, cols, released=0, superblock=SUPERBLOCK, plane=ScalePlaneKind.S6B):
    if cols > W.shape[1]:
        raise GrammarError(f"{cols} columns asked of a {W.shape[1]}-column tensor")
    rates = bresenham_rate_schedule(root_from_q256(Q256), cols)
    forests = {r: build_forest(r) for r in sorted(set(rates))}
    u = encode_unit(
        W[:, :cols].contiguous(), forests, rates, CC,
        rotation=RotationState.NONE, with_diagonals=False,
        released_positions=released, group=GROUP, half=HALF, superblock=superblock,
        scale_plane=plane,
    )
    return u, forests, rates


def part1_admissibility(W):
    print("\n" + "=" * 78)
    print("1. WHICH WIDTHS ARE ADMISSIBLE AT ALL")
    print("=" * 78)
    from tessera.serving.scheme import ROUTES, TESSERA_NVFP4, validate_tessera_scheme

    mult = ROUTES[TESSERA_NVFP4]["columns_multiple"]
    print(f"serving.scheme ROUTES[{TESSERA_NVFP4}]['columns_multiple'] = {mult}"
          "   (scheme.py:454 gates on it) -- the plugin's K quantum is 16, NOT 256,")
    print("so every width that is a whole number of 16 columns and not of 256 declares a")
    print("trailing PARTIAL superblock the plugin admits.  The load-cost question is live.")
    print(f"the NVFP4 route reads plane={ROUTES[TESSERA_NVFP4]['plane']!r} "
          f"span={ROUTES[TESSERA_NVFP4]['span']} grids={ROUTES[TESSERA_NVFP4]['grids']}; "
          f"export.DEFAULT_SCALE_PLANE is LUT.  loadcost.py encodes the S6B plane at span 1,")
    print("which no route reads -- see the scope note at the end.")
    print(f"\n{'cols':>6} {'%256':>5} {'%32':>4} {'%16':>4} {'blk':>4} {'last':>5}  "
          f"{'sched':<6} {'encode':<8} {'nvfp4 pack':<52} {'plugin K gate':<44} {'artifact':<12}")
    rows_out = []
    for cols in WIDTHS:
        blocks = superblock_count(cols, SUPERBLOCK)
        last = superblock_widths(cols, SUPERBLOCK)[-1]
        cells = {}
        try:
            bresenham_rate_schedule(root_from_q256(Q256), cols)
            cells["schedule"] = "ok"
        except Exception as exc:                                   # noqa: BLE001
            cells["schedule"] = head(exc)
        u = None
        try:
            u, forests, _ = encode(W, cols)
            cells["encode"] = "ok"
        except Exception as exc:                                   # noqa: BLE001
            cells["encode"] = head(exc)
        if u is None:
            cells["pack"] = cells["artifact"] = "(no unit)"
        else:
            codes = decode_codes_mixed(u, forests, CC)
            try:
                materialize_nvfp4(codes, u.scale_base, u.scale_refine, u.group, u.half)
                cells["pack"] = "ok"
            except Exception as exc:                               # noqa: BLE001
                cells["pack"] = head(exc)
            try:
                _, _, blob = build_unit_artifact(u, f"w{cols}", forests, Q256, CC,
                                                 superblock=SUPERBLOCK)
                parse_unit_artifact(blob, device=DEV)
                cells["artifact"] = f"ok {len(blob)}B"
            except Exception as exc:                               # noqa: BLE001
                cells["artifact"] = head(exc)
        # Route-conforming in every field EXCEPT the width, so the only thing
        # that can refuse is the column gate.  (The NVFP4 route reads the LUT
        # plane and a span-2 TCQ body -- see the wire note printed below.)
        scheme = {"family": TESSERA_NVFP4, "grid": "E2M1x2", "body": "TCQ", "plane": "LUT",
                  "q256": 896, "rows": ROWS, "columns": cols, "wire_bytes": 4096,
                  "roles": [["weight", ROWS]]}
        try:
            validate_tessera_scheme(scheme, "t")
            cells["scheme"] = "ok"
        except Exception as exc:                                   # noqa: BLE001
            cells["scheme"] = head(exc)
        print(f"{cols:>6} {cols%256:>5} {cols%32:>4} {cols%16:>4} {blocks:>4} {last:>5}  "
              f"{cells['schedule']:<6} {cells['encode']:<8} {cells['pack']:<52} "
              f"{cells['scheme']:<44} {cells['artifact']:<12}")
        rows_out.append({"cols": cols, **cells})
    return rows_out


def part2_identity(W):
    print("\n" + "=" * 78)
    print("2. CROSS-WIDTH CODE IDENTITY, released_positions=0")
    print("=" * 78)
    base_cols = 5120
    base, base_forests, base_rates = encode(W, base_cols)
    base_codes = decode_codes_mixed(base, base_forests, CC)
    print(f"baseline {base_cols} columns = {superblock_count(base_cols, SUPERBLOCK)} whole "
          f"superblocks; rates all-{base_rates[0]} ({len(set(base_rates))} distinct)")
    print(f"{'cols':>6} {'last blk':>9} {'%32':>4}  {'rates':>7} {'body':>7} {'compl':>7} "
          f"{'sbase':>7} {'srefine':>8} {'codes':>7}  {'max |dcode|':>11} {'trailing-block err':>19}")
    out = []
    for cols in [c for c in WIDTHS if c not in (5121, 5119)]:
        u, forests, rates = encode(W, cols)
        codes = decode_codes_mixed(u, forests, CC)
        n_groups = (ROWS * cols) // GROUP
        n_halves = (ROWS * cols) // HALF
        same = {
            "rates": rates == base_rates[:cols],
            "body": torch.equal(u.body_bits, base.body_bits[:, :cols]),
            "compl": torch.equal(u.completion_bits, base.completion_bits[:, :cols]),
            "sbase": bool(u.scale_base.numel() == n_groups),
            "srefine": bool(u.scale_refine.numel() == n_halves),
            "codes": torch.equal(codes, base_codes[:, :cols]),
        }
        dcode = int((codes.long() - base_codes[:, :cols].long()).abs().max()) if cols <= base_cols else -1
        # relative reconstruction error over the trailing block's columns only
        rec = reconstruct_unit(u, forests, CC)
        lo = (superblock_count(cols, SUPERBLOCK) - 1) * SUPERBLOCK
        tgt = W[:, lo:cols]
        tail = float((rec[:, lo:cols] - tgt).pow(2).sum() / tgt.pow(2).sum()).__pow__(0.5)
        whole = float((rec - W[:, :cols]).pow(2).sum() / W[:, :cols].pow(2).sum()) ** 0.5
        print(f"{cols:>6} {superblock_widths(cols, SUPERBLOCK)[-1]:>9} {cols%32:>4}  "
              f"{str(same['rates']):>7} {str(same['body']):>7} {str(same['compl']):>7} "
              f"{str(same['sbase']):>7} {str(same['srefine']):>8} {str(same['codes']):>7}  "
              f"{dcode:>11} {tail:>10.6f} vs {whole:.6f}")
        out.append({"cols": cols, **{k: bool(v) for k, v in same.items()},
                    "tail_rel_err": tail, "whole_rel_err": whole})
    return out


def part2b_partition_control(W):
    """The sharpest form of question (a), and the only one that works on EVERY
    scale plane: hold the tensor, the width and the rate schedule fixed, and
    move ONLY the superblock partition.

    At ``superblock=cols`` the unit is one whole superblock; at
    ``superblock=256`` the same width is 19 whole blocks plus a partial one.
    Nothing else differs -- same bytes in, same global statistics, same
    flattening -- so any difference in the body IS the partition's effect.
    Matched pair, one treatment.

    This is the control the cross-width test cannot be on the LUT plane, whose
    ``_pack_scales_lut`` fits a table and a global scale over the WHOLE tensor:
    narrowing a LUT-plane tensor moves a tensor-global quantity for a reason
    that has nothing to do with block position, so cross-width identity is not
    the available claim there.  The partition control is.
    """
    print("\n" + "=" * 78)
    print("2b. PARTITION CONTROL: one width, one tensor, only the superblock moves")
    print("=" * 78)
    print(f"{'plane':>7} {'cols':>6} {'rel':>6}  {'blocks 256':>10} {'last':>5}  "
          f"{'body same':>9} {'compl same':>10} {'codes same':>10} {'relidx same':>11} "
          f"{'rt exact':>8}")
    out = []
    for plane in (ScalePlaneKind.S6B, ScalePlaneKind.LUT):
        for cols in (4896, 5088):
            for frac in (0.0, 0.125):
                n_rel = int(frac * ROWS * cols)
                whole, wf, _ = encode(W, cols, released=n_rel, superblock=cols, plane=plane)
                part, pf, _ = encode(W, cols, released=n_rel, superblock=SUPERBLOCK, plane=plane)
                cw = decode_codes_mixed(whole, wf, CC)
                cp = decode_codes_mixed(part, pf, CC)
                rec_w = reconstruct_unit(whole, wf, CC)
                rec_p = reconstruct_unit(part, pf, CC)
                row = {
                    "plane": plane.name, "cols": cols, "rel": frac,
                    "body": bool(torch.equal(whole.body_bits, part.body_bits)),
                    "compl": bool(torch.equal(whole.completion_bits, part.completion_bits)),
                    "codes": bool(torch.equal(cw, cp)),
                    "relidx": bool(torch.equal(torch.sort(whole.release_index)[0],
                                               torch.sort(part.release_index)[0])),
                    "rt": bool(torch.equal(rec_w, rec_p)),
                    "sse_whole": whole.sse, "sse_part": part.sse,
                }
                print(f"{plane.name:>7} {cols:>6} {frac:>6} {superblock_count(cols, SUPERBLOCK):>10} "
                      f"{superblock_widths(cols, SUPERBLOCK)[-1]:>5}  "
                      f"{str(row['body']):>9} {str(row['compl']):>10} {str(row['codes']):>10} "
                      f"{str(row['relidx']):>11} {str(row['rt']):>8}   "
                      f"sse {whole.sse:.6g} / {part.sse:.6g}")
                out.append(row)
    return out


def part3_lut_plane(W):
    """The export-default plane (``export.DEFAULT_SCALE_PLANE = LUT``), where
    the cross-width test is confounded and the round trip is still the claim."""
    print("\n" + "=" * 78)
    print("3b. THE EXPORT-DEFAULT PLANE (LUT): round trip and release density")
    print("=" * 78)
    print(f"{'cols':>6} {'last':>5} {'released':>9} {'dens last/unit':>17} "
          f"{'reader==writer':>14} {'rt exact':>8}  {'blob sha16':>16} {'tail/whole err':>17}")
    out = []
    for cols in PAIR:
        n_rel = int(0.125 * ROWS * cols)
        u, forests, _ = encode(W, cols, released=n_rel, plane=ScalePlaneKind.LUT)
        counts = release_quota(n_rel, cols, SUPERBLOCK)
        widths = superblock_widths(cols, SUPERBLOCK)
        _, _, blob = build_unit_artifact(u, f"l{cols}", forests, Q256, CC, superblock=SUPERBLOCK)
        parsed = parse_unit_artifact(blob, device=DEV)
        agree = bool(torch.equal(torch.sort(parsed.unit.release_index)[0],
                                 torch.sort(u.release_index)[0]))
        rec_a = reconstruct_unit(u, forests, CC)
        rec_b = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
        rt = bool(torch.equal(rec_a, rec_b))
        lo = (superblock_count(cols, SUPERBLOCK) - 1) * SUPERBLOCK
        tgt = W[:, lo:cols]
        tail = float((rec_a[:, lo:cols] - tgt).pow(2).sum() / tgt.pow(2).sum()) ** 0.5
        whole = float((rec_a - W[:, :cols]).pow(2).sum() / W[:, :cols].pow(2).sum()) ** 0.5
        print(f"{cols:>6} {widths[-1]:>5} {n_rel:>9} "
              f"{counts[-1] / (ROWS * widths[-1]):>8.6f}/{sum(counts) / (ROWS * cols):.6f} "
              f"{str(agree):>14} {str(rt):>8}  {sha(blob):>16} {tail:>8.6f}/{whole:.6f}")
        out.append({"cols": cols, "reader_eq_writer": agree, "roundtrip_exact": rt,
                    "blob_sha16": sha(blob), "tail_rel_err": tail, "whole_rel_err": whole})
    return out


def part3_release(W):
    print("\n" + "=" * 78)
    print("3. RELEASE OVER THE TRAILING PARTIAL BLOCK, and the byte round trip")
    print("=" * 78)
    print("no call site in src/ passes released_positions != 0 (only unit_artifact's")
    print("pass-through, calculator's parameter and artifact.py's filler bisect), so this")
    print("is a HARNESS treatment -- it is what loadcost.py has always measured (12.5%).")
    print(f"{'cols':>6} {'blocks':>6} {'last':>5} {'released':>9} {'quota(last 3)':>22} "
          f"{'density last/unit':>18} {'reader==writer':>14}  {'blob sha16':>12} {'roundtrip':>10}")
    out = []
    for cols in PAIR + [4880]:
        n_rel = int(0.125 * ROWS * cols)
        u, forests, rates = encode(W, cols, released=n_rel)
        blocks = superblock_count(cols, SUPERBLOCK)
        widths = superblock_widths(cols, SUPERBLOCK)
        counts = release_quota(n_rel, cols, SUPERBLOCK)
        dens_last = counts[-1] / (ROWS * widths[-1])
        dens_unit = sum(counts) / (ROWS * cols)
        # the reader regenerates the placement from bytes alone
        _, _, blob = build_unit_artifact(u, f"r{cols}", forests, Q256, CC, superblock=SUPERBLOCK)
        parsed = parse_unit_artifact(blob, device=DEV)
        agree = torch.equal(torch.sort(parsed.unit.release_index)[0],
                            torch.sort(u.release_index)[0])
        rec_a = reconstruct_unit(u, forests, CC)
        rec_b = reconstruct_unit(parsed.unit, parsed.forests, parsed.code)
        rt = torch.equal(rec_a, rec_b)
        # and the writer's own placement equals the general reader form at these counts
        pre = decode_codes_mixed(u, forests, CC, apply_release=False)
        ro = release_order(pre, cols, SUPERBLOCK, counts)
        ca = _canonical_release_order(pre, cols, SUPERBLOCK, n_rel)
        bound = torch.equal(ro, ca)
        print(f"{cols:>6} {blocks:>6} {widths[-1]:>5} {n_rel:>9} {str(counts[-3:]):>22} "
              f"{dens_last:>8.6f}/{dens_unit:.6f} {str(bool(agree)):>14}  {sha(blob):>12} "
              f"{str(bool(rt)):>10}   quota==canonical:{bool(bound)}")
        out.append({"cols": cols, "released": n_rel, "counts_tail": counts[-3:],
                    "density_last": dens_last, "density_unit": dens_unit,
                    "reader_eq_writer": bool(agree), "roundtrip_exact": bool(rt),
                    "quota_eq_canonical": bool(bound), "blob_sha16": sha(blob),
                    "blob_bytes": len(blob)})
    return out


def part4_digests(W):
    print("\n" + "=" * 78)
    print("4. BYTE DIGEST MATRIX (re-encode, not a diff)")
    print("=" * 78)
    print(f"{'cols':>6} {'rel':>4}  {'body':>16} {'compl':>16} {'sbase':>16} {'srefine':>16} "
          f"{'relidx':>16} {'blob':>16} {'bytes':>8}")
    out = []
    for cols in PAIR:
        for rel_frac in (0.0, 0.125):
            n_rel = int(rel_frac * ROWS * cols)
            u, forests, _ = encode(W, cols, released=n_rel)
            _, _, blob = build_unit_artifact(u, f"d{cols}_{n_rel}", forests, Q256, CC,
                                             superblock=SUPERBLOCK)
            row = {"cols": cols, "rel": rel_frac, "body": sha(u.body_bits),
                   "compl": sha(u.completion_bits), "sbase": sha(u.scale_base),
                   "srefine": sha(u.scale_refine), "relidx": sha(u.release_index),
                   "blob": sha(blob), "bytes": len(blob)}
            print(f"{cols:>6} {rel_frac:>4} {row['body']:>17} {row['compl']:>16} "
                  f"{row['sbase']:>16} {row['srefine']:>16} {row['relidx']:>16} "
                  f"{row['blob']:>16} {row['bytes']:>8}")
            out.append(row)
    return out


def main():
    sha_head = os.environ.get("PSI_HEAD") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    W = load()
    print(f"tessera HEAD {sha_head}")
    print(f"source {SOURCE}  rows {ROWS} of 17408  device {DEV}  torch {torch.__version__}")
    print(f"wire: E2M1 grid, TCQ body at the cap, S6B plane, span 1, q256={Q256} (root 3), "
          f"superblock {SUPERBLOCK}, group {GROUP}, half {HALF}")
    report = {"head": sha_head, "rows": ROWS, "device": DEV, "q256": Q256,
              "admissibility": part1_admissibility(W),
              "identity": part2_identity(W),
              "partition_control": part2b_partition_control(W),
              "release": part3_release(W),
              "lut_plane": part3_lut_plane(W),
              "digests": part4_digests(W)}
    dest = os.environ.get("PSI_JSON")
    if dest:
        with open(dest, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
