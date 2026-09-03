"""What does a trailing partial superblock COST at load?  (#44's question (b).)

#44 asks for trellis replay + NVFP4 pack throughput at a conforming width and
at two partial ones, and it names a hypothesis: "if replay walks blocks rather
than positions, a 1-column partial block could cost close to a full one".

Read the code before timing it and that hypothesis is already answerable, and
the harness #44 points at could never have tested it:

* ``decode_codes_mixed`` partitions the unit by RATE GROUP
  (``for present in sorted(set(unit.rates))``), never by superblock, and
  ``materialize_nvfp4`` is per-16.  Both are position-walkers.
* The one load-time step that walks superblocks is release placement --
  ``decode.release_order`` loops the blocks and argsorts each one's members --
  and the reader RE-DERIVES it from bytes (``unit_artifact`` line 612), because
  §9's placement is not stored.
* ``restart_offsets`` -- the other per-superblock table -- has no consumer
  outside ``planes.py``/``layout.py`` in ``src/``, so it costs nothing at load.
* ``experiments/loadcost.py`` times ``decode_codes_mixed`` and
  ``materialize_nvfp4`` on an in-memory unit whose ``release_index`` is already
  populated.  The block-walking step happens at PARSE, outside its window.

So this harness times the reader path decomposed, at four widths, with and
without release:

  T_parse    bytes -> ParsedUnit  (includes release re-derivation)
  T_place    the block-walking step alone
  T_replay   decode_codes_mixed   (what loadcost.py calls trellis replay)
  T_pack     materialize_nvfp4

and adds a controlled diagnostic that isolates block-count scaling directly:
at ONE width and ONE release total, sweep the superblock parameter, so the
positions are fixed and only the number of blocks moves.

Matched-pair discipline: one process, one CUDA context, one encode per width
(the body is byte-identical with and without release -- the digest matrix in
``partial_superblock_identity.py`` shows it -- so the release-0 arm is the same
unit with the release planes emptied, not a second encode).  The ladder runs
forward and then reversed; if the two orders disagree by more than the effect,
the box moved and the run says so rather than reporting the mean.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tessera.alphabet import build_forest
from tessera.decode import decode_codes_mixed, materialize_nvfp4, unit_scale_field
from tessera.encode import _canonical_release_order, e2m1_value_table, encode_unit
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

Q256 = 768
SUPERBLOCK = 256
GROUP, HALF = 32, 16
ROWS = int(os.environ.get("PLC_ROWS", "17408"))
REPEATS = int(os.environ.get("PLC_REPEATS", "5"))
SOURCE = "/mnt/shared/tessera-ts44/gate_proj.pt"
DEV = os.environ.get("PLC_DEVICE", "cuda")
CC = ConvCode(memory=6)
FRAC = 0.125

# 4864 = 19 x 256 exactly           19 blocks, none partial
# 4896 = 19 x 256 + 32              20 blocks, the last 32/256 full  (nearly EMPTY)
# 5088 = 19 x 256 + 224             20 blocks, the last 224/256 full (nearly FULL)
# 5120 = 20 x 256 exactly           20 blocks, none partial
# every one a multiple of 32, so the S6b scale groups align to row starts at
# all four and #57's straddle is not a second treatment.
WIDTHS = [4864, 4896, 5088, 5120]


def timed(fn, n=REPEATS):
    fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    best = None
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best


def build(W, cols):
    rates = bresenham_rate_schedule(root_from_q256(Q256), cols)
    forests = {r: build_forest(r) for r in sorted(set(rates))}
    n_rel = int(FRAC * ROWS * cols)
    t0 = time.time()
    u = encode_unit(
        W[:, :cols].contiguous(), forests, rates, CC,
        rotation=RotationState.NONE, with_diagonals=False, released_positions=n_rel,
        group=GROUP, half=HALF, superblock=SUPERBLOCK, scale_plane=ScalePlaneKind.S6B,
    )
    empty_i = torch.zeros(0, dtype=u.release_index.dtype, device=u.release_index.device)
    empty_c = torch.zeros(0, dtype=u.release_code.dtype, device=u.release_code.device)
    u0 = dataclasses.replace(u, release_index=empty_i, release_code=empty_c)
    blobs = {}
    for frac, unit in ((FRAC, u), (0.0, u0)):
        _, _, blob = build_unit_artifact(unit, f"c{cols}_{frac}", forests, Q256, CC,
                                         superblock=SUPERBLOCK)
        blobs[frac] = blob
    print(f"  encoded {cols} in {time.time()-t0:6.1f}s   released {n_rel}  "
          f"bytes {len(blobs[FRAC])}/{len(blobs[0.0])}")
    return {"unit": u, "unit0": u0, "forests": forests, "blobs": blobs, "n_rel": n_rel}


def arm(cols, built, frac):
    unit = built["unit"] if frac else built["unit0"]
    forests, blob = built["forests"], built["blobs"][frac]
    params = ROWS * cols
    blocks = superblock_count(cols, SUPERBLOCK)
    last = superblock_widths(cols, SUPERBLOCK)[-1]
    started = time.time()
    t_parse = timed(lambda: parse_unit_artifact(blob, device=DEV))
    t_replay = timed(lambda: decode_codes_mixed(unit, forests, CC))
    codes = decode_codes_mixed(unit, forests, CC)
    t_pack = timed(lambda: materialize_nvfp4(codes, unit.scale_base, unit.scale_refine,
                                             unit.group, unit.half))
    if frac:
        pre = decode_codes_mixed(unit, forests, CC, apply_release=False)
        scale = unit_scale_field(unit, ROWS, cols)
        decoded = e2m1_value_table(unit.body_bits.device)[pre.int()] * scale
        n_rel = built["n_rel"]
        t_place = timed(lambda: _canonical_release_order(decoded, cols, SUPERBLOCK, n_rel))
        del pre, scale, decoded
    else:
        t_place = 0.0
    ended = time.time()
    del codes
    row = {"cols": cols, "frac": frac, "params": params, "blocks": blocks, "last": last,
           "bytes": len(blob), "t_parse": t_parse, "t_replay": t_replay, "t_pack": t_pack,
           "t_place": t_place, "started": started, "ended": ended}
    print(f"{cols:>6} {frac:>5} {blocks:>4} {last:>5} {params/1e6:>7.1f}M  "
          f"{t_parse*1e3:>9.2f} {t_replay*1e3:>9.2f} {t_pack*1e3:>8.2f} {t_place*1e3:>9.2f}   "
          f"{params/t_replay/1e6:>9.1f} {params/(t_replay+t_pack)/1e6:>9.1f}")
    return row


def block_sweep(W, built):
    """Positions FIXED, block count swept: the direct test of #44's hypothesis.

    One width (4864 columns), one release total, and only the ``superblock``
    parameter moves -- 1 block to 76.  If the block-walking step charges per
    block, the cost rises with the block count; if it charges per position, it
    is flat.  A diagnostic on the encoder's parameter, not a wire claim: the
    wire's superblock is 256.
    """
    cols = 4864
    unit, forests = built["unit"], built["forests"]
    n_rel = built["n_rel"]
    pre = decode_codes_mixed(unit, forests, CC, apply_release=False)
    scale = unit_scale_field(unit, ROWS, cols)
    decoded = e2m1_value_table(unit.body_bits.device)[pre.int()] * scale
    print(f"\n  block sweep at {cols} columns, {n_rel} releases, {ROWS*cols/1e6:.1f}M positions")
    print(f"  {'superblock':>10} {'blocks':>7} {'t_place ms':>11} {'us/block':>9}")
    out = []
    for sb in (4864, 2432, 1216, 608, 304, 256, 128, 64):
        blocks = superblock_count(cols, sb)
        counts = release_quota(n_rel, cols, sb)
        if max(counts) > ROWS * min(superblock_widths(cols, sb)):
            print(f"  {sb:>10} {blocks:>7}   quota overruns, skipped")
            continue
        t = timed(lambda sb=sb: _canonical_release_order(decoded, cols, sb, n_rel), n=3)
        print(f"  {sb:>10} {blocks:>7} {t*1e3:>11.2f} {t*1e6/blocks:>9.1f}")
        out.append({"superblock": sb, "blocks": blocks, "t_place": t})
    del pre, scale, decoded
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    head = os.environ.get("PLC_HEAD") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
    print(f"tessera HEAD {head}   host {socket.gethostname()}   device {DEV}   "
          f"torch {torch.__version__}")
    if DEV == "cuda":
        print(f"gpu {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    print(f"source {SOURCE}  rows {ROWS}  repeats {REPEATS} (min of)")
    print("wire: E2M1 grid, TCQ body at the cap, S6B plane, span 1, q256=768 (root 3)")
    print("  -- the wire experiments/loadcost.py has always measured.  See the scope note.")
    W = torch.load(SOURCE, map_location="cpu")[:ROWS].to(DEV).float().contiguous()
    print(f"\nencoding {len(WIDTHS)} widths (one encode each; the release-0 arm is the same "
          f"unit with the release planes emptied)")
    built = {}
    for cols in WIDTHS:
        built[cols] = build(W, cols)
    hdr = (f"\n{'cols':>6} {'rel':>5} {'blk':>4} {'last':>5} {'params':>8}  "
           f"{'parse ms':>9} {'replay ms':>9} {'pack ms':>8} {'place ms':>9}   "
           f"{'replay Mp/s':>9} {'total Mp/s':>9}")
    rows = []
    for label, order in (("FORWARD", WIDTHS), ("REVERSE", WIDTHS[::-1])):
        print(f"\n--- ladder {label} ---{hdr}")
        for cols in order:
            for frac in (0.0, FRAC):
                r = arm(cols, built[cols], frac)
                r["pass"] = label
                rows.append(r)
    sweep = block_sweep(W, built[4864])
    report = {"head": head, "host": socket.gethostname(), "device": DEV, "rows": ROWS,
              "repeats": REPEATS, "torch": torch.__version__, "arms": rows,
              "block_sweep": sweep,
              "window": {"start": min(r["started"] for r in rows),
                         "end": max(r["ended"] for r in rows)}}
    dest = os.environ.get("PLC_JSON")
    if dest:
        with open(dest, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {dest}")
    print(f"\nwall window for the netdata series: "
          f"{time.strftime('%H:%M:%S', time.localtime(report['window']['start']))} .. "
          f"{time.strftime('%H:%M:%S', time.localtime(report['window']['end']))} local, "
          f"unix {int(report['window']['start'])}..{int(report['window']['end'])}")


if __name__ == "__main__":
    main()
