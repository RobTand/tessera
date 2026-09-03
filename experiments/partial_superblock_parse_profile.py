"""Where does the reader's time go?  A profile, not a bench decomposition.

``partial_superblock_loadcost.py`` times the reader path in four components and
finds ``parse_unit_artifact`` ~50x the sum of the two steps
``experiments/loadcost.py`` measures.  A component bench says WHICH call is
slow; it does not say where the time goes inside it, and a claim about that is
not licensed by a bench number.  So: two profilers on one call.

  * ``cProfile``      -- Python-level attribution, cumtime.
  * ``torch.profiler`` -- the host/device split, which cProfile cannot see.

One width (4864, all-full superblocks), two arms (no release / 12.5% release),
so the release re-derivation shows up as a diff between two profiles of the
same function rather than as a separate timing.
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import socket
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tessera.alphabet import build_forest
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import RotationState, ScalePlaneKind
from tessera.trellis import ConvCode
from tessera.unit_artifact import build_unit_artifact, parse_unit_artifact

Q256, SUPERBLOCK, GROUP, HALF = 768, 256, 32, 16
ROWS = int(os.environ.get("PSP_ROWS", "17408"))
COLS = int(os.environ.get("PSP_COLS", "4864"))
DEV = os.environ.get("PSP_DEVICE", "cuda")
SOURCE = "/mnt/shared/tessera-ts44/gate_proj.pt"
CC = ConvCode(memory=6)


def main() -> None:
    print(f"host {socket.gethostname()}  device {DEV}  torch {torch.__version__}  "
          f"rows {ROWS} cols {COLS}  HEAD {os.environ.get('PSP_HEAD', '?')}")
    W = torch.load(SOURCE, map_location="cpu")[:ROWS, :COLS].to(DEV).float().contiguous()
    rates = bresenham_rate_schedule(root_from_q256(Q256), COLS)
    forests = {r: build_forest(r) for r in sorted(set(rates))}
    blobs = {}
    for frac in (0.0, 0.125):
        unit = encode_unit(
            W, forests, rates, CC, rotation=RotationState.NONE, with_diagonals=False,
            released_positions=int(frac * W.numel()), group=GROUP, half=HALF,
            superblock=SUPERBLOCK, scale_plane=ScalePlaneKind.S6B,
        )
        _, _, blob = build_unit_artifact(unit, f"c{COLS}_{frac}", forests, Q256, CC,
                                         superblock=SUPERBLOCK)
        blobs[frac] = blob
        print(f"  encoded frac {frac}: {len(blob)} bytes")

    for frac, blob in blobs.items():
        parse_unit_artifact(blob, device=DEV)          # warm: no JIT, no lazy import
        if DEV == "cuda":
            torch.cuda.synchronize()
        print(f"\n{'='*78}\nPARSE PROFILE  cols {COLS}  release fraction {frac}\n{'='*78}")
        pr = cProfile.Profile()
        pr.enable()
        parse_unit_artifact(blob, device=DEV)
        if DEV == "cuda":
            torch.cuda.synchronize()
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(14)
        print("\n".join(buf.getvalue().splitlines()[4:26]))

        acts = [torch.profiler.ProfilerActivity.CPU]
        if DEV == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=acts) as prof:
            parse_unit_artifact(blob, device=DEV)
            if DEV == "cuda":
                torch.cuda.synchronize()
        key = "self_cuda_time_total" if DEV == "cuda" else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=12))
        tot = prof.key_averages()
        cpu = sum(e.self_cpu_time_total for e in tot)
        cuda = sum(getattr(e, "self_device_time_total", 0.0) or 0.0 for e in tot)
        print(f"  self time totals: host {cpu/1e3:9.1f} ms   device {cuda/1e3:9.1f} ms")


if __name__ == "__main__":
    main()
