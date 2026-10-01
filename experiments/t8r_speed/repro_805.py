"""tessera#805: the dense split launch's descriptor-slot window, on poisoned workspaces.

The dense launch splits K into ``S`` items per (row block, superblock).  Its
producers write an item's descriptor into slot ``item_idx & 1`` when they
claim it, and wait only per chunk, for the chunk two before.  The consumers
read the slot once, at item start.  So two consecutive one-chunk items in one
CTA (``floor(nk / S) < 2``) let the producers rewrite slot ``j & 1`` with item
``j + 2``'s descriptor before the consumers read item ``j``'s.  Item ``j``'s
chunk is then reduced into item ``j + 2``'s partial, and item ``j``'s partial
is never written.  The E2M1 launch refuses that split; this one does not.

A never-written partial holds whatever the workspace held.
``routed_fused.dense_forward`` allocates it with ``torch.empty`` on every call,
and the caching allocator returns the previous identical launch's block, whose
value is correct.  A repeat-the-launch hash check is blind to the race.  So
every launch here calls the native op directly and fills ``partial`` and
``out`` with NaN first.  A never-written partial then reaches the output as
NaN.

Per (library, shape, M, S, grid) cell, ``--reps`` launches are counted for:

* ``nan_launches``: launches whose output holds a NaN;
* ``mismatch_launches``: launches whose bits differ from the first launch;
* ``first_over_bound``: the first output against the fp64 reference, as the
  largest ratio to the derived bound for that split (``fused_bound``);
* ``sha256``: the first output's bytes, so two arms can be compared cell by cell.

``S`` takes master's own pick (``dense_k_split``) and the forced splits
``nk // 2`` (every item two chunks or more), ``nk // 2 + 1`` and ``nk`` (every
item one chunk).  Grids below the SM count put more items on each CTA in
sequence, which is the race's precondition.  The output is one JSON line per
cell, plus a summary.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

from tessera import routed_fused as rf

import fused_bound as fb
from test_dense_fused_window import _a64, _inputs, _role


def _family(library):
    return rf.LIBRARIES[library][1]


def _prebuild(libraries):
    """Build the libraries in parallel subprocesses (one nvcc each), so this
    process only loads them."""
    procs = [subprocess.Popen([sys.executable, "-c", f"from tessera import routed_fused as rf; rf._ext({lib!r})"])
             for lib in libraries]
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise SystemExit(f"library build failed: {dict(zip(libraries, rcs))}")


def _launch(lib, role, xq, a, out, counter, s, partial, grid, empty):
    lib.dense_forward(
        bool(role.fp8), xq, a if a is not None else empty,
        role.words, role.table16, role.init, role.has_init, role.wscale,
        role.runs, role.bdesc, int(role.tile_words), int(role.slot_words), counter, int(s),
        partial if s > 1 else empty, out, int(grid), int(rf.BM))


def _sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def _splits(kinds, m, rows, cols, sms, tile_words):
    nk = cols // rf.BK
    master = rf.dense_k_split(m, rows, cols, sms, tile_words=tile_words)
    named = {"master": master, "half": nk // 2, "half+1": nk // 2 + 1, "full": nk}
    seen, out = set(), []
    for kind in kinds:
        s = named[kind]
        if 1 < s <= nk and s not in seen:
            seen.add(s)
            out.append((kind, s))
    return master, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, help="label recorded on every line (e.g. master, mutant)")
    ap.add_argument("--libraries", default="value,e4m3,e4m3mma")
    ap.add_argument("--shapes", default="32x4096,64x4096,4096x128,128x256,256x256")
    ap.add_argument("--ms", default="1,2,3,4,5,6")
    ap.add_argument("--splits", default="master,half,half+1,full")
    ap.add_argument("--grids", default="1,2,4,8,16,0", help="0 = the SM count (master's grid)")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    libraries = args.libraries.split(",")
    t_build = time.time()
    _prebuild(libraries)
    t_build = time.time() - t_build
    dev = torch.cuda.current_device()
    sms = rf._sm_count(dev)
    grids = [sms if int(g) == 0 else int(g) for g in args.grids.split(",")]
    lines = open(os.path.join(args.out, f"repro_805-{args.arm}.jsonl"), "w")
    meta = {"arm": args.arm, "sms": sms, "device": torch.cuda.get_device_name(dev),
            "head": os.environ.get("TESSERA_HEAD"), "kernel_sha": os.environ.get("KERNEL_SHA"),
            "reps": args.reps, "grids": grids, "build_s": round(t_build, 1), "torch": torch.__version__}
    print(json.dumps({"meta": meta}), flush=True)
    lines.write(json.dumps({"meta": meta}) + "\n")
    summary = {"cells": 0, "nan_cells": [], "mismatch_cells": [], "over_bound_cells": [], "api_mismatch": []}
    t0 = time.time()
    for library in libraries:
        family = _family(library)
        if family == "e4m3":
            os.environ[rf.ENV_E4M3_MMA] = "e4m3" if rf.LIBRARIES[library][2] else "f16"
        lib = rf._ext(library)
        for shape in args.shapes.split(","):
            rows, cols = (int(v) for v in shape.split("x"))
            nk = cols // rf.BK
            expert, bundle = _role(family, rows=rows, cols=cols, seed=805 + rows // 32 + cols // 32)
            role = rf.prepare_dense_role(bundle)
            assert role.library == library, (role.library, library)
            w64 = fb.fp64_weight(expert, family)
            for m in (int(v) for v in args.ms.split(",")):
                _x, xq, a = _inputs(family, m, cols, 8050 + m)
                a64 = _a64(family, xq, a)
                empty = xq.new_empty(0, dtype=torch.float32)
                # The public path at master's split, unpoisoned: the launch the
                # native calls below must reproduce at (master's S, grid = sms).
                api = torch.empty(m, rows, dtype=torch.bfloat16, device="cuda")
                rf.dense_forward(role, xq, a, api, torch.zeros(1, dtype=torch.int32, device="cuda"))
                master_s, splits = _splits(args.splits.split(","), m, rows, cols, sms, role.tile_words)
                for kind, s in splits:
                    r, bound = fb.dense_bound(family, a64, w64, cols, s)
                    for grid in grids:
                        out = torch.empty(m, rows, dtype=torch.bfloat16, device="cuda")
                        partial = torch.empty(s * m * rows, dtype=torch.float32, device="cuda")
                        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                        nan_l = torch.zeros((), dtype=torch.int64, device="cuda")
                        mis_l = torch.zeros((), dtype=torch.int64, device="cuda")
                        first = None
                        for _ in range(args.reps):
                            partial.fill_(float("nan"))
                            out.fill_(float("nan"))
                            counter.zero_()
                            _launch(lib, role, xq, a, out, counter, s, partial, grid, empty)
                            nan_l += out.isnan().any().long()
                            if first is None:
                                first = out.clone()
                            else:
                                mis_l += (out.view(torch.int16) != first.view(torch.int16)).any().long()
                        torch.cuda.synchronize()
                        d = (first.double() - r).abs()
                        ratio = float((d / bound).nan_to_num(float("inf")).max())
                        cell = {"arm": args.arm, "library": library, "rows": rows, "cols": cols, "nk": nk,
                                "m": m, "split_kind": kind, "s": s, "master_s": master_s,
                                "min_chunks": nk // s, "grid": grid, "reps": args.reps,
                                "nan_launches": int(nan_l), "mismatch_launches": int(mis_l),
                                "first_over_bound": ratio, "sha256": _sha(first)}
                        if s == master_s and grid == sms:
                            cell["equals_public_api"] = bool(torch.equal(first.view(torch.int16),
                                                                         api.view(torch.int16)))
                            if not cell["equals_public_api"]:
                                summary["api_mismatch"].append([library, shape, m, s])
                        lines.write(json.dumps(cell) + "\n")
                        lines.flush()
                        key = [library, shape, m, kind, s, grid]
                        summary["cells"] += 1
                        if cell["nan_launches"]:
                            summary["nan_cells"].append(key + [cell["nan_launches"]])
                        if cell["mismatch_launches"]:
                            summary["mismatch_cells"].append(key + [cell["mismatch_launches"]])
                        if ratio > 1.0:
                            summary["over_bound_cells"].append(key + [ratio])
                print(f"REPRO {args.arm} {library} {shape} M={m} master_S={master_s} "
                      f"splits={[s for _, s in splits]} elapsed={time.time() - t0:.0f}s "
                      f"nan_cells={len(summary['nan_cells'])} mismatch_cells={len(summary['mismatch_cells'])}",
                      flush=True)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["meta"] = meta
    with open(os.path.join(args.out, f"repro_805-{args.arm}-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({"summary": {k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()
                                  if k != "meta"}}), flush=True)


if __name__ == "__main__":
    main()
