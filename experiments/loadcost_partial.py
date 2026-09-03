"""What does a trailing partial superblock COST at load?  (issue #44)

``experiments/loadcost.py`` answers "what does Tessera cost at load" for one
shape.  This answers it for a *set* of shapes that differ only in whether the
last superblock is complete, which is the question #44 asks and which no
published figure has ever covered -- the one tensor that harness reads,
``Qwen3.8-27B layers.0.mlp.gate_proj``, is ``5120 = 20 x 256`` exactly.

**The matched set, and why these widths.**  #44 asks for ``k*256 + 1`` and
``k*256 - 1``.  Neither is measurable and neither is servable: ``decode.
materialize_nvfp4`` refuses an odd column count ("cannot pack 2 nibbles to a
byte") and ``kernel._require_column_groups`` refuses anything that is not a
whole number of 16-column scale groups.  A third alignment then rules out the
next candidates: ``encode._pack_scales`` cuts the S6b plane's 32-weight groups
out of the **flattened** tensor, so a width that is not a whole number of 32
straddles a row boundary and changes the encode of columns that were already
there.  Timing ``4880`` against ``4864`` would therefore vary the superblock
*and* the scale group in one arm and measure neither (principle: two
treatments are not a control).

So the widths below are all multiples of 32, and the only thing that varies
across them is how full the last superblock is:

    4864 = 19 x 256          complete -- control
    4896 = 19 x 256 +  32    partial, 32 of 256 -- thinnest clean partial
    5088 = 19 x 256 + 224    partial, 224 of 256 -- fattest clean partial
    5120 = 20 x 256          complete -- the shape every published figure covers
    4864                     the control REPEATED, last

**Why the control is repeated, and why the arms interleave** (the #13
lesson): two arms minutes apart on a shared box are two measurements of the
box as much as of the code, and #13's retracted numbers all came from
dividing one run by another run taken at a different time under different
load.  Two defences here.  The control is run first *and* last within every
pass, so drift inside a pass shows up as disagreement between two arms that
must agree.  And the whole sequence is repeated ``PASSES`` times rather than
each arm being run once to completion, so a load excursion lands on every
arm instead of on whichever arm was unlucky -- the per-pass numbers are kept
so an excursion is visible rather than averaged into the answer.

**The prediction, stated before the run.**  ``decode.decode_codes_mixed``
groups columns by rate and replays the trellis over rows; the release plane is
applied as a flat scatter (``codes.reshape(-1)[release_index] = ...``); and
``materialize_nvfp4`` is a stride-2 slice plus a scale relabelling.  The
string "superblock" does not appear in either.  Nothing walks blocks, so
M param/s should be flat across the set and #44's throughput cliff should not
exist.  A measurement that agrees with a read of the code is worth more than
either alone; a measurement that disagrees means the read is wrong.

**Scope.**  This is the wire ``loadcost.py`` builds -- E2M1 at the TCQ cap,
span 1, S6b block plane, no rotation, whole unit, TP=1 -- not the wire
``export.wire_recipe`` selects (span 2, LUT plane) and not the window body
that has been the default since 2026-09-02.  It licenses nothing about those.

    python experiments/loadcost_partial.py out.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))     # this checkout, not a shared one

import torch
from safetensors import safe_open

from tessera.alphabet import build_forest
from tessera.decode import decode_codes_mixed, materialize_nvfp4
from tessera.encode import encode_unit
from tessera.grammar import bresenham_rate_schedule, root_from_q256
from tessera.manifest import RotationState
from tessera.trellis import ConvCode

SUPERBLOCK = 256
ROWS = 4096             # a slice; big enough that the replay is not launch-bound
REPEATS = 15            # loadcost.py uses 3; a cliff hunt needs a spread
#: The box is shared, and a shared GB10 is BIMODAL, not merely noisy: at
#: PASSES=5 every arm's five per-pass medians alternated between roughly
#: 8.5 ms and 10.2 ms of replay, a ~20% swing that has nothing to do with the
#: width and everything to do with what else held the SMs.  Five samples of a
#: two-state process is not a measurement of either state; it is a measurement
#: of which state each arm happened to land in, and reporting its median as a
#: width effect is exactly the error #13 retracted three times.  The fix is
#: not a quieter box -- it is enough interleaved samples that both states land
#: on every arm in the same proportion, and a contrast reported with the
#: uncertainty that leaves.
PASSES = 200
RELEASE_FRACTION = 0.125

#: ``(cols, label)``.  Control first and last -- see the module docstring.
ARMS = [
    (4864, "complete, 19 superblocks (control)"),
    (4896, "partial, 32 of 256"),
    (5088, "partial, 224 of 256"),
    (5120, "complete, 20 superblocks"),
    (4864, "complete, 19 superblocks (control, repeated)"),
]

#: Where the harness may find ``gate_proj``.  ``/mnt/shared`` first so both
#: boxes read identical bytes; the digest is printed either way, so a reader
#: can tell whether two runs measured the same numbers.
SOURCES = [
    "/mnt/shared/tessera-ts44/gate_proj.pt",
    "/home/rob/tmp/ts44-gate_proj.pt",
]
HUB = ("/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
       "snapshots/*/model-*.safetensors")


def load_source() -> "tuple[torch.Tensor, str, str]":
    for path in SOURCES:
        if os.path.exists(path):
            W = torch.load(path, map_location="cpu")
            return W, path, hashlib.sha256(W.numpy().tobytes()).hexdigest()
    for path in sorted(glob.glob(HUB))[:4]:
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k.endswith("layers.0.mlp.gate_proj.weight"):
                    W = f.get_tensor(k).float().contiguous()
                    return W, path, hashlib.sha256(W.numpy().tobytes()).hexdigest()
    raise SystemExit("gate_proj not found; stage it to /mnt/shared/tessera-ts44/")


def timed(fn, repeats: int = REPEATS):
    """Wall and DEVICE time for one callable.

    Wall alone cannot separate "the kernel got slower" from "the box was
    busy", and on GB10 ``gpu_utilization`` reads ~96% for a stalled kernel
    and a saturated one alike.  CUDA events time the stream itself, so a
    disagreement between the two columns is the contention, made visible
    instead of argued about.
    """
    fn(); torch.cuda.synchronize()
    walls, devs = [], []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        t0 = time.perf_counter()
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        devs.append(start.elapsed_time(end) / 1e3)
    walls.sort(); devs.sort()
    return {
        "wall_median": walls[len(walls) // 2], "wall_min": walls[0],
        "dev_median": devs[len(devs) // 2], "dev_min": devs[0],
        "wall_p90": walls[int(0.9 * (len(walls) - 1))],
    }


def main() -> int:
    dev = "cuda"
    code = ConvCode(memory=6)
    forests = {r: build_forest(r) for r in (1, 2, 3)}
    W, source, digest = load_source()
    W = W[:ROWS].to(dev).float().contiguous()
    # The commit is the provenance of the number.  Read it from the tree
    # when there is one, and from the environment when this checkout was
    # rsynced to the other box without its ``.git`` -- never leave it blank.
    head = os.environ.get("TESSERA_COMMIT", "") or subprocess.run(
        ["git", "-C", str(HERE.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if not head:
        raise SystemExit("no commit: set TESSERA_COMMIT or run in a git tree")
    started = time.time()
    out = {
        "host": platform.node(), "commit": head, "source": source,
        "source_sha256": digest, "rows": ROWS, "repeats": REPEATS,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "started_unix": started, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loadavg": os.getloadavg(),
        "wire": "E2M1 grid, TCQ at cap (q256=768), span 1, S6b plane, "
                "no rotation, releases 12.5%, whole unit, TP=1",
        "arms": [],
    }
    print(f"host {out['host']}  commit {head[:9]}  gpu {out['gpu']}")
    print(f"source {source}\n  sha256 {digest[:32]}...")
    print(f"loadavg at start {out['loadavg']}   rows={ROWS}  repeats={REPEATS}")
    print(f"wire: {out['wire']}\n")
    # Encode once per width, outside the timing loop: the question is what a
    # LOAD costs, and a load reads bytes that were encoded long ago.
    built = {}
    for cols, _note in ARMS:
        if cols in built:
            continue
        Wc = W[:, :cols].contiguous()
        rates = bresenham_rate_schedule(root_from_q256(768), cols)
        built[cols] = encode_unit(
            Wc, forests, rates, code, rotation=RotationState.NONE,
            with_diagonals=False,
            released_positions=int(RELEASE_FRACTION * Wc.numel()),
        )
        print(f"  encoded {ROWS} x {cols}  "
              f"released {int(built[cols].release_index.numel())}")
    print()

    samples: dict = {}
    for p in range(PASSES):
        for slot, (cols, note) in enumerate(ARMS):
            unit = built[cols]
            replay = timed(lambda: decode_codes_mixed(unit, forests, code))
            codes = decode_codes_mixed(unit, forests, code)
            pack = timed(lambda: materialize_nvfp4(
                codes, unit.scale_base, unit.scale_refine, unit.group, unit.half))
            total_wall = replay["wall_median"] + pack["wall_median"]
            total_dev = replay["dev_median"] + pack["dev_median"]
            samples.setdefault((slot, cols, note), []).append({
                "pass": p, "replay": replay, "pack": pack,
                "total_wall": total_wall, "total_dev": total_dev,
                "m_param_per_s_wall": ROWS * cols / total_wall / 1e6,
                "m_param_per_s_dev": ROWS * cols / total_dev / 1e6,
            })
        if (p + 1) % 25 == 0 or p == 0:
            print(f"  pass {p + 1}/{PASSES} done at "
                  f"{time.strftime('%H:%M:%S')}  loadavg {os.getloadavg()[0]:.2f}")
    print()

    print(f"{'cols':>6} {'k*256+p':>11} {'Mparam':>8} "
          f"{'replay ms':>10} {'pack ms':>9} {'total ms':>9} "
          f"{'mean M/s':>10} {'+/-sem':>8} {'cv':>7} {'dev/wall':>9}  note")
    print("-" * 126)
    for (slot, cols, note), runs in samples.items():
        params = ROWS * cols
        med = lambda key: sorted(r[key] for r in runs)[len(runs) // 2]
        rates_ = sorted(r["m_param_per_s_wall"] for r in runs)
        n = len(rates_)
        mean = sum(rates_) / n
        var = sum((v - mean) ** 2 for v in rates_) / (n - 1) if n > 1 else 0.0
        sem = (var / n) ** 0.5
        arm = {
            "slot": slot, "cols": cols, "note": note,
            "partial": cols % SUPERBLOCK, "superblocks": -(-cols // SUPERBLOCK),
            "params": params, "passes": runs,
            "released": int(built[cols].release_index.numel()),
            "replay_ms": med("total_wall") and sorted(
                r["replay"]["wall_median"] for r in runs)[len(runs) // 2] * 1e3,
            "pack_ms": sorted(
                r["pack"]["wall_median"] for r in runs)[len(runs) // 2] * 1e3,
            "total_wall_median": med("total_wall"),
            "total_dev_median": med("total_dev"),
            # The number the question is actually about: cost PER PARAMETER.
            # A block-walking decoder would make this fall as the last
            # superblock empties; a position-walking one leaves it flat.
            "m_param_per_s_wall": rates_[len(rates_) // 2],
            "m_param_per_s_dev": sorted(
                r["m_param_per_s_dev"] for r in runs)[len(runs) // 2],
            "m_param_per_s_spread": (rates_[-1] - rates_[0]) / rates_[len(rates_) // 2],
            # Mean and its standard error, not the median: the per-pass
            # distribution is bimodal, and a median of a two-state process
            # reports whichever state took the middle sample rather than the
            # mixture the arm actually ran in.
            "mean": mean, "sem": sem, "n": n,
            "cv": (var ** 0.5) / mean,
        }
        out["arms"].append(arm)
        print(f"{cols:>6} {cols // SUPERBLOCK}*256+{cols % SUPERBLOCK:<5} "
              f"{params / 1e6:>8.2f} {arm['replay_ms']:>10.3f} "
              f"{arm['pack_ms']:>9.3f} {arm['total_wall_median'] * 1e3:>9.3f} "
              f"{arm['mean']:>10.1f} {arm['sem']:>8.1f} "
              f"{arm['cv'] * 100:>6.1f}% "
              f"{arm['total_dev_median'] / arm['total_wall_median']:>9.3f}  {note}")

    out["finished_unix"] = time.time()
    out["loadavg_end"] = os.getloadavg()

    # The control, twice.  Their disagreement is the box; anything smaller
    # than it is not a difference this run can see.
    out["passes"] = PASSES
    ctrl = [a for a in out["arms"] if a["cols"] == 4864]

    # The two control arms are the SAME width run in two slots of every pass.
    # Whatever separates them is the box, not the code, so their difference
    # -- with its own uncertainty -- is the floor below which this run cannot
    # resolve anything.  A contrast smaller than the floor is not a small
    # effect; it is no measurement at all.
    drift = abs(ctrl[0]["mean"] - ctrl[-1]["mean"])
    floor = drift / ctrl[0]["mean"]
    ctrl_sem = (ctrl[0]["sem"] ** 2 + ctrl[-1]["sem"] ** 2) ** 0.5 / ctrl[0]["mean"]
    out["control_drift_frac"] = floor
    out["control_drift_sem_frac"] = ctrl_sem
    print(f"\ncontrol: two 4864 arms, same width, different slots -- "
          f"{ctrl[0]['mean']:.1f} vs {ctrl[-1]['mean']:.1f} M param/s")
    print(f"  they differ by {floor * 100:.2f}% +/- {ctrl_sem * 100:.2f}% "
          f"-- the resolution floor of this run")

    partial = [a for a in out["arms"] if a["partial"]]
    complete = [a for a in out["arms"] if not a["partial"]]
    pm = sum(a["mean"] for a in partial) / len(partial)
    cm = sum(a["mean"] for a in complete) / len(complete)
    psem = (sum(a["sem"] ** 2 for a in partial) ** 0.5) / len(partial)
    csem = (sum(a["sem"] ** 2 for a in complete) ** 0.5) / len(complete)
    ratio_sem = (pm / cm) * ((psem / pm) ** 2 + (csem / cm) ** 2) ** 0.5
    out["partial_vs_complete_ratio"] = pm / cm
    out["partial_vs_complete_ratio_sem"] = ratio_sem
    print(f"partial mean {pm:.1f} +/- {psem:.1f} M param/s vs complete "
          f"{cm:.1f} +/- {csem:.1f}")
    print(f"  ratio {pm / cm:.4f} +/- {ratio_sem:.4f}  "
          f"({(pm / cm - 1) * 100:+.2f}% +/- {ratio_sem * 100:.2f}%)")
    # #44's hypothesis in its own terms: if replay walked BLOCKS, a unit
    # whose last block holds 32 of 256 columns would pay for 256 of them.
    thin = next(a for a in out["arms"] if a["cols"] == 4896)
    base = ctrl[0]
    # If replay charged per BLOCK, a 4896-column unit would pay for 20 blocks
    # while holding 19.125 blocks' worth of columns: 4.58% more per parameter
    # than the 4864-column control, which holds exactly 19.  That is the
    # effect size #44 hypothesises, and it is the number the resolution floor
    # has to be compared against.
    out["block_walk_predicted_excess_frac"] = (
        thin["superblocks"] / (thin["cols"] / SUPERBLOCK)
        / (base["superblocks"] / (base["cols"] / SUPERBLOCK)) - 1)
    out["observed_excess_frac"] = base["mean"] / thin["mean"] - 1
    print(f"\n#44's hypothesis, as an effect size: if replay charged per "
          f"BLOCK, 4896 would pay for")
    print(f"  {thin['superblocks']} blocks while holding "
          f"{thin['cols'] / SUPERBLOCK:.3f} -- "
          f"{out['block_walk_predicted_excess_frac'] * 100:+.2f}% cost per "
          f"parameter vs the 4864 control.")
    print(f"  Observed: {out['observed_excess_frac'] * 100:+.2f}%.")
    charged = base["total_wall_median"] / base["superblocks"] * thin["superblocks"]
    out["block_walk_prediction_ms"] = charged * 1e3
    out["block_walk_observed_ms"] = thin["total_wall_median"] * 1e3
    print(f"\n#44's block-walk hypothesis, priced: a 4896-column unit has "
          f"{thin['superblocks']} superblocks")
    print(f"  and would cost {charged * 1e3:.3f} ms if replay walked blocks; "
          f"it costs {thin['total_wall_median'] * 1e3:.3f} ms.")
    print(f"  Per-parameter cost is what a block walk would inflate: "
          f"{thin['m_param_per_s_wall']:.1f} vs {base['m_param_per_s_wall']:.1f} "
          f"M param/s.")
    print(f"\nwindow for netdata: {time.strftime('%H:%M:%S', time.localtime(started))}"
          f" .. {time.strftime('%H:%M:%S')}  "
          f"(unix {int(started)}..{int(out['finished_unix'])})")
    print(f"loadavg at end {out['loadavg_end']}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {sys.argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
