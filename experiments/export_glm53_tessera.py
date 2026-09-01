"""Export GLM-5.3-Flash in Tessera E2M1_K2, everything eligible.

The plan is WIDE on purpose.  Mia's EXL3 artifact quantizes routed experts and
leaves 15.44 GiB at BF16 on 2.6% of the params; the whole reason to have an
allocator is that this is leaving bytes on the floor.  Every 2-D weight the
wire can carry goes through Tessera at the family's top rung.

Excluded, and why -- each one structural, none of them a judgement call:
  * ``embed_tokens``  -- vLLM's VocabParallelEmbedding has no quantized path.
  * ``model.visual.*``-- inherit Mia's handling: BF16, outside the size match.
  * ``mlp.gate``      -- the router picks experts; its output is an argmax over
                         a 43-way softmax and a flipped route costs far more
                         than the ~0.09 GiB it saves.
  * non-2-D and non-``.weight`` -- norms, biases, ``A_log``, ``dt_bias``,
                         ``k_conv1d``: not Linears.
  * odd row counts    -- arity 2 pairs consecutive rows.

MTP (layer 45) is quantized like any other layer, which is what inheriting
Mia's treatment means: her artifact carries 288 EXL3 experts on that layer at
~4.31 bpw.  Vision stays BF16.  Both sit outside the body-to-body size match.

``export_checkpoint_streaming`` copies anything absent from the plan through
verbatim, so the exclusions above are expressed by omission rather than by a
second code path -- there is no branch that can disagree with the census.

``verify=True``: every unit is decoded back and compared before its bytes are
accepted.  The bytes are the claim (today's lesson: never a formula where an
accountant exists), and the accountant is the thing that writes them.
"""
import argparse
import json, sys, time
from pathlib import Path

import torch

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.export import export_checkpoint_streaming
from tessera.manifest import RotationState

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
OUT = "/mnt/shared/models/GLM-5.3-Flash-Tessera-E2M1K2-20260901"
PLAN = ("/mnt/shared/dq-runs/glm53-tessera-alloc-20260901/artifacts/"
        "glm53_tessera_plan.json")   # shared: both boxes read one plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="",
                    help="inclusive 1-based input-shard range, e.g. 61-120; "
                         "empty means all of them")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    plan = {k: int(v) for k, v in json.load(open(PLAN)).items()}
    shard_filter = None
    if args.shards:
        lo, _, hi = args.shards.partition("-")
        lo, hi = int(lo), int(hi or lo)
        # The 1:1 shard mapping is what makes a split safe: shards share no
        # state, so a disjoint subset writes exactly the files one box would.
        shard_filter = {f"model-{n:05d}-of-00120.safetensors"
                        for n in range(lo, hi + 1)}
    grid = tuple_grid(E2M1_GRID, 2)          # the serialisable arity-2 grid
    started = time.time()
    print(f"plan: {len(plan):,} tensors -> {args.out}  "
          f"shards={args.shards or 'all'}", flush=True)

    def progress(position, total, shard, n_units):
        elapsed = time.time() - started
        rate = elapsed / max(1, position)
        print(f"  shard {position:>3}/{total}  {shard}  units={n_units:,}  "
              f"{elapsed/60:.1f} min  eta {rate*(total-position)/60:.1f} min",
              flush=True)

    report = export_checkpoint_streaming(
        SRC, args.out, plan, grid=grid, rotation=RotationState.NONE,
        with_diagonals=False, device="cuda", verify=True, copy_aux=True,
        progress=progress, shard_filter=shard_filter,
        extra_config={"prismaquant_plan": "everything-eligible",
                      "source_model": SRC,
                      "inherits": {"vision": "bf16 passthrough (Mia)",
                                   "mtp_layer_45": "quantized like any layer (Mia)"}},
    )
    gib = lambda b: b / 2 ** 30
    print(f"\nunits            {len(report.units):,}")
    print(f"quantized params {report.quantized_params:,}")
    print(f"quantized bytes  {gib(report.quantized_bytes):.3f} GiB  "
          f"= {report.quantized_bytes*8/report.quantized_params:.4f} bpp")
    print(f"passthrough      {gib(report.passthrough_bytes):.3f} GiB")
    print(f"TOTAL            {gib(report.total_bytes):.3f} GiB")
    print(f"Mia total        163.560 GiB  -> "
          f"{(1-report.total_bytes/(163.560*2**30))*100:+.1f}%")
    print(f"elapsed          {(time.time()-started)/60:.1f} min")
    json.dump({"units": len(report.units),
               "quantized_params": report.quantized_params,
               "quantized_bytes": report.quantized_bytes,
               "passthrough_bytes": report.passthrough_bytes,
               "total_bytes": report.total_bytes,
               "grid_digest": report.grid_digest},
              open(f"{args.out}/export_report.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
