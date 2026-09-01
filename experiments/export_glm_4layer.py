"""End-to-end Tessera export of the 4-layer GLM-5.3-Flash cut.

Small-scale-first, on the *target* architecture rather than a stand-in: same
tensor names, same packed-expert shapes, same attention layout as the model the
goal names.  What it establishes is narrow and mechanical -- that the walk
covers every eligible Linear, that every unit round-trips through its own
bytes, and what the artifact actually weighs.  It is not a quality result.

Eligibility follows what vLLM can structurally load, not what looks
quantizable: ``embed_tokens`` stays BF16 because ``VocabParallelEmbedding``
falls through ``get_quant_method`` to ``None``, so quantizing it would be
silently ignored at best.  ``lm_head`` IS quantizable and is included.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, tuple_grid
from tessera.export import export_checkpoint_streaming

SRC = "/mnt/shared/models/GLM-5.3-Flash-4layer"


def eligible(name: str, shape) -> bool:
    """2-D Linears only, minus the ones the runtime cannot load quantized."""
    if len(shape) != 2:
        return False
    if "embed_tokens" in name:          # structurally unquantizable in vLLM
        return False
    if name.endswith("_log") or "norm" in name:
        return False
    return name.endswith(".weight")


def build_plan(src: Path, q256: int) -> "dict[str, int]":
    index = src / "model.safetensors.index.json"
    plan, skipped = {}, []
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shard_names: "dict[str, list[str]]" = {}
        for tensor, shard in weight_map.items():
            shard_names.setdefault(shard, []).append(tensor)
    else:
        shard_names = {}
        for path in sorted(src.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as h:
                shard_names[path.name] = list(h.keys())
    for shard, names in shard_names.items():
        with safe_open(str(src / shard), framework="pt") as h:
            for name in names:
                shape = h.get_slice(name).get_shape()
                if eligible(name, shape):
                    if shape[0] % 2:            # arity 2 needs even rows
                        skipped.append((name, tuple(shape)))
                        continue
                    plan[name] = q256
    return plan, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0,
                    help="encode only the first N eligible tensors (smoke)")
    args = ap.parse_args()

    src = Path(SRC)
    plan, skipped = build_plan(src, args.q256)
    if args.limit:
        plan = dict(list(plan.items())[:args.limit])
    print(f"plan: {len(plan)} tensors at q256={args.q256}"
          f"  ({len(skipped)} skipped for odd rows)", flush=True)
    for name, shape in skipped[:5]:
        print(f"  skipped {name} {shape}")

    started = time.time()
    seen = [0]

    def progress(i, total, shard, units):
        rate = (units - seen[0]) or 1
        seen[0] = units
        print(f"  [{i}/{total}] {shard}  units={units}  "
              f"{time.time()-started:.0f}s", flush=True)

    report = export_checkpoint_streaming(
        src, args.out, plan, grid=tuple_grid(E2M1_GRID, 2),
        device="cuda", progress=progress,
    )
    elapsed = time.time() - started
    print(f"\nunits          {len(report.units)}")
    print(f"quantized      {report.quantized_bytes/2**30:.3f} GiB "
          f"over {report.quantized_params/1e9:.3f} G params")
    print(f"passthrough    {report.passthrough_bytes/2**30:.3f} GiB")
    print(f"total          {report.total_bytes/2**30:.3f} GiB")
    print(f"body bpp       {float(report.body_bpp):.6f}  (exact "
          f"{report.body_bpp.numerator}/{report.body_bpp.denominator})")
    print(f"elapsed        {elapsed:.0f}s  "
          f"({report.quantized_params/1e6/max(elapsed,1e-9):.1f} Mparam/s)")


if __name__ == "__main__":
    sys.exit(main())
