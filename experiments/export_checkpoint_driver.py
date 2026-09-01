"""Export any BF16 safetensors checkpoint to Tessera at one uniform rung.

Model-agnostic on purpose: the GLM driver hardcoded a path and a layer count,
which made the exporter look model-specific when it is not.  What IS
model-specific is the eligibility rule, and that is one function here.
"""
import argparse
import json
import time
from pathlib import Path

from safetensors import safe_open

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.export import export_checkpoint_streaming

BASES = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID}


def eligible(name: str, shape) -> bool:
    """2-D Linears only, minus the ones the runtime cannot load quantized."""
    if len(shape) != 2:
        return False
    if "embed_tokens" in name:          # structurally unquantizable in vLLM
        return False
    if "visual" in name or "vision" in name:
        return False                    # inherit the source's vision handling
    if name.endswith("_log") or "norm" in name:
        return False
    return name.endswith(".weight")


def build_plan(src: Path, q256: int, arity: int):
    index = src / "model.safetensors.index.json"
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
    plan, skipped = {}, []
    for shard, names in shard_names.items():
        with safe_open(str(src / shard), framework="pt") as h:
            for name in names:
                shape = h.get_slice(name).get_shape()
                if not eligible(name, shape):
                    continue
                if shape[0] % arity:
                    # A k-tuple spans `arity` consecutive rows; an odd-row
                    # tensor has no whole last tuple. Passthrough, loudly.
                    skipped.append((name, tuple(shape)))
                    continue
                plan[name] = q256
    return plan, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--base", default="E2M1", choices=sorted(BASES))
    ap.add_argument("--arity", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    plan, skipped = build_plan(args.src, args.q256, args.arity)
    print(f"plan: {len(plan)} tensors at {args.base} K{args.arity} q256={args.q256}"
          f"  ({len(skipped)} skipped for rows % {args.arity})", flush=True)
    for name, shape in skipped[:5]:
        print(f"    skipped {name} {shape}", flush=True)

    started = time.time()

    def progress(i, n, shard, units):
        print(f"  [{i}/{n}] {shard}  units={units}  {time.time() - started:.0f}s",
              flush=True)

    report = export_checkpoint_streaming(
        args.src, args.out, plan,
        grid=tuple_grid(BASES[args.base], args.arity),
        device=args.device, progress=progress,
    )
    print(f"\nquantized      {report.quantized_bytes / 2**30:.3f} GiB over "
          f"{report.quantized_params / 1e9:.3f} G params")
    print(f"passthrough    {report.passthrough_bytes / 2**30:.3f} GiB")
    print(f"total          {report.total_bytes / 2**30:.3f} GiB")
    print(f"body bpp       {float(report.body_bpp):.6f}  (exact "
          f"{report.body_bpp.numerator}/{report.body_bpp.denominator})")
    print(f"elapsed        {time.time() - started:.0f}s  "
          f"({report.quantized_params / 1e6 / (time.time() - started):.1f} Mparam/s)")


if __name__ == "__main__":
    main()
