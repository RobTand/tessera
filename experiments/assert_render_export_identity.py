"""The producer/consumer seam: does the price the allocator paid equal the bytes?

PrismaQuant's ``render_tessera_weight`` is what the surrogate scores and what a
KL validation would evaluate.  ``tessera.export`` is what actually gets written.
Principle 8 says those must be the *same rendering* -- if they drift, the
allocator optimises one artifact and the exporter ships another, and every
number upstream is measuring a checkpoint that was never built.

Nothing structurally forces them to agree: they are two call sites, in two
repositories, that each independently build a rate schedule and a forest.  This
asserts the identity on real exported bytes rather than assuming it.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from tessera.export import BLOB_SUFFIX, load_tessera_weight, read_checkpoint_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("source", type=Path, help="the BF16 checkpoint it was built from")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from prismaquant.tessera_render import render_tessera_weight

    config = read_checkpoint_config(args.artifact)
    suffix = config.get("blob_suffix", BLOB_SUFFIX)
    grid = config["grid"]
    rungs = config["rungs_q256"]

    names = sorted(n for n in config["plan"])
    random.Random(args.seed).shuffle(names)
    names = names[: args.samples]

    src_index = json.loads(
        (args.source / "model.safetensors.index.json").read_text()
    )["weight_map"]

    from safetensors import safe_open

    print(f"grid {grid['base']} K{grid['arity']}  rungs {rungs}")
    print(f"checking {len(names)} of {len(config['plan'])} units\n")

    failures = 0
    for name in names:
        q256 = config["plan"][name]
        rung = f"TESSERA_{grid['base']}_K{grid['arity']}_R{q256}"
        with safe_open(str(args.source / src_index[name]), framework="pt") as h:
            weight = h.get_tensor(name).cuda()

        rendered = render_tessera_weight(weight, rung)
        exported = load_tessera_weight(args.artifact, name, device="cuda")

        same = torch.equal(rendered.float(), exported.float())
        if not same:
            failures += 1
            delta = (rendered.float() - exported.float()).abs().max().item()
            print(f"  MISMATCH {name}  ({rung})  max|d|={delta:.6e}")
        else:
            print(f"  ok  {name}  ({rung})  {tuple(weight.shape)}")

    print()
    if failures:
        raise SystemExit(
            f"{failures}/{len(names)} units differ between the render the "
            "allocator prices and the bytes the exporter wrote. The surrogate "
            "is scoring a checkpoint that was never built (principle 8)."
        )
    print(f"identity holds on {len(names)}/{len(names)} sampled units: the "
          "rendered price and the exported bytes are the same tensor.")


if __name__ == "__main__":
    main()
