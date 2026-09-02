#!/usr/bin/env python
"""Byte-compare two compressed-tensors checkpoints' quantized tensors.

Used to ask whether a fresh Tessera encode reproduces an earlier one (a
deterministic encoder makes the earlier stock receipt the comparator of the
fresh wire outright); exit 0 when every shared quantized tensor is identical,
1 otherwise, with the differing tensors named.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open

KEYS = (".weight", ".weight_scale", ".weight_packed", ".weight_global_scale", ".input_global_scale")


def load(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("model.layers.") and name.endswith(KEYS):
                    out[name] = handle.get_tensor(name)
    return out


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    ta, tb = load(a), load(b)
    shared = sorted(set(ta) & set(tb))
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    quantized = [n for n in shared if ta[n].dtype != torch.bfloat16]
    same = [n for n in quantized if ta[n].dtype == tb[n].dtype and ta[n].shape == tb[n].shape
            and torch.equal(ta[n].view(torch.uint8) if ta[n].dtype == torch.float8_e4m3fn else ta[n],
                            tb[n].view(torch.uint8) if tb[n].dtype == torch.float8_e4m3fn else tb[n])]
    diff = [n for n in quantized if n not in same]
    print(f"{a}\n{b}\nshared quantized tensors: {len(quantized)}  identical: {len(same)}  different: {len(diff)}")
    print(f"only in first: {len(only_a)}  only in second: {len(only_b)}")
    for n in diff[:12]:
        x, y = ta[n], tb[n]
        if x.dtype == torch.float8_e4m3fn:
            x, y = x.view(torch.uint8), y.view(torch.uint8)
        frac = float((x != y).float().mean()) if x.shape == y.shape else float("nan")
        print(f"  {n}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}, differing fraction {frac:.4f}")
    return 0 if not diff and not only_a and not only_b else 1


if __name__ == "__main__":
    sys.exit(main())
