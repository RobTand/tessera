"""Render a checkpoint through a PrismaQuant format and write it back as BF16.

The comparator arm for ``decode_back_to_bf16.py``.  To compare Tessera against
NVFP4 **on the serving metric** both arms must reach a real runtime, and the
decode-back trick gets Tessera there without a backend.  This does the same for
any registry format, so the two arms differ in the quantizer and in nothing
else -- same corpus, same serve, same dump, same tokenizer, same box.

**This gives the format's W4A16 error, and for NVFP4 that is GENEROUS.**  On
the GLM route NVFP4 serves W4A4: ``flashinfer_b12x`` quantizes the activation
to FP4 as well, and measured on real expert activations that took the error
from 0.0667 to 0.1092 (+64%).  A weight-only NVFP4 arm therefore understates
NVFP4's served error, so a Tessera win here is a win stated against the
comparator's better face, not its worse one.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import prismaquant.format_registry as fr


def eligible(name, shape):
    if len(shape) != 2 or "embed_tokens" in name:
        return False
    if "visual" in name or "vision" in name:
        return False
    if name.endswith("_log") or "norm" in name:
        return False
    return name.endswith(".weight")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--format", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = fr.get_format(args.format)
    print(f"{args.format}: {spec.effective_bits_for_shape((4096, 2048)):.4f} bpp "
          f"(family {spec.family})", flush=True)

    index_path = args.src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = {}
        for t, s in weight_map.items():
            shards.setdefault(s, []).append(t)
    else:
        shards = {}
        for p in sorted(args.src.glob("*.safetensors")):
            with safe_open(str(p), framework="pt") as h:
                shards[p.name] = list(h.keys())

    args.out.mkdir(parents=True, exist_ok=True)
    new_map, total, rendered = {}, 0, 0
    started = time.time()
    for i, shard in enumerate(sorted(shards), start=1):
        payload = {}
        with safe_open(str(args.src / shard), framework="pt") as h:
            for name in shards[shard]:
                t = h.get_tensor(name)
                if eligible(name, tuple(t.shape)):
                    q = spec.quantize_dequantize(t.to(args.device))
                    payload[name] = q.to(torch.bfloat16).cpu()
                    rendered += 1
                else:
                    payload[name] = t.contiguous()
        for n, t in payload.items():
            new_map[n] = shard
            total += t.numel() * t.element_size()
        save_file(payload, str(args.out / shard), metadata={"format": "pt"})
        del payload
        torch.cuda.empty_cache()
        print(f"  [{i}/{len(shards)}] {shard}  rendered={rendered}  "
              f"{time.time() - started:.0f}s", flush=True)

    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": new_map}, indent=2))
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in args.src.glob(pattern):
            if aux.name != "model.safetensors.index.json":
                shutil.copy2(aux, args.out / aux.name)
    cfg = args.out / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        c.pop("quantization_config", None)
        c["prismaquant_rendered_as"] = args.format
        cfg.write_text(json.dumps(c, indent=2))
    print(f"\nrendered {rendered} tensors as {args.format} -> "
          f"{total / 2**30:.2f} GiB BF16 in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
