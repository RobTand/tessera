"""Decode a Tessera checkpoint back into a plain BF16 checkpoint.

**Why this exists.** Tessera has no serving backend yet, so there is no way to
put its weights in front of a real runtime -- and until something does, every
number the format has is a weight-space or offline screen, which principle 3
says is triage and not a result.

A decode-back serve closes that gap without a backend.  The bytes are decoded
by ``read_unit_artifact`` -- the same reader the format defines, and the one
``tests/test_kernel.py`` pins the Triton decode against with ``torch.equal``
-- so the tensors written here *are* the artifact's meaning.  A KL measured on
them is the KL of the Tessera artifact, and it carries to the kernel lane
because both lanes serve the identical W4A16 contract: the kernel decodes the
same codes to the same values, just later and in registers.

**One caveat, stated rather than glossed.**  ``read_unit_artifact`` returns
fp32 and this writes bf16, so the served weights carry a bf16 rounding of the
reconstruction that a kernel lane feeding an fp32 accumulator would not.  That
rounding is ~2^-9 relative against a reconstruction error of ~9e-2, i.e. ~0.2%
of the error energy -- small, and in the direction of making this arm slightly
*worse* than the kernel lane, not better.  It is not "exact"; it is bounded and
conservative.

**What it is not.**  The output is BF16-resident, so it is the same size as the
source and proves *nothing* about the size target.  The size claim lives only
in the kernel lane.  This measures quality, and only quality.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tessera.export import BLOB_SUFFIX, read_checkpoint_config
from tessera.unit_artifact import read_unit_artifact


def decode_back(src: Path, out: Path, *, device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16) -> dict:
    config = read_checkpoint_config(src)
    suffix = config.get("blob_suffix", BLOB_SUFFIX)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards: "dict[str, list[str]]" = {}
    for key, shard in index["weight_map"].items():
        shards.setdefault(shard, []).append(key)

    out.mkdir(parents=True, exist_ok=True)
    new_map: "dict[str, str]" = {}
    total_bytes = 0
    decoded = 0
    started = time.time()

    for position, shard in enumerate(sorted(shards), start=1):
        payload: "dict[str, torch.Tensor]" = {}
        with safe_open(str(src / shard), framework="pt") as handle:
            for key in shards[shard]:
                tensor = handle.get_tensor(key)
                if key.endswith(suffix):
                    name = key[: -len(suffix)]
                    blob = bytes(tensor.numpy().tobytes())
                    payload[name] = (
                        read_unit_artifact(blob, device=device).to(dtype).cpu()
                    )
                    decoded += 1
                else:
                    payload[key] = tensor.contiguous()
        for name, tensor in payload.items():
            new_map[name] = shard
            total_bytes += tensor.numel() * tensor.element_size()
        save_file(payload, str(out / shard), metadata={"format": "pt"})
        del payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{position}/{len(shards)}] {shard}  decoded={decoded}  "
              f"{time.time() - started:.0f}s", flush=True)

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=2))

    for pattern in ("*.json", "*.txt", "*.jinja", "*.model", "*.py"):
        for aux in src.glob(pattern):
            if aux.name in {"model.safetensors.index.json", "tessera_config.json"}:
                continue
            shutil.copy2(aux, out / aux.name)

    # The output is a plain BF16 model.  A leftover quantization_config would
    # make the runtime look for a scheme that is not in these bytes.
    config_path = out / "config.json"
    if config_path.exists():
        model_config = json.loads(config_path.read_text())
        model_config.pop("quantization_config", None)
        # Provenance: what these weights are, so a served number can never be
        # mistaken for a BF16 baseline reading.
        model_config["tessera_decoded_from"] = {
            "grid_digest": config.get("grid_digest"),
            "encoder": config.get("encoder", {}),
            "body_bpp": config.get("accounting", {}).get("body_bpp"),
            "units": decoded,
        }
        config_path.write_text(json.dumps(model_config, indent=2))

    return {"decoded": decoded, "bytes": total_bytes,
            "seconds": time.time() - started}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")

    stats = decode_back(args.src, args.out, device=args.device)
    print(f"\ndecoded {stats['decoded']} units -> "
          f"{stats['bytes'] / 2**30:.2f} GiB BF16 in {stats['seconds']:.0f}s")
    print(f"artifact: {args.out}")


if __name__ == "__main__":
    main()
