"""Compare one historical dense unit with today's default encode (#198).

This is a weight-space screen, not a served KL or a checkpoint re-export.
Both arms are read from their wire bytes by the same reference decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tessera.alphabet import SERIALISABLE_GRIDS
from tessera.container import parse
from tessera.export import encode_linear_planes
from tessera.fused import parse_fused
from tessera.unit_artifact import read_unit_artifact


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare(checkpoint: Path, source: Path, target: str, encoder_commit: str) -> dict:
    config_bytes = (checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    groups = config["quantization_config"]["config_groups"].values()
    (group,) = [g for g in groups if g["targets"] == [target]]
    scheme = group["scheme"]
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
        fused = f.get_tensor(target + ".wire_bytes").numpy().tobytes()
    (member,) = parse_fused(fused)
    with safe_open(str(source), framework="pt") as f:
        original = f.get_tensor(target + ".weight").contiguous()
    weight = original.to("cuda", torch.float32)
    grid = next(g for g in SERIALISABLE_GRIDS.values() if g.name == scheme["grid"])
    exported, _, _ = encode_linear_planes(
        weight, grid=grid, q256=scheme["q256"], name=target,
    )
    old, new = parse(member.blob), parse(exported.blob)
    reference = weight.double()
    energy = reference.square().sum().item()
    arms = {}
    for name, blob, parsed in (("historical", member.blob, old),
                               ("current", exported.blob, new)):
        rendered = read_unit_artifact(blob, device="cuda").double()
        sse = (rendered - reference).square().sum().item()
        arms[name] = {
            "blob_sha256": sha(blob), "header_minor": blob[10],
            "plane_region_sha256": sha(parsed.plane_region),
            "plane_region_bytes": len(parsed.plane_region),
            "encoder_fixture_id": (parsed.manifest.encoder_fixture_id.hex()
                                   if parsed.manifest.encoder_fixture_id is not None else None),
            "blob_bytes": len(blob), "blob_bpp": len(blob) * 8 / weight.numel(),
            "sse": sse, "relative_sse": sse / energy,
            "relative_rmse": (sse / energy) ** 0.5,
        }
    differences = [i for i, (a, b) in enumerate(zip(old.plane_region, new.plane_region))
                   if a != b]
    differences.extend(range(min(len(old.plane_region), len(new.plane_region)),
                             max(len(old.plane_region), len(new.plane_region))))
    return {
        "schema": "tessera.encoder-evidence-screen.v1",
        "encoder_commit": encoder_commit,
        "checkpoint": str(checkpoint), "source": str(source), "target": target,
        "source_tensor_sha256": sha(original.view(torch.uint8).numpy().tobytes()),
        "source_dtype": str(original.dtype), "shape": list(weight.shape),
        "config_sha256": sha(config_bytes), "scheme": scheme,
        "encoding": "encode_linear_planes defaults; no Hessian; verify=True",
        "metric": "sum((read_unit_artifact(blob) - source_weight)**2), float64 reduction",
        "source_energy": energy, "arms": arms,
        "differing_region_bytes": len(differences),
        "first_difference": differences[0] if differences else None,
        "last_difference": differences[-1] if differences else None,
        "same_payload": old.plane_region == new.plane_region,
        "current_over_historical_sse": arms["current"]["sse"] / arms["historical"]["sse"],
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "served_kl_measured": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--target", default="model.layers.0.mlp.down_proj")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--encoder-commit", required=True,
                   help="Full commit of the mounted encoder, resolved on the host")
    args = p.parse_args()
    if len(args.encoder_commit) != 40 or any(c not in "0123456789abcdef" for c in args.encoder_commit):
        p.error("--encoder-commit must be a full lowercase Git SHA-1")
    result = compare(args.checkpoint, args.source, args.target, args.encoder_commit)
    text = json.dumps(result, indent=2) + "\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
