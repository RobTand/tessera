"""What the served expert bytes reconstruct, on the wire that was actually served.

The KL beside a 4-layer cut is not a quality number
(``docs/measurements/tessera-moe-served-2026-09-04.md``): the reference has no
opinion at any of 4088 positions, so nothing downstream of the loader can be
graded by it.  What CAN be graded on that checkpoint is the bytes -- the routed
expert wires the exporter wrote, decoded by the reader the plugin's expert
route calls, against the BF16 rows they were made from.

THIS IS A WEIGHT-SPACE SCREEN, NOT A SERVING METRIC (principle 3).  It cannot
promote anything.  What it can do is say that the round trip -- encode, write,
index, read back, decode -- preserved what the encoder claims, on real GLM
expert weights, and say it against controls rather than against nothing.  Two
controls, both RTN on the same rows, because one alone would be a treatment and
not a control:

* **NVFP4 RTN** (E2M1 values, one E4M3 scale per 16 inputs, 4.5 bpp) -- the
  4-bit format this box serves natively, at MORE bytes than the wire, so the
  comparison is not flattered by residency.  It is RTN: no GPTQ, no JSO, so it
  is the format's floor and not what the production recipe ships.
* **FP8 RTN** (E4M3 values, one fp32 scale per output channel, 8 bpp) -- twice
  the wire's bytes, so it brackets the answer from above rather than matching
  it.

The wire's own bits per weight are measured here from the container lengths
rather than assumed from the grid name, and printed beside both controls, so a
reader can see what was compared at what residency.

    python experiments/moe_wire_weight_error.py \
        --tessera /mnt/shared/tessera-runs/ts5/glm53-4layer-e16-tessera \
        --source  /mnt/shared/tessera-runs/ts5/glm53-4layer-e16 \
        --out experiments/results/moe_wire_weight_error.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tessera.serving.moe_route import prepare_tessera_moe_experts  # noqa: E402
from tessera.serving.scheme import validate_tessera_moe_scheme  # noqa: E402

FP8_MAX = 448.0
E2M1_MAX = 6.0
#: The E2M1 alphabet, magnitudes only; the sign is carried separately.
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_GROUP = 16
ROLE_ORDER = {"w13": ("gate_proj", "up_proj"), "w2": ("down_proj",)}


class Shards:
    """The tensors of a sharded checkpoint, opened once and kept open."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self._open: dict = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        handle = self._open.get(shard)
        if handle is None:
            handle = self._open[shard] = safe_open(str(self.root / shard), framework="pt")
        return handle.get_tensor(name)


def rtn_fp8_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """One E4M3 value per weight, one fp32 scale per output row -- 8 bpp."""
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32) * scale


def rtn_nvfp4(weight: torch.Tensor) -> torch.Tensor:
    """One E2M1 value per weight, one E4M3 scale per 16 inputs -- 4.5 bpp.

    The block scale is itself stored in E4M3, which is what the format does and
    what makes it a fair floor: quantising the scale exactly would be a
    different format with the same name.
    """
    w = weight.to(torch.float32)
    out, cols = w.shape
    if cols % NVFP4_GROUP:
        raise ValueError(f"{cols} inputs is not a whole number of {NVFP4_GROUP}-wide groups")
    g = w.reshape(out, cols // NVFP4_GROUP, NVFP4_GROUP)
    scale = (g.abs().amax(dim=-1, keepdim=True) / E2M1_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp_min(1e-12)
    levels = torch.tensor(E2M1_LEVELS, device=w.device, dtype=torch.float32)
    normalised = (g / scale).clamp(-E2M1_MAX, E2M1_MAX)
    snapped = levels[(normalised.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)]
    return (torch.copysign(snapped, normalised) * scale).reshape(out, cols)


def relative_error(source: torch.Tensor, approx: torch.Tensor) -> float:
    src = source.to(torch.float32)
    return float(torch.linalg.norm(approx - src) / torch.linalg.norm(src))


def geomean(values) -> float:
    t = torch.tensor(list(values), dtype=torch.float64)
    return float(torch.exp(torch.log(t).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tessera", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--layers", default="", help="comma-separated; default every routed stack")
    ap.add_argument("--experts", default="", help="comma-separated; default every expert")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tess_root, src_root = pathlib.Path(args.tessera), pathlib.Path(args.source)
    config = json.loads((tess_root / "config.json").read_text())
    stacks = [(name, g) for name, g in config["quantization_config"]["config_groups"].items()
              if g["scheme"].get("structure") == "routed_moe"]
    if not stacks:
        raise SystemExit(f"{tess_root}: no routed_moe stack in config_groups; nothing to grade")
    wanted = {int(x) for x in args.layers.split(",") if x.strip()}
    if wanted:
        stacks = [(n, g) for n, g in stacks
                  if int(re.search(r"layers[._](\d+)", g["targets"][0]).group(1)) in wanted]

    tess, src = Shards(tess_root), Shards(src_root)
    device = torch.device(args.device)
    records = []
    for _name, group in stacks:
        target = group["targets"][0]
        declared = dict(validate_tessera_moe_scheme(group["scheme"], target))
        experts = ([int(x) for x in args.experts.split(",") if x.strip()]
                   or list(range(int(declared["experts"]))))
        declared["experts"] = len(experts)
        wire_bytes = {}
        blobs = {}
        for g, roles in ROLE_ORDER.items():
            per_expert = []
            for e in experts:
                containers = []
                for role in roles:
                    raw = tess.get(f"{target}.{e}.{role}.wire")
                    wire_bytes[(e, role)] = int(raw.numel())
                    containers.append(raw.numpy().tobytes())
                per_expert.append(containers)
            blobs[g] = per_expert
        prepared = prepare_tessera_moe_experts(blobs, declared, target, device=device)
        stacked = {"w13": (prepared.w13_weight, prepared.w13_weight_scale),
                   "w2": (prepared.w2_weight, prepared.w2_weight_scale)}
        for g, roles in ROLE_ORDER.items():
            weight, scale = stacked[g]
            served = weight.to(torch.float32) * scale.to(torch.float32)
            rows_per_role = served.shape[1] // len(roles)
            for r, role in enumerate(roles):
                lo, hi = r * rows_per_role, (r + 1) * rows_per_role
                for i, e in enumerate(experts):
                    source = src.get(f"{target}.{e}.{role}.weight").to(device)
                    records.append({
                        "stack": target, "expert": e, "role": role,
                        "shape": list(source.shape),
                        "wire_bits_per_weight": 8.0 * wire_bytes[(e, role)] / source.numel(),
                        "wire_rel_err": relative_error(source, served[i, lo:hi]),
                        "nvfp4_rtn_rel_err": relative_error(source, rtn_nvfp4(source)),
                        "fp8_rtn_rel_err": relative_error(source, rtn_fp8_per_channel(source)),
                    })
                    del source
            del served
        del prepared, stacked

    for rec in records:
        rec["ratio_wire_over_nvfp4_rtn"] = rec["wire_rel_err"] / rec["nvfp4_rtn_rel_err"]
        rec["ratio_wire_over_fp8_rtn"] = rec["wire_rel_err"] / rec["fp8_rtn_rel_err"]
    summary = {
        "tessera": str(tess_root), "source": str(src_root),
        "note": "weight-space screen, not a serving metric",
        "stacks": [g["targets"][0] for _n, g in stacks],
        "rows_compared": len(records),
        "wire_bits_per_weight_max": max(r["wire_bits_per_weight"] for r in records),
        "wire_rel_err_geomean": geomean(r["wire_rel_err"] for r in records),
        "nvfp4_rtn_rel_err_geomean": geomean(r["nvfp4_rtn_rel_err"] for r in records),
        "fp8_rtn_rel_err_geomean": geomean(r["fp8_rtn_rel_err"] for r in records),
        "ratio_wire_over_nvfp4_rtn_geomean": geomean(
            r["ratio_wire_over_nvfp4_rtn"] for r in records),
        "ratio_wire_over_nvfp4_rtn_max": max(r["ratio_wire_over_nvfp4_rtn"] for r in records),
        "ratio_wire_over_fp8_rtn_geomean": geomean(r["ratio_wire_over_fp8_rtn"] for r in records),
        "records": records,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
