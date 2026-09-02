#!/usr/bin/env python
"""Export a dense checkpoint as Tessera wires for Gridbook's ``TESSERA_NVFP4`` lane.

One blob per vLLM module: every role a vLLM fusion stacks (q/k/v, gate/up)
is encoded as its own Tessera unit at the wire's own rate and framed into a
``tessera.fused`` container in stacking order; unfused Linears are a
one-member container.  The lane decodes the roles into row slices of one
stock NVFP4 tile and moves their LUT tables onto one global by an exact
binade shift at load -- nothing here rewrites a unit for that; this script
only *checks* the shift exists (``shared_lut_global``) so an unserveable
group is refused at export, not at load.

The checkpoint holds the wire's bytes (4.0 bpp at the E2M1x2 cap), and the
manifest states both what is on disk and what the lane holds resident per
mode.  ``lm_head`` and the embeddings stay BF16 (PrismaQuant's body-only
convention); a Linear whose rows are not a whole number of tuples is
passed through as BF16 and named in ``ignore``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.alphabet import E2M1_GRID, tuple_grid  # noqa: E402
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.fused import pack_fused, shared_lut_global  # noqa: E402
from tessera.stock import materialize_stock, stock_bytes  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402

FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(.*\.mlp\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj", ("gate_proj", "up_proj")),
)
FAMILY = "TESSERA_NVFP4"


def grid_for(name: str):
    match = re.fullmatch(r"(E2M1)(?:x(\d+))?", name)
    if not match:
        raise SystemExit(f"unknown grid {name!r}; the NVFP4 lane holds E2M1 or E2M1x2")
    return tuple_grid(E2M1_GRID, int(match.group(2) or 1))


def module_of(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")]


def fused_module(tensor_name: str):
    """``(fused module name, ordered member tensor names)`` or ``None``."""
    for pattern, fused, members in FUSED:
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def quantizable(src: Path):
    """Every 2-D ``.weight`` under ``model.layers`` (body only), with its shape."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for tensor, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor)
    else:
        shards = {}
        for path in sorted(src.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                shards[path.name] = list(handle.keys())
    shapes = {}
    for shard, names in shards.items():
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                if name.startswith("model.layers.") and name.endswith(".weight"):
                    shape = tuple(handle.get_slice(name).get_shape())
                    if len(shape) == 2:
                        shapes[name] = shape
    return shards, shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--input-scales", type=Path, required=True,
                    help="safetensors carrying <module>.input_global_scale per NVFP4 Linear (a stock NVFP4 export)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="encode only the first N layers (smoke)")
    args = ap.parse_args()

    grid = grid_for(args.grid)
    recipe = wire_recipe(grid, args.q256)
    if recipe.body.name != "TCQ" or recipe.span != 2 or recipe.scale_plane.name != "LUT":
        raise SystemExit(
            f"the Gridbook lane decodes the span-2 TCQ body over the LUT plane today; "
            f"{grid.name} q256={args.q256} is {recipe.body.name} span {recipe.span} over {recipe.scale_plane.name}")
    shards, shapes = quantizable(args.src)
    plan = {}
    passthrough = []
    for name, (rows, cols) in shapes.items():
        layer = int(name.split(".")[2])
        if args.layers is not None and layer >= args.layers:
            passthrough.append(name); continue
        if rows % (grid.arity * 32) or cols % 16:
            passthrough.append(name); continue
        plan[name] = (rows, cols)
    # group by vLLM module
    modules: dict[str, list[str]] = {}
    for name in plan:
        fused = fused_module(name)
        if fused is not None:
            key, members = fused
            if all(m in plan for m in members):
                modules[key] = list(members)
            else:
                modules[module_of(name)] = [name]
        else:
            modules[module_of(name)] = [name]
    # a fused module whose members are only partly quantizable falls back per role above;
    # a role listed under its fused module must not also appear alone
    owned = {m for members in modules.values() for m in members}
    assert owned == set(plan)

    input_scales = {}
    with safe_open(str(args.input_scales), framework="pt") as handle:
        for key in handle.keys():
            if key.endswith(".input_global_scale"):
                input_scales[key] = float(handle.get_tensor(key).float().reshape(-1)[0])

    args.out.mkdir(parents=True, exist_ok=True)
    K2 = grid
    started = time.time()
    payload: dict[str, torch.Tensor] = {}
    new_weight_map: dict[str, str] = {}
    config_groups: dict[str, dict] = {}
    units: dict[str, dict] = {}
    module_records: dict[str, dict] = {}
    ignore = ["lm_head", "model.embed_tokens"]
    passthrough_bytes = 0
    weights_cache: dict[str, torch.Tensor] = {}
    done = 0
    total = len(plan)

    # Load everything we need lazily per shard, encode per module once all roles are seen.
    pending_modules = dict(modules)
    out_shards = {}
    for shard, names in sorted(shards.items()):
        shard_payload: dict[str, torch.Tensor] = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    weights_cache[name] = tensor
                else:
                    shard_payload[name] = tensor
                    passthrough_bytes += tensor.numel() * tensor.element_size()
                    if name in passthrough:
                        ignore.append(module_of(name))
        # encode every module whose roles are all cached
        for module, members in list(pending_modules.items()):
            if not all(m in weights_cache for m in members):
                continue
            roles = []
            role_records = []
            for member in members:
                weight = weights_cache.pop(member).to(args.device, torch.float32).contiguous()
                exported, unit, forests = encode_linear_planes(
                    weight, grid=K2, q256=args.q256, name=member, verify=not args.no_verify)
                parsed = parse_unit_artifact(exported.blob, device=args.device)
                role = module_of(member).rsplit(".", 1)[-1]
                roles.append((role, exported.rows, exported.blob, unit, forests))
                role_records.append({
                    "tensor": member, "role": role, "rows": exported.rows, "cols": exported.columns,
                    "wire_bytes": exported.exact_bytes, "blob_bytes": len(exported.blob),
                    "wire_bpp": float(exported.bpp), "own_global": float(unit.scale_global),
                    "resident_bytes_stock": stock_bytes(materialize_stock(unit, forests, DEFAULT_CODE)),
                })
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {member}  {time.time() - started:.0f}s", flush=True)
            shared, _moved = shared_lut_global(
                [u.scale_lut for _, _, _, u, _ in roles], [float(u.scale_global) for _, _, _, u, _ in roles],
                [r for r, *_ in roles])
            blob = pack_fused([(r, rows, b) for r, rows, b, _u, _f in roles])
            rows_total = sum(r[1] for r in roles)
            cols = role_records[0]["cols"]
            # the A-side static scale: vLLM's NVFP4 scheme takes the MAX over fused shards
            scale_keys = [f"{module_of(m)}.input_global_scale" for m in members]
            missing = [k for k in scale_keys if k not in input_scales]
            if missing:
                raise SystemExit(f"no {missing} in {args.input_scales}; W4A4 cannot serve {module}")
            a_scale = max(input_scales[k] for k in scale_keys)
            shard_payload[f"{module}.wire_bytes"] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            shard_payload[f"{module}.trellis_input_global_scale"] = torch.tensor([a_scale], dtype=torch.float32)
            scheme = {
                "family": FAMILY, "grid": K2.name, "body": recipe.body.name, "plane": recipe.scale_plane.name,
                "q256": args.q256, "rows": rows_total, "columns": cols, "wire_bytes": len(blob),
                "roles": [[r, rows] for r, rows, *_ in roles],
            }
            config_groups[f"tessera_{module.replace('.', '_')}"] = {"format": "TESSERA", "targets": [module], "scheme": scheme}
            module_records[module] = {
                "roles": role_records, "shared_global": shared, "container_bytes": len(blob),
                "input_global_scale": a_scale, "rows": rows_total, "cols": cols,
                "wire_bytes": sum(r["wire_bytes"] for r in role_records),
                "resident_bytes_resident_mode": rows_total * cols // 2 + rows_total * cols // 16,
            }
            for r in role_records:
                units[r["tensor"]] = r
            del pending_modules[module]
        save_file({k: v.contiguous() for k, v in shard_payload.items()}, str(args.out / shard), metadata={"format": "pt"})
        out_shards[shard] = shard_payload.keys()
        for key in shard_payload:
            new_weight_map[key] = shard
        print(f"wrote {shard}: {len(shard_payload)} tensors", flush=True)
    if pending_modules:
        raise SystemExit(f"modules never completed: {sorted(pending_modules)}")

    config = json.loads((args.src / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "gridbook", "format": "mixed-precision",
        "config_groups": config_groups, "ignore": sorted(set(ignore)),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if len(shards) > 1:
        size = sum((args.out / s).stat().st_size for s in shards)
        (args.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size}, "weight_map": new_weight_map}, indent=2))
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in args.src.glob(pattern):
            if aux.name in ("config.json", "model.safetensors.index.json"):
                continue
            shutil.copy2(aux, args.out / aux.name)

    params = sum(r["rows"] * r["cols"] for r in units.values())
    wire = sum(r["wire_bytes"] for r in units.values())
    on_disk = sum(m["container_bytes"] for m in module_records.values())
    resident = sum(m["resident_bytes_resident_mode"] for m in module_records.values())
    totals = {
        "quantized_params": params, "modules": len(module_records), "units": len(units),
        "wire_bytes": wire, "wire_bpp": float(Fraction(wire * 8, params)) if params else None,
        "on_disk_bytes": on_disk, "on_disk_bpp": float(Fraction(on_disk * 8, params)) if params else None,
        "resident_mode_bytes": resident, "resident_mode_bpp": float(Fraction(resident * 8, params)) if params else None,
        "streamed_mode_note": "the prepared planes (~wire bytes + per-unit tables) plus one shared decode tile per device",
        "passthrough_bytes": passthrough_bytes,
        "checkpoint_bytes": sum((args.out / s).stat().st_size for s in shards),
    }
    manifest = {
        "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": f"tessera {K2.name} q256={args.q256} -> gridbook {FAMILY}",
        "recipe": {"body": recipe.body.name, "span": recipe.span, "plane": recipe.scale_plane.name},
        "input_scales_from": str(args.input_scales), "totals": totals, "modules": module_records,
    }
    (args.out / "tessera_gridbook_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(totals, indent=2))
    print(f"elapsed {time.time() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
