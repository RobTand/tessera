#!/usr/bin/env python
"""Export a dense checkpoint as Tessera wires for Gridbook's Tessera lane.

TWO FAMILIES, ONE LANE.  Gridbook's ``tessera_scheme`` names a family by the
stock tile a module decodes to: ``TESSERA_NVFP4`` (an E2M1-based grid over the
LUT plane, span-2 TCQ body -> the NVFP4 tile, W4A4) or ``TESSERA_FP8`` (the
E4M3 grid over the CHANNEL plane, window body -> the per-channel FP8 pair,
W8A8).  ``--grid``/``--q256`` set the default per Linear and ``--plan-json``
overrides it per tensor, so one checkpoint may carry both families and a
single serve executes each on its own tensor-core path: that is the product
an allocator targets.

ONE BLOB PER vLLM MODULE.  Every role a vLLM fusion stacks (q/k/v, gate/up)
is encoded as its own Tessera unit and framed into a ``tessera.fused``
container in stacking order; unfused Linears are a one-member container.  A
fused module's roles must share one family (vLLM builds one method per
module); an NVFP4 module's roles are checked at export for the exact binade
shift the lane applies at load (``shared_lut_global``), so an unserveable
group is refused here, not there.

THE STOCK TWIN.  ``--stock-twin DIR`` also writes the compressed-tensors
materialisation of the SAME wires (``tessera.stock.materialize_stock``; NVFP4
groups moved onto one shared global) that vanilla vLLM serves with no plugin,
so a served comparison is between one encode and its two servings rather than
between two encodes.

The manifest states what is on disk and what the lane holds resident per
mode.  ``lm_head`` and the embeddings stay BF16 (PrismaQuant's body-only
convention); a Linear whose rows are not a whole number of tuples, or a
fused module not all of whose roles are quantizable, is passed through as
BF16 and named in ``ignore``.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_stock_compressed import (  # noqa: E402
    FP8_INPUTS, FP8_WEIGHTS, NVFP4_INPUTS, NVFP4_WEIGHTS, regex_target)
from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid  # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.fused import pack_fused, shared_lut_global  # noqa: E402
from tessera.stock import materialize_stock, share_global, stock_bytes  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402

FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(.*\.mlp\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj", ("gate_proj", "up_proj")),
)
NVFP4 = "TESSERA_NVFP4"
FP8 = "TESSERA_FP8"


def grid_for(name: str):
    if name == "E4M3":
        return E4M3_GRID
    match = re.fullmatch(r"(E2M1)(?:x(\d+))?", name)
    if not match:
        raise SystemExit(f"unknown grid {name!r}; one of E2M1, E2M1x2 (NVFP4 route) or E4M3 (FP8 route)")
    return tuple_grid(E2M1_GRID, int(match.group(2) or 1))


def family_for(grid) -> str:
    return FP8 if grid.name == "E4M3" else NVFP4


def check_recipe(grid, q256: int):
    """The recipe the lane decodes for this family, or a refusal."""
    recipe = wire_recipe(grid, q256)
    family = family_for(grid)
    if family == NVFP4 and not (recipe.body.name == "TCQ" and recipe.span == 2 and recipe.scale_plane.name == "LUT"):
        raise SystemExit(
            f"the NVFP4 route decodes the span-2 TCQ body over the LUT plane today; "
            f"{grid.name} q256={q256} is {recipe.body.name} span {recipe.span} over {recipe.scale_plane.name}")
    if family == FP8 and not (recipe.body.name == "WINDOW" and recipe.scale_plane.name == "CHANNEL"):
        raise SystemExit(
            f"the FP8 route decodes the window body over the CHANNEL plane; "
            f"{grid.name} q256={q256} is {recipe.body.name} over {recipe.scale_plane.name}")
    return recipe


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


def stock_targets(modules):
    """compressed-tensors targets for member modules plus their fused names (the stock exporter's rule)."""
    found = sorted(set(modules))
    fused_names = set()
    by_prefix: dict[str, set[str]] = {}
    for m in found:
        for pattern, fused, member_names in FUSED:
            match = pattern.match(m + ".weight")
            if match:
                by_prefix.setdefault(match.group(1) + fused, set()).add(match.group(2))
    for fused_name, present in by_prefix.items():
        for pattern, fused, member_names in FUSED:
            if fused_name.endswith(fused) and present == set(member_names):
                fused_names.add(fused_name)
    return [regex_target(m) for m in sorted(set(found) | fused_names)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", default="E2M1x2", help="default grid per Linear: E2M1, E2M1x2 (NVFP4) or E4M3 (FP8)")
    ap.add_argument("--q256", type=int, default=896, help="default body bits per 256 weights")
    ap.add_argument("--plan-json", type=Path, default=None,
                    help='{"tensor.weight": {"grid": "E4M3", "q256": 1024} | "BF16", ...} per-tensor overrides')
    ap.add_argument("--input-scales", type=Path, default=None,
                    help="safetensors carrying <module>.input_global_scale per NVFP4 Linear (a stock NVFP4 "
                         "export); required when any module takes the NVFP4 route")
    ap.add_argument("--stock-twin", type=Path, default=None,
                    help="also write the compressed-tensors materialisation of the same wires here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="encode only the first N layers (smoke)")
    ap.add_argument("--hessian", type=Path, default=None,
                    help="capture_h_full.py payload: full input Hessians keyed by the tensor's module "
                         "name.  Enables the activation-aware encoder settings below; an encode that "
                         "uses them is not reproducible from the weights alone, so the file's own "
                         "provenance is copied into the manifest.")
    ap.add_argument("--ldlq-sigma", type=float, default=None,
                    help="Hessian regulariser for LDLQ cross-column feedback; unset means no LDLQ")
    ap.add_argument("--ldlq-block", type=int, default=128, help="LDLQ input-feature block")
    ap.add_argument("--refit-metric", default="plain",
                    help="error the row-scale refit minimises: plain | hessian | h^ALPHA")
    ap.add_argument("--refit-reach-floor", action="store_true",
                    help="hold every refit row scale high enough that the pass's target stays inside the body's reach")
    args = ap.parse_args()

    hessians, h_provenance = {}, None
    if args.hessian:
        payload = torch.load(args.hessian, map_location="cpu", weights_only=False)
        hessians, h_provenance = payload["H"], payload.get("provenance")
    activation_aware = args.ldlq_sigma is not None or args.refit_metric != "plain" or args.refit_reach_floor
    if activation_aware and not hessians:
        raise SystemExit("--ldlq-sigma / --refit-metric / --refit-reach-floor need --hessian")

    default_grid = grid_for(args.grid)
    check_recipe(default_grid, args.q256)
    overrides = {}
    if args.plan_json:
        for name, spec in json.loads(args.plan_json.read_text()).items():
            if spec == "BF16":
                overrides[name] = None
            else:
                g = grid_for(spec["grid"])
                check_recipe(g, int(spec["q256"]))
                overrides[name] = (g, int(spec["q256"]))
    shards, shapes = quantizable(args.src)
    unknown = sorted(set(overrides) - set(shapes))
    if unknown:
        raise SystemExit(f"plan names tensors that are not 2-D body weights here: {unknown[:5]}")
    plan: dict[str, tuple] = {}          # tensor -> (grid, q256, rows, cols)
    passthrough: list[str] = []
    for name, (rows, cols) in shapes.items():
        layer = int(name.split(".")[2])
        if args.layers is not None and layer >= args.layers:
            passthrough.append(name); continue
        if name in overrides and overrides[name] is None:
            passthrough.append(name); continue
        grid, q256 = overrides.get(name, (default_grid, args.q256))
        if rows % (grid.arity * 32) or cols % 16:
            passthrough.append(name); continue
        plan[name] = (grid, q256, rows, cols)
    # group by vLLM module: a fused module needs every role quantizable on ONE recipe
    modules: dict[str, list[str]] = {}
    for name in list(plan):
        fused = fused_module(name)
        if fused is None:
            modules[module_of(name)] = [name]
            continue
        key, members = fused
        if key in modules:
            continue
        recipes = {(plan[m][0].name, plan[m][1]) for m in members if m in plan}
        if all(m in plan for m in members) and len(recipes) == 1:
            modules[key] = list(members)
        else:
            why = "not every role is quantizable" if not all(m in plan for m in members) else f"roles disagree {sorted(recipes)}"
            print(f"  passthrough {key}: {why}; vLLM builds one method per fused module", flush=True)
            for m in members:
                if m in plan:
                    del plan[m]
                    passthrough.append(m)
    owned = {m for members in modules.values() for m in members}
    assert owned == set(plan)

    input_scales = {}
    if args.input_scales:
        with safe_open(str(args.input_scales), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".input_global_scale"):
                    input_scales[key] = float(handle.get_tensor(key).float().reshape(-1)[0])
    needs_scales = [m for m, members in modules.items() if family_for(plan[members[0]][0]) == NVFP4]
    if needs_scales and not input_scales:
        raise SystemExit(f"{len(needs_scales)} modules take the NVFP4 route (W4A4 needs a static input scale) "
                         "but no --input-scales was given")

    args.out.mkdir(parents=True, exist_ok=True)
    twin = args.stock_twin
    if twin is not None:
        twin.mkdir(parents=True, exist_ok=True)
    started = time.time()
    new_weight_map: dict[str, str] = {}
    twin_weight_map: dict[str, str] = {}
    config_groups: dict[str, dict] = {}
    units: dict[str, dict] = {}
    module_records: dict[str, dict] = {}
    twin_modules: dict[str, list[str]] = {NVFP4: [], FP8: []}
    twin_records: dict[str, dict] = {}
    ignore = ["lm_head", "model.embed_tokens"]
    passthrough_bytes = 0
    weights_cache: dict[str, torch.Tensor] = {}
    done = 0
    total = len(plan)

    pending_modules = dict(modules)
    for shard, names in sorted(shards.items()):
        shard_payload: dict[str, torch.Tensor] = {}
        twin_payload: dict[str, torch.Tensor] = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    weights_cache[name] = tensor
                else:
                    shard_payload[name] = tensor
                    twin_payload[name] = tensor
                    passthrough_bytes += tensor.numel() * tensor.element_size()
                    if name in passthrough:
                        ignore.append(module_of(name))
        for module, members in list(pending_modules.items()):
            if not all(m in weights_cache for m in members):
                continue
            grid, q256, _r, _c = plan[members[0]]
            family = family_for(grid)
            recipe = wire_recipe(grid, q256)
            roles = []
            role_records = []
            stock_tensors: dict[str, dict] = {}
            for member in members:
                weight = weights_cache.pop(member).to(args.device, torch.float32).contiguous()
                extra = {}
                if activation_aware:
                    key = module_of(member)
                    if key not in hessians:
                        # A wrong key renders RTN and raises nothing; refuse instead.
                        raise SystemExit(
                            f"no Hessian for {key}: the capture's keys must be the encoder's unit names")
                    H = hessians[key].to(args.device, torch.float32)
                    if H.shape[0] != weight.shape[1]:
                        raise SystemExit(f"{key}: H is {tuple(H.shape)} for {weight.shape[1]} inputs")
                    if args.ldlq_sigma is not None:
                        extra["ldl"] = block_ldl(
                            regularize_hessian(H, sigma_reg=args.ldlq_sigma), args.ldlq_block)
                        extra["ldl_block"] = args.ldlq_block
                    if args.refit_metric == "hessian":
                        extra["refit_metric"] = H
                    elif args.refit_metric != "plain":
                        alpha = float(args.refit_metric.removeprefix("h^"))
                        h = H.diagonal()
                        extra["refit_metric"] = (h / h.mean()).pow(alpha)
                    extra["refit_reach_floor"] = args.refit_reach_floor
                exported, unit, forests = encode_linear_planes(
                    weight, grid=grid, q256=q256, name=member, verify=not args.no_verify, **extra)
                extra.clear()
                parse_unit_artifact(exported.blob, device=args.device)      # the reader accepts what we wrote
                role = module_of(member).rsplit(".", 1)[-1]
                roles.append((role, exported.rows, exported.blob, unit, forests))
                stock_tensors[member] = materialize_stock(unit, forests, DEFAULT_CODE)
                role_records.append({
                    "tensor": member, "role": role, "rows": exported.rows, "cols": exported.columns,
                    "grid": grid.name, "q256": q256, "family": family,
                    "wire_bytes": exported.exact_bytes, "blob_bytes": len(exported.blob),
                    "wire_bpp": float(exported.bpp), "own_global": float(unit.scale_global),
                    "resident_bytes_stock": stock_bytes(stock_tensors[member]),
                })
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {member}  {time.time() - started:.0f}s", flush=True)
            blob = pack_fused([(r, rows, b) for r, rows, b, _u, _f in roles])
            rows_total = sum(r[1] for r in roles)
            cols = role_records[0]["cols"]
            record = {
                "family": family, "grid": grid.name, "q256": q256, "roles": role_records,
                "container_bytes": len(blob), "rows": rows_total, "cols": cols,
                "wire_bytes": sum(r["wire_bytes"] for r in role_records),
            }
            shard_payload[f"{module}.wire_bytes"] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            if family == NVFP4:
                shared, _moved = shared_lut_global(
                    [u.scale_lut for _, _, _, u, _ in roles], [float(u.scale_global) for _, _, _, u, _ in roles],
                    [r for r, *_ in roles])
                # the A-side static scale: vLLM's NVFP4 scheme takes the MAX over fused shards
                scale_keys = [f"{module_of(m)}.input_global_scale" for m in members]
                missing = [k for k in scale_keys if k not in input_scales]
                if missing:
                    raise SystemExit(f"no {missing} in {args.input_scales}; W4A4 cannot serve {module}")
                a_scale = max(input_scales[k] for k in scale_keys)
                shard_payload[f"{module}.trellis_input_global_scale"] = torch.tensor([a_scale], dtype=torch.float32)
                record.update({"shared_global": shared, "input_global_scale": a_scale,
                               "resident_bytes_resident_mode": rows_total * cols // 2 + rows_total * cols // 16})
                if twin is not None:
                    moved, divisor = share_global({module_of(m): stock_tensors[m] for m in members})
                    for m in members:
                        for key, value in moved[module_of(m)].items():
                            twin_payload[f"{module_of(m)}.{key}"] = value.cpu()
                        twin_payload[f"{module_of(m)}.input_global_scale"] = torch.tensor(
                            [input_scales[f"{module_of(m)}.input_global_scale"]], dtype=torch.float32)
                    record["twin_shared_divisor"] = divisor
            else:
                record["resident_bytes_resident_mode"] = rows_total * cols + rows_total * 4
                if twin is not None:
                    for m in members:
                        for key, value in stock_tensors[m].items():
                            twin_payload[f"{module_of(m)}.{key}"] = value.cpu()
            if twin is not None:
                twin_modules[family].extend(module_of(m) for m in members)
                twin_records[module] = {"family": family, "members": [module_of(m) for m in members],
                                        "resident_bytes": sum(stock_bytes(stock_tensors[m]) for m in members)}
            scheme = {
                "family": family, "grid": grid.name, "body": recipe.body.name, "plane": recipe.scale_plane.name,
                "q256": q256, "rows": rows_total, "columns": cols, "wire_bytes": len(blob),
                "roles": [[r, rows] for r, rows, *_ in roles],
            }
            config_groups[f"tessera_{module.replace('.', '_')}"] = {"format": "TESSERA", "targets": [module], "scheme": scheme}
            module_records[module] = record
            for r in role_records:
                units[r["tensor"]] = r
            del pending_modules[module]
        save_file({k: v.contiguous() for k, v in shard_payload.items()}, str(args.out / shard), metadata={"format": "pt"})
        for key in shard_payload:
            new_weight_map[key] = shard
        print(f"wrote {shard}: {len(shard_payload)} tensors", flush=True)
        if twin is not None:
            save_file({k: v.contiguous() for k, v in twin_payload.items()}, str(twin / shard), metadata={"format": "pt"})
            for key in twin_payload:
                twin_weight_map[key] = shard
            print(f"wrote twin {shard}: {len(twin_payload)} tensors", flush=True)
    if pending_modules:
        raise SystemExit(f"modules never completed: {sorted(pending_modules)}")

    ignore = sorted(set(ignore))
    config = json.loads((args.src / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "gridbook", "format": "mixed-precision",
        "config_groups": config_groups, "ignore": ignore,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if len(shards) > 1:
        size = sum((args.out / s).stat().st_size for s in shards)
        (args.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size}, "weight_map": new_weight_map}, indent=2))
    aux_patterns = ("*.json", "*.txt", "*.jinja", "*.model")
    for pattern in aux_patterns:
        for aux in args.src.glob(pattern):
            if aux.name in ("config.json", "model.safetensors.index.json"):
                continue
            shutil.copy2(aux, args.out / aux.name)
            if twin is not None:
                shutil.copy2(aux, twin / aux.name)

    by_family = {}
    for fam in (NVFP4, FP8):
        recs = [m for m in module_records.values() if m["family"] == fam]
        if not recs:
            continue
        params = sum(r["rows"] * r["cols"] for m in recs for r in m["roles"])
        wire = sum(m["wire_bytes"] for m in recs)
        resident = sum(m["resident_bytes_resident_mode"] for m in recs)
        by_family[fam] = {
            "modules": len(recs), "units": sum(len(m["roles"]) for m in recs), "quantized_params": params,
            "wire_bytes": wire, "wire_bpp": float(Fraction(wire * 8, params)),
            "resident_mode_bytes": resident, "resident_mode_bpp": float(Fraction(resident * 8, params)),
        }
    params = sum(r["rows"] * r["cols"] for r in units.values())
    wire = sum(r["wire_bytes"] for r in units.values())
    on_disk = sum(m["container_bytes"] for m in module_records.values())
    resident = sum(m["resident_bytes_resident_mode"] for m in module_records.values())
    totals = {
        "quantized_params": params, "modules": len(module_records), "units": len(units),
        "wire_bytes": wire, "wire_bpp": float(Fraction(wire * 8, params)) if params else None,
        "on_disk_bytes": on_disk, "on_disk_bpp": float(Fraction(on_disk * 8, params)) if params else None,
        "resident_mode_bytes": resident, "resident_mode_bpp": float(Fraction(resident * 8, params)) if params else None,
        "streamed_mode_note": "the prepared planes (~wire bytes + per-unit tables) plus one transient decoded tile per forward",
        "by_family": by_family,
        "passthrough_bytes": passthrough_bytes,
        "checkpoint_bytes": sum((args.out / s).stat().st_size for s in shards),
    }
    families = sorted({m["family"] for m in module_records.values()})
    manifest = {
        "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": f"tessera {default_grid.name} q256={args.q256}" + (f" + plan {args.plan_json}" if args.plan_json else "")
               + f" -> gridbook {'+'.join(families)}",
        "default": {"grid": default_grid.name, "q256": args.q256}, "plan_json": str(args.plan_json) if args.plan_json else None,
        "input_scales_from": str(args.input_scales) if args.input_scales else None,
        "activation_aware": None if not activation_aware else {
            "ldlq_sigma": args.ldlq_sigma, "ldlq_block": args.ldlq_block,
            "refit_metric": args.refit_metric, "refit_reach_floor": args.refit_reach_floor,
            "hessian": str(args.hessian), "hessian_provenance": h_provenance,
            "note": "encoder-side only: the wire, the decoder and the lane are unchanged, "
                    "but this encode is not reproducible from the weights alone",
        },
        "stock_twin": str(twin) if twin is not None else None,
        "totals": totals, "modules": module_records,
    }
    (args.out / "tessera_gridbook_manifest.json").write_text(json.dumps(manifest, indent=2))

    if twin is not None:
        twin_groups = {}
        if twin_modules[NVFP4]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": "nvfp4-pack-quantized", "weights": dict(NVFP4_WEIGHTS),
                "input_activations": dict(NVFP4_INPUTS), "targets": stock_targets(twin_modules[NVFP4])}
        if twin_modules[FP8]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": "float-quantized", "weights": dict(FP8_WEIGHTS),
                "input_activations": dict(FP8_INPUTS), "targets": stock_targets(twin_modules[FP8])}
        twin_config = json.loads((args.src / "config.json").read_text())
        twin_config["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "mixed-precision",
            "config_groups": twin_groups, "ignore": ignore, "quantization_status": "compressed",
        }
        (twin / "config.json").write_text(json.dumps(twin_config, indent=2))
        if len(shards) > 1:
            size = sum((twin / s).stat().st_size for s in shards)
            (twin / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": size}, "weight_map": twin_weight_map}, indent=2))
        twin_resident = sum(r["resident_bytes"] for r in twin_records.values())
        (twin / "tessera_stock_twin_manifest.json").write_text(json.dumps({
            "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wire_checkpoint": str(args.out), "arm": manifest["arm"] + " (stock twin of the same wires)",
            "totals": {"quantized_params": params, "modules": len(twin_records),
                       "resident_bytes": twin_resident,
                       "resident_bpp": float(Fraction(twin_resident * 8, params)) if params else None,
                       "checkpoint_bytes": sum((twin / s).stat().st_size for s in shards)},
            "modules": twin_records,
        }, indent=2))
    print(json.dumps(totals, indent=2))
    print(f"elapsed {time.time() - started:.0f}s -> {args.out}" + (f" (twin -> {twin})" if twin is not None else ""))


if __name__ == "__main__":
    main()
