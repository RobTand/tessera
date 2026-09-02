#!/usr/bin/env python
"""Export a dense checkpoint as Tessera wires for Tessera's own vLLM plugin.

THREE FAMILIES, ONE PLUGIN.  ``tessera.serving.scheme`` names a family by the
stock tile a module decodes to: ``TESSERA_NVFP4`` (an E2M1-based grid over the
LUT plane, span-2 TCQ body -> the NVFP4 tile, W4A4), ``TESSERA_FP8`` (the
E4M3 grid over the CHANNEL plane, window body -> the per-channel FP8 pair,
W8A8), or ``TESSERA_BF16`` (the BF16 grid over the CHANNEL plane, window body
-> a plain bfloat16 tile, W16A16).  ``--grid``/``--q256`` set the default per
Linear and ``--plan-json`` overrides it per tensor, so one checkpoint may
carry all three and a single serve executes each on its own path: that is the
product an allocator targets.  The 16-bit route exists because the E4M3
*alphabet* -- not the trellis -- floors the window body's error from R=6
upward, so above ~6 bpp an 8-bit tile has nothing left to buy
(``docs/measurements/tessera-bf16-route-2026-09-02.md``).

The checkpoint declares ``quantization_config.quant_method: "tessera"``, which
is what selects the plugin: there is no serve flag to enable it, only
``TESSERA_SERVE_MODE`` to declare the residency.

ONE BLOB PER vLLM MODULE.  Every role a vLLM fusion stacks (q/k/v, gate/up)
is encoded as its own Tessera unit and framed into a ``tessera.fused``
container in stacking order; unfused Linears are a one-member container.  A
fused module's roles must share one family (vLLM builds one method per
module); an NVFP4 module's roles are checked at export for the exact binade
shift the lane applies at load (``shared_lut_global``), so an unserveable
group is refused here, not there.

THE STOCK TWIN.  ``--stock-twin DIR`` also writes the materialisation of the
SAME wires (``tessera.stock.materialize_stock``; NVFP4 groups moved onto one
shared global) that vanilla vLLM serves with no plugin, so a served
comparison is between one encode and its two servings rather than between two
encodes.  A BF16 module's twin tensor is a plain ``<module>.weight`` bfloat16
tile and rides no config group; a twin all of whose modules are BF16 carries
**no** ``quantization_config`` at all, because it is then an ordinary BF16
checkpoint and declaring one would only invite a runtime to look for
tensors that are not there.

The manifest states what is on disk and what the plugin holds resident per
mode.  ``lm_head`` and the embeddings stay BF16 (PrismaQuant's body-only
convention); a Linear whose rows are not a whole number of tuples, or a
fused module not all of whose roles are quantizable, is passed through as
BF16 and named in ``ignore``.  Naming it there is not bookkeeping: the plugin
REFUSES a Linear that is neither declared nor ignored, so one mistyped target
is a refusal rather than a silently BF16 artifact.

THE ARTIFACT IS TENSOR-PARALLELISM-AGNOSTIC, and this exporter never encodes
per rank.  One whole unit per role is written once; a serve with tp_size > 1
loads the whole unit on every rank and cuts it at load
(``tessera.serving.sharding``), so the same checkpoint serves any TP degree and
re-sharding is a serve flag rather than a re-export.  Encoding per rank would
make the bytes a function of the machine they were built for, and a unit cut
for 4 ranks could not be re-cut for 8.

ROUTED-MoE EXPERTS ARE NOT EXPORTED YET, and are refused by name rather than
skipped.  A transformers-5 checkpoint packs a layer's experts into 3-D tensors
(``mlp.experts.gate_up_proj`` ``[E, K, 2N]``, ``...down_proj`` ``[E, N, K]``);
``quantizable`` separates them from the 2-D body and ``main`` either passes
them through as BF16 (named in ``ignore``, so the plugin serves the MoE layer
unquantized instead of refusing it) or, if a plan names one, refuses with the
tensor and its shape.  When the expert route lands, the work is: encode one
unit per expert per role, frame them into one container per vLLM MoE module
(the fused rule below already groups by module name and is rank-agnostic),
declare ``structure: "routed_moe"`` on the scheme, and decode to the stock
PACKED expert layouts vLLM's fused-MoE kernels read.  The grouping rule that a
fused module's roles must share ONE family holds for experts exactly as it does
for q/k/v: vLLM builds one method per module.
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
from tessera.alphabet import (  # noqa: E402
    BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid)
from tessera.bf16_route import BF16_FAMILY  # noqa: E402
from tessera.export import (  # noqa: E402
    DEFAULT_CODE, DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA, DEFAULT_REFIT_OBJECTIVE,
    ActivationSource, encode_linear_planes, wire_recipe)
from tessera.fused import pack_fused, shared_lut_global  # noqa: E402
from tessera.stock import materialize_stock, share_global, stock_bytes  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402

FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(.*\.mlp\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj", ("gate_proj", "up_proj")),
)
NVFP4 = "TESSERA_NVFP4"
FP8 = "TESSERA_FP8"
BF16 = BF16_FAMILY


def grid_for(name: str):
    if name == "E4M3":
        return E4M3_GRID
    if name == "BF16":
        return BF16_GRID
    match = re.fullmatch(r"(E2M1)(?:x(\d+))?", name)
    if not match:
        raise SystemExit(
            f"unknown grid {name!r}; one of E2M1, E2M1x2 (NVFP4 route), E4M3 "
            "(FP8 route) or BF16 (the 16-bit route)")
    return tuple_grid(E2M1_GRID, int(match.group(2) or 1))


def family_for(grid) -> str:
    if grid.name == "BF16":
        return BF16
    return FP8 if grid.name == "E4M3" else NVFP4


def check_recipe(grid, q256: int):
    """The recipe the lane decodes for this family, or a refusal."""
    recipe = wire_recipe(grid, q256)
    family = family_for(grid)
    if family == NVFP4 and not (recipe.body.name == "TCQ" and recipe.span == 2 and recipe.scale_plane.name == "LUT"):
        raise SystemExit(
            f"the NVFP4 route decodes the span-2 TCQ body over the LUT plane today; "
            f"{grid.name} q256={q256} is {recipe.body.name} span {recipe.span} over {recipe.scale_plane.name}")
    if family in (FP8, BF16) and not (recipe.body.name == "WINDOW" and recipe.scale_plane.name == "CHANNEL"):
        raise SystemExit(
            f"the {family} route decodes the window body over the CHANNEL plane; "
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
    """The commit this build came from -- from git, or from the environment.

    A build that runs on a synced copy of the tree has no ``.git`` and used to
    stamp ``unknown``, which is a provenance hole in an artifact whose whole
    claim is that the surrogate, the KL and the bytes are one rendering.
    ``TESSERA_GIT`` is how the caller supplies it when git cannot.
    """
    import os

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return os.environ.get("TESSERA_GIT", "unknown")


def quantizable(src: Path):
    """The body's tensors under ``model.layers``, split by RANK.

    Returns ``(shards, shapes, expert_shapes)``: ``shapes`` is every 2-D
    ``.weight`` (a dense Linear, one blob per vLLM module) and
    ``expert_shapes`` every ``.weight`` of rank 3 or more -- a transformers-5
    packed expert stack.  The split is explicit because dropping the second
    kind silently is how an MoE checkpoint would export as "fully quantized"
    while its experts stayed BF16 and nothing said so.
    """
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
    shapes, expert_shapes = {}, {}
    for shard, names in shards.items():
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                if name.startswith("model.layers.") and name.endswith(".weight"):
                    shape = tuple(handle.get_slice(name).get_shape())
                    if len(shape) == 2:
                        shapes[name] = shape
                    elif len(shape) >= 3:
                        expert_shapes[name] = shape
    return shards, shapes, expert_shapes


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
    ap.add_argument("--grid", default="E2M1x2",
                    help="default grid per Linear: E2M1, E2M1x2 (NVFP4), E4M3 (FP8) or BF16 (W16A16)")
    ap.add_argument("--q256", type=int, default=896, help="default body bits per 256 weights")
    ap.add_argument("--plan-json", type=Path, default=None,
                    help='{"tensor.weight": {"grid": "E4M3"|"BF16", "q256": 1024} | "PASSTHROUGH", ...} per-tensor overrides')
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
    ap.add_argument("--ldlq-sigma", type=float, default=DEFAULT_LDLQ_SIGMA,
                    help="Hessian regulariser for LDLQ cross-column feedback; a negative value turns LDLQ off")
    ap.add_argument("--ldlq-block", type=int, default=DEFAULT_LDLQ_BLOCK, help="LDLQ input-feature block")
    ap.add_argument("--refit-metric", default=DEFAULT_REFIT_OBJECTIVE,
                    help="error the row-scale refit minimises: plain | hessian | h^ALPHA")
    ap.add_argument("--refit-reach-floor", action="store_true",
                    help="hold every refit row scale high enough that the pass's target stays inside the body's reach")
    args = ap.parse_args()

    # The activation-aware settings fire when, and only when, a Hessian is
    # here: the encoder cannot invent one, and a weights-only export must stay
    # the byte-for-byte artifact it was.  Given one, the defaults are the
    # measured recipe (export.DEFAULT_LDLQ_*), overridable per run.  The recipe
    # itself lives in ``ActivationSource``, not here: this script and the
    # library exporters must not carry two copies of it, which is the drift
    # that let the library path encode weights-only while the script did not.
    activation = None
    if args.hessian:
        payload = torch.load(args.hessian, map_location="cpu", weights_only=False)
        if args.ldlq_sigma is not None and args.ldlq_sigma < 0:
            args.ldlq_sigma = None                  # `--ldlq-sigma -1` turns LDLQ off
        activation = ActivationSource(
            hessians=payload["H"], provenance=dict(payload.get("provenance") or {},
                                                   path=str(args.hessian)),
            ldlq_sigma=args.ldlq_sigma, ldlq_block=args.ldlq_block,
            refit_objective=args.refit_metric,
            refit_reach_floor=args.refit_reach_floor,
        )

    default_grid = grid_for(args.grid)
    check_recipe(default_grid, args.q256)
    overrides = {}
    if args.plan_json:
        for name, spec in json.loads(args.plan_json.read_text()).items():
            if spec in ("BF16", "PASSTHROUGH"):
                # The bare string is PASSTHROUGH -- copy the source tensor,
                # quantise nothing.  ``{"grid": "BF16", "q256": N}`` is the
                # opposite: the 16-bit trellis route at rate N, whose tile is
                # bf16 but whose wire is 4-8 bpp.  ``"BF16"`` is kept as a
                # spelling of passthrough because plans were written with it;
                # ``"PASSTHROUGH"`` is the one that cannot be misread.
                overrides[name] = None
            else:
                g = grid_for(spec["grid"])
                check_recipe(g, int(spec["q256"]))
                overrides[name] = (g, int(spec["q256"]))
    shards, shapes, expert_shapes = quantizable(args.src)
    planned_experts = sorted(set(overrides) & set(expert_shapes))
    if planned_experts:
        first = planned_experts[0]
        raise SystemExit(
            f"the plan names {len(planned_experts)} packed expert tensor(s), e.g. {first} "
            f"{list(expert_shapes[first])}: routed-MoE expert export is a follow-up. The wires, "
            "the container framing and the scheme's structure field are all in place; what is "
            "missing is the per-expert encode and the decode to vLLM's packed expert layouts. "
            "Remove them from the plan to pass them through as BF16.")
    unknown = sorted(set(overrides) - set(shapes))
    if unknown:
        raise SystemExit(f"plan names tensors that are not 2-D body weights here: {unknown[:5]}")
    if expert_shapes:
        print(f"  {len(expert_shapes)} packed expert tensors stay BF16 and are named in ignore "
              f"(routed-MoE export is a follow-up); e.g. {sorted(expert_shapes)[0]}", flush=True)
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
    twin_modules: dict[str, list[str]] = {NVFP4: [], FP8: [], BF16: []}
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
                        # ``ignore`` is read against vLLM's OWN Linear names, and
                        # vLLM builds one Linear per FUSED module: ignoring
                        # q_proj/k_proj/v_proj leaves ``qkv_proj`` neither
                        # declared nor ignored, and the plugin refuses that
                        # checkpoint at load.  A passed-through role therefore
                        # ignores the fused module it belongs to.
                        fused_here = fused_module(name)
                        ignore.append(fused_here[0] if fused_here else module_of(name))
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
                # A missing key renders RTN and raises nothing; ``for_unit``
                # refuses instead, and is the same call the library exporters make.
                extra = ({} if activation is None else
                         activation.for_unit(member, weight.shape[1], args.device))
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
            elif family == BF16:
                # Resident here is the DECODED tile -- 16 bits a weight, the
                # source precision.  It is the correctness path and not a size
                # claim; the product mode is streamed, and the wire it streams
                # is ``wire_bytes`` above.
                record["resident_bytes_resident_mode"] = rows_total * cols * 2
                if twin is not None:
                    for m in members:
                        # One tensor, under the ORIGINAL name: the twin is an
                        # ordinary BF16 checkpoint, not a compressed one.
                        twin_payload[m] = stock_tensors[m]["weight"].cpu()
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
                # ``structure`` names what kind of vLLM layer this is: ``dense``
                # (a LinearBase, one blob per module) today.  The expert route
                # will declare ``routed_moe`` here rather than change the schema.
                "family": family, "structure": "dense",
                "grid": grid.name, "body": recipe.body.name, "plane": recipe.scale_plane.name,
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

    # A packed expert stack stays BF16; naming its MODULE in ``ignore`` is what
    # lets the plugin serve that MoE layer unquantized instead of refusing it.
    for name in expert_shapes:
        ignore.append(module_of(name).rsplit(".", 1)[0] if name.endswith(".weight") else name)
    ignore = sorted(set(ignore))
    config = json.loads((args.src / "config.json").read_text())
    config["quantization_config"] = {
        # The field that selects Tessera's own vLLM plugin (entry point
        # ``tessera = tessera.serving:register``).  No serve flag enables it.
        "quant_method": "tessera", "format": "mixed-precision",
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
    for fam in (NVFP4, FP8, BF16):
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
               + f" -> tessera.serving {'+'.join(families)}",
        "default": {"grid": default_grid.name, "q256": args.q256}, "plan_json": str(args.plan_json) if args.plan_json else None,
        "input_scales_from": str(args.input_scales) if args.input_scales else None,
        "activation_aware": None if activation is None else activation.config_block(),
        "stock_twin": str(twin) if twin is not None else None,
        "totals": totals, "modules": module_records,
    }
    (args.out / "tessera_serving_manifest.json").write_text(json.dumps(manifest, indent=2))

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
        if twin_groups:
            # A BF16 module is in no group: its twin tensor is the ordinary
            # ``<module>.weight``, so it is ignored by the quantization config
            # exactly as ``lm_head`` is.
            twin_config["quantization_config"] = {
                "quant_method": "compressed-tensors", "format": "mixed-precision",
                "config_groups": twin_groups,
                "ignore": sorted(set(ignore) | set(twin_modules[BF16])),
                "quantization_status": "compressed",
            }
        else:
            # Every module decoded to a plain bf16 tile: this IS a BF16
            # checkpoint.  Declaring an empty config would tell a runtime to
            # look for compressed tensors that do not exist.
            twin_config.pop("quantization_config", None)
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
