"""Export a checkpoint through Tessera to the stock compressed-tensors formats.

The wire's rate lives on the kernel lane.  This exporter writes the other
thing a Tessera encoding is: a checkpoint vanilla vLLM serves with no plugin,
no fork and no custom kernel, because every unit materialises to a stock
tensor (``tessera.stock``) -- E2M1/E2M1x2 over a LUT plane as NVFP4 at 4.5
bits resident, E4M3 over the CHANNEL plane as per-channel FP8 at 8 bits
resident.  The served bytes decode to the reader's reconstruction bit for
bit; the *rate* is the stock format's, and the manifest says so per unit.

Arms this writes, one per invocation:

  --grid E2M1x2 --q256 896 --activations w4a4 --input-scales DONOR
        Tessera as an NVFP4 encoder, served W4A4 on the CUTLASS FP4 route.
        The static ``input_global_scale`` every W4A4 Linear needs is copied
        per Linear from DONOR (a production PrismaQuant NVFP4 export of the
        same model: the same calibration the comparator arm serves with).
  --grid E4M3 --q256 1024
        Tessera-8 as a per-channel FP8 encoder, served W8A8.
  --fp8-rtn
        The per-channel FP8 round-to-nearest comparator (amax/448 per row,
        PrismaQuant's ``quantize_dequantize_fp8_dynamic``), through the same
        writer, so the two FP8 arms differ in nothing but the encoder.

Fused groups (q/k/v, gate/up) are one vLLM Linear with one
``weight_global_scale``; ``share_global`` moves each group onto one power of
two exactly or the export refuses.  A ``--plan-json`` may name a grid and
rung per tensor (or ``"BF16"``) so an allocation drives the export; the
default is one rung everywhere.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_checkpoint_driver import BASES, build_plan  # noqa: E402
from tessera.alphabet import E4M3_GRID, tuple_grid  # noqa: E402
from tessera.export import DEFAULT_CODE, encode_linear_planes, wire_recipe  # noqa: E402
from tessera.stock import (  # noqa: E402
    FLOAT_QUANTIZED, NVFP4_PACK_QUANTIZED, declared_format, materialize_stock,
    share_global, stock_bytes, stock_dequant, stock_kind, vllm_fp4_predicate)

FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj",
     ("q_proj", "k_proj", "v_proj")),
    (re.compile(r"^(.*\.mlp\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj",
     ("gate_proj", "up_proj")),
)

#: vLLM's NVFP4 config group, verbatim from a production PrismaQuant export
#: (fc45-0p6b-nvfp4): tensor_group 16 with an E4M3 block scale, static
#: per-tensor input global with local (dynamic) per-16 input scales.
NVFP4_WEIGHTS = {
    "num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
    "symmetric": True, "dynamic": False,
    "scale_dtype": "torch.float8_e4m3fn", "zp_dtype": "torch.float8_e4m3fn",
    "observer": "memoryless_minmax",
}
NVFP4_INPUTS = {
    "num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16,
    "symmetric": True, "dynamic": "local", "observer": "static_minmax",
    "scale_dtype": "torch.float8_e4m3fn", "zp_dtype": "torch.float8_e4m3fn",
}
#: PrismaQuant's ``FP8_E4M3_SCHEME``: per-channel static weights, per-token
#: dynamic activations -- vLLM's W8A8 FP8 route.
FP8_WEIGHTS = {
    "num_bits": 8, "type": "float", "strategy": "channel",
    "symmetric": True, "dynamic": False, "observer": "memoryless_minmax",
}
FP8_INPUTS = {
    "num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True,
}
FP8_MAX = 448.0


def grid_for(name: str):
    if name == "E4M3":
        return E4M3_GRID
    match = re.fullmatch(r"(E2M1)(?:x(\d+))?", name)
    if not match:
        raise SystemExit(f"unknown grid {name!r}; one of E2M1, E2M1x2, E4M3")
    return tuple_grid(BASES["E2M1"], int(match.group(2) or 1))


def module_of(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")]


def regex_target(module: str) -> str:
    return f"re:^{module.replace('.', '[.]')}$"


def fused_key(tensor_name: str):
    for pattern, fused, members in FUSED:
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def fp8_rtn(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel FP8 round-to-nearest: ``scale = amax(row) / 448``."""
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return {"weight": q.contiguous(), "weight_scale": scale.to(torch.float32).contiguous()}


def check_fp8_rtn_against_prismaquant(weight: torch.Tensor) -> str:
    try:
        from prismaquant.export_native_compressed import quantize_dequantize_fp8_dynamic
    except Exception as exc:  # pragma: no cover - PrismaQuant absent
        return f"PrismaQuant not importable ({type(exc).__name__}); local formula unchecked"
    q, s = quantize_dequantize_fp8_dynamic(weight.float())
    mine = fp8_rtn(weight)
    same_q = torch.equal(q.to(torch.float8_e4m3fn).view(torch.uint8), mine["weight"].view(torch.uint8))
    same_s = torch.equal(s.float().reshape(-1), mine["weight_scale"].reshape(-1))
    if not (same_q and same_s):
        raise SystemExit("the local FP8 RTN is not PrismaQuant's quantize_dequantize_fp8_dynamic")
    return "identical to PrismaQuant's quantize_dequantize_fp8_dynamic on the first Linear"


def stock_quantization_config(config_groups, ignore):
    """The ``quantization_config`` block for these groups, and what it resolves to.

    The top-level ``format`` is DERIVED from the groups (``declared_format``)
    rather than fixed.  It used to be the constant ``"mixed-precision"``, and
    the stock NVFP4 twin this exporter writes for the comparator arm is not
    mixed: one group, every target.  That constant is the whole of vLLM's
    FP4-model predicate, so our comparator answered False where a uniform-NVFP4
    checkpoint from anyone else answers True, and the two arms of a speed
    comparison were not the same compiled graph (#92).

    The resolved predicate travels back beside the block so the manifest records
    it.  A genuinely mixed artifact still declares ``mixed-precision`` -- that
    is the honest label -- and this record is what turns the fusion it gives up
    into a priced property of mixing instead of a silent one.

    Returns ``(None, None)`` when nothing was quantized: such a checkpoint
    declares no ``quantization_config`` at all, rather than one telling a
    runtime to look for compressed tensors it does not hold.
    """
    if not config_groups:
        return None, None
    fmt = declared_format(config_groups)
    return {
        "quant_method": "compressed-tensors",
        "format": fmt,
        "config_groups": config_groups,
        "ignore": list(ignore),
        "quantization_status": "compressed",
    }, vllm_fp4_predicate("compressed-tensors", fmt)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--plan-json", type=Path, default=None,
                    help='{"tensor.weight": {"grid": "E4M3", "q256": 1024} | "BF16", ...}')
    ap.add_argument("--fp8-rtn", action="store_true", help="the FP8 per-channel RTN comparator arm")
    ap.add_argument("--activations", choices=("w4a4", "w4a16"), default="w4a4")
    ap.add_argument("--input-scales", type=Path, default=None,
                    help="safetensors carrying <module>.input_global_scale per NVFP4 Linear")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    grid = grid_for(args.grid)
    default_plan, skipped = build_plan(args.src, args.q256, grid.arity)
    # The wire lane quantizes lm_head; the production comparators ignore it
    # (PrismaQuant's body-only convention, principle 12, and the donor carries
    # no input scale for it).  Matched arms, so it stays BF16 here.
    default_plan = {n: q for n, q in default_plan.items() if not n.startswith("lm_head")}
    if args.plan_json:
        override = json.loads(args.plan_json.read_text())
        plan = {}
        for name, spec in override.items():
            if name not in default_plan and spec != "BF16":
                raise SystemExit(f"plan names {name}, which is not a quantizable Linear here")
            if spec == "BF16":
                continue
            plan[name] = (grid_for(spec["grid"]), int(spec["q256"]))
        for name in default_plan:
            if name not in override:
                plan[name] = (grid, args.q256)
    else:
        plan = {name: (grid, q) for name, q in default_plan.items()}
    for name, shape in skipped:
        print(f"    passthrough (rows % arity) {name} {shape}")

    # FUSED NVFP4 GROUPS, expected roster derived from the PLAN before the
    # first encode (#212).  vLLM serves q/k/v and gate/up as ONE Linear with
    # one weight_global_scale, so the group is complete when every PLANNED
    # sibling has been encoded -- however many source shards they arrived
    # from.  The per-shard completion loop below therefore WAITS for members
    # a later shard holds instead of comparing a partial group against the
    # roster at an arbitrary shard boundary; a population that can never
    # complete (a sibling absent from the source, planned BF16, or planned
    # onto the FP8 kind) is refused here, before anything is encoded or
    # written.
    fused_expected: dict[str, set[str]] = {}
    if not args.fp8_rtn:
        for name, (unit_grid, _q) in sorted(plan.items()):
            fused = fused_key(name)
            if fused is None or unit_grid.name == "E4M3":
                continue
            key, roster = fused
            nvfp4 = {m for m in roster if m in plan and plan[m][0].name != "E4M3"}
            if nvfp4 != set(roster):
                absent = sorted(set(roster) - nvfp4)
                raise SystemExit(
                    f"fused group {key} cannot share one weight_global_scale: "
                    f"{sorted(nvfp4)} are planned NVFP4 but {absent} are not (BF16, FP8, "
                    "or not a quantizable Linear in this source). vLLM serves the group as "
                    "ONE Linear with one scale, so every member must be planned NVFP4 "
                    "together. Refusing before the first encode rather than at a shard "
                    "boundary.")
            fused_expected[key] = set(roster)

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for tensor, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor)
    else:
        shards = {}
        for path in sorted(args.src.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                shards[path.name] = list(handle.keys())

    input_scales = None
    if not args.fp8_rtn and args.activations == "w4a4" and any(g.arity and g.name != "E4M3" for g, _ in plan.values()):
        if args.input_scales is None:
            raise SystemExit("W4A4 needs --input-scales (a donor export's per-Linear input_global_scale)")
        input_scales = {}
        with safe_open(str(args.input_scales), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".input_global_scale"):
                    input_scales[key] = handle.get_tensor(key).to(torch.float32)

    units: dict[str, dict] = {}
    groups: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    nvfp4_modules, fp8_modules, ignored = [], [], []
    passthrough_bytes = 0
    new_weight_map: dict[str, str] = {}
    rtn_check = None
    started = time.time()
    total_units = len(plan)
    done = 0

    for shard, names in sorted(shards.items()):
        payload: dict[str, torch.Tensor] = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name not in plan:
                    payload[name] = tensor
                    passthrough_bytes += tensor.numel() * tensor.element_size()
                    if name in default_plan or any(name == s for s, _ in skipped):
                        ignored.append(module_of(name))
                    continue
                unit_grid, q256 = plan[name]
                weight = tensor.to(args.device, torch.float32).contiguous()
                if args.fp8_rtn:
                    tensors = fp8_rtn(weight)
                    if rtn_check is None:
                        rtn_check = check_fp8_rtn_against_prismaquant(weight)
                    record = {"kind": "fp8", "encoder": "fp8-rtn-per-channel", "wire_bytes": None}
                else:
                    exported, unit, forests = encode_linear_planes(
                        weight, grid=unit_grid, q256=q256, name=name, verify=not args.no_verify,
                    )
                    tensors = materialize_stock(unit, forests, DEFAULT_CODE)
                    recipe = wire_recipe(unit_grid, q256)
                    record = {
                        "kind": stock_kind(tensors), "encoder": "tessera",
                        "grid": unit_grid.name, "q256": q256,
                        "recipe": {"body": recipe.body.name, "span": recipe.span,
                                   "plane": recipe.scale_plane.name, "window_bits": recipe.window_bits},
                        "wire_bytes": exported.exact_bytes,
                        "wire_bpp": float(exported.bpp),
                    }
                rows, cols = weight.shape
                record.update({"rows": rows, "cols": cols})
                module = module_of(name)
                fused = fused_key(name)
                if record["kind"] == "nvfp4" and fused is not None:
                    key, _members = fused
                    groups.setdefault(key, {})[name] = tensors
                else:
                    record["resident_bytes"] = stock_bytes(tensors)
                    for suffix, value in tensors.items():
                        payload[f"{module}.{suffix}"] = value.cpu()
                if record["kind"] == "nvfp4":
                    nvfp4_modules.append(module)
                    if input_scales is not None:
                        scale_key = f"{module}.input_global_scale"
                        if scale_key not in input_scales:
                            raise SystemExit(f"no {scale_key} in {args.input_scales}; W4A4 cannot serve it")
                        payload[scale_key] = input_scales[scale_key]
                else:
                    fp8_modules.append(module)
                units[name] = record
                done += 1
                if done % 20 == 0 or done == total_units:
                    print(f"  [{done}/{total_units}] {name}  {time.time() - started:.0f}s", flush=True)
        # Fused groups: one global per vLLM Linear, exactly.  A group whose
        # planned siblings sit in a later source shard is left pending here
        # and completed -- one exact global share, one write, into whichever
        # output shard was open when the last member arrived (#212).
        for key, members in list(groups.items()):
            if set(members) != fused_expected[key]:
                continue
            before = {n: stock_dequant({k: v.to(args.device) for k, v in t.items()}) for n, t in members.items()}
            shared, divisor = share_global(members)
            for n, tensors in shared.items():
                after = stock_dequant({k: v.to(args.device) for k, v in tensors.items()})
                if not torch.equal(after, before[n]):
                    raise SystemExit(f"{n}: sharing the global changed the served weights")
                units[n]["resident_bytes"] = stock_bytes(tensors)
                units[n]["shared_global_divisor"] = divisor
                units[n]["own_global_divisor"] = float(members[n]["weight_global_scale"])
                for suffix, value in tensors.items():
                    payload[f"{module_of(n)}.{suffix}"] = value.cpu()
            del groups[key]
        out_shard = shard
        save_file(payload, str(args.out / out_shard), metadata={"format": "pt"})
        for key in payload:
            new_weight_map[key] = out_shard
        print(f"wrote {out_shard}: {len(payload)} tensors", flush=True)

    if groups:
        # Unreachable when the source holds every planned tensor (the roster
        # check above refused anything that cannot complete), so this names an
        # index that promised tensors no shard delivered.
        missing = {key: sorted(fused_expected[key] - set(members))
                   for key, members in sorted(groups.items())}
        raise SystemExit(
            f"fused groups never completed at end of input, missing members {missing}: "
            "the source shards never delivered tensors the plan and the source index name.")

    # --- config.json ---------------------------------------------------------
    config = json.loads((args.src / "config.json").read_text())
    config_groups = {}

    def targets(modules):
        found = sorted(set(modules))
        fused_names = set()
        by_prefix: dict[str, set[str]] = {}
        for m in found:
            for pattern, fused, member_names in FUSED:
                match = pattern.match(m + ".weight")
                if match:
                    by_prefix.setdefault(match.group(1) + fused, set()).add(match.group(2))
        for fused_module, present in by_prefix.items():
            for pattern, fused, member_names in FUSED:
                if fused_module.endswith(fused) and present == set(member_names):
                    fused_names.add(fused_module)
        return [regex_target(m) for m in sorted(set(found) | fused_names)]

    if nvfp4_modules:
        group = {"format": NVFP4_PACK_QUANTIZED, "weights": dict(NVFP4_WEIGHTS)}
        if input_scales is not None:
            group["input_activations"] = dict(NVFP4_INPUTS)
        group["targets"] = targets(nvfp4_modules)
        config_groups[f"group_{len(config_groups)}"] = group
    if fp8_modules:
        config_groups[f"group_{len(config_groups)}"] = {
            "format": FLOAT_QUANTIZED, "weights": dict(FP8_WEIGHTS),
            "input_activations": dict(FP8_INPUTS), "targets": targets(fp8_modules),
        }
    ignore = ["lm_head", "model.embed_tokens"] + sorted(set(ignored) - {"lm_head", "model.embed_tokens"})
    quantization_config, fp4_predicate = stock_quantization_config(config_groups, ignore)
    if quantization_config is None:
        config.pop("quantization_config", None)
    else:
        config["quantization_config"] = quantization_config
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if len(shards) > 1:
        total = sum((args.out / s).stat().st_size for s in shards)
        (args.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": new_weight_map}, indent=2))
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in args.src.glob(pattern):
            if aux.name in ("config.json", "model.safetensors.index.json"):
                continue
            shutil.copy2(aux, args.out / aux.name)

    # --- the accounting: wire and resident, both stated ------------------------
    params = sum(u["rows"] * u["cols"] for u in units.values())
    resident = sum(u["resident_bytes"] for u in units.values())
    wire = sum(u["wire_bytes"] for u in units.values() if u["wire_bytes"] is not None)
    totals = {
        "quantized_params": params,
        "resident_bytes": resident,
        "resident_bpp": float(Fraction(resident * 8, params)) if params else None,
        "wire_bytes": wire if not args.fp8_rtn else None,
        "wire_bpp": float(Fraction(wire * 8, params)) if (params and not args.fp8_rtn) else None,
        "passthrough_bytes": passthrough_bytes,
        "checkpoint_bytes": sum((args.out / s).stat().st_size for s in shards),
    }
    manifest = {
        "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": "fp8-rtn" if args.fp8_rtn else f"tessera {grid.name} q256={args.q256}",
        "activations": None if args.fp8_rtn else args.activations,
        "input_scales_from": str(args.input_scales) if input_scales is not None else None,
        "fp8_rtn_check": rtn_check,
        # What the declaration above resolves to in the pinned serving runtime.
        # ``None`` when this checkpoint quantized nothing and so declares no
        # quantization_config.
        "vllm_fp4_predicate": fp4_predicate,
        "resident_format_note": (
            "resident bytes are the stock format's (NVFP4 4.5 bpp, FP8 per-channel 8 bpp); "
            "wire_bpp is what the same unit costs on Tessera's kernel lane and is NOT what this checkpoint holds"
        ),
        "totals": totals,
        "units": units,
    }
    (args.out / "tessera_stock_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(totals, indent=2))
    print(f"elapsed {time.time() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
