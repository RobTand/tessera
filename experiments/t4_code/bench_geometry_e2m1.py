"""D41 E2M1 measurement adapter; never substitutes an E4M3 or BF16 reader.

The serializable scalar and tuple grids are enumerated at q256 step one.
Current serving uses uniform E2M1x2 span-two TCQ, not the experimental
routed_fused_e2m1 WINDOW/LUT16 library (L14). Research tuple subcap wires
have L12; routed served_recipe promotes them to TCQ, while dense keeps L12.
Owner refusals remain missing measurements, not a family-wide exclusion.
--packed-reader explicitly enables actual mixed TCQ and dense L12 WINDOW
geometry without changing a serving default. --correctness checks its code
and scale bytes against stock and its native arithmetic on bounded shapes.
--quality-structure binds the CPU screen to the actual routed or dense recipe.

--prepare-inputs extracts the real layer-three expert-zero BF16 source tiles.
--cpu-preflight is the same entry point's D38 slice. --quality screens actual
sample tiles on CPU with the recipe measured, not a Gaussian quality proxy.
GPU timings use fully encoded seeded weights at the actual projection shape;
all experts share those synthetic weights, as in bench_e2m1.py. No quality,
served KL, serving admission, default or encoded-wire changes are implied.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import zlib

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t8r_speed"))
from bench_geometry import pick_recorded, time_call
from bench_rates import sha
from bench_t8r import (PowerSampler, balanced_routing, kernel_profile,
                       recorded_routing)
from tessera.alphabet import SERIALISABLE_GRIDS, grid_for_name
from tessera.calculator import terminal_rate
from tessera.export import encode_linear, served_recipe, wire_recipe
from tessera.grammar import bresenham_rate_schedule
from tessera.manifest import BodyKind, body_rate_cap, scale_plane_terminal_flags
from tessera.structure import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE
from tessera.unit_artifact import read_unit_artifact

MS = (1, 16, 2048, 4096)
SHAPES = (("routed", "gate_up", 1024, 4096, 0),
          ("routed", "down", 4096, 1024, 2),
          ("dense", "o_proj", 4096, 4096, 2),
          ("dense", "q_b", 8192, 1536, 2))
GRID_OWNER = "prismaquant.tessera_menu.menu_families; tessera_formats.family_q256_bounds/realisable_rungs(step_q256=1); tessera.alphabet.SERIALISABLE_GRIDS"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False))
    temporary.replace(path)


def rational(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def family_grids():
    return sorted((g for g in SERIALISABLE_GRIDS.values()
                   if g.name == "E2M1" or g.name.startswith("E2M1x")), key=lambda g: g.arity)


def bounds(grid):
    recipe = wire_recipe(grid)
    low = Fraction(256, grid.arity)
    high = Fraction(body_rate_cap(recipe.body, grid) * 256, grid.arity)
    if low.denominator != 1 or high.denominator != 1:
        raise ValueError(f"{grid.name}: arity does not divide the q256 grid")
    return int(low), int(high)


def recipe_kwargs(recipe):
    return {"body": recipe.body, "span": recipe.span,
            "scale_plane": recipe.scale_plane, "window_bits": recipe.window_bits,
            "window_seed": recipe.window_seed, "window_sigma": recipe.window_sigma,
            "channel_sigma": recipe.channel_sigma}


def exact_bits(grid, q, rows, cols, recipe):
    base, refine, row_scale = scale_plane_terminal_flags(recipe.scale_plane)
    rate = terminal_rate(q * grid.arity, rows, cols, arity=grid.arity,
                         cap=body_rate_cap(recipe.body, grid), span=recipe.span,
                         window_bits=recipe.window_bits, code_bytes=grid.code_bytes,
                         with_scale_base=base, with_scale_refine=refine,
                         with_row_scale=row_scale, with_diagonals=False, completion=0,
                         with_forest=recipe.body is BodyKind.TCQ)
    return Fraction(rate) * rows * cols


def owner_refusal(grid, q, kind, cols=256):
    structure = STRUCTURE_ROUTED_MOE if kind == "routed" else STRUCTURE_DENSE
    recipe = served_recipe(grid, q, structure)
    rates = bresenham_rate_schedule(Fraction(q * grid.arity, 256), cols,
                                   cap=body_rate_cap(recipe.body, grid))
    if grid.arity != 2:
        return {"owner": "tessera.kernel_a4.build_code_nibbles", "reason":
                f"a span-2 code table is defined for arity 2; this unit has arity {grid.arity}",
                "serving_owner": "serving.runtime_contract.formats has no scalar E2M1 reader"}
    if recipe.body is not BodyKind.TCQ:
        return {"owner": "tessera.compact_prep.prepare_span2_compact", "reason":
                "prepare_span2_compact takes a TCQ unit; this one has no forest",
                "experimental_owner": "tessera.routed_fused_e2m1 dense/routed readers require WINDOW_BITS=14",
                "actual_window_bits": recipe.window_bits,
                "missing_evidence": "current L12 WINDOW is not the L14 fused specialization or the served TCQ path"}
    if len(set(rates)) != 1:
        return {"owner": "tessera.compact_prep.prepare_span2_compact", "reason":
                f"the span-2 planes take one forest per unit; rates {sorted(set(rates))}",
                "missing_evidence": "no current native span-2 mixed-rate specialization"}
    return None


def catalog():
    families = {}
    for grid in family_grids():
        lo, hi = bounds(grid)
        rows = []
        for q in range(lo, hi + 1):
            research = wire_recipe(grid, q)
            rows.append({"q256": q, "arity": grid.arity,
                         "research_recipe": research.to_config(),
                         "served_recipes": {kind: served_recipe(grid, q, structure).to_config()
                            for kind, structure in (("routed", STRUCTURE_ROUTED_MOE), ("dense", STRUCTURE_DENSE))},
                         "owner_refusals": {kind: owner_refusal(grid, q, kind) for kind in ("routed", "dense")}})
        families[f"TESSERA_E2M1_K{grid.arity}"] = {"rung_min": lo, "rung_max": hi,
            "grid_step_q256": 1, "grid_owner": GRID_OWNER, "rungs": rows}
    return {"families": families, "scope": "reader coverage, not canonical admission or dominance"}


def prepare_inputs(model, out):
    from safetensors import safe_open
    mapping = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    samples, metadata = {}, {}
    for role in ("gate", "up", "down"):
        name = f"model.language_model.layers.3.mlp.experts.0.{role}_proj.weight"
        with safe_open(str(Path(model) / mapping[name]), framework="pt", device="cpu") as handle:
            source = handle.get_slice(name)
            shape = source.get_shape()
            weight = source[:32, :256].contiguous()
        expected = [4096, 2048] if role == "down" else [2048, 4096]
        if shape != expected or weight.dtype != torch.bfloat16:
            raise ValueError((name, shape, weight.dtype, expected))
        samples[role] = weight
        metadata[role] = {"tensor": name, "file": mapping[name], "source_shape": shape,
            "slice": [[0, 32], [0, 256]], "source_dtype": "bfloat16",
            "source_sha256": hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "source_squared_norm": float(weight.double().square().sum())}
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(samples, Path(out) / "source-tiles.pt")
    dump(Path(out) / "source-tiles.json", metadata)
    dump(Path(out) / "family-catalog.json", catalog())


def load_samples(path):
    samples = torch.load(path, map_location="cpu", weights_only=True)
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    for role in ("gate", "up", "down"):
        weight = samples[role]
        if tuple(weight.shape) != (32, 256) or weight.dtype != torch.bfloat16:
            raise ValueError(f"{role}: expected real BF16 32x256 sample")
        actual = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        if actual != metadata[role]["source_sha256"]:
            raise ValueError(f"{role}: sample bytes do not match their own digest")
    return samples, metadata


def quality(args, grid, samples, metadata):
    result = {"schema": "tessera.rung_quality.v1", "source_kind": "actual_sampled_expert_weights",
              "device": "cpu", "format": f"TESSERA_E2M1_K{grid.arity}",
              "objective": "unweighted weight-space relative SSE; not served KL or H-weighted quality",
              "codec": "encode_linear with actual served_recipe; verified exact accountant",
              "structure": args.quality_structure, "grid": grid.name,
              "rungs": {}, "start_unix": time.time()}
    for q in args.qs:
        recipe = served_recipe(grid, q, args.quality_structure)
        row = {"measurement_status": "measured", "source_kind": result["source_kind"], "device": "cpu",
               "arity": grid.arity, "grid": grid.name, "rung_q256": q,
               "structure": args.quality_structure, "recipe": recipe.to_config(), "samples": [], "anomaly_flags": []}
        for role, weight in samples.items():
            unit = encode_linear(weight, grid=grid, q256=q, **recipe_kwargs(recipe))
            decoded = read_unit_artifact(unit.blob).double()
            if not bool(torch.isfinite(decoded).all()):
                raise ValueError("nonfinite quality reconstruction")
            bits = exact_bits(grid, q, 32, 256, recipe)
            if bits != unit.exact_bytes * 8:
                raise ValueError((grid.name, q, bits, unit.exact_bytes))
            sse = float((decoded - weight.double()).square().sum())
            row["samples"].append({**metadata[role], "relative_sse": sse / metadata[role]["source_squared_norm"],
                "squared_error": sse, "exact_bytes": unit.exact_bytes, "accounted_bits": rational(bits),
                "grid": grid.name, "arity": grid.arity, "rung_q256": q,
                "structure": args.quality_structure, "recipe": recipe.to_config(),
                "encoder_blob_sha256": hashlib.sha256(unit.blob).hexdigest()})
        result["rungs"][str(q)] = row
        dump(Path(args.out) / "quality.json", result)
    result["end_unix"] = time.time()
    dump(Path(args.out) / "quality.json", result)


class CompilerCapture:
    """Retain metadata from the very Triton launch used, not a guessed kernel."""
    def __init__(self, module, out):
        self.observed = {}
        self.binaries = {}
        self.originals = []
        self.out = Path(out)
        modules = [(module, ("_a4_span2_gemm_kernel", "_a4_span2_grouped_kernel"))]
        from tessera import kernel_a4_wire
        modules.append((kernel_a4_wire, ("_a4_wire_gemm_kernel",)))
        for owner, symbol in ((owner, symbol) for owner, symbols in modules for symbol in symbols):
            jit = getattr(owner, symbol)
            original = jit.run
            self.originals.append((jit, original))
            def run(*args, _original=original, _symbol=symbol, **kwargs):
                compiled = _original(*args, **kwargs)
                if compiled is not None:
                    self.observed[_symbol] = compiled
                return compiled
            jit.run = run

    def geometry(self, symbol, grid, q, rows, cols, recipe, unit):
        compiled = self.observed[symbol]
        md = compiled.metadata
        resource = {"REG": compiled.n_regs, "SPILLS": compiled.n_spills,
                    "SHARED": md.shared, "compiler_symbol": md.name}
        folder = self.out / "compiler"
        folder.mkdir(exist_ok=True)
        artifacts = {}
        for language in ("ptx", "ttgir", "cubin"):
            body = compiled.asm.get(language)
            if body:
                raw = body.encode() if isinstance(body, str) else bytes(body)
                digest = hashlib.sha256(raw).hexdigest()
                filename = folder / f"{digest}.{language}"
                if not filename.exists():
                    filename.write_bytes(raw)
                artifacts[language] = {"path": str(filename), "sha256": digest}
                if language == "cubin":
                    self.binaries[digest] = {"path": str(filename), "symbol": md.name}
        available = torch.cuda.get_device_properties("cuda").shared_memory_per_block_optin
        if hasattr(unit, "body_kind"):
            return {"bits_per_256_weight_tile": rational(exact_bits(grid, q, rows, cols, recipe) * 256 / (rows * cols)),
                "alignment": {"arity": 2, "span": unit.layout["span"],
                    "inputs": {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                        "bytes": t.numel() * t.element_size()} for name, t in unit.named_tensors()},
                    "column_layout": unit.layout},
                "shared_memory": {"requested_bytes": md.shared, "available_bytes": available, "fits": md.shared <= available},
                "register_pressure": resource,
                "decode_width": {"window_bits": unit.window_bits, "value_bits": 4, "arity": 2,
                    "run_widths": sorted(set(unit.layout["column_rates"])), "memory": unit.memory,
                    "span": unit.layout["span"], "block_m": 64, "block_n": 64, "block_k": 128,
                    "mma_k": 64, "scale_group": 16},
                "raw": {"owner": "tessera.compact_prep.prepare_a4_wire_compact; tessera.kernel_a4_wire.PreparedA4Wire",
                    "decoder_kind": "native_packed_" + unit.body_kind,
                    "compiler_artifacts": artifacts, "lut_entries": 16,
                    "global_bytes_per_unit": 4, "descriptors": "packed BODY, column rates and bit starts; no WINDOW word ring"}}
        return {"bits_per_256_weight_tile": rational(exact_bits(grid, q, rows, cols, recipe) * 256 / (rows * cols)),
            "alignment": {"rate": unit.rate, "arity": unit.arity, "span": 2,
                "plane_shapes": {name: list(getattr(unit, name).shape) for name in
                    ("select", "label", "point", "nibbles", "lut_bytes", "label_lut", "code_nibbles")},
                "plane_bytes": {name: getattr(unit, name).numel() * getattr(unit, name).element_size() for name in
                    ("select", "label", "point", "nibbles", "lut_bytes", "label_lut", "code_nibbles")}},
            "shared_memory": {"requested_bytes": md.shared, "available_bytes": available, "fits": md.shared <= available},
            "register_pressure": resource,
            "decode_width": {"window_bits": recipe.window_bits, "value_bits": 4, "arity": grid.arity,
                "run_widths": [unit.rate], "memory": unit.memory, "span": 2,
                "block_m": 64, "block_n": 64, "block_k": 128, "mma_k": 64, "scale_group": 16},
            "raw": {"owner": "terminal_rate; compact_prep.prepare_span2_compact; kernel_a4; launched Triton CompiledKernel",
                "compiler_artifacts": artifacts, "lut_entries": 16,
                "global_bytes_per_unit": 4, "descriptors": "span2 planes and expert CSR offsets; no WINDOW chunk_desc"}}

    def close(self):
        for jit, original in self.originals:
            jit.run = original


def encoded_unit(grid, q, rows, cols, seed, kind, packed=False):
    from tessera.compact_prep import parse_compact_wire
    from tessera.serving.native_a4 import prepare_a4_unit
    recipe = served_recipe(grid, q, STRUCTURE_ROUTED_MOE if kind == "routed" else STRUCTURE_DENSE)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(rows, cols, generator=generator, device="cuda") * 0.02).to(torch.bfloat16)
    encoded = encode_linear(weight, grid=grid, q256=q, **recipe_kwargs(recipe))
    if exact_bits(grid, q, rows, cols, recipe) != encoded.exact_bytes * 8:
        raise ValueError("projection bytes differ from the actual recipe accountant")
    wire = parse_compact_wire(encoded.blob, device="cuda", name=kind)
    if packed:
        from tessera.compact_prep import prepare_a4_wire_compact
        return prepare_a4_wire_compact(wire, device="cuda"), recipe
    return prepare_a4_unit(wire), recipe


def build_group(grid, q, shape, args, recorded, capture):
    from tessera.kernel_a4 import A4UnitStack
    from tessera.serving.native_a4 import a4_dense_apply, a4_grouped_apply, stack_epilogues
    kind, name, rows, cols, mode = shape
    refusal = owner_refusal(grid, q, kind, cols)
    recipe = served_recipe(grid, q, STRUCTURE_ROUTED_MOE if kind == "routed" else STRUCTURE_DENSE)
    head = {"kind": kind, "shape": name, "mode": mode, "q256": q, "rows": rows, "cols": cols,
            "arity": grid.arity, "body_kind": recipe.body.name.lower(), "window_bits": recipe.window_bits,
            "recipe": recipe.to_config(), "cells": {}}
    packed = bool(args.packed_reader and grid.arity == 2 and refusal is not None)
    if refusal and not packed:
        head["owner_refusal"] = refusal
        return head, None
    seed = zlib.crc32(f"weight:{name}".encode())
    units = [encoded_unit(grid, q, rows, cols, seed + i, kind, packed=packed)[0] for i in range(2 if mode == 0 else 1)]
    gs = torch.tensor(448.0 * 6.0 / 3.0, device="cuda", dtype=torch.float32)
    if packed:
        from tessera.kernel_a4_wire import PreparedA4Wire
        prepared = [PreparedA4Wire([unit] * (288 if kind == "routed" else 1), gs) for unit in units]
        stacks = epilogues = []
        dense_epilogue = None
        head["decoder_kind"] = "native_packed_" + recipe.body.name.lower()
    else:
        stacks = [A4UnitStack.stack([unit] * 288) for unit in units] if kind == "routed" else []
        epilogues = [stack_epilogues(stack, gs) for stack in stacks]
        dense_epilogue = units[0].epilogue_for(gs) if kind == "dense" else None

    def make(m, how):
        generator = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"x:{name}:{m}:{how}".encode()))
        xrows = m * 8 if kind == "routed" and mode == 2 else m
        x = (torch.randn(xrows, cols, device="cuda", generator=generator) * 0.5).to(torch.bfloat16)
        holder = {}
        if kind == "routed":
            ids, weights = recorded_routing(recorded[m]["path"], m, "cuda") if how == "recorded" else balanced_routing(m, "cuda")
            flat = ids.flatten().long()
            order = torch.argsort(flat, stable=True)
            counts = torch.bincount(flat, minlength=288).to(torch.int32)
            offsets = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), torch.cumsum(counts, 0).to(torch.int32)))
            tokens = (order // 8 if mode == 0 else order).to(torch.int32)
            def call():
                if packed:
                    values = [item(x, expert_offsets=offsets, route_ids=tokens, num_routes=m * 8) for item in prepared]
                else:
                    values = [a4_grouped_apply(x, stack, gs, expert_offsets=offsets, route_ids=tokens,
                                num_routes=m * 8, epilogues=ep) for stack, ep in zip(stacks, epilogues)]
                if mode == 0:
                    gate, up = values
                    holder["out"] = (torch.nn.functional.silu(gate.float().clamp(max=10)) * up.float().clamp(-10, 10)).to(torch.bfloat16)
                else:
                    holder["out"] = (values[0].float() * weights.flatten()[order, None]).to(torch.bfloat16)
            symbol = "_a4_wire_gemm_kernel" if packed else "_a4_span2_grouped_kernel"
            owner_symbol = "tessera.kernel_a4_wire.PreparedA4Wire" if packed else "tessera.kernel_a4.a4_span2_grouped_gemm"
        else:
            def call():
                holder["out"] = prepared[0](x) if packed else a4_dense_apply(x, units[0], gs, epilogue=dense_epilogue)
            symbol = "_a4_wire_gemm_kernel" if packed else "_a4_span2_gemm_kernel"
            owner_symbol = "tessera.kernel_a4_wire.PreparedA4Wire" if packed else "tessera.kernel_a4.a4_span2_gemm"
        call()
        torch.cuda.synchronize()
        geo = capture.geometry(symbol, grid, q, rows, cols, recipe, units[0])
        return call, holder, geo, owner_symbol
    return head, make


def run_gpu(args, grid):
    from tessera import kernel_a4
    from tessera.serving.runtime_image import CENSUS_IMAGE_ENV
    power = PowerSampler()
    if power.source != "pynvml":
        raise RuntimeError("D41 requires actual NVML board power; pynvml not available")
    capture = CompilerCapture(kernel_a4, args.out)
    recorded = pick_recorded(args.routing, args.ms_values) if args.routing else {}
    source_paths = [Path(kernel_a4.__file__)]
    if args.packed_reader:
        from tessera import kernel_a4_wire, compact_prep
        source_paths += [Path(kernel_a4_wire.__file__), Path(compact_prep.__file__)]
    source_sha = hashlib.sha256(b"".join(path.read_bytes() for path in source_paths)).hexdigest()
    props = torch.cuda.get_device_properties("cuda")
    lo, hi = bounds(grid)
    meta = {"format": f"TESSERA_E2M1_K{grid.arity}", "family": "e2m1", "arity": grid.arity,
            "rung_min": lo, "rung_max": hi, "grid_owner": GRID_OWNER, "grid_step_q256": 1,
            "architecture": f"sm_{props.major}{props.minor}", "library": "native_packed" if args.packed_reader else "native_span2",
            "kernel_sha": source_sha, "library_sha256": None, "tessera_head": os.environ.get("TESSERA_HEAD"),
            "library_kind": "actual Triton compiled CUDA binary set, not an ELF extension",
            "activation_contract": "e2m1_group16_ue4m3_static; BF16 inputs, fixed static global448*6/3, native quantizer",
            "image": os.environ.get(CENSUS_IMAGE_ENV, os.environ.get("ORACLE_IMAGE")),
            "requested_image": os.environ.get("ORACLE_IMAGE"),
            "torch": torch.__version__, "host": os.environ.get("HOST_NAME"),
            "pb_action": os.environ.get("PB_ACTION_KEY", os.environ.get("PRISMABUILD_ACTION_KEY")),
            "paired_seed_contract": "fixed full-shape weight and activation seeds independent of rung; identical routing IDs and uniform weights1/8",
            "statistic": "mean of forward and reverse pass medians; graph replay with event fallback recorded",
            "cases": ",".join(f"q{q}" for q in args.qs), "ms": args.ms, "part": args.part,
            "recorded": recorded, "power_source": power.source, "envelope_w": 140,
            "execution_scope": "actual native span2 decoded projections; prepared routing, gate/up activation and route-weighted down; no complete served MoE or TP collective",
            "start_unix": time.time()}
    groups = {}
    path = Path(args.out) / f"bench_geometry_{args.part}.json"
    def save():
        meta["compiled_binaries"] = capture.binaries
        meta["library_sha256"] = (hashlib.sha256(json.dumps(sorted(capture.binaries)).encode()).hexdigest()
                                   if capture.binaries else None)
        dump(path, {"meta": meta, "groups": groups})
    try:
        specs = [(q, shape) for q in args.qs for shape in SHAPES if args.part == "all" or shape[0] == args.part]
        for pas in ("F", "R"):
            for q, shape in (specs if pas == "F" else reversed(specs)):
                head, make = build_group(grid, q, shape, args, recorded, capture)
                key = f"{shape[0]}:{q}:{shape[1]}"
                rec = groups.setdefault(key, head)
                if make is None:
                    save()
                    continue
                variants = [(m, "balanced" if shape[0] == "routed" else None) for m in args.ms_values]
                if shape[0] == "routed":
                    variants += [(m, "recorded") for m in sorted(recorded)]
                for m, how in (variants if pas == "F" else reversed(variants)):
                    ckey = str(m) + (f":{how}" if how else "")
                    cell = rec["cells"].setdefault(ckey, {})
                    call, holder, geo, symbol = make(m, how)
                    timer, samples = time_call(call, args.warmup, args.iters)
                    cell[pas] = {"median_ms": statistics.median(samples), "samples_ms": samples,
                                 "min_ms": min(samples), "timer": timer, "unix": time.time(),
                                 "clock": power.read_clock_temperature()}
                    if pas == "F":
                        cell.update(normalized_geometry=geo, kernel_path=symbol,
                                    out_sha256=sha(holder["out"]), profile=kernel_profile(call, reps=1, full_names=True),
                                    power=power.sample_during(call, args.power_s))
                    else:
                        cell["ms"] = (cell["F"]["median_ms"] + cell["R"]["median_ms"]) / 2
                        cell["spread"] = abs(cell["F"]["median_ms"] - cell["R"]["median_ms"]) / cell["ms"]
                    print(json.dumps({"group": key, "M": ckey, "pass": pas, "ms": cell[pas]["median_ms"]}), flush=True)
                    save()
                del make
                torch.cuda.empty_cache()
        meta["end_unix"] = time.time()
        save()
    finally:
        capture.close()


def _rounding_gamma(steps, epsilon):
    """Outward-rounded finite gamma_n; no first-order remainder is dropped."""
    if type(steps) is not int or steps < 0:
        raise ValueError("the rounding bound needs a non-negative integer operation count")
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("the rounding bound needs a finite precision term between zero and one")
    if steps >= 1.0 / epsilon:
        raise ValueError("the rounding bound needs steps * epsilon below one")
    scaled = steps * epsilon
    if scaled == 0:
        return 0.0
    return math.nextafter(scaled / (1.0 - scaled), math.inf)


def dense_packed_fp4_operand_magnitude(rendered_x, weight):
    """Conditional upper magnitude for the actual float32 operands.

    Float32 operands convert exactly to float64. Their products remain
    normal and finite in float64. ASSUMPTION: the positive GEMM has at most
    2*K local errors, each at most float64 epsilon times its exact magnitude.
    Inflate that conditional contraction and round its scalar value upward.
    Grouped gates take the maximum over their actual expert segments.
    """
    if rendered_x.dim() != 2 or weight.dim() != 2:
        raise ValueError("operand magnitude needs two-dimensional activation and weight tiles")
    if rendered_x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("operand magnitude needs the actual float32 reference operands")
    if rendered_x.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight contraction lengths differ")
    if rendered_x.shape[0] < 1 or weight.shape[0] < 1 or rendered_x.shape[1] < 1:
        raise ValueError("operand magnitude needs at least one row and one contraction step")
    if not torch.isfinite(rendered_x).all() or not torch.isfinite(weight).all():
        raise ValueError("operand magnitude needs finite activation and weight tiles")
    gamma = _rounding_gamma(2 * int(rendered_x.shape[1]), torch.finfo(torch.float64).eps)
    if gamma >= 1:
        raise ValueError("the float64 magnitude reduction cannot establish an upper bound")
    measured = float((rendered_x.double().abs() @ weight.double().abs().T).max())
    if not math.isfinite(measured):
        raise ValueError("the operand magnitude reduction is not finite")
    if measured == 0:
        return 0.0
    denominator = math.nextafter(1.0 - gamma, -math.inf)
    upper = math.nextafter(measured / denominator, math.inf)
    if not math.isfinite(upper):
        raise ValueError("the operand magnitude upper bound is not finite")
    return upper


def _packed_fp4_bound_inputs(operand_magnitude, k):
    """Validate the scalar contract independently of the diagnostic receipt."""
    if type(k) is not int or k < 1:
        raise ValueError("the derived packed FP4 bound needs the contraction length as a positive integer")
    if k % 128:
        raise ValueError("the packed reader contraction length must be a multiple of 128")
    try:
        magnitude = float(operand_magnitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("the derived packed FP4 bound needs a real operand magnitude") from exc
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("the derived packed FP4 bound needs a finite non-negative operand magnitude")
    return magnitude


def _packed_fp4_conditional_terms(magnitude, k):
    """Compute the existing envelope without native qualification policy."""
    epsilon = torch.finfo(torch.float32).eps
    native_steps, reference_steps = 2 * k + 3, 2 * k + 2
    native_gamma = _rounding_gamma(native_steps, epsilon)
    reference_gamma = _rounding_gamma(reference_steps, epsilon)
    combined = math.nextafter(native_gamma + reference_gamma, math.inf)
    coefficient = math.nextafter(combined / (1.0 - 2.0 * epsilon), math.inf)
    atol = math.nextafter(coefficient * magnitude, math.inf) if magnitude else 0.0
    if not math.isfinite(atol):
        raise ValueError("the derived packed FP4 bound is not finite")
    return epsilon, native_steps, reference_steps, native_gamma, reference_gamma, coefficient, atol


def derive_packed_fp4_arithmetic_bound(operand_magnitude, *, k):
    """Return the unchanged conditional allowance and its explicit assumptions.

    The complete derivation is in section 17 of the serving contract.
    Let e = 2^-23 and gamma(n) = n*e/(1-n*e). The diagnostic uses::

        [gamma(2*K+3) + gamma(2*K+2)] * M_upper / (1-2*e)

    ASSUMPTION: each native scaled product and sum obeys the local relative
    error model, with at most 2*K errors along each output path. PTX does
    not guarantee either fact. ASSUMPTION: both normal finite divisions
    have error at most two ULPs. These are activation/896 and weight_global/896.
    ASSUMPTION: the reference GEMM and magnitude GEMM obey their stated models.
    ASSUMPTION: reference weight formation is exact and intermediate values
    obey the relative models, without overflow or subnormal exceptions.

    Both paths share one quantizer and return float32. No independent BF16
    term applies here. The fused WINDOW epilogue has separate BF16 terms.
    Neither finite outputs nor diagnostic agreement proves these assumptions.
    """
    magnitude = _packed_fp4_bound_inputs(operand_magnitude, k)
    (epsilon, native_steps, reference_steps, native_gamma, reference_gamma,
     coefficient, atol) = _packed_fp4_conditional_terms(magnitude, k)
    return ({"atol": atol, "rtol": 0.0}, {
        "schema": "tessera.packed_fp4_arithmetic_bound.v4",
        "status": "conditional_diagnostic_only",
        "arithmetic_qualified": False,
        "native_arithmetic_qualified": False,
        "intermediate_domain_established": False,
        "missing_native_contract": "PTX supplies no scaled-product error inequality, accumulation depth bound, or subnormal rule for E2M1 MMA.",
        "bound": "[gamma(2*K+3) + gamma(2*K+2)] * max output sum_k |ax||w| / (1-2*epsilon_fp32)",
        "u_fp32": 2.0**-24,
        "epsilon_fp32": epsilon,
        "packed_reader_k_tiles": k // 128,
        "native_rounding": "ASSUMPTION: each scaled product and sum has local relative error at most epsilon_fp32. Each output path has at most 2*K errors.",
        "normalization_rounding": "ASSUMPTION: each activation/896 and weight_global/896 division has at most two ULPs of error. Each exact quotient is normal and finite.",
        "reference_rounding": "ASSUMPTION: the actual float32 reference GEMM has at most 2*K product and sum errors. Each local relative error is at most epsilon_fp32.",
        "magnitude_rounding": "ASSUMPTION: the positive float64 GEMM has at most 2*K local errors. Each local relative error is at most epsilon_fp64.",
        "epilogue_rounding": "ASSUMPTION: the native float32 multiplication has local relative error at most epsilon_fp32. Its exact result is normal and finite.",
        "weight_formation": "ASSUMPTION: the UE4M3 scales are finite and nonnegative. An exact float32 power-of-two weight global produces exact normal finite reference weights.",
        "quantization_error": "excluded: both paths use the same represented activation codes, group scales, weight codes, and weight scales",
        "derivation_owner": "docs/tessera-serving-and-moe-contract.md#17-packed-t4-arithmetic-contract-2026-10-07-issue-1007",
        "native_steps": native_steps,
        "reference_steps": reference_steps,
        "native_gamma": native_gamma,
        "reference_gamma": reference_gamma,
        "k": k,
        "coefficient": coefficient,
        "operand_magnitude_upper": magnitude,
        "atol": atol,
        "rtol_is_zero_because": "the bound scales with the operand magnitude sum, not with the expected output",
        "scope": "conditional normal finite intermediate arithmetic; intermediate domain not established by finite outputs; no overflow or subnormal qualification",
        "bf16_terms": "none: one shared BF16 quantizer; float32 reference, accumulation and output",
    })


def check_packed_fp4_arithmetic(actual, expected, operand_magnitude, *, k):
    """Reject diagnostic discrepancies without certifying the native model."""
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        raise ValueError("the packed FP4 arithmetic gate needs float32 native and reference outputs")
    if actual.dim() != 2 or actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError("the packed FP4 arithmetic gate needs equal nonempty two-dimensional output shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("the packed FP4 arithmetic gate needs finite native and reference outputs")
    tolerances, receipt = derive_packed_fp4_arithmetic_bound(operand_magnitude, k=k)
    # Float32 subtraction/threshold casting can admit an error just beyond
    # the outward-rounded allowance. Compare the actual float32 values in
    # float64 instead; significant near-boundary subtraction is exact there.
    torch.testing.assert_close(actual.double(), expected.double(), **tolerances)
    return receipt


def run_correctness(args, grid, samples):
    """Byte oracles and conditional arithmetic diagnostics, 64 by 256."""
    from tessera.compact_prep import parse_compact_wire, prepare_a4_wire_compact
    from tessera.kernel_a4_wire import PreparedA4Wire, decode_wire_codes
    from tessera.kernel_a4 import a4_quantize_activation
    from tessera.stock import materialize_stock, stock_dequant, _nvfp4_values
    from tessera.unit_artifact import parse_unit_artifact
    if not args.packed_reader or grid.name != "E2M1x2":
        raise ValueError("correctness needs the explicit packed E2M1 pair reader")
    torch.backends.cuda.matmul.allow_tf32 = False
    gs = torch.tensor(896.0, dtype=torch.float32, device="cuda")
    result = {"schema": "tessera.a4_packed_correctness.v2", "status": "running",
              "native_arithmetic_qualified": False, "rows": []}
    structures = (STRUCTURE_ROUTED_MOE, STRUCTURE_DENSE) if args.part == "all" else ((STRUCTURE_ROUTED_MOE,) if args.part == "routed" else (STRUCTURE_DENSE,))
    for q in args.qs:
        for structure in structures:
            recipe = served_recipe(grid, q, structure)
            units, weights = [], []
            for role in ("gate", "up"):
                encoded = encode_linear(samples[role].repeat(2, 1).to("cuda"),
                    grid=grid, q256=q, **recipe_kwargs(recipe))
                unit = prepare_a4_wire_compact(parse_compact_wire(encoded.blob, device="cuda"))
                parsed = parse_unit_artifact(encoded.blob, device="cpu")
                stock = materialize_stock(parsed.unit, parsed.forests, parsed.code)
                codes, scales = decode_wire_codes(unit)
                if not torch.equal(codes.cpu(), stock["weight_packed"]) or not torch.equal(scales.cpu(), stock["weight_scale"].view(torch.uint8)):
                    raise ValueError(f"GPU packed code or scale bytes differ: {q} {structure} {role}")
                from tessera.slicing import slice_unit
                from tessera.unit_artifact import build_unit_artifact
                shard = slice_unit(parsed, rows=(32, 64))
                _, _, shard_blob = build_unit_artifact(shard, "history", parsed.forests, q * grid.arity, parsed.code)
                shard_unit = prepare_a4_wire_compact(parse_compact_wire(shard_blob, device="cuda"))
                shard_parsed = parse_unit_artifact(shard_blob, device="cpu")
                shard_stock = materialize_stock(shard_parsed.unit, shard_parsed.forests, shard_parsed.code)
                shard_codes, shard_scales = decode_wire_codes(shard_unit)
                if not torch.equal(shard_codes.cpu(), shard_stock["weight_packed"]) or not torch.equal(shard_scales.cpu(), shard_stock["weight_scale"].view(torch.uint8)):
                    raise ValueError(f"GPU incoming history bytes differ: {q} {structure} {role}")
                units.append(unit)
                weights.append(stock_dequant(stock).cuda())
            dense = PreparedA4Wire([units[0]], gs)
            grouped = PreparedA4Wire([units[0], units[0], units[1]], gs)
            errors = []
            magnitudes = []
            gate_atols = []
            comparisons = []
            bound_template = None
            for m in (1, 16, 33, 65):
                generator = torch.Generator(device="cuda").manual_seed(3100 + m)
                x = torch.randn(m, 256, generator=generator, device="cuda").to(torch.bfloat16)
                a, scale = a4_quantize_activation(x, gs)
                nib = torch.stack((a & 15, a >> 4), dim=-1).reshape(m, 256)
                rendered_x = _nvfp4_values(nib) * scale.float().repeat_interleave(16, dim=1) / gs
                k = int(x.shape[1])
                if k != int(weights[0].shape[1]) or k != int(rendered_x.shape[1]):
                    raise ValueError(f"the gate contraction lengths differ: {k}")
                magnitude = dense_packed_fp4_operand_magnitude(rendered_x, weights[0])

                expected = rendered_x @ weights[0].T
                actual = dense(x, out_dtype=torch.float32)
                bound_template = check_packed_fp4_arithmetic(actual, expected, magnitude, k=k)
                errors.append(float((actual.double() - expected.double()).abs().max()))
                magnitudes.append(float(magnitude))
                gate_atols.append(bound_template["atol"])
                comparisons.append({"kind": "dense", "M": m, "max_abs_error": errors[-1],
                    "operand_magnitude_upper": magnitude, "atol": gate_atols[-1]})
                if m >= 4:
                    offsets = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device="cuda")
                    tokens = torch.tensor([2, 0, 2, 1, 3], dtype=torch.int32, device="cuda")
                    grouped_magnitude = max(
                        dense_packed_fp4_operand_magnitude(rendered_x[tokens[:3].long()], weights[0]),
                        dense_packed_fp4_operand_magnitude(rendered_x[tokens[3:].long()], weights[1]))

                    actual = grouped(x, expert_offsets=offsets, route_ids=tokens, num_routes=5, out_dtype=torch.float32)
                    expected = torch.cat((rendered_x[tokens[:3].long()] @ weights[0].T,
                                          rendered_x[tokens[3:].long()] @ weights[1].T))
                    bound_template = check_packed_fp4_arithmetic(actual, expected, grouped_magnitude, k=k)
                    errors.append(float((actual.double() - expected.double()).abs().max()))
                    magnitudes.append(float(grouped_magnitude))
                    gate_atols.append(bound_template["atol"])
                    comparisons.append({"kind": "grouped", "M": m, "routes": 5, "max_abs_error": errors[-1],
                        "operand_magnitude_upper": grouped_magnitude, "atol": gate_atols[-1]})
            row = {"q256": q, "structure": structure, "recipe": recipe.to_config(),
                "codes_and_scales": "byte identical to materialize_stock", "native_max_abs_error": max(errors),
                "derived_max_atol": max(gate_atols), "derived_atol_per_gate": gate_atols,
                "operand_magnitude_upper_per_gate": magnitudes,
                "arithmetic_comparisons": comparisons, "native_arithmetic_qualified": False,
                "arithmetic_bound": {name: value for name, value in bound_template.items()
                    if name not in ("atol", "operand_magnitude_upper")},
                "M": [1, 16, 33, 65], "grouped": "three experts, one empty, distinct last expert, duplicate and reordered tokens",
                "inputs": {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                    "bytes": t.numel() * t.element_size()} for name, t in units[0].named_tensors()}}
            result["rows"].append(row)
            dump(Path(args.out) / "correctness.json", result)
            print(json.dumps(row), flush=True)
    result["status"] = "diagnostic_passed"
    dump(Path(args.out) / "correctness.json", result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="E2M1x2")
    ap.add_argument("--cases", default="896")
    ap.add_argument("--part", choices=("routed", "dense", "all"), default="all")
    ap.add_argument("--ms", default="1,16,2048,4096")
    ap.add_argument("--routing", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--data-manifest", default="", help="PB staged whole-file readset")
    ap.add_argument("--prepare-inputs", action="store_true")
    ap.add_argument("--cpu-preflight", action="store_true")
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--quality-structure", choices=(STRUCTURE_ROUTED_MOE, STRUCTURE_DENSE), default=STRUCTURE_ROUTED_MOE)
    ap.add_argument("--packed-reader", action="store_true", help="Opt-in actual mixed TCQ and dense WINDOW reader")
    ap.add_argument("--correctness", action="store_true", help="Bounded byte and numerical native reader oracle")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--power-s", type=float, default=0.1)
    args = ap.parse_args()
    torch.set_num_threads(1)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.prepare_inputs:
        prepare_inputs(args.model, args.out)
        return 0
    if args.data_manifest and not args.cpu_preflight and not args.quality:
        # Consume the supported PB lease reader before any CUDA work.
        # Original shared inputs are never reopened by the GPU path.
        from pb_staged_store import StagedInputs
        inputs = StagedInputs(args.data_manifest)
        local_root = Path(args.out) / "staged-inputs"
        try:
            for path, offset in inputs.entries:
                if offset != 0:
                    raise ValueError("E2M1 adapter reads whole-file metadata inputs")
                local = local_root / path.lstrip("/")
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(inputs.read(path))
            dump(Path(args.out) / "staged-reads.json", inputs.reads)
        finally:
            inputs.close()
        for field in ("samples", "config", "routing"):
            path = getattr(args, field)
            if path:
                setattr(args, field, str(local_root / path.lstrip("/")))
    grid = grid_for_name(args.grid)
    if grid not in family_grids():
        raise ValueError("not a serializable E2M1 family")
    args.qs = [int(q.removeprefix("q")) for q in args.cases.split(",")]
    lo, hi = bounds(grid)
    if not args.qs or any(q < lo or q > hi for q in args.qs):
        raise ValueError((args.qs, lo, hi))
    args.ms_values = [int(m) for m in args.ms.split(",")]
    if not args.ms_values or any(m <= 0 for m in args.ms_values):
        raise ValueError("M must be positive")
    samples, metadata = load_samples(args.samples)
    config = json.loads(Path(args.config).read_text())
    text = config.get("text_config", config)
    if (text["hidden_size"], text["moe_intermediate_size"] // 2,
        text["n_routed_experts"], text["num_experts_per_tok"]) != (4096, 1024, 288, 8):
        raise ValueError("actual GLM config does not match the representative TP2 shapes")
    if args.cpu_preflight:
        if args.data_manifest:
            from prismabuild.client import read_data_manifest
            manifest, _encoding = read_data_manifest(args.data_manifest)
            for entry in manifest["entries"]:
                with open(entry["path"], "rb") as stream:
                    stream.seek(entry["offset"])
                    if not stream.read(min(16, entry["bytes"])):
                        raise ValueError("declared input has no readable bytes")
        tiny = []
        from tessera.compact_prep import parse_compact_wire, prepare_a4_wire_compact
        from tessera.kernel_a4_wire import decode_wire_codes
        from tessera.stock import materialize_stock
        from tessera.unit_artifact import parse_unit_artifact
        structures = (STRUCTURE_ROUTED_MOE, STRUCTURE_DENSE) if args.part == "all" else ((STRUCTURE_ROUTED_MOE,) if args.part == "routed" else (STRUCTURE_DENSE,))
        for q in args.qs:
            for structure in structures:
                recipe = served_recipe(grid, q, structure)
                encoded = encode_linear(samples["gate"], grid=grid, q256=q, **recipe_kwargs(recipe))
                wire = parse_compact_wire(encoded.blob, device="cpu", name="D38")
                if (wire.rows, wire.metadata.columns) != (32, 256):
                    raise ValueError("tiny parsed wire has wrong shape")
                row = {"q256": q, "arity": grid.arity, "structure": structure,
                       "recipe": recipe.to_config(), "exact_bytes": encoded.exact_bytes}
                if args.packed_reader and grid.arity == 2:
                    unit = prepare_a4_wire_compact(wire, device="cpu")
                    parsed = parse_unit_artifact(encoded.blob, device="cpu")
                    reference = materialize_stock(parsed.unit, parsed.forests, parsed.code)
                    codes, scales = decode_wire_codes(unit)
                    if not torch.equal(codes, reference["weight_packed"]) or not torch.equal(scales, reference["weight_scale"].view(torch.uint8)):
                        raise ValueError("CPU packed reader differs from actual stock bytes")
                    row["prepared"] = {"body_kind": unit.body_kind, "layout": unit.layout,
                        "memory": unit.memory, "window_bits": unit.window_bits,
                        "preparation_owner": "tessera.compact_prep.prepare_a4_wire_compact",
                        "inputs": {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                            "bytes": t.numel() * t.element_size()} for name, t in unit.named_tensors()}}
                    if args.correctness:
                        from tessera.stock import stock_dequant
                        magnitude = dense_packed_fp4_operand_magnitude(
                            samples["gate"][:1].float(), stock_dequant(reference))
                        _, row["arithmetic_bound_cpu_preparation"] = derive_packed_fp4_arithmetic_bound(
                            magnitude, k=int(wire.metadata.columns))
                        row["arithmetic_bound_cpu_preparation"]["operand_source"] = "unquantized CPU sample; helper preparation only, not the GPU FP4 contraction"
                        from tessera.slicing import slice_unit
                        from tessera.unit_artifact import build_unit_artifact
                        shard = slice_unit(parsed, rows=(16, 32))
                        _, _, shard_blob = build_unit_artifact(shard, "history", parsed.forests, q * grid.arity, parsed.code)
                        shard_unit = prepare_a4_wire_compact(parse_compact_wire(shard_blob, device="cpu"), device="cpu")
                        shard_parsed = parse_unit_artifact(shard_blob, device="cpu")
                        shard_stock = materialize_stock(shard_parsed.unit, shard_parsed.forests, shard_parsed.code)
                        shard_codes, shard_scales = decode_wire_codes(shard_unit)
                        if not torch.equal(shard_codes, shard_stock["weight_packed"]) or not torch.equal(shard_scales, shard_stock["weight_scale"].view(torch.uint8)):
                            raise ValueError("CPU incoming history bytes differ from actual stock bytes")
                        row["incoming_history"] = {"rows": shard_unit.rows, "cols": shard_unit.cols,
                            "nonzero_start": bool(shard_unit.initial.any()), "codes_and_scales": "byte identical to stock"}
                tiny.append(row)
        recorded = pick_recorded(args.routing, args.ms_values) if args.routing else {}
        if args.routing:
            for m in (2048, 4096):
                if m in args.ms_values and m not in recorded:
                    raise ValueError(f"missing recorded routing at M{m}")
            for m, record in recorded.items():
                ids, _ = recorded_routing(record["path"], m, "cpu")
                if int(ids.min()) < 0 or int(ids.max()) >= 288:
                    raise ValueError("recorded expert IDs outside the expert axis")
        dump(Path(args.out) / "cpu-preflight.json", {"status": "passed", "tiny_wire_reads": tiny,
             "samples": metadata, "recorded": recorded, "M": args.ms_values,
             "gpu_measurements": False})
        return 0
    if args.quality:
        quality(args, grid, samples, metadata)
        return 0
    if args.correctness:
        run_correctness(args, grid, samples)
        return 0
    run_gpu(args, grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
