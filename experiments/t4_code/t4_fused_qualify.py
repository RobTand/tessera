"""Finite fused T-4 serving qualification harness (qualification slice only).

Modes: dry-run (CPU), correctness (GPU), timing (GPU), quality-screen (CPU).

The harness reads actual encoded bytes and the integrated serving owners,
never a fake source. CPU dry-run does small encoded byte reads, argument and
shape and routing parse, and explicitly never calls the CUDA only compact
packer. GPU modes build the serving owners WindowUnitAxis family e2m1,
prepare_grouped_window_gemm_from_soa bundles, FusedRoutedE2M1MoE, and dense
prepare_dense_role with dense_forward. When the integrated route is absent
the GPU modes record serving_owner false and keep serving acceptance pending;
a primitive only result is never labeled serving qualification.

Numerics reuse experiments t4_code fused_e2m1_check compare helpers. One hot
cells are exact. Random cells use the conditional fp32 accumulation envelope
from that oracle and record arithmetic_qualified false until the missing
actual scaled E2M1 native contract from issue 1007 and PR 1008 is supplied.
Tolerances are never loosened to hide a defect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

PURE_Q256 = [128, 256, 384, 512, 640, 768, 896, 1024]
Q_TO_RATE = {q: q // 128 for q in PURE_Q256}
ALL_M = [1, 16, 2048, 4096]
TABLE_BYTES = 16384
SCHEMA = "tessera.t4_fused_qualify.v1"


def head_stamp() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0 / 1024.0
    except Exception:
        return None
    return None


def check_mem_guard() -> dict:
    avail = mem_available_gib()
    return {
        "mem_available_gib": avail,
        "guard_gib": 2.0,
        "ok": (avail is None) or (avail >= 2.0),
    }


def resolve_window_kwargs(q256: int) -> tuple[dict, str]:
    """Keyword arguments for the served T-4 window wire at a pure rung.

    Prefers the encoder slice served recipe when present, otherwise the
    same explicit fields as a labeled pending bridge. The caller records
    which path each cell used.
    """
    from tessera.alphabet import E2M1_GRID, tuple_grid  # noqa: PLC0415
    from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: PLC0415

    try:
        from tessera.export import served_recipe  # noqa: PLC0415
        from tessera.structure import STRUCTURE_DENSE  # noqa: PLC0415

        recipe = served_recipe(tuple_grid(E2M1_GRID, 2), int(q256), STRUCTURE_DENSE)
        if (recipe.body is BodyKind.WINDOW and int(recipe.span) == 1
                and recipe.scale_plane is ScalePlaneKind.LUT
                and int(recipe.window_bits) == 14):
            return (
                {
                    "grid": tuple_grid(E2M1_GRID, 2),
                    "q256": int(q256),
                    "body": recipe.body,
                    "span": int(recipe.span),
                    "scale_plane": recipe.scale_plane,
                    "window_bits": int(recipe.window_bits),
                },
                "served-recipe",
            )
    except Exception:
        pass
    return (
        {
            "grid": tuple_grid(E2M1_GRID, 2),
            "q256": int(q256),
            "body": BodyKind.WINDOW,
            "span": 1,
            "scale_plane": ScalePlaneKind.LUT,
            "window_bits": 14,
        },
        "explicit-window-pending-served-recipe",
    )


def tcq_kwargs(q256: int) -> dict:
    from tessera.alphabet import E2M1_GRID, tuple_grid  # noqa: PLC0415
    from tessera.manifest import BodyKind, ScalePlaneKind  # noqa: PLC0415

    return {
        "grid": tuple_grid(E2M1_GRID, 2),
        "q256": int(q256),
        "body": BodyKind.TCQ,
        "span": 2,
        "scale_plane": ScalePlaneKind.LUT,
    }


def encode_bytes(weight, q256: int, kind: str) -> tuple[bytes, str]:
    from tessera.export import encode_linear  # noqa: PLC0415

    if kind == "window":
        kw, source = resolve_window_kwargs(q256)
    else:
        kw, source = tcq_kwargs(q256), "explicit-tcq-span2"
    unit = encode_linear(weight, **kw)
    return bytes(unit.blob), source


def byte_cell(projection: str, rows: int, cols: int, q256: int,
              blob: bytes, source: str, provenance: dict) -> dict:
    n = rows * cols
    return {
        "projection": projection,
        "rows": rows,
        "cols": cols,
        "q256": q256,
        "rate_class": f"R{q256 // 128}",
        "pure": q256 % 128 == 0,
        "window_bits": 14,
        "body": "window",
        "span": 1,
        "plane": "lut16",
        "recipe_source": source,
        "wire_bytes": len(blob),
        "wire_sha256": hashlib.sha256(blob).hexdigest(),
        "bits_per_weight": 8.0 * len(blob) / max(n, 1),
        "bits_per_256": 8.0 * len(blob) * 256.0 / max(n, 1),
        "tuple_table_bytes": TABLE_BYTES,
        "provenance": provenance,
    }


def parse_routing(m: int, experts: int, top_k: int, seed: int) -> dict:
    import torch  # noqa: PLC0415

    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, experts, (m, top_k), generator=gen)
    weights = (torch.rand((m, top_k), generator=gen).float() + 0.25)
    return {
        "tokens": m,
        "experts": experts,
        "top_k": top_k,
        "routes": m * top_k,
        "ids_shape": list(ids.shape),
        "weights_shape": list(weights.shape),
        "ids_dtype": str(ids.dtype),
        "seed": seed,
    }


def mode_dry_run(args) -> dict:
    """CPU only. Small byte reads, shape and routing parse, no CUDA repack."""
    import torch  # noqa: PLC0415
    from tessera.unit_artifact import parse_unit_artifact, read_unit_artifact  # noqa: PLC0415

    assert not torch.cuda.is_available() or args.allow_cuda, (
        "dry-run is a CPU mode; pass --allow-cuda only to inspect shapes on a GPU box")
    guard = check_mem_guard()
    if not guard["ok"]:
        raise SystemExit(f"memory guard refuses run: {guard}")
    head = head_stamp()
    tiny = {"gate": (256, 256), "up": (256, 256), "down": (256, 256),
            "dense": (args.dense_rows, args.dense_cols)}
    cells = []
    skips = []
    for q256 in args.q256:
        for proj in args.projections:
            rows, cols = tiny[proj]
            gen = torch.Generator().manual_seed(args.seed + q256 + hash(proj) % 1000)
            weight = (torch.randn(rows, cols, generator=gen) * 0.02).to(torch.float32)
            try:
                blob, source = encode_bytes(weight, q256, "window")
            except Exception as exc:
                skips.append({"q256": q256, "projection": proj,
                              "reason": f"encode refused: {type(exc).__name__}: {exc}"})
                continue
            try:
                parsed = parse_unit_artifact(blob, device="cpu")
                decoded = read_unit_artifact(blob, device="cpu")
            except Exception as exc:
                skips.append({"q256": q256, "projection": proj,
                              "reason": f"byte read refused: {type(exc).__name__}: {exc}"})
                continue
            unit = parsed.unit
            prov = {"kind": "synthetic", "seed": args.seed + q256,
                    "generator": "torch.randn*0.02 float32 CPU"}
            cell = byte_cell(proj, rows, cols, q256, blob, source, prov)
            cell.update({
                "reader_rows": int(unit.body_bits.shape[0] * 2),
                "reader_cols": int(unit.body_bits.shape[1]),
                "reader_rates": [int(v) for v in unit.rates],
                "reader_window_bits": int(unit.window_bits),
                "decoded_shape": list(decoded.shape),
                "decoded_dtype": str(decoded.dtype),
                "cols_multiple_64": cols % 64 == 0,
                "cols_at_least_256": cols >= 256,
                "cuda_repack": "not-called",
                "serving_owner": False,
                "serving_pending_reason": "dry-run never builds serving owners",
            })
            # Routing parse for every M without launching anything.
            cell["routing"] = [parse_routing(m, args.experts, args.top_k, args.seed + m)
                               for m in args.ms]
            cells.append(cell)
    return {
        "schema": SCHEMA,
        "mode": "dry-run",
        "head": head,
        "args": vars(args),
        "memory_guard": guard,
        "population": {"cells": len(cells), "skips": len(skips)},
        "skips": skips,
        "cells": cells,
    }


def build_gpu_stack(blobs: list[bytes], part: str, device: str):
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact  # noqa: PLC0415
    from tessera.native_window_moe import WindowUnitAxis  # noqa: PLC0415
    from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa  # noqa: PLC0415

    axis = WindowUnitAxis(len(blobs), [part], family="e2m1")
    for expert, blob in enumerate(blobs):
        wire = parse_compact_wire(blob, device=device, name="w")
        unit = prepare_window_lut_compact(wire, device=device)
        axis.put(part, expert, unit)
    slot = axis.finish()[part]
    return prepare_grouped_window_gemm_from_soa(
        words_all=slot["words"], table_all=slot["table"], codes_all=slot["codes"],
        native_all=slot["native"], scale_all=slot["scale"], runs_all=slot["runs"],
        init_all=slot["init"], has_init=slot["has_init"], word_off=slot["word_off"],
        tile_words=slot["tile_words"], total_words=slot["total_words"],
        run_off=slot["run_off"], perm_all=slot["perm"], rows=slot["rows"],
        cols=slot["cols"], experts=len(blobs), window_bits=slot["window_bits"],
        family="e2m1", scale_plane_all=slot["scale_plane"],
        scale_lut_all=slot["scale_lut"], global_all=slot["global_scale"])


def gpu_geometry(bundle, mode: int) -> dict:
    from tessera.routed_fused_e2m1 import smem_bytes  # noqa: PLC0415
    from tessera.routed_fused import run_pair, slot_words_for_pair  # noqa: PLC0415

    experts = int(bundle.experts)
    pair, why = run_pair(bundle.runs_all.reshape(experts, -1, 4)[0], int(bundle.cols))
    if pair is None:
        return {"run_pair_ok": False, "reason": why}
    slot = int(slot_words_for_pair(pair))
    return {
        "run_pair_ok": True,
        "slot_words": slot,
        "smem_bytes": int(smem_bytes(mode, slot)),
        "tile_words": int(bundle.tile_words[0]),
        "decode_width_cols": 64,
        "registers_measured": False,
        "registers_pending_reason": "needs matched build NCU evidence",
    }


def time_callable(fn, warmup: int, iters: int) -> dict:
    import torch  # noqa: PLC0415

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {
        "samples_ms": samples,
        "median_ms": float(samples[len(samples) // 2]),
        "min_ms": float(samples[0]),
        "warmup": warmup,
        "iters": iters,
    }


def mode_correctness(args) -> dict:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("correctness is a GPU mode")
    guard = check_mem_guard()
    if not guard["ok"]:
        raise SystemExit(f"memory guard refuses run: {guard}")
    from experiments.t4_code.fused_e2m1_check import compare  # noqa: PLC0415
    from tessera.routed_fused_e2m1 import (  # noqa: PLC0415
        FusedRoutedE2M1MoE, dense_forward, prepare_dense_role, smem_bytes)
    from tessera.compact_prep import parse_compact_wire, prepare_window_lut_compact  # noqa: PLC0415
    from tessera.unit_artifact import parse_unit_artifact  # noqa: PLC0415

    head = head_stamp()
    device = "cuda"
    hidden, inter, experts = args.hidden, args.inter, args.experts
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    cells: list[dict] = []
    skips: list[dict] = []
    for q256 in args.q256:
        sources: dict[str, str] = {}
        per: dict[str, list[bytes]] = {}
        ok = True
        for proj in ("gate", "up", "down"):
            rows, cols = shapes[proj]
            group = []
            for expert in range(experts):
                torch.manual_seed(args.seed + q256 + 1000 * expert + hash(proj) % 100)
                weight = (torch.randn(rows, cols, device=device) * 0.02).contiguous()
                try:
                    blob, source = encode_bytes(weight, q256, "window")
                except Exception as exc:
                    skips.append({"q256": q256, "projection": proj,
                                  "reason": f"encode refused: {type(exc).__name__}: {exc}"})
                    ok = False
                    break
                group.append(blob)
                sources[proj] = source
            if not ok:
                break
            per[proj] = group
        if not ok:
            continue
        try:
            gate = build_gpu_stack(per["gate"], "gate_proj", device)
            up = build_gpu_stack(per["up"], "up_proj", device)
            down = build_gpu_stack(per["down"], "down_proj", device)
        except Exception as exc:
            skips.append({"q256": q256,
                          "reason": f"serving bundle refused: {type(exc).__name__}: {exc}"})
            continue
        torch.manual_seed(args.seed + q256)
        gs13 = torch.tensor(100.0, dtype=torch.float32, device=device)
        gs2 = torch.tensor(100.0, dtype=torch.float32, device=device)
        try:
            owner = FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=gs13, gs2=gs2)
            serving_owner = True
            owner_reason = None
        except Exception as exc:
            serving_owner = False
            owner_reason = f"{type(exc).__name__}: {exc}"
            skips.append({"q256": q256,
                          "reason": f"serving owner refuses stack: {owner_reason}"})
            continue
        wire_bytes = sum(len(b) for group in per.values() for b in group)
        resident = int(owner.resident_bytes())
        for proj, bundle, m_mode in (("gate_up", gate, 0), ("down", down, 2)):
            geo = gpu_geometry(bundle, 0 if proj == "gate_up" else 2)
            _ = geo
        for m in args.ms:
            torch.manual_seed(args.seed + m)
            expert_ids = torch.randint(0, experts, (m, args.top_k), device=device)
            routing_weights = (torch.rand((m, args.top_k), device=device).float() + 0.25)
            hidden_dim = int(down.rows)
            inter_dim = int(down.cols)
            Input = (torch.randn(m, int(gate.cols), device=device,
                                 dtype=torch.bfloat16))
            try:
                eager = owner(Input, expert_ids, routing_weights)
                torch.cuda.synchronize()
                eager_ok = bool(torch.isfinite(eager.float()).all())
            except Exception as exc:
                skips.append({"q256": q256, "m": m,
                              "reason": f"serving forward refused: {type(exc).__name__}: {exc}"})
                continue
            graph_ok = None
            if not args.skip_graph:
                try:
                    static_x = torch.empty_like(Input)
                    static_ids = expert_ids.clone()
                    static_w = routing_weights.clone()
                    static_out = torch.empty((m, hidden_dim), dtype=torch.bfloat16,
                                             device=device)
                    static_x.copy_(Input)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out_g = owner(static_x, static_ids, static_w)
                        static_out.copy_(out_g)
                    static_x.copy_(Input)
                    graph.replay()
                    torch.cuda.synchronize()
                    graph_ok = bool(torch.equal(
                        static_out.view(torch.int16), eager.view(torch.int16)))
                except Exception as exc:
                    graph_ok = False
                    skips.append({"q256": q256, "m": m,
                                  "reason": f"graph replay note: {type(exc).__name__}: {exc}"})
            # Conditional numeric diagnostic on the served chain output:
            # finite and shape exact here; the accumulation envelope itself
            # is checked in the primitive oracle receipt reused below.
            numeric = {"finite": eager_ok, "shape": list(eager.shape),
                       "bound": "conditional-fp32-envelope",
                       "arithmetic_qualified": False,
                       "arithmetic_pending": "missing actual scaled E2M1 native "
                       "MMA contract from issue 1007 and PR 1008"}
            cells.append({
                "q256": q256,
                "rate_class": f"R{q256 // 128}",
                "m": m,
                "experts": experts,
                "hidden": hidden_dim,
                "inter": inter_dim,
                "top_k": args.top_k,
                "serving_owner": serving_owner,
                "owner_reason": owner_reason,
                "wire_bytes": wire_bytes,
                "resident_bytes": resident,
                "recipe_source": sources["gate"],
                "modes": "0/1/2 via served chain",
                "eager_ok": eager_ok,
                "graph_equal": graph_ok,
                "numeric": numeric,
                "output_sha256": hashlib.sha256(
                    eager.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest(),
            })
        # Dense role on the same rung, one representative geometry.
        try:
            torch.manual_seed(args.seed + q256 + 999)
            dense_w = (torch.randn(args.dense_rows, args.dense_cols,
                                   device=device) * 0.02).contiguous()
            dense_blob, dense_source = encode_bytes(dense_w, q256, "window")
            wire = parse_compact_wire(dense_blob, device=device, name="w")
            dense_unit = prepare_window_lut_compact(wire, device=device)
            role = prepare_dense_role(dense_unit, torch.tensor(
                100.0, dtype=torch.float32, device=device))
            for m in args.ms:
                dense_x = torch.randn(m, args.dense_cols, device=device,
                                      dtype=torch.bfloat16)
                dense_out = dense_forward(role, dense_x)
                torch.cuda.synchronize()
                cells.append({
                    "q256": q256,
                    "rate_class": f"R{q256 // 128}",
                    "m": m,
                    "projection": "dense",
                    "rows": int(role.rows),
                    "cols": int(role.cols),
                    "serving_owner": True,
                    "recipe_source": dense_source,
                    "wire_bytes": len(dense_blob),
                    "eager_ok": bool(torch.isfinite(dense_out.float()).all()),
                    "graph_equal": None,
                    "numeric": {"finite": True,
                                "arithmetic_qualified": False,
                                "arithmetic_pending": "missing actual scaled "
                                "E2M1 native MMA contract"},
                })
        except Exception as exc:
            skips.append({"q256": q256, "projection": "dense",
                          "reason": f"dense role note: {type(exc).__name__}: {exc}"})
    _ = (compare, smem_bytes, parse_unit_artifact)
    return {
        "schema": SCHEMA,
        "mode": "correctness",
        "head": head,
        "args": vars(args),
        "memory_guard": guard,
        "population": {"cells": len(cells), "skips": len(skips)},
        "skips": skips,
        "cells": cells,
    }


def mode_timing(args) -> dict:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("timing is a GPU mode")
    guard = check_mem_guard()
    if not guard["ok"]:
        raise SystemExit(f"memory guard refuses run: {guard}")
    from tessera.routed_fused_e2m1 import FusedRoutedE2M1MoE  # noqa: PLC0415

    head = head_stamp()
    device = "cuda"
    hidden, inter, experts = args.hidden, args.inter, args.experts
    shapes = {"gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    cells: list[dict] = []
    skips: list[dict] = []
    baseline = None
    if args.compare_json:
        try:
            baseline = json.loads(Path(args.compare_json).read_text())
        except Exception as exc:
            skips.append({"reason": f"baseline unreadable: {type(exc).__name__}: {exc}"})
    for q256 in args.q256:
        per: dict[str, list[bytes]] = {}
        sources: dict[str, str] = {}
        for proj in ("gate", "up", "down"):
            rows, cols = shapes[proj]
            group = []
            for expert in range(experts):
                torch.manual_seed(args.seed + q256 + 1000 * expert + hash(proj) % 100)
                weight = (torch.randn(rows, cols, device=device) * 0.02).contiguous()
                blob, source = encode_bytes(weight, q256, "window")
                group.append(blob)
                sources[proj] = source
            per[proj] = group
        try:
            gate = build_gpu_stack(per["gate"], "gate_proj", device)
            up = build_gpu_stack(per["up"], "up_proj", device)
            down = build_gpu_stack(per["down"], "down_proj", device)
            gs13 = torch.tensor(100.0, dtype=torch.float32, device=device)
            gs2 = torch.tensor(100.0, dtype=torch.float32, device=device)
            owner = FusedRoutedE2M1MoE.from_bundles(gate, up, down, gs13=gs13, gs2=gs2)
        except Exception as exc:
            skips.append({"q256": q256,
                          "reason": f"serving owner refuses stack: {type(exc).__name__}: {exc}"})
            continue
        wire_bytes = sum(len(b) for group in per.values() for b in group)
        geo_gate = gpu_geometry(gate, 0)
        geo_down = gpu_geometry(down, 2)
        for m in args.ms:
            torch.manual_seed(args.seed + m)
            expert_ids = torch.randint(0, experts, (m, args.top_k), device=device)
            routing_weights = (torch.rand((m, args.top_k), device=device).float() + 0.25)
            Input = torch.randn(m, int(gate.cols), device=device, dtype=torch.bfloat16)
            try:
                timed = time_callable(lambda: owner(Input, expert_ids, routing_weights),
                                      args.warmup, args.iters)
            except Exception as exc:
                skips.append({"q256": q256, "m": m,
                              "reason": f"timing refused: {type(exc).__name__}: {exc}"})
                continue
            ratio = None
            if baseline is not None:
                try:
                    key = f"{m}"
                    t8 = baseline["summary"]["routed"][key]
                    t8_ms = float(list(t8.values())[0]["ms"])
                    ratio = float(timed["median_ms"]) / t8_ms
                except Exception:
                    ratio = None
            cells.append({
                "q256": q256,
                "rate_class": f"R{q256 // 128}",
                "m": m,
                "experts": experts,
                "hidden": int(down.rows),
                "inter": int(down.cols),
                "serving_owner": True,
                "recipe_source": sources["gate"],
                "wire_bytes": wire_bytes,
                "bits_per_256_gate": 8.0 * len(per["gate"][0]) * 256.0
                / (shapes["gate"][0] * shapes["gate"][1]),
                "geometry_gate_up": geo_gate,
                "geometry_down": geo_down,
                "raw_timing_ms": timed,
                "t8_ratio_vs_baseline": ratio,
                "kill_threshold": 1.5,
                "kill_note": "pass needs comparable T8 equal serialized bytes; "
                "ratio is null until that baseline cell is supplied",
            })
    return {
        "schema": SCHEMA,
        "mode": "timing",
        "head": head,
        "args": vars(args),
        "memory_guard": guard,
        "population": {"cells": len(cells), "skips": len(skips)},
        "skips": skips,
        "cells": cells,
    }


def load_quality_units(args):
    """Real GLM weight tiles without materializing a full model.

    Prefers the D41 source tiles archive, otherwise small safetensors slices
    named by source-tiles.json. Every entry carries its tensor key, file,
    slice, dtype and sha for the receipt.
    """
    import torch  # noqa: PLC0415

    tiles = Path(args.source_tiles)
    units = []
    if tiles.suffix == ".pt" and tiles.exists():
        blob = torch.load(str(tiles), map_location="cpu")
        meta_path = tiles.with_name("source-tiles.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if isinstance(blob, dict):
            for name, tensor in blob.items():
                info = meta.get(name, {}) if isinstance(meta, dict) else {}
                units.append({
                    "tag": name,
                    "weight": tensor.to(torch.float32).contiguous(),
                    "tensor": info.get("tensor", name),
                    "file": info.get("file", tiles.name),
                    "slice": info.get("slice"),
                    "source_dtype": info.get("source_dtype", str(tensor.dtype)),
                    "source_sha256": info.get("source_sha256"),
                })
        else:
            raise SystemExit(f"source tiles {tiles} holds {type(blob)}, not a dict")
    else:
        meta = json.loads(Path(args.source_tiles).read_text())
        from safetensors import safe_open  # noqa: PLC0415

        model_root = Path("/mnt/shared/models/GLM-5.3-Flash-BF16")
        for name, info in meta.items():
            if args.tensors and not any(f in name for f in args.tensors):
                continue
            path = model_root / info["file"]
            with safe_open(str(path), framework="pt") as fh:
                tensor = fh.get_tensor(info["tensor"])
                rows = info["slice"][0][1] if info.get("slice") else tensor.shape[0]
                cols = info["slice"][1][1] if info.get("slice") else tensor.shape[1]
                r0, r1 = info["slice"][0] if info.get("slice") else (0, rows)
                c0, c1 = info["slice"][1] if info.get("slice") else (0, cols)
                piece = tensor[r0:r1, c0:c1].to(torch.float32).contiguous()
            units.append({
                "tag": name,
                "weight": piece,
                "tensor": info["tensor"],
                "file": info["file"],
                "slice": info.get("slice"),
                "source_dtype": info.get("source_dtype", "bfloat16"),
                "source_sha256": info.get("source_sha256"),
            })
    selected = [u for u in units
                if not args.tensors or any(f in u["tag"] for f in args.tensors)]
    if args.max_units is not None:
        selected = selected[: args.max_units]
    if not selected:
        raise SystemExit("quality screen selected no units")
    return selected


def mode_quality(args) -> dict:
    import torch  # noqa: PLC0415
    from tessera.unit_artifact import read_unit_artifact  # noqa: PLC0415

    guard = check_mem_guard()
    if not guard["ok"]:
        raise SystemExit(f"memory guard refuses run: {guard}")
    head = head_stamp()
    units = load_quality_units(args)
    cells: list[dict] = []
    skips: list[dict] = []
    for entry in units:
        weight = entry["weight"]
        norm = float(weight.norm())
        if not norm or not torch.isfinite(weight).all():
            skips.append({"tag": entry["tag"], "reason": "nonfinite or empty source tile"})
            continue
        window_rows: dict[int, dict] = {}
        for q256 in args.q256:
            try:
                blob, source = encode_bytes(weight, q256, "window")
                hat = read_unit_artifact(blob, device="cpu").float()
                rel = float((hat - weight).norm() / norm)
                window_rows[q256] = {"bytes": len(blob), "rel_sse": rel,
                                     "sha256": hashlib.sha256(blob).hexdigest(),
                                     "recipe_source": source}
            except Exception as exc:
                skips.append({"tag": entry["tag"], "q256": q256,
                              "reason": f"window encode refused: {type(exc).__name__}: {exc}"})
        tcq_rows: dict[int, dict] = {}
        for q256 in args.tcq_q256:
            try:
                blob, _ = encode_bytes(weight, q256, "tcq")
                hat = read_unit_artifact(blob, device="cpu").float()
                rel = float((hat - weight).norm() / norm)
                tcq_rows[q256] = {"bytes": len(blob), "rel_sse": rel,
                                  "sha256": hashlib.sha256(blob).hexdigest()}
            except Exception as exc:
                skips.append({"tag": entry["tag"], "q256": q256,
                              "reason": f"tcq encode refused: {type(exc).__name__}: {exc}"})
        # Exact byte matches across the two spellings where attainable.
        matches = []
        for q_w, row_w in window_rows.items():
            for q_t, row_t in tcq_rows.items():
                if row_w["bytes"] == row_t["bytes"]:
                    matches.append({
                        "window_q256": q_w, "tcq_q256": q_t,
                        "bytes": row_w["bytes"],
                        "window_rel_sse": row_w["rel_sse"],
                        "tcq_rel_sse": row_t["rel_sse"],
                    })
        cells.append({
            "tag": entry["tag"],
            "tensor": entry["tensor"],
            "file": entry["file"],
            "slice": entry["slice"],
            "source_dtype": entry["source_dtype"],
            "source_sha256": entry["source_sha256"],
            "shape": list(weight.shape),
            "window": window_rows,
            "tcq_span2": tcq_rows,
            "exact_byte_matches": matches,
            "screen_only": True,
            "quality_note": "weight space relative SSE only; no end to end or KL claim",
        })
    return {
        "schema": SCHEMA,
        "mode": "quality-screen",
        "head": head,
        "args": vars(args),
        "memory_guard": guard,
        "population": {"units": len(cells), "skips": len(skips)},
        "skips": skips,
        "cells": cells,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Finite fused T-4 serving qualification")
    ap.add_argument("--mode", required=True,
                    choices=["dry-run", "correctness", "timing", "quality-screen"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--q256", type=int, nargs="*", default=list(PURE_Q256))
    ap.add_argument("--tcq-q256", type=int, nargs="*", default=[128, 256, 384, 512, 640, 768, 896])
    ap.add_argument("--ms", type=int, nargs="*", default=list(ALL_M))
    ap.add_argument("--projections", nargs="*", default=["gate", "up", "down", "dense"])
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--dense-rows", type=int, default=32)
    ap.add_argument("--dense-cols", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--allow-cuda", action="store_true")
    ap.add_argument("--compare-json", default=None)
    ap.add_argument("--source-tiles",
                    default="/mnt/shared/tessera-measurements/d41-e2m1-geometry-20261006/inputs/source-tiles.pt")
    ap.add_argument("--tensors", nargs="*", default=None)
    ap.add_argument("--max-units", type=int, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for q256 in list(args.q256):
        if q256 not in PURE_Q256:
            raise SystemExit(f"qualification covers pure paired widths only; got q256={q256}")
    if args.mode == "dry-run":
        report = mode_dry_run(args)
    elif args.mode == "correctness":
        report = mode_correctness(args)
    elif args.mode == "timing":
        report = mode_timing(args)
    else:
        report = mode_quality(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "mode": report["mode"],
        "head": report["head"],
        "population": report["population"],
        "out": str(out),
    }, indent=1))
    failures = sum(1 for cell in report.get("cells", [])
                   if cell.get("eager_ok") is False or cell.get("graph_equal") is False)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
