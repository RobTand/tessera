"""Build exact cases and coverage from the reader and the frozen input lists."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

from experiments.numerics_rows.common import check_runtime, check_source, sha256, write_json
from experiments.numerics_rows.quantizers import PHASES
from experiments.numerics_rows.reference import bf16, bf16_array, exact_input, f32_bits, f32_value, multiply, output

D2 = (1, 2, 8, 64, 65, 128, 512, 2048)
COUNTS = {1: (1,), 2: (1, 2), 8: (1, 5, 8), 64: (1, 37, 64),
          65: (1, 64, 65), 128: (1, 64, 65, 128), 512: (1, 64, 65, 128, 129, 512)}


def configurations(rows, cols, library, sms, structure, *, tile_words=None, triton_shape=None):
    from tessera.routed_fused import dense_k_split, superblock_rows

    selected, seen = set(D2), set()
    records = {}
    for m in range(1, 2049):
        width = 64 if library in ("e2m1", "triton") else superblock_rows(library, 0, m)
        split = 1 if structure == "routed_moe" or library in ("e2m1", "triton") else dense_k_split(m, rows, cols, sms, tile_words=tile_words)
        config = {"k_split": split, "last_tile_partial": m % 64 != 0,
                  "superblock_width": width, "chunk_size": 64 if library in ("e2m1", "triton") else 32}
        if library == "triton":
            config["triton_shape"] = list(triton_shape(m)) if triton_shape else [64, 64, 64]
        key = json.dumps(config, sort_keys=True)
        if m >= 2 and key not in seen:
            selected.add(m)
            seen.add(key)
        records[m] = config
    return [{"M": m, **records[m], "route_counts": list(route_counts(m, records[m]["superblock_width"]))
             if structure == "routed_moe" else []} for m in sorted(selected)]


def route_counts(m, width):
    if m == 2048:
        return 1, width - 1, width, width + 1, 2 * width, m
    return COUNTS.get(m, tuple(sorted({1, min(m, width), min(m, width + 1), m})))


def routing(m, experts, count, selected, top_k=8, seed=0):
    """Give the selected expert exactly count routes, with distinct IDs per token."""
    selected = [selected] if isinstance(selected, int) else list(selected)
    if not 0 < count <= m or not selected or len(set(selected)) != len(selected) or any(not 0 <= e < experts for e in selected) or len(selected) > top_k or top_k >= experts:
        raise ValueError("routing needs 0 < count <= M and top_k < experts")
    rng = random.Random(seed)
    tokens = {e: set(rng.sample(range(m), count)) for e in selected}
    choices = [e for e in range(experts) if e not in selected]
    ids, weights = [], []
    for token in range(m):
        row = [e for e in selected if token in tokens[e]]
        row += rng.sample(choices, top_k - len(row))
        rng.shuffle(row)
        raw = [rng.uniform(0.1, 1.1) for _ in row]
        ids.append(row)
        weights.append([x / sum(raw) for x in raw])
    return ids, weights


def chunk_pairs(cols, chunk):
    return [(chunk * c + c % chunk, chunk * (c + 1) + (7 * (c + 1)) % chunk)
            for c in range(cols // chunk - 1)]


def split_columns(cols, chunk, splits):
    chunks = cols // chunk
    return [chunk * (i * chunks // splits) for i in range(splits)]


def sweep_values(family, *, gs=1.0, normal_j=None, small_product=False, gate_max=None, limit=10.0):
    if family == "e2m1" and normal_j is None:
        raise ValueError("E2M1 sweep requires Q1's normal, unsaturated exponent")
    base = 6 * 2.0**normal_j / gs if family == "e2m1" else (2.0**64 if small_product else 1.0)
    values = [bf16(sign * base * phase) for phase in PHASES for sign in (1, -1)]
    if gate_max:
        while base * abs(gate_max) >= limit:
            base /= 2
            if bf16(base) == 0:
                break
            values.extend((bf16(base), bf16(-base)))
    return list(dict.fromkeys(values))


def alphabet(family, lut=()):
    if family == "value":
        return sorted({f32_value(i << 16) for i in range(65536) if math.isfinite(f32_value(i << 16))})
    if family == "e4m3":
        from experiments.t4_code.fp4_arithmetic_attest import ue4m3
        return sorted({sign * float(ue4m3(i)) for i in range(127) for sign in (1, -1)})
    e2 = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    return sorted({sign * float(v) * float(scale) for v in e2 for scale in lut for sign in (1, -1)})


def separation(weight, values, evaluate, *, stage="dense"):
    """Evaluate adjacent values through the served stage, including its clamp."""
    import bisect

    pos = bisect.bisect_left(values, weight)
    if pos == len(values) or values[pos] != weight:
        raise ValueError("decoded weight is outside the value set")
    lower, upper = pos == 0, pos == len(values) - 1
    by_sign = {"positive": [False, False], "negative": [False, False]}
    cases = []
    for case, activation in evaluate.inputs:
        actual = evaluate(weight, activation)
        if not math.isfinite(actual):
            raise ValueError("reference is not finite")
        left = pos == 0 or evaluate(values[pos - 1], activation) != actual
        right = pos == len(values) - 1 or evaluate(values[pos + 1], activation) != actual
        if stage == "gate" and actual != 0:
            gf, uf, limit = evaluate.gate_state(weight, activation)
            if not (0 < gf < limit and actual * uf > 0):
                left = right = False
        lower, upper = lower or left, upper or right
        sign = "positive" if activation > 0 else "negative"
        by_sign[sign][0] |= left
        by_sign[sign][1] |= right
        cases.append(case)
        if lower and upper:
            break
    return {"lower_excluded": lower, "upper_excluded": upper, "both": lower and upper, "by_sign": by_sign, "cases": cases}


class DTable:
    def __init__(self, record):
        import numpy as np

        for field, digest in (("dom_path", "dom_sha256"), ("d_table_path", "d_table_sha256")):
            if sha256(record[field]) != record[digest]:
                raise ValueError("changed D table")
        domain = np.fromfile(record["dom_path"], dtype="<f4")
        values = np.fromfile(record["d_table_path"], dtype="<f4")
        self.table = {f32_bits(float(x)): float(y) for x, y in zip(domain, values)}
        self.limit = f32_value(record["L_bits"])

    def __call__(self, value):
        return self.table[f32_bits(value)]


def swiglu(g, u, table, limit=10.0):
    gf, uf = min(bf16(g), limit), max(min(bf16(u), limit), -limit)
    return bf16(multiply(table(gf), uf))


def decode_roles(blob, scheme, target, *, device="cpu", plan=None):
    """Keep raw dot operands separate from the reader's epilogue scales."""
    import torch
    from tessera.serving.scheme import parse_tessera_blob_for_scheme
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.decode import materialize_bf16, materialize_fp8
    from experiments.t4_code.fused_e2m1_check import ref_weight
    from tessera.fused import parse_fused

    parsed = parse_tessera_blob_for_scheme(blob, scheme, target, device=device)
    if plan is not None:
        parsed = shard_parsed_roles(parsed, plan)
    result = []
    for name, item in parsed:
        if scheme["family"] == "TESSERA_NVFP4":
            member = next(m for m in parse_fused(blob) if m.name == name)
            cut = None if plan is None or plan.is_whole else (plan.role(name).lo, plan.role(name).hi)
            axis = "cols" if plan is not None and plan.axis == "column" else "rows"
            weight = ref_weight(member.blob, cut, device, axis)
            scale = torch.tensor([item.unit.scale_global], dtype=torch.float32, device=device)
        else:
            decoder = materialize_fp8 if scheme["family"] == "TESSERA_FP8" else materialize_bf16
            weight, scale = decoder(item.unit, item.forests, item.code)
            if scheme["family"] == "TESSERA_FP8":
                weight = weight.view(torch.float8_e4m3fn)
            weight = weight.double()
        result.append((name, weight, scale, item))
    return result


def wire_stamp(parsed):
    from tessera.manifest import BodyKind, ScalePlaneKind

    unit = parsed.unit
    manifest = parsed.manifest
    return {"q256": int(manifest.branch.root_q256) // int(parsed.grid.arity),
            "body": parsed.body.name.lower(), "span": int(unit.span),
            "plane": "lut16" if manifest.scale_plane.kind == ScalePlaneKind.LUT else "channel",
            "window_bits": int(unit.window_bits), "seed": int(unit.window_seed),
            "sigma": unit.window_sigma, "channel_sigma": unit.channel_sigma}


def admitted_units(contract, image_digest, family, structure, rung, pair, m):
    cells = contract["lane_eligibility"]["cells"]
    result = []
    for cell in cells:
        runtime = cell.get("runtime", {})
        if not runtime.get("image", "").endswith(image_digest) or cell["family"] != family or cell["structure"] != structure:
            continue
        if cell["regime"] != ("decode" if m == 1 else "batch"):
            continue
        for executes in cell["executes"]:
            if (executes["symbol"], executes["decoder"]) != tuple(pair):
                continue
            rungs = executes.get("rungs_q256", cell["rungs_q256"])
            result.append({"cell": cell["id"], "admitted_by_v68": rung in rungs,
                           "role": "unit" if rung in rungs else "observation"})
    return result




class QuantizerTable:
    def __init__(self, q1, family, gs=1.0):
        import numpy as np

        record, = [r for r in q1["quantizers"] if r["family"] == family and r["gs"] == gs]
        if sha256(record["path"]) != record["sha256"] or not record["repeat_equal"]:
            raise ValueError("Q1 bytes changed or did not repeat")
        arrays = np.load(record["path"])
        bits = arrays["input_bits"][:, 0].astype(np.uint16)
        codes = arrays["code_bytes"].reshape(len(bits), -1)[:, 0]
        if family == "e4m3":
            from experiments.t4_code.fp4_arithmetic_attest import ue4m3
            values = [(-1 if code & 128 else 1) * float(ue4m3(int(code) & 127)) for code in codes]
            scales = arrays["scale_bytes"].copy().reshape(len(bits), -1)[:, :4].copy().view("<f4").reshape(-1)
        elif family == "e2m1":
            from experiments.t4_code.fp4_arithmetic_attest import e2m1, ue4m3
            sf = arrays["scale_bytes"].reshape(len(bits), -1)[:, 0]
            values = [float(e2m1(int(code) & 15) * ue4m3(int(scale))) for code, scale in zip(codes, sf)]
            self.codes = {int(bit): float(e2m1(int(code) & 15)) for bit, code in zip(bits, codes)}
            scales = [1.0] * len(bits)
        else:
            values = [f32_value(int(bit) << 16) for bit in bits]
            scales = [1.0] * len(bits)
        self.values = {int(bit): (float(value), float(scale)) for bit, value, scale in zip(bits, values, scales)}
        self.family = family
        self.normal_j = record["smallest_normal_unsaturated_j"]
        self.gs = gs

    def __call__(self, value):
        return (bf16(value), 1.0) if self.family == "value" else self.values[f32_bits(value) >> 16]

    def raw(self, value):
        return self.codes[f32_bits(value) >> 16]


def column_coverage(weights, scales, family, quantizer, *, peer=None, peer_scales=None, table=None, stage="dense", lut=(), router_weight=1.0):
    """Check all elements. Retain only the tested input recipe and coverage counts."""
    import numpy as np
    import torch
    from experiments.numerics_rows.row import x_products

    weights, scales = np.asarray(weights), np.asarray(scales).reshape(-1)
    values = np.asarray(alphabet(family, lut))
    position = np.searchsorted(values, weights)
    lower, upper = values[np.maximum(position - 1, 0)], values[np.minimum(position + 1, len(values) - 1)]
    left, right = position == 0, position == len(values) - 1
    positive_left, positive_right, negative_left, negative_right = [np.zeros_like(left) for _ in range(4)]
    recipes, x_false = [], 0
    for column in range(weights.shape[1]):
        w, lo, hi = weights[:, column], lower[:, column], upper[:, column]
        small = family == "value" and bool(np.any((np.abs(w) > 0) & (np.abs(w) < 2**-126)))
        gate_max = float(np.max(np.abs((w if stage == "gate" else peer[:, column]) * scales))) if peer is not None else None
        inputs = sweep_values(family, gs=quantizer.gs, normal_j=quantizer.normal_j,
                              small_product=small, gate_max=gate_max)
        tried = []
        for activation in inputs:
            aq, a_scale = quantizer(activation)
            product = aq * w
            valid = x_products(torch.from_numpy(product[:, None])).numpy()
            if peer is not None:
                valid &= x_products(torch.from_numpy((aq * peer[:, column])[:, None])).numpy()
            x_false += int((~valid).sum())
            if not valid.all():
                continue
            def evaluate(candidate):
                acc = (aq * candidate).astype(np.float32)
                acc = (acc * np.float32(a_scale)).astype(np.float32)
                acc = (acc * scales).astype(np.float32)
                if peer is None:
                    return bf16_array((acc * np.float32(router_weight)).astype(np.float32))
                other = ((aq * peer[:, column]).astype(np.float32) * np.float32(a_scale)).astype(np.float32)
                other = (other * peer_scales).astype(np.float32)
                g, u = (acc, other) if stage == "gate" else (other, acc)
                gf, uf = np.minimum(bf16_array(g), table.limit), np.clip(bf16_array(u), -table.limit, table.limit)
                if not np.isfinite(gf).all():
                    return np.full(gf.shape, np.nan)
                d = np.asarray([table(float(x)) for x in gf], dtype=np.float32)
                return bf16_array((d * uf).astype(np.float32))
            actual, below, above = evaluate(w), evaluate(lo), evaluate(hi)
            legal = np.isfinite(actual) & np.isfinite(below) & np.isfinite(above)
            if not legal.all():
                continue
            if stage == "gate":
                gf = bf16_array(((aq * w).astype(np.float32) * np.float32(a_scale) * scales).astype(np.float32))
                legal &= ((gf > 0) & (gf < table.limit)) | (actual == 0)
            if stage == "up":
                g = bf16_array(((aq * peer[:, column]).astype(np.float32) * np.float32(a_scale) * peer_scales).astype(np.float32))
                legal &= np.asarray([table(float(min(x, table.limit))) != 0 for x in g])
            lhit, rhit = legal & (actual != below), legal & (actual != above)
            pl, pr = (positive_left, positive_right) if activation > 0 else (negative_left, negative_right)
            pl[:, column] |= lhit
            pr[:, column] |= rhit
            if stage == "gate":
                lhit &= (actual != 0) | (positive_left[:, column] & negative_left[:, column])
                rhit &= (actual != 0) | (positive_right[:, column] & negative_right[:, column])
            left[:, column] |= lhit
            right[:, column] |= rhit
            tried.append(activation)
            if (left[:, column] & right[:, column]).all():
                break
        recipes.append({"kind": "sweep", "column": column, "values": tried, "stage": stage})
    both = left & right
    return {"stage": stage, "elements": weights.size, "lower_excluded": int(left.sum()), "upper_excluded": int(right.sum()),
            "both": int(both.sum()), "by_sign": {"positive": [int(positive_left.sum()), int(positive_right.sum())],
            "negative": [int(negative_left.sum()), int(negative_right.sum())]}, "complete": bool(both.all()), "x_false_candidates": x_false, "recipes": recipes,
            "uncovered": np.argwhere(~both)[:512].tolist()}


def separation_route(m, experts, expert):
    """Choose one real route from the count-one census for this expert."""
    selected = list(range(expert // 8 * 8, min(expert // 8 * 8 + 8, experts)))
    ids, weights = routing(m, experts, 1, selected)
    flat = [e for row in ids for e in row]
    index = flat.index(expert)
    return {"M": m, "count": 1, "selected": selected, "flat_index": index,
            "weight": f32_value(f32_bits(weights[index // 8][index % 8]))}


def routed_separation(weight, scale, family, quantizer, *, expert, experts, configs, **kwargs):
    records = []
    for config in configs:
        m = config["M"]
        if m not in (1, 64, 2048):
            continue
        route = separation_route(m, experts, expert)
        records.append({**column_coverage(weight, scale, family, quantizer, router_weight=route["weight"], **kwargs),
                        "route": route})
    return {**records[0], "complete": all(record["complete"] for record in records), "by_m": records}



def exact_chunks(weight, quantizer, configs, peer=None):
    import torch
    from experiments.numerics_rows.row import x_products
    aq, _ = quantizer(1.0)
    chunk = configs[0]["chunk_size"]
    pairs = chunk_pairs(weight.shape[1], chunk)
    split_cases = [split_columns(weight.shape[1], chunk, s) for s in sorted({c["k_split"] for c in configs}) if s > 1]
    records = []
    for columns in [*pairs, *split_cases]:
        mask = x_products(weight[:, list(columns)].double() * aq)
        if peer is not None:
            mask &= x_products(peer[:, list(columns)].double() * aq)
        records.append({"columns": list(columns), "x_elements": int(mask.sum()), "holds_for_a_row": bool(mask.any())})
    return records

def checkpoint_routed(store, entry):
    from types import SimpleNamespace
    from experiments.routed_pair_oracle import load_wires, load_input_scales

    scheme = store.schemes[entry["name"]]
    experts = int(scheme["experts"])
    if "wire_root" in entry:
        args = SimpleNamespace(wire_root=entry["wire_root"], layer=entry["layer"], scales=entry.get("scales"))
        blobs, _ = load_wires(args, entry["wire_family"], list(range(experts)))
        scales = load_input_scales(args, list(range(experts)))[0] if scheme["family"] == "TESSERA_NVFP4" else {}
    else:
        blobs = {p: [store.get(f"{entry['name']}.{e}.{p}.wire").contiguous().numpy().tobytes()
                     for e in range(experts)] for p in ("gate_proj", "up_proj", "down_proj")}
        scales = {p: [float(store.get(f"{entry['name']}.{e}.{p}.input_global_scale")) for e in range(experts)]
                  for p in blobs} if scheme["family"] == "TESSERA_NVFP4" else {}
    return scheme, blobs, scales


def projection_scheme(scheme, projection, wire):
    from tessera.fused import parse_fused
    from tessera.unit_artifact import parse_unit_metadata

    member, = parse_fused(wire)
    meta = parse_unit_metadata(member.blob)
    return {**{k: scheme[k] for k in ("family", "grid", "body", "plane")},
            "q256": int(meta.manifest.branch.root_q256) // int(meta.grid.arity), "rows": int(meta.rows),
            "columns": int(meta.columns), "wire_bytes": len(wire), "roles": [[projection, int(meta.rows)]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "population", "q1", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image", choices=("G", "N"), required=True)
    args = p.parse_args()
    check_source(args.source)
    check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    import tessera
    import torch
    from experiments.dense_fused_oracle import Store
    from tessera.lane_planes import lut_scale_bytes

    torch.set_num_threads(1)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    population, q1 = (json.loads(p.read_text()) for p in (args.population, args.q1))
    contract = json.loads((args.source / "src/tessera/serving/runtime_contract.json").read_text())
    manifest = {"schema": "tessera.numerics_cases.v1", "population_sha256": sha256(args.population),
                "q1_table_sha256": sha256(args.q1), "modules": [], "complete": True}
    for entry in population["modules"]:
        store = Store(entry["root"])
        scheme = store.schemes[entry["name"]]
        family = {"TESSERA_FP8": "e4m3", "TESSERA_NVFP4": "e2m1", "TESSERA_BF16": "value"}[scheme["family"]]
        structure = scheme.get("structure", "dense")
        facts = {"name": entry["name"], "root": entry["root"], "family": family, "structure": structure,
                 "roles": [n for n, _ in scheme.get("roles", [])], "experts": int(scheme.get("experts", 1)),
                 "rows": int(scheme["rows"]) if structure == "dense" else int(scheme["groups"]["w13"]["rows"]),
                 "cols": int(scheme["columns"]) if structure == "dense" else int(scheme["groups"]["w13"]["columns"]),
                 "complete": True}
        manifest["modules"].append(facts)
        if structure == "routed_moe":
            _, blobs, input_scales = checkpoint_routed(store, entry)
            facts["wire_sha256"] = {p: [hashlib.sha256(blob).hexdigest() for blob in wires] for p, wires in blobs.items()}
            from tessera.serving.nvfp4_moe_route import _static_input_global
            gs13 = float(_static_input_global(torch.tensor(input_scales["gate_proj"] + input_scales["up_proj"], dtype=torch.float32), "cpu")) if family == "e2m1" else 1.0
            gs2 = float(_static_input_global(torch.tensor(input_scales["down_proj"], dtype=torch.float32), "cpu")) if family == "e2m1" else 1.0
            groups = [{p: (blobs[p][e], projection_scheme(scheme, p, blobs[p][e]), gs2 if p == "down_proj" else gs13)
                       for p in blobs} for e in range(scheme["experts"])]

        else:
            wire = store.wire(entry["name"]).contiguous().numpy().tobytes()
            gs = float(store.get(entry["name"] + ".input_global_scale")) if family == "e2m1" else 1.0
            groups = [{"dense": (wire, scheme, gs)}]

            facts["wire_sha256"] = hashlib.sha256(wire).hexdigest()
        table = DTable(next(t for t in q1["tables"] if t["key"] == entry["library"])) if structure == "routed_moe" else None
        facts["cuts"] = []
        for size, rank in ((1, 0), (2, 0), (2, 1)):
            local = {"structure": structure, "family": family, "tp": {"size": size, "rank": rank}, "roles": [], "complete": True}
            facts["cuts"].append(local)
            # Ask the loader's pure shard planner for each original projection.
            for expert, group in enumerate(groups):
                decoded = []
                stamps = {}
                plans = {}
                for projection, (wire, declaration, gs) in group.items():
                    from tessera.serving.sharding import plan_shard
                    is_down = projection == "down_proj" or structure == "dense" and entry["name"].endswith("down_proj")
                    role_rows = [int(r) for _, r in declaration["roles"]]
                    plan = plan_shard(entry["name"], roles=declaration["roles"], columns=declaration["columns"],
                        out_partitions=role_rows if is_down else [r // size for r in role_rows],
                        in_size=declaration["columns"] // size if is_down else declaration["columns"], tp_rank=rank, tp_size=size,
                        input_size=declaration["columns"], output_size=sum(role_rows))
                    from tessera.serving.scheme import parse_tessera_blob_for_scheme
                    whole = parse_tessera_blob_for_scheme(wire, declaration, entry["name"], device="cpu")
                    stamps.update({name: wire_stamp(parsed) for name, parsed in whole})
                    del whole
                    for name, w, scale, parsed in decode_roles(wire, declaration, entry["name"], plan=plan):
                        decoded.append((name, w, scale, parsed, gs))
                        plans[name] = {"cut_axis": plan.axis, "cut_range": [plan.role(name).lo, plan.role(name).hi]}
                if expert == 0:
                    local_rows = sum(w.shape[0] for _, w, _, _, _ in decoded) if structure == "dense" else next(w.shape[0] for name, w, _, _, _ in decoded if name == "gate_proj")
                    local_cols = decoded[0][1].shape[1]
                    local["m_configs"] = configurations(local_rows, local_cols, entry["library"], population["sms"], structure)
                mapping = {name: (weight, scale) for name, weight, scale, _, _ in decoded}
                for name, weight, scale, parsed, gs in decoded:
                    quantizer = QuantizerTable(q1, family, gs)
                    stage = "gate" if structure == "routed_moe" and name == "gate_proj" else "up" if structure == "routed_moe" and name == "up_proj" else "down" if structure == "routed_moe" else "dense"
                    peer = mapping["up_proj" if stage == "gate" else "gate_proj"] if stage in ("gate", "up") else None
                    ratio = (scale.float() / torch.tensor(gs, dtype=torch.float32)) if family == "e2m1" else scale
                    if ratio.numel() == 1:
                        ratio = ratio.expand(weight.shape[0])
                    peer_scale = None if peer is None else peer[1].float() / (gs if family == "e2m1" else 1)
                    if peer_scale is not None and peer_scale.numel() == 1:
                        peer_scale = peer_scale.expand(weight.shape[0])
                    lut = lut_scale_bytes(parsed.unit.scale_lut, "cpu").view(torch.float8_e4m3fn).float().tolist() if family == "e2m1" else ()
                    separation_fn = routed_separation if stage == "down" else column_coverage
                    route_args = {"expert": expert, "experts": len(groups), "configs": local["m_configs"]} if stage == "down" else {}
                    separated = separation_fn(weight.numpy(), ratio.numpy(), family, quantizer, **route_args,
                        peer=None if peer is None else peer[0].numpy(), peer_scales=None if peer_scale is None else peer_scale.numpy(),
                        table=table, stage=stage, lut=lut)
                    chunks = exact_chunks(weight, quantizer, local["m_configs"], None if peer is None else peer[0])
                    stamp = stamps[name]
                    format_entry = next(f for f in contract["formats"] if f["family"] == entry["contract_family"])
                    stamped = stamp in format_entry["attested_wire"]
                    if not stamped and "allowable_rungs" in format_entry:
                        rule = format_entry["allowable_rungs"]
                        lo, hi = rule["range_q256"]
                        expected = {**rule["wire"], "q256": stamp["q256"]}
                        stamped = stamp == expected and lo <= stamp["q256"] <= hi and (stamp["q256"] - lo) % rule["step_q256"] == 0
                    file = args.out / f"{len(manifest['modules'])}-tp{size}-{rank}-{expert}-{name}.pt"
                    torch.save({"weight": weight, "scale": scale}, file)
                    index = scale_index_manifest(weight, ratio, family, stage, table, peer, peer_scale, quantizer, len(groups), expert, local["m_configs"])
                    local["roles"].append({"name": name, "expert": expert, "gs": gs, "reader_path": str(file), "reader_sha256": sha256(file),
                        "wire_stamp": stamp, "matches_attested_wire": stamped, "rows": weight.shape[0], "cols": weight.shape[1],
                        "runs_table": sorted(set(int(r) for r in parsed.unit.rates)), "lut": lut, **plans[name],
                        "separation": separated, "scale_index": index, "chunks": chunks})
                    local["complete"] &= stamped and separated["complete"] and all(i["reads"] == i["reads_covered"] for i in index) and all(c["holds_for_a_row"] for c in chunks)
                    facts["q256"] = stamp["q256"]
                    write_json(args.out / "case_manifest.json", manifest)
            facts["complete"] &= local["complete"]
        manifest["complete"] &= facts["complete"]
        write_json(args.out / "progress.json", {"arm": population.get("arm", args.population.stem),
            "modules_completed": len(manifest["modules"]), "last_module": entry["name"],
            "reference_seconds": time.monotonic() - started,
            "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
    manifest.update(reference_seconds=time.monotonic() - started,
                    host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(args.out / "case_manifest.json", manifest)


def scale_index_manifest(weight, scale, family, stage, table, peer, peer_scale, quantizer, experts, expert, configs):
    """Generate read tuples on exact one-hot cases, including activation scale arrays."""
    import numpy as np
    from experiments.numerics_rows.scale_index import make_plan
    from experiments.numerics_rows.reference import rn32

    w = weight.numpy()
    other = None if peer is None else peer[0].numpy()
    scales = scale.numpy().reshape(-1)
    other_scales = None if peer_scale is None else peer_scale.numpy().reshape(-1)
    mode0 = stage in ("gate", "up")
    if stage == "up":
        return []  # The gate role owns the joint gate/up array and its read tuples.
    rows, cols = w.shape
    target = "ratio" if family == "e2m1" else "row"
    cases = []
    for row in range(rows if target == "row" else 1):
        candidates = np.flatnonzero((w[row] != 0) & ((other[row] != 0) if mode0 else True))
        column = int(candidates[0]) if len(candidates) else 0
        source = sweep_values(family, gs=quantizer.gs, normal_j=quantizer.normal_j,
                              small_product=family == "value" and abs(w[row, column]) < 2**-126)[0]
        if mode0 and w[row, column] < 0:
            source = -source
        aq, a_scale = quantizer(source)
        g, u = rn32(F(aq) * F(float(w[row, column]))), rn32(F(aq) * F(float(other[row, column]))) if mode0 else 0
        index = expert * rows + row if target == "row" else expert
        count = experts * rows if target == "row" else experts
        read = [index, count + index] if mode0 else [index]
        cases.append({"read": read, "g": g, "u": u, "a_scale": a_scale, "g_scale": float(scales[min(row, len(scales)-1)]),
                      "u_scale": float(other_scales[min(row, len(other_scales)-1)]) if mode0 else 1.0,
                      "scale": target, "column": column, "input": source, "output_row": row, "input_row": 0})
    n = count * (2 if mode0 else 1)
    metadata = {"N": n, "target": target, "mode": 0 if mode0 else 2, "expert": expert, "projection": "gate_proj" if mode0 else "down_proj"}
    plans = [] if experts > 1 and not mode0 else [make_plan(target, n, cases, table=table, gate=range(count) if mode0 else (), up=range(count, n) if mode0 else (), metadata=metadata)]
    if family == "value":
        activation_targets = []
    else:
        activation_targets = ["a_s" if family == "e4m3" else "sfa"]
    if not mode0 and experts > 1:
        activation_targets.append("router")
    for config in configs:
        m = config["M"]
        routes = m * 8 if experts > 1 and not mode0 else m
        selected_routes, router_weights = list(range(routes)), [1.0] * routes
        if experts > 1 and not mode0:
            ids, routing_weights = routing(m, experts, m, expert)
            flat = [e for row in ids for e in row]
            flat_weights = [v for row in routing_weights for v in row]
            order = sorted(range(len(flat)), key=flat.__getitem__)
            selected_routes = [i for i, index in enumerate(order) if flat[index] == expert]
            router_weights = [flat_weights[index] for index in order]
            local_cases = [{**c, "input_row": selected_routes[0], "router_weight": f32_value(f32_bits(router_weights[selected_routes[0]]))} for c in cases]
            plans.append(make_plan(target, n, local_cases, table=table, metadata={**metadata, "M": m}))
        for name in activation_targets:
            if experts > 1 and (mode0 and expert != 0 or not mode0 and expert >= 8):
                continue
            if experts > 1 and not mode0:
                ids, routing_weights = routing(m, experts, m, list(range(8)))
                flat = [e for row in ids for e in row]
                flat_weights = [v for row in routing_weights for v in row]
                order = sorted(range(len(flat)), key=flat.__getitem__)
                selected_routes = [i for i, index in enumerate(order) if flat[index] == expert]
                router_weights = [flat_weights[index] for index in order]
            groups = cols // 16 if name == "sfa" else 1
            selected = selected_routes
            read_cases = []
            for input_row in selected:
                for group in range(groups):
                    candidates = np.argwhere((w[:, group*16:(group+1)*16] != 0) & ((other[:, group*16:(group+1)*16] != 0) if mode0 else True)) if name == "sfa" else np.argwhere((w != 0) & ((other != 0) if mode0 else True))
                    row, column = (int(x) for x in candidates[0]) if len(candidates) else (0, 0)
                    if name == "sfa":
                        column += group * 16
                    source = sweep_values(family, gs=quantizer.gs, normal_j=quantizer.normal_j)[0]
                    if mode0 and w[row, column] < 0:
                        source = -source
                    aq, a_scale = quantizer(source)
                    if name == "sfa":
                        # Q1 retains the packed nibble independently of its block scale.
                        aq = quantizer.raw(source)
                    index = input_row * groups + group
                    read_cases.append({"read": [index, index] if mode0 else [index], "g": rn32(F(aq) * F(float(w[row, column]))),
                                       "u": rn32(F(aq) * F(float(other[row, column]))) if mode0 else 0,
                                       "a_scale": 1.0 if name in ("a_s", "sfa") else a_scale,
                                       "g_scale": float(scales[min(row, len(scales)-1)]), "u_scale": float(other_scales[min(row, len(other_scales)-1)]) if mode0 else 1.0,
                                       "scale": name, "column": column, "input": source, "output_row": row, "input_row": input_row,
                                       "router_weight": f32_value(f32_bits(router_weights[input_row]))})
            metadata = {"N": routes * groups, "M": m, "target": name, "mode": 0 if mode0 else 2,
                        "expert": expert, "projection": "gate_proj" if mode0 else "down_proj", "shared": mode0,
                        **({"route_experts": list(range(8))} if experts > 1 and not mode0 else {})}
            plans.append(make_plan(name, routes * groups, read_cases, table=table, ue4m3=name == "sfa", metadata=metadata))
    return plans


if __name__ == "__main__":
    main()
