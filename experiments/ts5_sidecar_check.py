#!/usr/bin/env python3
"""Spot-check an exported routed-MoE Tessera checkpoint's sidecar BEFORE a serve.

A serve costs a GPU slot and twenty minutes.  Everything this reads is a
safetensors header or a JSON field, costs seconds, and each check is a failure
mode that would otherwise surface only at load -- after the teacher arm has
already been paid for:

  1. ``wire_stride`` is the MAXIMUM blob length in its group.  The exporter
     derives it from the lengths it wrote and ``unpack_moe_wires`` re-derives it
     at load, so this recomputes it a THIRD time, from the shard headers alone,
     with no decoder code in common with either. Every expert projection must
     be present exactly once. A stride below or above the real maximum would
     be refused by the loader.
  2. every module left in BF16 is in ``ignore``.  A module that is neither
     declared nor ignored is a ``TesseraConfig`` refusal at load.
  3. the serving manifest asks for no kernel lane.  A stamped GEMV lane makes
     the census refuse on engagement in ``resident`` mode however the routes
     come out, which would make the census unreadable rather than negative.

usage: ts5_sidecar_check.py <exported-dir>
"""
import json
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera.serving.scheme import MOE_GROUP_SHARDS
from tessera.serving_parts import validate_explicit_plan


def headers(d):
    """``{tensor name: (dtype, shape, nbytes)}`` across every shard, header-only."""
    out = {}
    owners = {}
    index = d / "model.safetensors.index.json"
    shards = (sorted({d / name for name in
                      json.loads(index.read_text())["weight_map"].values()})
              if index.exists() else sorted(d.glob("*.safetensors")))
    for shard in shards:
        with open(shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if name in owners:
                raise ValueError(
                    f"duplicate tensor name {name!r} in {owners[name]} and {shard}; "
                    "each checkpoint tensor must have exactly one shard owner")
            beg, end = meta["data_offsets"]
            out[name] = (meta["dtype"], meta["shape"], end - beg)
            owners[name] = shard
    return out


def main(path, plan_json=None):
    d = Path(path)
    cfg = json.loads((d / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    groups = qc.get("config_groups") or {}
    ignore = set(qc.get("ignore") or [])
    tensors = headers(d)
    problems = []

    print(f"quant_method={qc.get('quant_method')!r} "
          f"config_groups={len(groups)} ignore={len(ignore)} tensors={len(tensors)}")

    moe = {k: g for k, g in groups.items()
           if (g.get("scheme") or {}).get("structure") == "routed_moe"}
    dense = {k: g for k, g in groups.items() if k not in moe}
    print(f"routed_moe groups={len(moe)} other groups={len(dense)}")

    # --- 1. the stride is the max blob, recomputed from the headers ----------
    for key, g in sorted(moe.items()):
        scheme = g["scheme"]
        stack = (g.get("targets") or [None])[0]
        print(f"  {key}: target={stack} experts={scheme.get('experts')} "
              f"grid={scheme.get('grid')} body={scheme.get('body')} "
              f"plane={scheme.get('plane')}")
        for gname, spec in sorted((scheme.get("groups") or {}).items()):
            shards = MOE_GROUP_SHARDS.get(gname)
            if shards is None:
                problems.append(f"{key}: unknown group {gname!r}")
                continue
            roles = spec.get("roles") or []
            if len(roles) != len(shards):
                problems.append(f"{key}/{gname}: roles do not match runtime shards {shards}")
                continue
            lens = []
            for expert in range(scheme["experts"]):
                for shard, (role, _rows) in zip(shards, roles):
                    # LFM preserves w1/w3/w2; other checkpoints spell the
                    # same runtime shards gate_proj/up_proj/down_proj.
                    candidates = {f"{stack}.{expert}.{name}.wire"
                                  for name in (role, shard)}
                    found = sorted(candidates.intersection(tensors))
                    if len(found) != 1:
                        problems.append(
                            f"{key}/{gname}: expert {expert} role {role} requires exactly "
                            f"one wire from {sorted(candidates)}, found {found}")
                        continue
                    lens.append(tensors[found[0]][2])
            stride = spec.get("wire_stride")
            if not lens:
                problems.append(f"{key}/{gname}: no wires found under {stack}")
                continue
            hi, lo = max(lens), min(lens)
            ok = stride == hi
            print(f"    {gname}: n={len(lens)} stride={stride} max={hi} "
                  f"spread={hi - lo} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                problems.append(f"{key}/{gname}: wire_stride {stride} != max blob {hi}")

    # --- 2. the BF16 remainder is declared ignored ---------------------------
    wires = [n for n in tensors if n.endswith(".wire")]
    bf16_modules = sorted({n.rsplit(".", 1)[0] for n, (dt, _s, _n) in tensors.items()
                           if dt in ("BF16", "F16") and n.endswith(".weight")})
    quantized = {n.rsplit(".", 1)[0] for n in wires}
    declared = set()
    for g in groups.values():
        declared.update(g.get("targets") or [])
    unaccounted = [m for m in bf16_modules
                   if m not in ignore
                   and m not in declared
                   and not any(m.startswith(f"{t}.") or t.startswith(f"{m}.")
                               for t in declared)
                   and not any(m == i or m.startswith(f"{i}.") for i in ignore)]
    print(f"wires={len(wires)} quantized modules={len(quantized)} "
          f"bf16 .weight modules={len(bf16_modules)} unaccounted={len(unaccounted)}")
    for m in unaccounted[:12]:
        print(f"    unaccounted: {m}")
    # An unaccounted BF16 module is only a refusal if the runtime OFFERS it to
    # the quant config; this script cannot know that, so it reports rather than
    # refuses.  The construction census is what says which are offered.

    # --- 3. no kernel lane demanded ------------------------------------------
    man = d / "tessera_serving_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        lanes = m.get("requires_lanes") or m.get("lanes") or []
        print(f"manifest requires_lanes={lanes}")
        if lanes:
            problems.append(f"manifest demands kernel lanes {lanes}: the census refuses "
                            "on engagement in resident mode whatever the routes are")
        identity = m.get("export_identity", {})
        embedded_plan = identity.get("options", {}).get("plan")
        explicit_plan = json.loads(Path(plan_json).read_text()) if plan_json is not None else None
        if explicit_plan is not None and embedded_plan is not None and explicit_plan != embedded_plan:
            problems.append("explicit --plan-json differs from the merged export identity plan")
        plan = explicit_plan if explicit_plan is not None else embedded_plan
        if plan is not None:
            try:
                validate_explicit_plan(plan, m.get("modules", {}), groups,
                                      source_tensors=identity.get("source", {}).get("tensors"))
                recorded = m.get("routed_moe", {}).get("quantized_stacks")
                actual = {name for name, record in m.get("modules", {}).items()
                          if record.get("structure") == "routed_moe"}
                if recorded is None or len(recorded) != len(actual) or set(recorded) != actual:
                    raise ValueError("explicit plan stack population disagrees with routed_moe summary")
                print(f"explicit plan: {len(actual)} routed stacks match config, manifest, "
                      "source projections and requested role grid/rung")
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(str(exc))
    else:
        print("manifest: absent (no lane demanded)")
        if plan_json is not None:
            problems.append("explicit --plan-json requires a serving manifest for coverage proof")

    print("PROBLEMS:" if problems else "NO PROBLEMS")
    for p in problems:
        print("  -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--plan-json", type=Path,
                        help="require this exact export plan; otherwise use the merged identity plan")
    args = parser.parse_args()
    sys.exit(main(args.checkpoint, plan_json=args.plan_json))
